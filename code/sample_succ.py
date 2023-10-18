import os
import shutil
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import utils
from utils import get_global_position_from_camera
from sapien.core import Pose
from env import Env
from camera import Camera
from robots.panda_robot import Robot

from pointnet2_ops.pointnet2_utils import furthest_point_sample

import matplotlib.pyplot as plt
cmap = plt.cm.get_cmap("jet")

class ContactError(Exception):
    pass

# test parameters
parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--shape_id', type=str, help='shape id')
parser.add_argument('--start_epoch', type=int, help='start epoch')
parser.add_argument('--end_epoch', type=int, help='end epoch')
parser.add_argument('--model_version', type=str, help='model version')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--state', type=str, default='random-closed-middle')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if result_dir exists [default: False]')
parser.add_argument('--render_gt', action='store_true', default=False)
parser.add_argument('--AIP', action='store_true', default=False)
parser.add_argument('--load_dir', action='store_true', default=False)
eval_conf = parser.parse_args()

# load train config
train_conf = torch.load(os.path.join('logs', eval_conf.exp_name, 'conf.pth'))
primact_type = train_conf.primact_type

# set up device
device = torch.device(eval_conf.device)
print(f'Using device: {device}')

# create models
model_def = utils.get_model_module(eval_conf.model_version)
network = model_def.Network(train_conf.feat_dim)
network.to(device)
network.eval()
if eval_conf.AIP:
    AIP_def = utils.get_model_module('model_AIP')
    AIP = AIP_def.AIP(train_conf.feat_dim)
    AIP.to(device)
    AIP.eval()

# setup env
env = Env()

# setup camera
# cam = Camera(env, random_position=True)
cam = Camera(env, fixed_position=True)
#cam = Camera(env)
env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)
mat33 = cam.mat44[:3, :3]

# load shape
object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % eval_conf.shape_id
object_material = env.get_material(4, 4, 0.01)
state = eval_conf.state
# if np.random.random() < 0.5:
#     state = 'closed'
print('Object State: %s' % state)
joint_angles = env.load_object(object_urdf_fn, object_material, state=state)
cur_qpos = env.get_object_qpos()

# simulate some steps for the object to stay rest
still_timesteps = 0
wait_timesteps = 0
while still_timesteps < 50000 and wait_timesteps < 200000:
    env.step()
    env.render()
    cur_new_qpos = env.get_object_qpos()
    invalid_contact = False
    for c in env.scene.get_contacts():
        for p in c.points:
            if abs(p.impulse @ p.impulse) > 1e-4:
                invalid_contact = True
                break
        if invalid_contact:
            break
    if np.max(np.abs(cur_new_qpos - cur_qpos)) < 1e-6 and (not invalid_contact):
        still_timesteps += 1
    else:
        still_timesteps = 0
    cur_qpos = cur_new_qpos
    wait_timesteps += 1

if still_timesteps < 5000:
    print('Object Not Still!')
    env.close()
    exit(1)

### use the GT vision
rgb, depth = cam.get_observation()
object_link_ids = env.movable_link_ids
gt_movable_link_mask = cam.get_movable_link_mask(object_link_ids)
xs, ys = np.where(gt_movable_link_mask > 0)
# print('gt_movable_link_mask: ', gt_movable_link_mask.shape)
if len(xs) == 0:
    print('No Movable Pixel! Quit!')
    env.close()
    exit(1)
# print(len(xs))

# generate prediction on all points

# get original pc
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = cam.compute_camera_XYZA(depth)
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts, depth.shape[0], depth.shape[1])
# prepare input pc

# sample 30000 masked points
mask = (cam_XYZA[:, :, 3] > 0.5)
# print('mask: ', mask.shape)
pc = cam_XYZA[mask, :3]
grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)    # 2 x 448 x 448
pcids = grid_xy[:, mask].T
# print('pcids1: ', pcids.shape)
pc_movable = (gt_movable_link_mask > 0)[mask]
# print('pc_movable: ', pc_movable.shape)
idx = np.arange(pc.shape[0])
np.random.shuffle(idx)
while len(idx) < 30000:
    idx = np.concatenate([idx, idx])
idx = idx[:30000-1]
pc = pc[idx, :]
pc_movable = pc_movable[idx]
pcids = pcids[idx, :]
pc[:, 0] -= 5

x_mean = pc[:, 0].mean()
y_mean = pc[:, 1].mean()
z_mean = pc[:, 2].mean()
x_max = pc[:, 0].max()
y_max = pc[:, 1].max()
z_max = pc[:, 2].max()
x_min = pc[:, 0].min()
y_min = pc[:, 1].min()
z_min = pc[:, 2].min()
print('mean: ', x_mean, y_mean, z_mean)
print('max: ', x_max, y_max, z_max)
print('min: ', x_min, y_min, z_min)

pc = torch.from_numpy(pc).unsqueeze(0).to(device)

# furthest sample
input_pcid = furthest_point_sample(pc, train_conf.num_point_per_shape).long().reshape(-1)
pc = pc[:, input_pcid, :3]  # 1 x N x 3
pc_movable = pc_movable[input_pcid.cpu().numpy()]     # N
pcids = pcids[input_pcid.cpu().numpy()]
pccolors = rgb[pcids[:, 0], pcids[:, 1]]

# get action direction (normal direction of each point)
dirs1 = []
dirs2 = []
gt_nor = cam.get_normal_map()

# sample a random forward direction
gripper_forward_direction_camera = torch.tensor([[0., 1., 1.]]).to(device)
# print('forward shape: ', gripper_forward_direction_camera.shape)
gripper_forward_direction_camera = F.normalize(gripper_forward_direction_camera, dim=1)

for x, y in pcids:
    direction_cam = gt_nor[x, y, :3]
    direction_cam /= np.linalg.norm(direction_cam)
    # using normal direction of the random point as up direction
    # if 'pushing' in primact_type:
    #     direction_cam = - direction_cam
    action_direction_cam = -direction_cam

    # generate query
    up = torch.tensor(action_direction_cam.reshape(1, 3)).to(device)
    forward = gripper_forward_direction_camera
    left = torch.cross(up, forward)
    forward = torch.cross(left, up)
    forward = F.normalize(forward, dim=1)

    dirs1.append(up[0])
    dirs2.append(forward[0])
    # if dirs1 is None:
    #     dirs1 = up
    #     dirs2 = forward
    # else:
    #     torch.cat(dirs1, up)
    #     torch.cat(dirs2, forward)

dirs1 = torch.stack(dirs1)
dirs2 = torch.stack(dirs2)
# print('dirs: ', dirs1.shape, dirs2.shape)
# print('dirs: ', dirs1[0], dirs2[1])

for model_epoch in range(eval_conf.start_epoch, eval_conf.end_epoch + 1):
    # check if eval results already exist. If so, delete it.
    result_dir = os.path.join('logs', eval_conf.exp_name, f'visu_critic_heatmap-{eval_conf.shape_id}-model_epoch_{model_epoch}-{eval_conf.result_suffix}')
    if os.path.exists(result_dir):
        shutil.rmtree(result_dir)

    os.mkdir(result_dir)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(os.path.join(result_dir, 'rgb.png'))
    print(f'\nTesting under directory: {result_dir}\n')

    # load pretrained model
    print('Loading ckpt from ', os.path.join('logs', eval_conf.exp_name, 'ckpts'), model_epoch)
    data_to_restore = torch.load(os.path.join('logs', eval_conf.exp_name, 'ckpts', '%d-network.pth' % model_epoch))
    network.load_state_dict(data_to_restore, strict=False)
    print('DONE\n')

    # create AIP and load
    if eval_conf.AIP:
        print('Loading AIP ckpt from ', os.path.join('logs', eval_conf.exp_name, 'ckpts'), model_epoch)
        data_to_restore = torch.load(os.path.join('logs', eval_conf.exp_name, 'ckpts', '%d-AIP.pth' % model_epoch))
        AIP.load_state_dict(data_to_restore, strict=False)

    # push through pointnet
    feats = network.pointnet2(pc.repeat(1, 1, 2))[0].permute(1, 0)    # N x F

    # infer for all pixels
    with torch.no_grad():
        input_queries = torch.cat([dirs1, dirs2], dim=1)
        print(input_queries.shape)
        net = network.critic(feats, input_queries)
        pred = net.cpu()
        result = torch.sigmoid(net).cpu().numpy()
        if eval_conf.model_version == 'model_3d_conf':
            confidence = torch.sigmoid(network.confidence(feats, input_queries)).cpu().numpy()
        else:
            confidence = abs(2 * result - 1)
        print(result.sum())
        result *= pc_movable
        confidence *= pc_movable

        sample_point = result.argmax()

    # get gt_label for points in input pc
    # setup robot
    robot_urdf_fn = './robots/panda_gripper.urdf'
    robot_material = env.get_material(4, 4, 0.01)
    robot = Robot(env, robot_urdf_fn, robot_material, open_gripper=('pulling' in primact_type))

    gt_label = np.zeros(len(input_pcid))
    # print('gt_label: ', gt_label.shape)
    print(len(pcids[pc_movable]))
    forward = forward.cpu().numpy().reshape(3)
    final_dist = 0.1
    if primact_type == 'pushing-left' or primact_type == 'pushing-up':
        final_dist = 0.11

    # movable_pc = pc[pc_movable, :]
    # print(len(movable_pc))
    # interact_pcid = furthest_point_sample(movable_pc, 200).long().reshape(-1)
    pt_id = sample_point
    pt = pcids[pt_id]

    env.set_object_joint_angles(joint_angles)
    rgb_origin, _ = cam.get_observation()
    Image.fromarray((rgb_origin * 255).astype(np.uint8)).save(os.path.join(result_dir, 'rgb_origin.png'))
    # if pt_id >= 10:
    #     break
    # get pixel 3D position (world)
    x, y = pt
    print('id: ', pt_id, 'pt ', x, y)
    position_world_xyz1 = get_global_position_from_camera(cam, depth, y, x)
    position_world = position_world_xyz1[:3]

    # get target part
    env.set_target_object_part_actor_id(object_link_ids[gt_movable_link_mask[x, y] - 1])
    target_object_part_joint_id = env.target_object_part_joint_id
    joint_angle_lower = env.joint_angles_lower[target_object_part_joint_id]
    joint_angle_upper = env.joint_angles_upper[target_object_part_joint_id]
    start_target_part_qpos = env.get_target_part_qpos()

    # compute pose
    action_direction_world = cam.get_metadata()['mat44'][:3, :3] @ dirs1[pt_id].cpu().numpy().reshape(3)
    up = np.array(action_direction_world, dtype=np.float32)
    forward_world = cam.get_metadata()['mat44'][:3, :3] @ dirs2[pt_id].cpu().numpy().reshape(3)
    left = np.cross(up, forward_world)
    left /= np.linalg.norm(left)
    forward_world = np.cross(left, up)
    forward_world /= np.linalg.norm(forward_world)
    rotmat = np.eye(4).astype(np.float32)
    rotmat[:3, 0] = forward_world
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up

    final_rotmat = np.array(rotmat, dtype=np.float32)
    final_rotmat[:3, 3] = position_world - action_direction_world * final_dist
    final_pose = Pose().from_transformation_matrix(final_rotmat)

    start_rotmat = np.array(rotmat, dtype=np.float32)
    start_rotmat[:3, 3] = position_world - action_direction_world * 0.15
    start_pose = Pose().from_transformation_matrix(start_rotmat)

    action_direction = None
    if 'left' in primact_type:
        action_direction = forward_world
    elif 'up' in primact_type:
        action_direction = left

    if action_direction is not None:
        end_rotmat = np.array(rotmat, dtype=np.float32)
        end_rotmat[:3, 3] = position_world - action_direction_world * final_dist + action_direction * 0.05

    ### viz the EE gripper position
    # move back
    robot.robot.set_root_pose(start_pose)
    env.render()

    # activate contact checking
    env.start_checking_contact(robot.hand_actor_id, robot.gripper_actor_ids, 'pushing' in primact_type)

    ### main steps
    target_link_mat44 = env.get_target_part_pose().to_transformation_matrix()
    position_local_xyz1 = np.linalg.inv(target_link_mat44) @ position_world_xyz1

    valid = True
    try:
        if 'pushing' in primact_type:
            robot.close_gripper()
        elif 'pulling' in primact_type:
            robot.open_gripper()

        # approach
        robot.move_to_target_pose(final_rotmat, 2000)
        robot.wait_n_steps(2000)

        if 'pulling' in primact_type:
            robot.close_gripper()
            robot.wait_n_steps(2000)

        if 'left' in primact_type or 'up' in primact_type:
            robot.move_to_target_pose(end_rotmat, 2000)
            robot.wait_n_steps(2000)

        if primact_type == 'pulling':
            robot.move_to_target_pose(start_rotmat, 2000)
            robot.wait_n_steps(2000)

    except Exception:
        print('error')
        valid = False

    if valid:
        final_target_part_qpos = env.get_target_part_qpos()
        abs_motion = abs(final_target_part_qpos - start_target_part_qpos)
        tot_motion = joint_angle_upper - joint_angle_lower + 1e-8
        success = (abs_motion > train_conf.abs_thres) or (abs_motion / tot_motion > train_conf.rel_thres)
        if success:
            print('succ')
            rgb_target, _ = cam.get_observation()
            Image.fromarray((rgb_target * 255).astype(np.uint8)).save(os.path.join(result_dir, 'rgb_target.png'))
        else:
            print('fail')

    pred_succ = result > 0.5
    pred_succ_and = []
    for i in range(len(pred_succ)):
        if pred_succ[i] == 1.0 and gt_label[i] == 1.0:
            pred_succ_and.append(1.)
        else:
            pred_succ_and.append(0.)
    pred_succ_and = np.array(pred_succ_and)

    precise = pred_succ_and.sum() / pred_succ.sum()
    recall = pred_succ_and.sum() / gt_label.sum()
    F1 = 2 * (precise * recall) / (precise + recall)
    print('score from: ', result_dir)
    print('Score: ', precise, recall, F1)

# close env
env.close()

