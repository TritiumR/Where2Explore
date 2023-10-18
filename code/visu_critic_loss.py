import os
import shutil
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import utils
from env import Env
from camera import Camera

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

# setup env
env = Env()

# setup camera
cam = Camera(env, fixed_position=True)
env.set_controller_camera_pose(cam.pos[0], cam.pos[1], cam.pos[2], np.pi + cam.theta, -cam.phi)
mat33 = cam.mat44[:3, :3]

# load shape
object_urdf_fn = '../data/where2act_original_sapien_dataset/%s/mobility_vhacd.urdf' % eval_conf.shape_id
object_material = env.get_material(4, 4, 0.01)
state = eval_conf.state
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
gripper_forward_direction_camera = torch.tensor([[0., 1., 0.]]).to(device)
# print('forward shape: ', gripper_forward_direction_camera.shape)
gripper_forward_direction_camera = F.normalize(gripper_forward_direction_camera, dim=1)

for x, y in pcids:
    direction_cam = gt_nor[x, y, :3]
    direction_cam /= np.linalg.norm(direction_cam)
    # using normal direction of the random point as up direction
    action_direction_cam = -direction_cam

    # generate query
    up = torch.tensor(action_direction_cam.reshape(1, 3)).to(device)
    forward = gripper_forward_direction_camera
    left = torch.cross(up, forward)
    forward = torch.cross(left, up)
    forward = F.normalize(forward, dim=1)

    dirs1.append(up[0])
    dirs2.append(forward[0])

dirs1 = torch.stack(dirs1)
dirs2 = torch.stack(dirs2)

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

        fn = os.path.join(result_dir, 'pred')
        resultcolors = cmap(result)[:, :3]
        pccolors = pccolors * (1 - np.expand_dims(result, axis=-1)) + resultcolors * np.expand_dims(result, axis=-1)
        utils.export_pts_color_pts(fn,  pc[0].cpu().numpy(), pccolors)
        utils.export_pts_color_obj(fn,  pc[0].cpu().numpy(), pccolors)
        utils.render_pts_label_png(fn,  pc[0].cpu().numpy(), result)

        fn = os.path.join(result_dir, 'similarity')
        utils.render_pts_label_png(fn, pc[0].cpu().numpy(), confidence)

# close env
env.close()

