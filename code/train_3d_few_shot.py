"""
    Train the Action Scoring Module only
"""

import os
import time
import sys
import shutil
import random
from time import strftime
from argparse import ArgumentParser
import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
from datagen import DataGen
from data import SAPIENVisionDataset
import utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))


def train(conf, ruler_data_list):
    # create training and validation datasets and data loaders
    data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction', 'gripper_forward_direction',
                     'result', 'cur_dir', 'shape_id', 'cnt_id', 'trial_id', 'is_original', 'obj_state', 'category', 'camera_direction']

    # load network model
    model_def = utils.get_model_module(conf.model_version)

    # create models
    network = model_def.Network(conf.feat_dim)
    utils.printout(conf.flog, '\n' + str(network) + '\n')

    # create optimizers
    network_opt = torch.optim.Adam(network.parameters(), lr=conf.lr, weight_decay=conf.weight_decay)

    # learning rate scheduler
    network_lr_scheduler = torch.optim.lr_scheduler.StepLR(network_opt, step_size=conf.lr_decay_every,
                                                           gamma=conf.lr_decay_by)

    # create logs
    if not conf.no_console_log:
        header = '     Time    Epoch     Dataset    Iteration    Progress(%)       LR    TotalLoss'
    if not conf.no_tb_log:
        # https://github.com/lanpa/tensorboard-pytorch
        from tensorboardX import SummaryWriter
        train_writer = SummaryWriter(os.path.join(conf.exp_dir, 'train'))
        val_writer = SummaryWriter(os.path.join(conf.exp_dir, 'val'))

    # send parameters to device
    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)

    # load dataset
    ruler_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num,
                                        abs_thres=conf.abs_thres, rel_thres=conf.rel_thres, dp_thres=conf.dp_thres,
                                        img_size=conf.img_size, no_true_false_equal=True,
                                        world_coordinate=conf.world_coordinate)

    ruler_dataset.load_data(ruler_data_list)

    ruler_dataloader = torch.utils.data.DataLoader(ruler_dataset, batch_size=conf.batch_size, shuffle=False,
                                                   pin_memory=True,
                                                   num_workers=0, drop_last=True, collate_fn=utils.collate_feats,
                                                   worker_init_fn=utils.worker_init_fn)

    train_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num,
                                        abs_thres=conf.abs_thres, rel_thres=conf.rel_thres, dp_thres=conf.dp_thres,
                                        img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal,
                                        world_coordinate=conf.world_coordinate)

    # create a data generator
    datagen = DataGen(conf.num_processes_for_datagen, conf.flog)

    # create a trial counter
    trial_counter = dict()

    # start training
    start_time = time.time()

    last_train_console_log_step, last_val_console_log_step = None, None
    correct_times = None

    start_epoch = 0

    # if load model
    if conf.load_model:
        data_to_restore = torch.load(os.path.join(conf.load_model, 'ckpts', '%d-network.pth' % conf.load_model_epoch))
        network.load_state_dict(data_to_restore)
        print('loading model from: ', os.path.join(conf.load_model, 'ckpts', '%d-network.pth' % conf.load_model_epoch))

    # train for every epoch
    for epoch in range(start_epoch, conf.epochs):
        ### collect data for the current epoch
        # sample conf
        sample_conf_list = []
        sample_conf_name = []
        sample_conf_value = []

        cur_sample_conf_dir = os.path.join(conf.data_dir, 'epoch-%04d_sample-conf' % epoch)
        utils.force_mkdir(cur_sample_conf_dir)
        ruler_batches = enumerate(ruler_dataloader, 0)

        for ruler_batch_ind, batch in ruler_batches:
            network.eval()
            with torch.no_grad():
                categories = batch[data_features.index('category')]
                input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(conf.device)  # B x 3N x 3
                input_pxids = torch.cat(batch[data_features.index('pc_pxids')], dim=0).to(conf.device)  # B x 3N x 2
                input_movables = torch.cat(batch[data_features.index('pc_movables')], dim=0).to(conf.device)  # B x 3N
                batch_size = input_pcs.shape[0]

                input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
                input_pcid2 = furthest_point_sample(input_pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
                input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
                whole_pxids = input_pxids[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
                whole_movables = input_movables[input_pcid1, input_pcid2].reshape(batch_size, conf.num_point_per_shape)

                # sample a random EE orientation
                random_up = torch.randn(conf.batch_size, 3).float().to(conf.device)
                random_forward = torch.randn(conf.batch_size, 3).float().to(conf.device)
                random_left = torch.cross(random_up, random_forward)
                random_forward = torch.cross(random_left, random_up)
                random_dirs1 = F.normalize(random_up, dim=1).float()
                random_dirs2 = F.normalize(random_forward, dim=1).float()

                # forward through the network
                _, _, whole_feats = network(input_pcs, random_dirs1, random_dirs2)  # B x 2, B x F x N

                # test over the entire image
                whole_pc_confidences1 = network.inference_whole_pc_confidence(whole_feats, random_dirs1, random_dirs2)  # B x N
                whole_pc_confidences2 = network.inference_whole_pc_confidence(whole_feats, -random_dirs1, random_dirs2)  # B x N

                # add to the sample_conf_list if wanted
                ss_cur_dir = batch[data_features.index('cur_dir')]
                ss_shape_id = batch[data_features.index('shape_id')]
                ss_obj_state = batch[data_features.index('obj_state')]
                ss_is_original = batch[data_features.index('is_original')]

                for i in range(conf.batch_size):
                    key = ss_shape_id[i] + '-' + ss_obj_state[i]
                    if (ss_is_original[i]) and (key not in sample_conf_name):
                        sample_conf_name.append(key)
                        if ss_shape_id[i] not in trial_counter.keys():
                            trial_counter[ss_shape_id[i]] = conf.num_interaction_data_offline

                        gt_movable = whole_movables[i].cpu().numpy()

                        whole_pc_confidence1 = whole_pc_confidences1[i].cpu().numpy() * gt_movable
                        whole_pc_confidence_max1 = np.max(whole_pc_confidence1)
                        whole_pc_confidence_sum1 = np.sum(whole_pc_confidence1) + 1e-12
                        whole_pc_confidence1[gt_movable == 0] = 1
                        whole_pc_confidence_min1 = np.min(whole_pc_confidence1)

                        whole_pc_confidence2 = whole_pc_confidences2[i].cpu().numpy() * gt_movable
                        whole_pc_confidence_max2 = np.max(whole_pc_confidence2)
                        whole_pc_confidence_sum2 = np.sum(whole_pc_confidence2) + 1e-12
                        whole_pc_confidence2[gt_movable == 0] = 1
                        whole_pc_confidence_min2 = np.min(whole_pc_confidence2)

                        random_dir1 = random_dirs1[i].cpu().numpy()
                        random_dir2 = random_dirs2[i].cpu().numpy()

                        # sample <X, Y> on each img
                        if whole_pc_confidence_sum1 < whole_pc_confidence_sum2:
                            ptid = np.argmin(whole_pc_confidence1)
                        else:
                            ptid = np.argmin(whole_pc_confidence2)

                        whole_pc_confidence_min_sum = min(whole_pc_confidence_sum1, whole_pc_confidence_sum2)

                        X = whole_pxids[i, ptid, 0].item()
                        Y = whole_pxids[i, ptid, 1].item()

                        # add job to the queue
                        str_cur_dir1 = ',' + ','.join(['%f' % elem for elem in random_dir1])
                        str_cur_dir2 = ',' + ','.join(['%f' % elem for elem in random_dir2])

                        sample_conf_list.append((conf.offline_data_dir, str_cur_dir1, str_cur_dir2,
                                                 ss_cur_dir[i].split('/')[-1], cur_sample_conf_dir, X, Y))

                        sample_conf_value.append(whole_pc_confidence_min_sum / gt_movable.sum())

                        print(f'check category {categories[i]} shape {ss_shape_id[i]} state {ss_obj_state[i]}')

                        print('min_sum: ', whole_pc_confidence_min_sum / gt_movable.sum())
                        print('min-max ', min(whole_pc_confidence_min1, whole_pc_confidence_min2),
                              max(whole_pc_confidence_max1, whole_pc_confidence_max2))

        sample_conf_value = np.array(sample_conf_value)
        sample_idx = np.argsort(sample_conf_value)[: conf.sample_max_num]
        for idx in sample_idx:
            item = sample_conf_list[idx]
            shape_id, category, _, _, _ = item[3].split('_')
            datagen.add_one_recollect_job(item[0], item[1], item[2], item[3], item[4], item[5], item[6], conf.num_interaction_data_offline * (epoch + 1))
            state = sample_conf_name[idx].split('-')[-1]
            trial_counter[shape_id] += 1
            print('category %s shape %s state %s confidence %d', category, shape_id, state, sample_conf_value[idx])

        torch.save(sample_conf_list, os.path.join(conf.exp_dir, 'ckpts', '%d-sample_conf_list.pth' % epoch))

        # start all jobs
        datagen.start_all()
        utils.printout(conf.flog, f'  [ {strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} Started generating epoch-{epoch} data ]')

        utils.printout(conf.flog, f'  [{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} Waiting epoch-{epoch} data ]')
        train_data_list = datagen.join_all()
        utils.printout(conf.flog, f'  [{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} Gathered epoch-{epoch} data ]')

        cur_data_folders = []
        for item in train_data_list:
            item = '/'.join(item.split('/')[:-1])
            if item not in cur_data_folders:
                cur_data_folders.append(item)
        for cur_data_folder in cur_data_folders:
            with open(os.path.join(cur_data_folder, 'data_tuple_list.txt'), 'w') as fout:
                for item in train_data_list:
                    if cur_data_folder == '/'.join(item.split('/')[:-1]):
                        fout.write(item.split('/')[-1] + '\n')

        train_dataset.load_data(train_data_list)

        utils.printout(conf.flog, str(train_dataset))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True,
                                                       pin_memory=True,
                                                       num_workers=0, drop_last=True, collate_fn=utils.collate_feats,
                                                       worker_init_fn=utils.worker_init_fn)

        train_num_batch = len(train_dataloader)
        print('train_num_batch: ', train_num_batch)
        train_batches = enumerate(train_dataloader, 0)

        # record correctness
        if epoch == start_epoch and correct_times is None:
            correct_times = dict()
            correct_times['size'] = conf.num_interaction_data_offline * (conf.epochs + 10)
            print("correct_times_size: ", correct_times['size'])

        ### print log
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        ### train critic for every batch
        for train_batch_ind, batch in train_batches:
            train_step = epoch * train_num_batch + train_batch_ind

            log_console = not conf.no_console_log and (last_train_console_log_step is None or
                                                       train_step - last_train_console_log_step >= conf.console_log_interval)
            if log_console:
                last_train_console_log_step = train_step

            # save checkpoint
            if train_batch_ind == 0:
                with torch.no_grad():
                    utils.printout(conf.flog, 'Saving checkpoint ...... ')
                    torch.save(network.state_dict(), os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % epoch))
                    torch.save(network_opt.state_dict(),
                               os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % epoch))
                    torch.save(network_lr_scheduler.state_dict(),
                               os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % epoch))
                    torch.save(train_dataset, os.path.join(conf.exp_dir, 'ckpts', '%d-train_dataset.pth' % epoch))
                    utils.printout(conf.flog, 'DONE')

                    np.save(os.path.join(conf.exp_dir, 'ckpts', '%d-correct_times.pth' % epoch), correct_times)
                    utils.printout(conf.flog, 'correct_times-DONE')

            # set models to training mode
            network.train()

            # forward pass (including logging)
            whole_feats, whole_pcs, whole_pxids, whole_movables = forward(batch=batch,
                                                                          data_features=data_features,
                                                                          network=network, conf=conf,
                                                                          step=train_step, epoch=epoch,
                                                                          batch_ind=train_batch_ind,
                                                                          num_batch=train_num_batch,
                                                                          start_time=start_time,
                                                                          log_console=log_console,
                                                                          log_tb=not conf.no_tb_log,
                                                                          tb_writer=train_writer,
                                                                          lr=network_opt.param_groups[0]['lr'],
                                                                          correctness=correct_times,
                                                                          network_opt=network_opt,
                                                                          network_lr_scheduler=network_lr_scheduler)



def forward(batch, data_features, network, conf,
            step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
            log_console=False, log_tb=False, tb_writer=None, lr=None, correctness=None, network_opt=None, network_lr_scheduler=None):
    shape_id = batch[data_features.index('shape_id')]
    cnt_id = batch[data_features.index('cnt_id')]
    trial_id = batch[data_features.index('trial_id')]

    # prepare input
    input_pcs = torch.cat(batch[data_features.index('pcs')], dim=0).to(conf.device)  # B x 3N x 3
    input_pxids = torch.cat(batch[data_features.index('pc_pxids')], dim=0).to(conf.device)  # B x 3N x 2
    input_movables = torch.cat(batch[data_features.index('pc_movables')], dim=0).to(conf.device)  # B x 3N
    batch_size = input_pcs.shape[0]

    input_pcid1 = torch.arange(batch_size).unsqueeze(1).repeat(1, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcid2 = furthest_point_sample(input_pcs, conf.num_point_per_shape).long().reshape(-1)  # BN
    input_pcs = input_pcs[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
    input_pxids = input_pxids[input_pcid1, input_pcid2, :].reshape(batch_size, conf.num_point_per_shape, -1)
    input_movables = input_movables[input_pcid1, input_pcid2].reshape(batch_size, conf.num_point_per_shape)

    input_dirs1 = torch.cat(batch[data_features.index('gripper_direction')], dim=0).to(conf.device)  # B x 3
    input_dirs2 = torch.cat(batch[data_features.index('gripper_forward_direction')], dim=0).to(conf.device)  # B x 3

    # forward through the network
    pred_result_logits, conf_result_logits, pred_whole_feats = network(input_pcs, input_dirs1, input_dirs2)  # B x 2, B x F x N

    # prepare gt
    gt_result = torch.Tensor(batch[data_features.index('result')]).long().to(conf.device)  # B

    total_loss = 0
    result_loss_per_data = network.critic.get_ce_loss(pred_result_logits, gt_result)

    result_loss = result_loss_per_data.mean()
    print('epoch', epoch, 'batch_id', batch_ind, 'result', result_loss.item())

    total_loss += result_loss.item()
    # optimize one step for critic
    network_opt.zero_grad()
    result_loss.backward()
    network_opt.step()
    network_lr_scheduler.step()

    # compute correctness and confidence_loss
    pred_result_logits, conf_result_logits, _ = network(input_pcs, input_dirs1, input_dirs2)  # B x 2, B x F x N

    pred_results = pred_result_logits.detach().cpu().numpy() > 0
    correct = pred_results == gt_result.cpu().numpy()

    correct_gt = []
    for idx in range(batch_size):
        key_id = shape_id[idx] + '-' + cnt_id[idx]
        if key_id not in correctness:
            correctness[key_id] = np.zeros((correctness['size'], 2))
        correctness[key_id][int(trial_id[idx])][0] += correct[idx]
        correctness[key_id][int(trial_id[idx])][1] += 1

        correct_gt.append(correctness[key_id][int(trial_id[idx])][0] / correctness[key_id][int(trial_id[idx])][1])

    if batch_ind % 100 == 0:
        print('correct in epoch', epoch, correct)
        print('correctness in epoch', epoch, correctness[shape_id[0] + '-' + cnt_id[0]][int(trial_id[0])],
              correctness[shape_id[1] + '-' + cnt_id[1]][int(trial_id[1])])

    correct_gt = torch.tensor(correct_gt).to(conf.device)

    confidence_loss = network.critic.get_ce_loss(conf_result_logits, correct_gt)
    confidence_loss = confidence_loss.mean()

    total_loss += confidence_loss.item()

    print('epoch', epoch, 'batch_id', batch_ind, 'confidence', confidence_loss.item())

    # optimize one step for confidence
    network_opt.zero_grad()
    confidence_loss.backward()
    network_opt.step()
    network_lr_scheduler.step()

    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog,
                           f'''{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                           f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                           f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                           f'''{100. * (1 + batch_ind + num_batch * epoch) / (num_batch * conf.epochs):>9.1f}%      '''
                           f'''{lr:>5.2E} '''
                           f'''{result_loss.item():>10.5f}''')
            conf.flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            if correctness is not None:
                tb_writer.add_scalar('confidence_loss', confidence_loss.item(), step)
            tb_writer.add_scalar('result_loss', result_loss.item(), step)
            tb_writer.add_scalar('total_loss', total_loss, step)
            tb_writer.add_scalar('lr', lr, step)

    return pred_whole_feats.detach(), input_pcs.detach(), input_pxids.detach(), input_movables.detach()


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--primact_type', type=str, help='the primact type')
    parser.add_argument('--category_types', type=str, help='list all categories [Default: None, meaning all 10 categories]', default=None)
    parser.add_argument('--data_dir_prefix', type=str, help='data directory')
    parser.add_argument('--offline_data_dir', type=str, help='data directory')
    parser.add_argument('--train_shape_fn', type=str, help='training shape file that indexs all shape-ids')
    parser.add_argument('--ins_cnt_fn', type=str, help='a file listing all category instance count')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    # parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--seed', type=int, default=-1, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_point_per_shape', type=int, default=10000)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--abs_thres', type=float, default=0.01, help='abs thres')
    parser.add_argument('--rel_thres', type=float, default=0.5, help='rel thres')
    parser.add_argument('--dp_thres', type=float, default=0.5, help='dp thres')
    parser.add_argument('--no_true_false_equal', action='store_true', default=False, help='if make the true/false data loaded equally [default: False]')
    parser.add_argument('--load_model', type=str, default=None, help='what model to load')
    parser.add_argument('--load_model_epoch', type=int, default=None, help='what model to load')

    # training parameters
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--buffer_max_num', type=int, default=20000)
    parser.add_argument('--num_processes_for_datagen', type=int, default=20)
    parser.add_argument('--num_interaction_data_offline', type=int, default=24)
    parser.add_argument('--sample_max_num', type=int, default=0)
    parser.add_argument('--sample_interval', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    parser.add_argument('--world_coordinate', action='store_true', default=False)

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10, help='number of optimization steps beween console log prints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=10, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # parse args
    conf = parser.parse_args()

    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.model_version}-{conf.primact_type}-{conf.category_types}-{conf.exp_suffix}'

    # mkdir exp_dir; ask for overwrite if necessary;
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    if conf.load_model:
        conf.load_model = os.path.join(conf.log_dir, conf.load_model)
    if os.path.exists(conf.exp_dir):
        if not conf.overwrite:
            response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
            if response != 'y':
                exit(1)
        shutil.rmtree(conf.exp_dir)

    os.mkdir(conf.exp_dir)
    os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
    if not conf.no_visu:
        os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # prepare data_dir
    conf.data_dir = conf.data_dir_prefix + '-' + conf.exp_name
    if os.path.exists(conf.data_dir):
        if not conf.overwrite:
            response = input('A data_dir named "%s" already exists, overwrite? (y/n) ' % conf.data_dir)
            if response != 'y':
                exit(1)
        shutil.rmtree(conf.data_dir)

    os.mkdir(conf.data_dir)

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # backup python files used for this training
    os.system('cp datagen.py data.py models/%s.py %s %s' % (conf.model_version, __file__, conf.exp_dir))

    # set training device
    device = torch.device(conf.device)
    utils.printout(flog, f'Using device: {conf.device}\n')
    conf.device = device

    # parse params
    utils.printout(flog, 'primact_type: %s' % str(conf.primact_type))

    if conf.category_types is None:
        conf.category_types = ['Box', 'Door', 'Faucet', 'Kettle', 'Microwave', 'Refrigerator', 'StorageFurniture',
                               'Switch', 'TrashCan', 'Window']
    else:
        conf.category_types = conf.category_types.split(',')
    utils.printout(flog, 'category_types: %s' % str(conf.category_types))

    # read cat2freq
    conf.cat2freq = dict()
    with open(conf.ins_cnt_fn, 'r') as fin:
        for l in fin.readlines():
            category, _, freq = l.rstrip().split()
            conf.cat2freq[category] = int(freq)
    utils.printout(flog, str(conf.cat2freq))

    shape_ids = []
    if conf.train_shape_fn is not None:
        with open(conf.train_shape_fn, 'r') as fin:
            for l in fin.readlines():
                shape_id, _, = l.rstrip().split()
                shape_ids.append(shape_id)
        print(shape_ids)

    with open(os.path.join(conf.offline_data_dir, 'data_tuple_list.txt'), 'r') as fin:
        all_train_data_list = [os.path.join(conf.offline_data_dir, l.rstrip()) for l in fin.readlines()]

    train_data_list = []
    for item in all_train_data_list:
        shape_id, category, _, _, trail_id = item.split('/')[-1].split('_')[-5:]
        if int(trail_id) < conf.num_interaction_data_offline and category in conf.category_types:
            if conf.train_shape_fn is None or (shape_id in shape_ids):
                print(shape_id, category)
                train_data_list.append(item)
    utils.printout(flog, 'len(train_data_list): %d' % len(train_data_list))

    ### start training
    train(conf, train_data_list)

    # close file log
    flog.close()

