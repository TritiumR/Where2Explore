"""
    Train the Where2Explore model
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
from data import SAPIENVisionDataset
import utils
from pointnet2_ops.pointnet2_utils import furthest_point_sample

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, 'blender_utils'))


def train(conf, train_data_list):
    # create training datasets and data loaders
    data_features = ['pcs', 'pc_pxids', 'pc_movables', 'gripper_img_target', 'gripper_direction', 'gripper_forward_direction', \
                     'result', 'cur_dir', 'shape_id', 'cnt_id', 'trial_id', 'is_original', 'obj_state', 'category']

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
    else:
        train_writer = None

    # send parameters to device
    network.to(conf.device)
    utils.optimizer_to_device(network_opt, conf.device)

    # load dataset
    train_dataset = SAPIENVisionDataset([conf.primact_type], conf.category_types, data_features, conf.buffer_max_num,
                                        abs_thres=conf.abs_thres, rel_thres=conf.rel_thres, dp_thres=conf.dp_thres,
                                        img_size=conf.img_size, no_true_false_equal=conf.no_true_false_equal,
                                        world_coordinate=conf.world_coordinate)

    train_dataset.load_data(train_data_list)

    total_data_num = len(train_dataset)
    print('total data num: ', total_data_num)

    # start training
    start_time = time.time()

    last_train_console_log_step, last_val_console_log_step = None, None
    correct_times = None

    # if resume
    start_epoch = 0
    if conf.resume:
        # figure out the latest epoch to resume
        for item in os.listdir(os.path.join(conf.exp_dir, 'ckpts')):
            if item.endswith('-train_dataset.pth'):
                start_epoch = max(start_epoch, int(item.split('-')[0]))

        # load states for network, optimizer, lr_scheduler, sample_conf_list
        data_to_restore = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-network.pth' % start_epoch))
        network.load_state_dict(data_to_restore)
        data_to_restore = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-optimizer.pth' % start_epoch))
        network_opt.load_state_dict(data_to_restore)
        data_to_restore = torch.load(os.path.join(conf.exp_dir, 'ckpts', '%d-lr_scheduler.pth' % start_epoch))
        network_lr_scheduler.load_state_dict(data_to_restore)

        # load correct_times
        correct_times = np.load(os.path.join(conf.exp_dir, 'ckpts', '%d-correct_times.pth.npy' % start_epoch), allow_pickle=True).item()

    # if load model
    if conf.load_model:
        data_to_restore = torch.load(os.path.join(conf.load_model, 'ckpts', '%d-network.pth' % conf.load_model_epoch))
        network.load_state_dict(data_to_restore)
        print('loading model from: ', os.path.join(conf.load_model, 'ckpts', '%d-network.pth' % conf.load_model_epoch))

    # train for every epoch
    for epoch in range(start_epoch, conf.epochs):
        utils.printout(conf.flog, str(train_dataset))
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=conf.batch_size, shuffle=True,
                                                       pin_memory=True,
                                                       num_workers=0, drop_last=True, collate_fn=utils.collate_feats,
                                                       worker_init_fn=utils.worker_init_fn)
        train_num_batch = len(train_dataloader)
        print('train_num_batch: ', train_num_batch)

        # record correctness
        if epoch == start_epoch and correct_times is None:
            total_data_num = len(train_dataset)
            correct_times = dict()
            correct_times['size'] = total_data_num

        ### print log
        if not conf.no_console_log:
            utils.printout(conf.flog, f'training run {conf.exp_name}')
            utils.printout(conf.flog, header)

        train_batches = enumerate(train_dataloader, 0)

        ### train for every batch
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
                    utils.printout(conf.flog, 'DONE')

                    np.save(os.path.join(conf.exp_dir, 'ckpts', '%d-correct_times.pth' % epoch), correct_times)
                    utils.printout(conf.flog, 'correct_times-DONE')

            # set models to training mode
            network.train()

            # forward pass (including logging)
            total_loss, whole_feats, whole_pcs, whole_pxids, whole_movables = forward(batch=batch,
                                                                                      data_features=data_features,
                                                                                      network=network, conf=conf,
                                                                                      step=train_step, epoch=epoch,
                                                                                      batch_ind=train_batch_ind,
                                                                                      num_batch=train_num_batch,
                                                                                      start_time=start_time,
                                                                                      log_console=log_console,
                                                                                      log_tb=not conf.no_tb_log,
                                                                                      tb_writer=train_writer, lr=
                                                                                      network_opt.param_groups[0]['lr'],
                                                                                      correctness=correct_times)

            # optimize one step
            network_opt.zero_grad()
            total_loss.backward()
            network_opt.step()
            network_lr_scheduler.step()


def forward(batch, data_features, network, conf,
            step=None, epoch=None, batch_ind=0, num_batch=1, start_time=0,
            log_console=False, log_tb=False, tb_writer=None, lr=None, correctness=None):
    shape_id = batch[data_features.index('shape_id')]
    cnt_id = batch[data_features.index('cnt_id')]
    trial_id = batch[data_features.index('trial_id')]
    categories = batch[data_features.index('category')]

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

    result_loss_per_data = network.critic.get_ce_loss(pred_result_logits, gt_result)

    result_loss = torch.tensor(0.).to(conf.device)
    result_loss_count = 0
    for idx in range(batch_size):
        if categories[idx] in conf.real_category_types:
            result_loss += result_loss_per_data[idx]
            result_loss_count += 1

    if result_loss_count != 0:
        result_loss /= result_loss_count

    # compute correctness and confidence_loss
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

    print('epoch', epoch, 'batch_id', batch_ind, 'confidence', confidence_loss.item())
    print('epoch', epoch, 'batch_id', batch_ind, 'result', result_loss.item())

    total_loss = result_loss + confidence_loss

    # display information
    data_split = 'train'

    with torch.no_grad():
        # log to console
        if log_console:
            utils.printout(conf.flog,
                           f'''{strftime("%H:%M:%S", time.gmtime(time.time() - start_time)):>9s} '''
                           f'''{epoch:>5.0f}/{conf.epochs:<5.0f} '''
                           f'''{data_split:^10s} '''
                           f'''{batch_ind:>5.0f}/{num_batch:<5.0f} '''
                           f'''{100. * (1 + batch_ind + num_batch * epoch) / (num_batch * conf.epochs):>9.1f}%      '''
                           f'''{lr:>5.2E} '''
                           f'''{total_loss.item():>10.5f}''')
            conf.flog.flush()

        # log to tensorboard
        if log_tb and tb_writer is not None:
            if correctness is not None:
                tb_writer.add_scalar('result_loss', result_loss.item(), step)
                tb_writer.add_scalar('confidence_loss', confidence_loss.item(), step)
            tb_writer.add_scalar('total_loss', total_loss.item(), step)
            tb_writer.add_scalar('lr', lr, step)

    return total_loss, pred_whole_feats.detach(), input_pcs.detach(), input_pxids.detach(), input_movables.detach()


if __name__ == '__main__':
    ### get parameters
    parser = ArgumentParser()

    # main parameters (required)
    parser.add_argument('--exp_suffix', type=str, help='exp suffix')
    parser.add_argument('--model_version', type=str, help='model def file')
    parser.add_argument('--primact_type', type=str, help='the primact type')
    parser.add_argument('--category_types', type=str, help='list all categories [Default: None, meaning all 10 categories]', default=None)
    parser.add_argument('--real_category_types', type=str, default=None)
    parser.add_argument('--data_dir_prefix', type=str, help='data directory')
    parser.add_argument('--offline_data_dir', type=str, help='data directory')
    parser.add_argument('--val_data_dir', type=str, help='data directory', default=None)
    parser.add_argument('--ins_cnt_fn', type=str, help='a file listing all category instance count')

    # main parameters (optional)
    parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
    # parser.add_argument('--seed', type=int, default=3124256514, help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--seed', type=int, default=-1,
                        help='random seed (for reproducibility) [specify -1 means to generate a random one]')
    parser.add_argument('--log_dir', type=str, default='logs', help='exp logs directory')
    parser.add_argument('--overwrite', action='store_true', default=False,
                        help='overwrite if exp_dir exists [default: False]')
    parser.add_argument('--resume', action='store_true', default=False,
                        help='resume if exp_dir exists [default: False]')

    # network settings
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--num_point_per_shape', type=int, default=10000)
    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--abs_thres', type=float, default=0.01, help='abs thres')
    parser.add_argument('--rel_thres', type=float, default=0.5, help='rel thres')
    parser.add_argument('--dp_thres', type=float, default=0.5, help='dp thres')
    parser.add_argument('--no_true_false_equal', action='store_true', default=False,
                        help='if make the true/false data loaded equally [default: False]')
    parser.add_argument('--load_model', type=str, default=None, help='what model to load')
    parser.add_argument('--load_model_epoch', type=int, default=None, help='what model to load')

    # training parameters
    parser.add_argument('--epochs', type=int, default=10000)
    parser.add_argument('--buffer_max_num', type=int, default=20000)
    parser.add_argument('--num_interaction_data_offline', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr_decay_by', type=float, default=0.9)
    parser.add_argument('--lr_decay_every', type=float, default=5000)
    parser.add_argument('--sample_confidence', action='store_true', default=False)
    parser.add_argument('--world_coordinate', action='store_true', default=False)
    parser.add_argument('--use_val', action='store_true', default=False)
    parser.add_argument('--only_val', action='store_true', default=False)

    # logging
    parser.add_argument('--no_tb_log', action='store_true', default=False)
    parser.add_argument('--no_console_log', action='store_true', default=False)
    parser.add_argument('--console_log_interval', type=int, default=10,
                        help='number of optimization steps beween console log prints')

    # visu
    parser.add_argument('--num_batch_every_visu', type=int, default=1, help='num batch every visu')
    parser.add_argument('--num_epoch_every_visu', type=int, default=10, help='num epoch every visu')
    parser.add_argument('--no_visu', action='store_true', default=False, help='no visu? [default: False]')

    # parse args
    conf = parser.parse_args()

    ### prepare before training
    # make exp_name
    conf.exp_name = f'exp-{conf.model_version}-{conf.primact_type}-{conf.category_types}-{conf.exp_suffix}'

    if conf.overwrite and conf.resume:
        raise ValueError('ERROR: cannot specify both --overwrite and --resume!')

    # mkdir exp_dir; ask for overwrite if necessary; or resume
    conf.exp_dir = os.path.join(conf.log_dir, conf.exp_name)
    if conf.load_model:
        conf.load_model = os.path.join(conf.log_dir, conf.load_model)
    if os.path.exists(conf.exp_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A training run named "%s" already exists, overwrite? (y/n) ' % conf.exp_name)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.exp_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no training run named %s to resume!' % conf.exp_name)
    if not conf.resume:
        os.mkdir(conf.exp_dir)
        os.mkdir(os.path.join(conf.exp_dir, 'ckpts'))
        if not conf.no_visu:
            os.mkdir(os.path.join(conf.exp_dir, 'val_visu'))

    # prepare data_dir
    conf.data_dir = conf.data_dir_prefix + '-' + conf.exp_name
    if os.path.exists(conf.data_dir):
        if not conf.resume:
            if not conf.overwrite:
                response = input('A data_dir named "%s" already exists, overwrite? (y/n) ' % conf.data_dir)
                if response != 'y':
                    exit(1)
            shutil.rmtree(conf.data_dir)
    else:
        if conf.resume:
            raise ValueError('ERROR: no data_dir named %s to resume!' % conf.data_dir)
    if not conf.resume:
        os.mkdir(conf.data_dir)

    # control randomness
    if conf.seed < 0:
        conf.seed = random.randint(1, 10000)
    random.seed(conf.seed)
    np.random.seed(conf.seed)
    torch.manual_seed(conf.seed)

    # save config
    if not conf.resume:
        torch.save(conf, os.path.join(conf.exp_dir, 'conf.pth'))

    # file log
    if conf.resume:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'a+')
    else:
        flog = open(os.path.join(conf.exp_dir, 'train_log.txt'), 'w')
    conf.flog = flog

    # backup command running
    utils.printout(flog, ' '.join(sys.argv) + '\n')
    utils.printout(flog, f'Random Seed: {conf.seed}')

    # backup python files used for this training
    if not conf.resume:
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

    if conf.real_category_types is None:
        conf.real_category_types = ['Box', 'Door', 'Faucet', 'Kettle', 'Microwave', 'Refrigerator', 'StorageFurniture',
                                    'Switch', 'TrashCan', 'Window']
    else:
        conf.real_category_types = conf.real_category_types.split(',')

    utils.printout(flog, 'real_types: %s' % str(conf.real_category_types))

    # read cat2freq
    conf.cat2freq = dict()
    with open(conf.ins_cnt_fn, 'r') as fin:
        for l in fin.readlines():
            category, _, freq = l.rstrip().split()
            conf.cat2freq[category] = int(freq)
    utils.printout(flog, str(conf.cat2freq))

    with open(os.path.join(conf.offline_data_dir, 'data_tuple_list.txt'), 'r') as fin:
        all_train_data_list = [os.path.join(conf.offline_data_dir, l.rstrip()) for l in fin.readlines()]

    train_data_list = []
    for item in all_train_data_list:
        if int(item.split('_')[-1]) < conf.num_interaction_data_offline:
            train_data_list.append(item)
    utils.printout(flog, 'len(train_data_list): %d' % len(train_data_list))

    ### start training
    train(conf, train_data_list)

    ### before quit
    # close file log
    flog.close()

