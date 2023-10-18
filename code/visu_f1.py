import os
from argparse import ArgumentParser
import numpy as np

# test parameters
parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--gt_dir', type=str, default=None)
parser.add_argument('--shape_id', type=str, help='shape id')
parser.add_argument('--model_epoch', type=int, help='epoch')
parser.add_argument('--result_suffix', type=str, default='nothing')
eval_conf = parser.parse_args()

model_dir = os.path.join('logs', eval_conf.exp_name, f'visu_critic_heatmap-{eval_conf.shape_id}-model_epoch_{eval_conf.model_epoch}-{eval_conf.result_suffix}')
gt_dir = os.path.join('logs', eval_conf.gt_dir, f'visu_critic_heatmap-{eval_conf.shape_id}')
if not os.path.exists(model_dir) or not os.path.exists(gt_dir):
    print(f'no directory {model_dir} or {gt_dir}')

print('model loading from: ', model_dir)
print('gt loading from: ', gt_dir)

pred_label_fn = os.path.join(model_dir, 'pred.label')
gt_label_fn = os.path.join(gt_dir, 'gt.label')

with open(pred_label_fn, 'r') as fin:
    pred_list = fin.readlines()
pred_list = np.array([float(line.split('\n')[0]) for line in pred_list]) > 0.5

with open(gt_label_fn, 'r') as fin:
    gt_list = fin.readlines()
gt_list = np.array([float(line.split('\n')[0]) for line in gt_list])

# print(gt_list[:10], len(pred_list), len(gt_list))

pred_succ_and = []
for i in range(len(pred_list)):
    if pred_list[i] == 1.0 and gt_list[i] == 1.0:
        pred_succ_and.append(1.)
    else:
        pred_succ_and.append(0.)
pred_succ_and = np.array(pred_succ_and)

precise = pred_succ_and.sum() / pred_list.sum()
recall = pred_succ_and.sum() / gt_list.sum()
F1 = (precise + recall) / 2

print('Score: ', precise, recall, F1)
print('**************************')

