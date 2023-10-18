list="40147 102739 7265 103369 102726 103311 103323 1028"
for shape_id in $list
do
  CUDA_VISIBLE_DEVICES=0 xvfb-run -a python visu_critic_loss.py \
    --exp_name exp-model_3d_conf-pushing-StorageFurniture,Window,Faucet-explore_model \
    --start_epoch 0 \
    --end_epoch 5 \
    --model_version model_3d_conf \
    --result_suffix visual \
    --shape_id "$shape_id" \
    --state 2
done
