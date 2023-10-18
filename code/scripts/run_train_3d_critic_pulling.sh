CUDA_VISIBLE_DEVICES=0 python train_3d_conf.py \
    --exp_suffix original_model \
    --model_version model_3d_critic \
    --primact_type pulling \
    --category_types StorageFurniture,Faucet,Window \
    --data_dir_prefix ../data/gt_data \
    --offline_data_dir ../data/gt_data-train_data-pulling \
    --ins_cnt_fn ../stats/ins_cnt_15cats.txt \
    --buffer_max_num 400000 \
    --epochs 20 \
    --num_interaction_data_offline 150 \
    --no_visu \
    --overwrite


