CUDA_VISIBLE_DEVICES=0 python train_3d_conf.py \
    --exp_suffix explore_model \
    --model_version model_3d_conf \
    --primact_type pushing \
    --category_types StorageFurniture,Faucet,Window \
    --real_category_types StorageFurniture \
    --data_dir_prefix ../data/gt_data \
    --offline_data_dir ../data/gt_data-train_data-pushing \
    --ins_cnt_fn ../stats/ins_cnt_15cats.txt \
    --buffer_max_num 400000 \
    --epochs 20 \
    --num_interaction_data_offline 150 \
    --no_visu \
    --overwrite


