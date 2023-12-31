CUDA_VISIBLE_DEVICES=0 python train_3d_few_shot.py \
    --exp_suffix few-shot_explore_model \
    --model_version model_3d_conf \
    --primact_type pulling \
    --category_types Switch,Bucket,KitchenPot,Box,Table,Refrigerator,TrashCan,Microwave,Door,WashingMachine,Kettle \
    --data_dir_prefix ../data/gt_data \
    --offline_data_dir ../data/gt_data-train_ruler_data-pulling \
    --ins_cnt_fn ../stats/ins_cnt_all_same.txt \
    --load_model exp-model_3d_conf-pulling-StorageFurniture,Window,Faucet-explore_model \
    --load_model_epoch 15 \
    --buffer_max_num 400000 \
    --epochs 5 \
    --num_processes_for_datagen 5 \
    --num_interaction_data_offline 6 \
    --sample_max_num 50 \
    --no_visu \
    --overwrite
