CUDA_VISIBLE_DEVICES=0 python test_3d_critic.py \
    --model_version model_3d_conf \
    --primact_type pushing \
    --category_types Switch,Bucket,KitchenPot,Box,Table,Refrigerator,TrashCan,Microwave,Door,WashingMachine,Kettle \
    --val_data_dir ../data/gt_data-test_data-pushing \
    --buffer_max_num 100000 \
    --no_true_false_equal \
    --load_model exp-model_3d_conf-pushing-Switch,Bucket,KitchenPot,Box,Table,Refrigerator,TrashCan,Microwave,Door,WashingMachine,Kettle-few-shot_explore_model \
    --start_epoch 0 \
    --end_epoch 5
