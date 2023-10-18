python gen_offline_data.py \
  --data_dir ../data/gt_data-train_data-pulling \
  --data_fn ../stats/train_10cats_train_data_list.txt \
  --primact_types pulling \
  --category_types StorageFurniture,Faucet,Window \
  --num_processes 10 \
  --num_epochs 150 \
  --ins_cnt_fn ../stats/ins_cnt_15cats.txt \
  --divide_state

python gen_offline_data.py \
  --data_dir ../data/gt_data-train_ruler_data-pulling \
  --data_fn ../stats/few-shot_train.txt \
  --primact_types pulling \
  --category_types Switch,Bucket,KitchenPot,Box,Table,Refrigerator,TrashCan,Microwave,Door,WashingMachine,Kettle \
  --num_processes 10 \
  --num_epochs 6 \
  --ins_cnt_fn ../stats/ins_cnt_all_same.txt \
  --divide_state

python gen_offline_data.py \
  --data_dir ../data/gt_data-test_data-pulling \
  --data_fn ../stats/few-shot_test.txt \
  --primact_types pulling \
  --category_types Switch,Bucket,KitchenPot,Box,Table,Refrigerator,TrashCan,Microwave,Door,WashingMachine,Kettle \
  --num_processes 10 \
  --num_epochs 20 \
  --ins_cnt_fn ../stats/ins_cnt_15cats.txt