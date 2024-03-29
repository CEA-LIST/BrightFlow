python train.py \
--ckpt_dir /path/to/checkpoint/dir/ \
--log_dir /path/to/log/dir/ \
--restore_ckpt /path/to/checkpoint.pth \
--name brightflow \
--dataset_train Sintel \
--dataset_test Sintel \
--model raft \
--batch_size 8 \
--num_steps 75000 \
--lr 0.0002 \
--lr_decay_step 15000 \
--census_weight_flow 1. \
--l1_weight_correc 1. \
--selfsup_starting_step 30000 \
--selfsup_end_rising_step 37500 \
--selfsup_weight_max 0.3 \
--mode flow_correc \
--correc_weight 0.1 \
--correc_in_photo_starting_step 25000 \
--correc_starting_step 20000 \
--use_full_size_warping \
--smart_clamp \
--occ_in_correc_inputs \
--sequentially \
--keep_good_corrections_only \