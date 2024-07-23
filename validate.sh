python3 imitate_episodes.py \
    --task_name insert_redness_relief_slanted \
    --ckpt_dir training_outputs/2024-04-05-10-42-07insert_redness_relief_slantedtest1 \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --eval \
    --run_name slanted



python3 imitate_episodes.py \
    --task_name insert_redness_relief_slanted_anywhere \
    --ckpt_dir training_outputs/2024-04-26-10-53-06insert_redness_relief_slanted_anywhereanywhere_slanted \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --eval \
    --run_name anywhere_Vincent

python3 imitate_episodes.py \
    --task_name insert_redness_relief_slanted_anywhere \
    --ckpt_dir training_outputs/2024-05-06-18-16-32pick_vialpick_vial \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --eval \
    --run_name pick_vialpick_vial



    /home/user/exp_checkpoints/closed_loop_demos_weight/2024-05-10-22-45-56place_vialplace_vial_05_10_bs128_lr8e-5


# placing works okay, fails when tolerances are tight
python3 imitate_episodes.py \
    --task_name insert_redness_relief_slanted_anywhere \
    --ckpt_dir /home/user/exp_checkpoints/closed_loop_demos_weight/2024-05-10-22-45-56place_vialplace_vial_05_10_bs128_lr8e-5 \
    --ckpt_name best_policy_epoch_2500_seed_0.ckpt \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --eval \
    --run_name eval


python3 imitate_episodes.py \
    --task_name insert_redness_relief_slanted_anywhere \
    --ckpt_dir /home/user/exp_checkpoints/closed_loop_demos_weight/2024-05-10-22-46-12pick_vialpick_vial_05_10_bs128_lr8e-5 \
    --ckpt_name best_policy_epoch_19700_seed_0.ckpt \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --eval \
    --run_name eval


python3 imitate_episodes.py \
    --task_name insert_redness_relief_slanted_anywhere \
    --ckpt_dir /home/user/code/act/training_outputs/2024-05-17-22-42-30insert_ibuprofeninsert_ibuprofen \
    --ckpt_name policy_epoch_5500_seed_0.ckpt \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --eval \
    --run_name eval

python3 imitate_episodes.py \
    --task_name insert_ibuprofen_zeroqpos \
    --ckpt_dir /home/user/code/act/training_outputs/2024-06-06-17-51-39insert_ibuprofen_zeroqposinsert_ibuprofen_zeroqpos \
    --ckpt_name policy_epoch_7500_seed_0.ckpt \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --eval \
    --run_name eval


#trained with bsize 128
python3 imitate_episodes.py \
    --task_name insert_ibuprofen_zeroqpos \
    --ckpt_dir /home/user/code/act/training_outputs/2024-06-09-00-46-29insert_ibuprofen_zeroqposrun_1 \
    --ckpt_name policy_epoch_10000_seed_0.ckpt \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --eval \
    --run_name eval

#trained with bsize 128, for much longer
python3 imitate_episodes.py \
    --task_name insert_ibuprofen_zeroqpos \
    --ckpt_dir /home/user/code/act/training_outputs/2024-06-10-21-57-10insert_ibuprofen_zeroqposb128_100kepoch \
    --ckpt_name policy_epoch_24500_seed_0.ckpt \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --eval \
    --run_name eval

# test on relative actions:
python3 imitate_episodes.py \
    --task_name insert_ibuprofen_rel \
    --ckpt_dir /home/user/data/act_training_runs/2024-06-24-18-01-33insert_ibuprofen_relinsert_ibuprofen_rel \
    --ckpt_name policy_epoch_18000_seed_0.ckpt \
    --policy_class ACT --kl_weight 10 --chunk_size 40 --hidden_dim 512 --batch_size 32 --dim_feedforward 3200 \
    --num_epochs 5000  --lr 2e-5 \
    --seed 0 \
    --eval \
    --run_name eval