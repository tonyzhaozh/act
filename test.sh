python3 ./imitate_episodes.py --task_name sim_stack_block_scripted --ckpt_dir ./ckpt --policy_class ACT --kl_weight 10 --chunk_size 100 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0


python3 ./imitate_episodes.py --task_name sim_stack_block_scripted --ckpt_dir /home/mlrig/Documents/act/ckpt/sim_stack_block_400_scripted_hidden_dim_512_bs_8_cs_32/ --policy_class ACT --kl_weight 10 --chunk_size 32 --hidden_dim 512 --batch_size 8 --dim_feedforward 3200 --num_epochs 2000 --lr 1e-5 --seed 0 --isaac_sim --eval

