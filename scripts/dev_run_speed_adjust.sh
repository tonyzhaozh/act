# default value for the --eval option
eval_arg="--eval"

echo "eval_arg: $eval_arg"

python3 imitate_episodes.py \
	--task_name sim_transfer_cube_scripted \
	--ckpt_dir /iris/u/davidy02/model/dev_cube_speed_var \
	--policy_class ACT \
       	--kl_weight 10 \
	--chunk_size 100 \
	--hidden_dim 512 \
	--batch_size 8 \
	--dim_feedforward 3200 \
	--num_epochs 10000  --lr 1e-5 \
	--seed 0 \
	--temporal_agg \
	--use_speed_var \
	--speed_model \
	--adjust_speed \
	$eval_arg

	# --use_speed_var \

