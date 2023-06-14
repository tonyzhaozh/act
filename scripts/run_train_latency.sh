# default value for the --eval option
#eval_arg=""
eval_arg="--eval"

python3 imitate_episodes.py \
	--task_name sim_transfer_cube_scripted \
	--ckpt_dir /iris/u/davidy02/model/cube_org_latency_test_0_s1 \
	--policy_class ACT \
       	--kl_weight 10 \
	--chunk_size 100 \
	--hidden_dim 512 \
	--batch_size 8 \
	--dim_feedforward 3200 \
	--num_epochs 2000  --lr 1e-5 \
	--seed 0 \
	--temporal_agg \
	$eval_arg
