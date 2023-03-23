declare -a arr=("1" "10" "30" "50" "75" "100" "150")

for i in "${arr[@]}"
do
	echo "running experiment for chunk size $i"
	python3 imitate_episodes.py \
	--task_name sim_transfer_cube_scripted \
	--ckpt_dir /iris/u/davidy02/model/cube_chunk_$i \
	--policy_class ACT \
       	--kl_weight 10 \
	--chunk_size $i \
	--hidden_dim 512 \
	--batch_size 8 \
	--dim_feedforward 3200 \
	--num_epochs 2000  --lr 1e-5 \
	--seed 0
done
