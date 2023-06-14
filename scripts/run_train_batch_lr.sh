# default value for the --eval option
eval_arg=""

# parse the command-line options
while getopts "e" opt; do
  case ${opt} in
    e )
      eval_arg="--eval"
      ;;
    \? )
      echo "Invalid option: -$OPTARG" 1>&2
      exit 1
      ;;
  esac
done

echo "eval_arg: $eval_arg"

declare -a arr_batch=("8" "16")
#declare -a arr_batch=("8" "16" "32" "64")
declare -a arr_lr=("1e-3" "1e-4" "5e-5" "1e-5" "5e-6")

for i in "${arr_batch[@]}"
do
for j in "${arr_lr[@]}"
do
	echo "running experiment for batch size $i, lr $j"
	python3 imitate_episodes.py \
	--task_name sim_transfer_cube_scripted \
	--ckpt_dir /iris/u/davidy02/model/cube_bs_${i}_lr_${j} \
	--policy_class ACT \
    --kl_weight 10 \
	--chunk_size 100 \
	--hidden_dim 512 \
	--batch_size $i \
	--dim_feedforward 3200 \
	--num_epochs 2000  \
	--lr $j \
	--seed 0 \
	$eval_arg
done
done
