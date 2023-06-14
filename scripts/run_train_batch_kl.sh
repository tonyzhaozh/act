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

# array of kl loss values to run experiments for
# declare -a arr=("0" "1" "5" "10" "20" "50" "100")
declare -a arr=("5" "10" "20" "50" "100")

# run experiments for each value of kl loss
for i in "${arr[@]}"
do
	echo "running experiment for kl loss $i"
	python3 imitate_episodes.py \
	--task_name sim_transfer_cube_scripted \
	--ckpt_dir /iris/u/davidy02/model/cube_kl_$i \
	--policy_class ACT \
       	--kl_weight $i \
	--chunk_size 100 \
	--hidden_dim 512 \
	--batch_size 8 \
	--dim_feedforward 3200 \
	--num_epochs 2000  --lr 1e-5 \
	--seed 0 \
	$eval_arg
done
