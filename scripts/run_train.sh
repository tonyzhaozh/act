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

python3 imitate_episodes.py \
	--task_name sim_transfer_cube_scripted \
	--ckpt_dir /iris/u/davidy02/model/cube_chunk_50 \
	--policy_class ACT \
       	--kl_weight 10 \
	--chunk_size 100 \
	--hidden_dim 512 \
	--batch_size 8 \
	--dim_feedforward 3200 \
	--num_epochs 2000  --lr 1e-5 \
	--seed 0\
	$eval_arg
