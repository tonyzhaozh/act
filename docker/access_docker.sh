docker run --gpus all -it --rm --shm-size 8G \
    --volume $CODE/act:/home/app/act \
    --volume $DATA:/home/app/data \
    -e "DATA=/home/app/data/" \
    emancro/act:latest /bin/bash -c "cd /home/app/act/detr && pip install -e . && cd /home/app/act && /bin/bash "
