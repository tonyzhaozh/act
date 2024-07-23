import os


GRAB_ITERATIONS = [
    'policy_epoch_12000_seed_0.ckpt',
    'policy_epoch_18000_seed_0.ckpt',
    'policy_epoch_5000_seed_0.ckpt'    
]

OTHER_FOLDERS_AND_FILES = [
    'dataset_stats.pkl'
]



def download_from_gcs(args):

    destination_path = f'/home/user/data/act_training_runs/{args.run_name}'
    source_path = f'/mnt/disks/extra_data/data/act_training/training_outputs/{args.run_name}'

    if not args.dry_run:
        os.makedirs(destination_path, exist_ok=True)
    else:
        print('would create', destination_path)

    for file in OTHER_FOLDERS_AND_FILES:
        # cmd = 'scp -r ' + args.machine + ':' + source_path + '/' + file + ' ' + destination_path
        cmd = 'gcloud compute scp --recurse ' + args.machine + ':' + source_path + '/' + file + ' ' + destination_path + ' --zone=us-central1-c'
        if not args.dry_run:
            os.system(cmd)
        else:
            print('would execute', cmd)

    for iteration in GRAB_ITERATIONS:
        # cmd = 'scp -r ' +  args.machine + ':' + source_path + '/' + iteration + ' ' + destination_path
        cmd = 'gcloud compute scp --recurse ' + args.machine + ':' + source_path + '/' + iteration + ' ' + destination_path + ' --zone=us-central1-c'
        if not args.dry_run:
            os.system(cmd)
        else:
            print('would execute', cmd)
    


if __name__ == "__main__":
    # example
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--machine', default='a100-2', type=str)
    parser.add_argument('--run_name', default='experiment_20240610_205136', type=str, required=True)   
    parser.add_argument('--dry_run', action='store_true')

    args = parser.parse_args()

    download_from_gcs(args)