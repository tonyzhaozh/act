#!/bin/bash

# Get the list of processes using CUDA and extract relevant information
cuda_processes=$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits)

# Check if there are no CUDA processes
if [ -z "$cuda_processes" ]; then
    echo "No CUDA processes found."
    exit 0
fi

# Loop through each line of the output
while IFS=, read -r pid process_name used_memory; do
    echo "Found process: PID=$pid, Name=$process_name, Used Memory=$used_memory MiB"

    # Kill the process
    echo "Killing process with PID $pid"
    kill -9 "$pid"

done <<< "$cuda_processes"
