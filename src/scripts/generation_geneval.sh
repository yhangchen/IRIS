# --- Configuration ---
cd t2i-r1/src

export dir="$HOME/projects/T2I-R1-IR/src/t2i-r1/src/janus/outputs"
export save_dir="$HOME/projects/T2I-geneval/outputs"
export with_cot=True

# --- Parallel Execution ---
available_gpus=(0 1 2 3 4 5 6 7)
# --- Parallel Execution ---
job_count=0
num_gpus=${#available_gpus[@]}         # Total number of GPUs you want to use (0 to 7)
max_processes=$num_gpus   # The desired number of concurrent processes

for method in "Janus-Pro-1B" "t2i-r1-all-reward-correct-template" "t2i-r1-all-image-1000-text-1000-correct-template" "t2i-r1-all-without-cot-correct-template"
  do
    if [[ "$method" == "Janus-Pro-1B" ]]; then
        TOTAL=100
    else
        TOTAL=800
    fi
    for (( NUM=100; NUM<=TOTAL; NUM+=100 ))
    do

      if [[ $(jobs -p | wc -l) -ge $max_processes ]]; then
        wait -n
      fi

      # Assign a GPU ID to the current job, cycling from 0 to 7
      gpu_index=$((job_count % num_gpus))
      gpu_id=${available_gpus[$gpu_index]}

      # Set the specific checkpoint for this job
      checkpoint="checkpoint-$NUM"

      echo "Spawning Job #$job_count on GPU $gpu_id -> Method: $method, Checkpoint: $checkpoint, Data: $dataname"

      # Run the python command in the background (&) on the assigned GPU
      CUDA_VISIBLE_DEVICES=$gpu_id python -m infer.reason_inference_geneval \
        --model_path "$dir/$method/$checkpoint" \
        --data_path ../../../data/evaluation_metadata.jsonl \
        --save_dir "$save_dir/$method/$checkpoint/generate_$with_cot" \
        --with_cot "$with_cot" &

      # Increment the job counter
      job_count=$((job_count + 1))
    done
  done
