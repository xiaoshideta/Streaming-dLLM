tasks="gsm8k_cot gsm8k_cot"
nshots="5 5"
lengths="256 512"
temperatures="0 0"
steps="8 16"
window_size="32 32"
confidence_alpha="0.3 0.3"
model=/root/autodl-tmp/model/Dream-base
# Create arrays from space-separated strings
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"
read -ra TEMP_NUM <<< "$steps"
read -ra WIN_SIZE <<< "$window_size"
read -ra COM_ALPHA <<< "$confidence_alpha"

export CUDA_VISIBLE_DEVICES=0
export HF_ALLOW_CODE_EVAL=1

for i in "${!TASKS_ARRAY[@]}"; do
    output_path=evals_results1/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path"
    accelerate launch eval.py --model dream \
        --model_args pretrained=${model},max_new_tokens=${LENGTH_ARRAY[$i]},diffusion_steps=${TEMP_NUM[$i]},add_bos_token=true,temperature=${TEMP_ARRAY[$i]},top_p=0.95,alg=confidence_threshold,threshold=0.9,use_cache=true,window_size=${WIN_SIZE[$i]},confidence_alpha=${COM_ALPHA[$i]} \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code
done