export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
tasks="mbpp mbpp gsm8k gsm8k minerva_math minerva_math"
nshots="3 3 5 5 4 4"
lengths="256 512 256 512 256 512"
temperatures="0 0 0 0 0 0"
steps="8 16 8 16 8 16"
window_size="96 96 96 128 96 192"
confidence_alpha="0.3 0.3 0.4 0.6 0.4 0.3"
block_length=32
# if you want to test llada-instruct, replace the model path
model_path='/root/autodl-tmp/model/llada-1.5'
read -ra TASKS_ARRAY <<< "$tasks"
read -ra NSHOTS_ARRAY <<< "$nshots"
read -ra LENGTH_ARRAY <<< "$lengths"
read -ra TEMP_ARRAY <<< "$temperatures"
read -ra TEMP_NUM <<< "$steps"
read -ra WIN_SIZE <<< "$window_size"
read -ra COM_ALPHA <<< "$confidence_alpha"

export CUDA_VISIBLE_DEVICES=0
accelerate launch --main_process_port 29510 eval_llada.py --model llada_dist \
    --model_args model_path=${model_path},gen_length=256,steps=8,temperature=0.0,block_length=${block_length},use_cache=true,threshold=0.9,show_speed=True,window_size=96,confidence_alpha=0.3 \
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size 1 \
    --output_path evals_results1/humaneval_256 \
    --log_samples \
    --confirm_run_unsafe_code
    
accelerate launch --main_process_port 29510 eval_llada.py --model llada_dist \
    --model_args model_path=${model_path},gen_length=512,steps=16,temperature=0.0,block_length=${block_length},use_cache=true,threshold=0.9,show_speed=True,window_size=96,confidence_alpha=0.4\
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size 1 \
    --output_path evals_results1/humaneval_512 \
    --log_samples \
    --confirm_run_unsafe_code

### NOTICE: use postprocess for humaneval
# python postprocess_code.py xxx.jsonl

for i in "${!TASKS_ARRAY[@]}"; do
    output_path=evals_results1/${TASKS_ARRAY[$i]}-ns${NSHOTS_ARRAY[$i]}
    echo "Task: ${TASKS_ARRAY[$i]}, Shots: ${NSHOTS_ARRAY[$i]}; Output: $output_path"
    accelerate launch eval_llada.py --model llada_dist \
        --model_args model_path=${model_path},gen_length=${LENGTH_ARRAY[$i]},steps=${TEMP_NUM[$i]},temperature=${TEMP_ARRAY[$i]},block_length=${block_length},show_speed=True,use_cache=true,threshold=0.9,window_size=${WIN_SIZE[$i]},confidence_alpha=${COM_ALPHA[$i]} \
        --tasks ${TASKS_ARRAY[$i]} \
        --num_fewshot ${NSHOTS_ARRAY[$i]} \
        --batch_size 1 \
        --output_path $output_path \
        --log_samples \
        --confirm_run_unsafe_code
done

# Fast-dllm prefix cache + parallel
accelerate launch --main_process_port 29510 eval_llada.py --model llada_dist \
    --model_args model_path=${model_path},gen_length=256,steps=8,temperature=0.0,block_length=${block_length},use_cache=true,threshold=0.9,show_speed=True,window_size=256,confidence_alpha=0.0 \
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size 1 \
    --output_path evals_results1/humaneval_256 \
    --log_samples \
    --confirm_run_unsafe_code

# 