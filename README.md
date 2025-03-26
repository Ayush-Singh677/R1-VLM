# R1-VLM - Codebase taken from [R1-V](https://github.com/Deep-Agent/R1-V)

**Blogs:**
[🎯 RLVR in Vision Language Models: Findings, Questions and Directions](https://deepagent.notion.site/rlvr-in-vlms)

## Setup

```bash
conda create -n r1-v python=3.11 
conda activate r1-v

bash setup.sh
```

> [!NOTE] 
> If you meet bug when running the script, first try align your environments with `./src/requirements.txt`

### Supported Models

1. Qwen2-VL
2. Qwen2.5-VL 

### Supported Training Datasets

1. [🤗 R1V Training Dataset: CLEVR-70k-Counting](https://huggingface.co/datasets/leonardPKU/clevr_cogen_a_train): Item Counting Problems

2. [🤗 R1V Training Dataset: CLEVR-70k-Complex](https://huggingface.co/datasets/MMInstruction/Clevr_CoGenT_TrainA_70K_Complex): Number Related Reasoning 

3. [🤗 R1V Training Dataset: GEOQA-8k](https://huggingface.co/datasets/leonardPKU/GEOQA_R1V_Train_8K): Geometry Reasoning


### Supported Evaluations

1. [SuperClevr-200](https://github.com/Deep-Agent/R1-V?tab=readme-ov-file#superclevr): Item Counting Problems
2. [GeoQA-Test-Direct-Answer-735](https://github.com/Deep-Agent/R1-V?tab=readme-ov-file#geoqa): Geometry Reasoning

## Training

### GRPO

```bash
cd src/r1-v

export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"

torchrun --nproc_per_node="8" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    src/open_r1/grpo.py \
    --output_dir <OUTPUT_DIR> \
    --model_name_or_path <PATH-TO-Qwen2-VL-2B-Instruct> \ 
    --dataset_name leonardPKU/clevr_cogen_a_train \  
    --deepspeed local_scripts/zero3.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-CLEVR-70k \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 8   # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance  

```

> [!NOTE] 
> 1. To reproduce the result, keep the per_device_train_batch_size to 1 for now, as there is a revealed bug about batched training. See the [reproduction report](https://github.com/Deep-Agent/R1-V/issues/4#issuecomment-2633348354) here. We realize it is important for effiency and are working on solving it with the community.
> 2. If you meet **OOM Error**, you can try reduce `--num_generations`
> 3. To use vLLM to speed up, please refer to this [script](https://github.com/Deep-Agent/R1-V/blob/main/src/scripts/run_grpo_vllm.sh).

### SFT

We also provide SFT code, please follow the script and edit the config to customize the sft task.

```bash
accelerate launch --config_file src/r1-v/configs/zero2.yaml src/r1-v/src/open_r1/sft.py --config src/r1-v/configs/qwen2vl_sft_config.yaml 
```

## Evaluation


### SuperCLEVR

![image](https://github.com/user-attachments/assets/4f48233c-0546-432f-94e6-723f91fbd086)

We provide the example script to evaluate OOD counting performance on a subset of SuperCLEVR within 1 minute. You can also modify the script and dataset to test on your own dataset.



```bash
cd ./src/eval
wget https://www.cs.jhu.edu/~zhuowan/zhuowan/SuperCLEVR/to_be_released/images.zip
unzip images.zip

# change the model path in the script
python test_qwen2vl_counting_superclevr.py 

# tested scores: 
# Qwen2VL-2B-Instruct: 48.0%
# Qwen2VL-2B-Instruct-GRPO-100step: 82.5%
```

### GEOQA

<img width="379" alt="截屏2025-02-11 13 38 50" src="https://github.com/user-attachments/assets/0282872d-bfe5-40fa-ac00-8986450a0b1e" />
<img width="379" alt="截屏2025-02-11 14 54 16" src="https://github.com/user-attachments/assets/053ebb99-5f19-4599-be51-a7c335ab2b8b" />



We provide the example script to evaluate on the test set (direct answer form) of [GEOQA](https://arxiv.org/abs/2312.11370).


```bash
# prepare images for testing
cd ./src/eval
git lfs install
git clone https://huggingface.co/datasets/Luckyjhg/Geo170K
cd Geo170K
unzip images.zip


# Evaluation Script
python test_qwen2vl_geoqa.py

# tested scores: 
# Qwen2VL-7B-Instruct: 30.63%
# Qwen2VL-7B-Instruct-GRPO-2epochs: 38.72%

# Qwen2.5VL-3B-Instruct: 35.41%
# Qwen2.5VL-3B-Instruct-GRPO-1epochs: 47.48%
```

To enable faster inference with multiple GPUs, you could also use the script in `R1-V/src/scripts/test_grpo_geoqa_multigpu.sh`
```
bash src/scripts/test_grpo_geoqa_multigpu.sh
```



