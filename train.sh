data_path="/data/datasets/chinese-alpaca-lora/trans_chinese_alpaca_data.json"
model_dir="/data/models/LLaMA/7B-hf/"

torchrun --nproc_per_node=8 finetune.py \
    --base_model $model_dir \
    --data_path $data_path \
    --output_dir './lora-alpaca'
    --batch_size 256 \
    --micro_batch_size 8 \
    --num_epochs 3 \
    --learning_rate 1e-4 \
    --cutoff_len 512 \
    --val_set_size 2000 \
    --lora_r 8 \
    --lora_alpha 16 \
    --lora_dropout 0.05 \
    --lora_target_modules '[q_proj,v_proj]' \
    --train_on_inputs \
    --group_by_length

