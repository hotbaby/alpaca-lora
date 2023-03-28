
model_dir="/data/models/LLaMA/7B-hf/"
lora_weights="./lora-alpaca"

python generate.py \
    --load_8bit \
    --base_model $model_dir \
    --lora_weights $lora_weights
