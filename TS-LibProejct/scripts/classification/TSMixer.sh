# 第1个参数：GPU ID（默认 0）
GPU_ID=${1:-0}

# 第2个参数：mode（默认 "base"）
MODE=${2:-"base"}

export CUDA_VISIBLE_DEVICES=$GPU_ID

python -u run.py \
  --task_name classification \
  --is_training 1 \
  --root_path ./myDataK \
  --model_id tessTSMixer \
  --model TSMixer \
  --data UEA \
  --seq_len 512 \
  --enc_in 1 \
  --batch_size 32 \
  --d_model 64 \
  --d_ff 128 \
  --e_layers 2 \
  --dropout 0.1 \
  --learning_rate 0.001 \
  --train_epochs 100 \
  --patience 10 \
  --itr 1 \
  --mode "$MODE"