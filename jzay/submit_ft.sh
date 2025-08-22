#!/bin/bash

MODELS=("WavLMplus" "Wav2Vec2XLSR")
LRS=(1e-4 1e-5)

for model in "${MODELS[@]}"; do
  for lr in "${LRS[@]}"; do
    sbatch train.sh \
      --network_name $model \
      --freeze True \
      --freeze_transformer False \
      --lr $lr \
      --batch_size 16 \
      --accumulate_grad_batches 4 \
      --hparams.max_epochs 18 \
      --wandb_project ft_wavlm_vs_wav2vec2xlsr \
      --limit_train_batches 1 \
      --early_stopping False \
      --log_freq_audio 1 \
      --precision 16 \
      --use_vad \
      --conditional_transformer_unfreezing \
      --transformer_unfreeze_step 10000 \
      --scheduler TriStage \
      --total_training_steps 100000 \
      --warmup_steps 10000
    sleep 1
  done
done
