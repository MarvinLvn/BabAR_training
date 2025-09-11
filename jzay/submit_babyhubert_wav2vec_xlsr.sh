#!/bin/bash

LRS=(1e-5 1e-6)
MODELS=("BabyHubert" "Wav2Vec2XLSR")

for LR in "${LRS[@]}"; do
  for model in "${MODELS[@]}"; do
      sbatch train.sh \
        --network_name $model \
        --freeze True \
        --freeze_transformer False \
        --lr ${LR} \
        --batch_size 16 \
        --accumulate_grad_batches 4 \
        --hparams.max_epochs 18 \
        --wandb_project ft_babyhubert_vs_wav2vec_xlsr \
        --early_stopping False \
        --log_freq_audio 3 \
        --precision 16 \
        --conditional_transformer_unfreezing \
        --transformer_unfreeze_step 10000 \
        --scheduler TriStage \
        --total_training_steps 100000 \
        --warmup_steps 10000 \
        --val_check_interval 0.5 \
        --limit_train_batches 1.0
      sleep 1
  done
done