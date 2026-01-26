#!/bin/bash

# MODELS=("WavLMplus" "Wav2Vec2XLSR") already trained
# MODELS=("Hubert" "WavLM" "Wav2Vec2" "WavLMplus" "Wav2Vec2XLSR")
MODELS=("Wav2Vec2 "Hubert" "WavLM" "Wav2Vec2XLSR" "W2VLB" "BabyHubert")
SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  for model in "${MODELS[@]}"; do
      sbatch train.sh \
        --network_name $model \
        --seed $SEED \
        --freeze True \
        --freeze_transformer False \
        --lr 1e-5 \
        --batch_size 32 \
        --accumulate_grad_batches 2 \
        --hparams.max_epochs 21 \
        --wandb_project babar_finetuning \
        --early_stopping False \
        --log_freq_audio 3 \
        --precision 16 \
        --conditional_transformer_unfreezing \
        --transformer_unfreeze_step 10000 \
        --scheduler TriStage \
        --total_training_steps 100000 \
        --warmup_steps 10000 \
        --val_check_interval 1.0 \
        --limit_train_batches 1.0 \
        --context_duration 0 \
        --num_workers 4
      sleep 1
  done
done
