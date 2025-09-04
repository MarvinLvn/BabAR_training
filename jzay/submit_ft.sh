#!/bin/bash

# MODELS=("WavLMplus" "Wav2Vec2XLSR") already trained
# MODELS=("Hubert" "WavLM" "Wav2Vec2" "WavLMplus" "Wav2Vec2XLSR")
MODELS=("Hubert" "WavLM" "Wav2Vec2")

for model in "${MODELS[@]}"; do
    sbatch train.sh \
      --network_name $model \
      --freeze True \
      --freeze_transformer False \
      --lr 1e-5 \
      --batch_size 16 \
      --accumulate_grad_batches 4 \
      --hparams.max_epochs 18 \
      --wandb_project ft_wavlm_vs_wav2vec2xlsr \
      --early_stopping False \
      --log_freq_audio 1 \
      --precision 16 \
      --use_vad \
      --conditional_transformer_unfreezing \
      --transformer_unfreeze_step 10000 \
      --scheduler TriStage \
      --total_training_steps 100000 \
      --warmup_steps 10000 \
      --val_check_interval 0.5 \
      --limit_train_batches 1.0
    sleep 1
done
