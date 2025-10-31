#!/bin/bash

LR=1e-5
CONTEXT_DURATION=15
BS=32
ACCUM=2
PROJECT=art_study

# Baseline model without articulatory heads
sbatch \
  --output=logs/parallel-%j.out \
  --error=logs/parallel-%j.err \
  train.sh \
  --network_name BabyHubert \
  --freeze True \
  --freeze_transformer False \
  --lr ${LR} \
  --batch_size 16 \
  --accumulate_grad_batches 4 \
  --hparams.max_epochs 30 \
  --wandb_project $PROJECT \
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
  --context_duration $CONTEXT_DURATION \
  --num_workers 4

sleep 1

# Models with articulatory heads at different weights
WEIGHTS=(0.001 0.01 0.1 1 10)

for WEIGHT in "${WEIGHTS[@]}"; do
  sbatch \
    --output=logs/parallel-%j.out \
    --error=logs/parallel-%j.err \
    train.sh \
    --network_name BabyHubert \
    --freeze True \
    --freeze_transformer False \
    --lr ${LR} \
    --batch_size $BS \
    --accumulate_grad_batches $ACCUM \
    --hparams.max_epochs 30 \
    --wandb_project $PROJECT \
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
    --context_duration $CONTEXT_DURATION \
    --num_workers 4 \
    --use_articulatory_heads \
    --articulatory_loss_weight $WEIGHT

  sleep 1
done

echo "Submitted 6 jobs total to project: $PROJECT"
echo "  1 baseline (no articulatory heads)"
echo "  5 with articulatory heads (weights: 0.001, 0.01, 0.1, 1, 10)"
echo "All with context_duration=$CONTEXT_DURATION, lr=$LR, batch_size=$BS, accumulate_grad_batches=$ACCUM"
