#!/bin/bash

DURATIONS=(5 10 15 20)
LR=1e-5
SEEDS=(0 1 2 3 4)

for SEED in "${SEEDS[@]}"; do
  for CONTEXT_DURATION in "${DURATIONS[@]}"; do
    ACCUM=2
    BS=32
    if [ "$CONTEXT_DURATION" -ge 20 ]; then
      ACCUM=4
      BS=16
    fi;
    sbatch train.sh \
      --network_name BabyHubert \
      --seed $SEED \
      --freeze True \
      --freeze_transformer False \
      --lr ${LR} \
      --batch_size $BS \
      --accumulate_grad_batches $ACCUM \
      --hparams.max_epochs 21 \
      --wandb_project babar_context \
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
  done;
done;
