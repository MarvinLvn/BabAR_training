
DURATIONS=(0 5 10 15 20)

for CONTEXT_DURATION in "${DURATIONS[@]}"; do
  sbatch train.sh \
    --network_name BabyHubert \
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
    --val_check_interval 1.0 \
    --limit_train_batches 1.0 \
    --context_duration $CONTEXT_DURATION
  sleep 1
done;
