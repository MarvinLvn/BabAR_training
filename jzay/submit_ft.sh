
MODELS=("WavLM" "WavLMplus")

for model in "${MODELS[@]}"; do
    echo "Submitting $model with freeze_transformer False"
    sbatch train.sh \
        --network_name $model \
        --freeze True \
        --freeze_transformer False \
        --lr 1e-3 \
        --batch_size 64 \
        --max_epochs 40 \
        --wandb_project transformers_ft \
        --limit_train_batches 0.5
    sleep 1
done