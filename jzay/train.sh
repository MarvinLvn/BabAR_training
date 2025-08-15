#!/bin/bash
#SBATCH --job-name=phoneme_train
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Initialize variables
NETWORK_NAME=""
FREEZE=""
FREEZE_TRANSFORMER=""
LR=""
BATCH_SIZE=""
MAX_EPOCHS=""
SCHEDULER=""
WANDB_PROJECT=""
LIMIT_TRAIN_BATCHES=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --network_name)
            NETWORK_NAME="$2"
            shift 2
            ;;
        --freeze)
            FREEZE="$2"
            shift 2
            ;;
        --freeze_transformer)
            FREEZE_TRANSFORMER="$2"
            shift 2
            ;;
        --lr)
            LR="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max_epochs)
            MAX_EPOCHS="$2"
            shift 2
            ;;
        --scheduler)
            SCHEDULER="$2"
            shift 2
            ;;
        --wandb_project)
            WANDB_PROJECT="$2"
            shift 2
            ;;
        --limit_train_batches)
            LIMIT_TRAIN_BATCHES="$2"
            shift 2
            ;;
        *)
            echo "Unknown option $1"
            exit 1
            ;;
    esac
done

# Check required arguments
if [ -z "$NETWORK_NAME" ]; then
    echo "Error: --network_name is required"
    exit 1
fi

if [ -z "$FREEZE" ]; then
    echo "Error: --freeze is required"
    exit 1
fi

if [ -z "$FREEZE_TRANSFORMER" ]; then
    echo "Error: --freeze_transformer is required"
    exit 1
fi

if [ -z "$LR" ]; then
    echo "Error: --lr is required"
    exit 1
fi

if [ -z "$BATCH_SIZE" ]; then
    echo "Error: --batch_size is required"
    exit 1
fi

if [ -z "$MAX_EPOCHS" ]; then
    echo "Error: --max_epochs is required"
    exit 1
fi

if [ -z "$WANDB_PROJECT" ]; then
    echo "Error: --wandb_project is required"
    exit 1
fi

if [ -z "$LIMIT_TRAIN_BATCHES" ]; then
    echo "Error: --limit_train_batches is required"
    exit 1
fi

mkdir -p logs
source ~/.bashrc
conda activate phorec

DATASET_PATH="/lustre/fsn1/projects/rech/xdz/uow84uh/DATA/TinyVox"
INVENTORY_PATH="/lustre/fsn1/projects/rech/xdz/uow84uh/DATA/TinyVox/unique_phonemes.json"

CMD="python main.py --gpu 1 --num_workers 8 --network_name $NETWORK_NAME --train True --lr $LR --batch_size $BATCH_SIZE --max_epochs $MAX_EPOCHS --wandb_project $WANDB_PROJECT --dataset_path $DATASET_PATH --inventory_path $INVENTORY_PATH --freeze $FREEZE --freeze_transformer $FREEZE_TRANSFORMER --limit_train_batches $LIMIT_TRAIN_BATCHES"

if [ -n "$SCHEDULER" ]; then
    CMD="$CMD --scheduler $SCHEDULER"
fi

echo "Running: $CMD"
$CMD