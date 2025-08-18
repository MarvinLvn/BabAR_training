#!/bin/bash
#SBATCH --job-name=phoneme_train
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#SBATCH --time=20:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=3
#SBATCH -C v100-32g
#SBATCH --hint=nomultithread
#SBATCH -A xdz@v100

WORK_DIR=/linkhome/rech/genscp01/uow84uh/Multilingual-PR
cd $WORK_DIR

# Initialize variables
NETWORK_NAME=""
FREEZE=""
FREEZE_TRANSFORMER=""
LR=""
BATCH_SIZE=""
ACCUMULATE_GRAD_BATCHES=""
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
        --accumulate_grad_batches)
            ACCUMULATE_GRAD_BATCHES="$2"
            shift 2
            ;;
        --hparams.max_epochs)
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

if [ -z "$ACCUMULATE_GRAD_BATCHES" ]; then
    echo "Error: --accumulate_grad_batches is required"
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

export WANDB_MODE=offline

# Build the command with all required parameters
CMD="python main.py \
    --gpu 1 \
    --num_workers 8 \
    --freeze $FREEZE \
    --freeze_transformer $FREEZE_TRANSFORMER \
    --limit_train_batches $LIMIT_TRAIN_BATCHES \
    --network_name $NETWORK_NAME \
    --train True \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --hparams.max_epochs $MAX_EPOCHS \
    --wandb_project $WANDB_PROJECT \
    --dataset_path $DATASET_PATH \
    --inventory_path $INVENTORY_PATH"

# Add optional scheduler parameter if provided
if [ -n "$SCHEDULER" ]; then
    CMD="$CMD --scheduler $SCHEDULER"
fi

echo "Running: $CMD"
$CMD