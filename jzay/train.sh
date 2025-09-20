#!/bin/bash
#SBATCH --job-name=phoneme_train
#SBATCH --output=logs/train-%j.out
#SBATCH --error=logs/train-%j.err
#SBATCH --qos=qos_gpu-t4
#SBATCH --time=100:00:00
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=10
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
EARLY_STOPPING=""
LOG_FREQ_AUDIO=""
PRECISION=""
USE_VAD=""
CONDITIONAL_TRANSFORMER_UNFREEZING=""
TRANSFORMER_UNFREEZE_STEP=""
TOTAL_TRAINING_STEPS=""
WARMUP_STEPS=""
VAL_CHECK_INTERVAL=""
CONTEXT_DURATION=""
NUM_WORKERS=""

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
        --early_stopping)
            EARLY_STOPPING="$2"
            shift 2
            ;;
        --log_freq_audio)
            LOG_FREQ_AUDIO="$2"
            shift 2
            ;;
        --precision)
            PRECISION="$2"
            shift 2
            ;;
        --num_workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --use_vad)
            USE_VAD="True"
            shift 1
            ;;
        --conditional_transformer_unfreezing)
            CONDITIONAL_TRANSFORMER_UNFREEZING="True"
            shift 1
            ;;
        --transformer_unfreeze_step)
            TRANSFORMER_UNFREEZE_STEP="$2"
            shift 2
            ;;
        --total_training_steps)
            TOTAL_TRAINING_STEPS="$2"
            shift 2
            ;;
        --warmup_steps)
            WARMUP_STEPS="$2"
            shift 2
            ;;
        --val_check_interval)
            VAL_CHECK_INTERVAL="$2"
            shift 2
            ;;
        --context_duration)
            CONTEXT_DURATION="$2"
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

# Set default num_workers if not provided
if [ -z "$NUM_WORKERS" ]; then
    NUM_WORKERS=4  # Default to 4 workers
fi

mkdir -p jzay/logs
source ~/.bashrc
conda activate phorec

DATASET_PATH="/lustre/fsn1/projects/rech/xdz/uow84uh/DATA/TinyVox"
INVENTORY_PATH="/lustre/fsn1/projects/rech/xdz/uow84uh/DATA/TinyVox/unique_phonemes.json"

export WANDB_MODE=offline
# Create job specific tmp dir (for audio logging)
TEMP_DIR="$WORK_DIR/tmp/job_${SLURM_JOB_ID}"
mkdir -p $TEMP_DIR
export TMPDIR=$TEMP_DIR

# Build the command with all required parameters
CMD="python main.py \
    --gpu 1 \
    --num_workers $NUM_WORKERS \
    --freeze $FREEZE \
    --freeze_transformer $FREEZE_TRANSFORMER \
    --limit_train_batches $LIMIT_TRAIN_BATCHES \
    --network_name $NETWORK_NAME \
    --train True \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --accumulate_grad_batches $ACCUMULATE_GRAD_BATCHES \
    --max_epochs $MAX_EPOCHS \
    --wandb_project $WANDB_PROJECT \
    --val_check_interval $VAL_CHECK_INTERVAL \
    --dataset_path $DATASET_PATH \
    --inventory_path $INVENTORY_PATH"

# Add optional parameters if provided
if [ -n "$SCHEDULER" ]; then
    CMD="$CMD --scheduler $SCHEDULER"
fi

if [ -n "$EARLY_STOPPING" ]; then
    CMD="$CMD --early_stopping $EARLY_STOPPING"
fi

if [ -n "$LOG_FREQ_AUDIO" ]; then
    CMD="$CMD --log_freq_audio $LOG_FREQ_AUDIO"
fi

if [ -n "$PRECISION" ]; then
    CMD="$CMD --precision $PRECISION"
fi

if [ "$USE_VAD" = "True" ]; then
    CMD="$CMD --use_vad"
fi

if [ "$CONDITIONAL_TRANSFORMER_UNFREEZING" = "True" ]; then
    CMD="$CMD --conditional_transformer_unfreezing"
fi

if [ -n "$TRANSFORMER_UNFREEZE_STEP" ]; then
    CMD="$CMD --transformer_unfreeze_step $TRANSFORMER_UNFREEZE_STEP"
fi

if [ -n "$TOTAL_TRAINING_STEPS" ]; then
    CMD="$CMD --total_training_steps $TOTAL_TRAINING_STEPS"
fi

if [ -n "$WARMUP_STEPS" ]; then
    CMD="$CMD --warmup_steps $WARMUP_STEPS"
fi

if [ -n "$CONTEXT_DURATION" ]; then
    CMD="$CMD --context_duration $CONTEXT_DURATION"
fi

echo "Running: $CMD"
$CMD