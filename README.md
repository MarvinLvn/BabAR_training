<p align="center">
  <img src="assets/BabAR_logo.png" alt="BabAR Logo" width="300">
</p>

Welcome to **BabAR_training**, a repository to fine-tune self-supervised speech models for phoneme recognition using a CTC loss.
It's built around [TinyVox](https://github.com/MarvinLvn/TinyVox), a large-scale dataset of IPA-transcribed children's utterances, but can be adapted to other datasets and languages.
This repo was initially forked from [Multilingual-PR](https://github.com/ASR-project/Multilingual-PR), but has been extensively modifed since then. 

## Installation

To install the dependencies, you can run: 

```shell
git clone https://github.com/ASR-project/Multilingual-PR.git
cd Multilingual-PR
conda create -n phorec python=3.10
conda activate phorec
pip install torch>=2.8.0 torchvision>=0.23.0 torchaudio>=2.8.0 --index-url https://download.pytorch.org/whl/cu128
module load ffmpeg # make sure ffmpeg is installed
pip install -r requirements.txt
```

## Data preparation 

If you want to resplit TinyVox into train/dev/test sets:

```shell
python utils/split_tinyvox.py --data ../tinyvox/TinyVox
```

Otherwise, you can use the same split as we did since we prove {train,val,test}.csv in TinyVox directly.

## Data formatting

The code expects TinyVox to have the following structure:

```
├── audio/
├── original/
├── metadata.csv
├── train.csv
├── val.csv
└── test.csv
```

with:
- `audio/` containing children's utterances (one .wav = one utterance). This folder is not used in my codebase as I train BabAR with surrounding audio context, but should be usable for any pipeline that trains ASR systems without context. 
- `original/` containing the original audio files (i.e., usually hour long audio files containing the recorded activity). These are the original audio files found in PhonBank.
- `metadata.csv` containing various metadta (one line = one utterance) as detailed in the [TinyVox git repository](https://github.com/MarvinLvn/tinyvox).
- `{train,val,test}.csv` containing the metadata split across training, validation, and test set as detailed in the [TinyVox git repository](https://github.com/MarvinLvn/tinyvox).

## Model training

To train a model, you can run:

```shell

python train.py --network_name <NETWORK_NAME> --context_duration <CONTEXT_DURATION> --wandb_project <WANDB_PROJECT> \
  --seed 0 --freeze True --freeze_transformer False --conditional_transformer_unfreezing \
  --transformer_unfreeze_step 10000 --scheduler TriStage --total_training_steps 100000 --warmup_steps 10000 \
  --val_check_interval 1.0 --limit_train_batches 1.0 \
  --lr 1e-4 --batch_size 16 --accumulate_grad_batches 4 \
  --max_epochs 21 --early_stopping False --log_freq_audio 3 --precision 16 --num_workers 4
```

where:
- <NETWORK_NAME> is Wav2Vec2, Hubert, WavLM, Wav2Vec2XLSR, W2VLB, or BabyHubert
- <CONTEXT_DURATION> is the size of the context window in seconds (we found 20s performs best on TinyVox)
- <WANDB_PROJECT> is the name of this experiment (for logging in wandb)

## Model evaluation 

Once you're done training, you can validate or evaluate the model using:

```shell
python evaluate.py --checkpoint_path <CHECKPOINT_PATH> --split <SPLIT> --context_duration <CONTEXT_DURATION> --batch_size 16 --save_details
```

where:
- <CHECKPOINT_PATH> is the path to the .pt files
- <SPLIT> is val or test
- <CONTEXT_DURATION> is the size of the context window (s) used during training


