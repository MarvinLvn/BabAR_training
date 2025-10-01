# Installation

To install the dependencies, you can run: 

```shell
git clone https://github.com/ASR-project/Multilingual-PR.git
cd Multilingual-PR
conda create -n phorec python=3.10
conda activate phorec
pip install torch>=2.8.0 torchvision>=0.23.0 torchaudio>=2.8.0 --index-url https://download.pytorch.org/whl/cu128
module load ffmpeg # make sure ffmpeg is installed
pip install -r requirements.txt
conda install conda-forge/label/kenlm_dev::kenlm # to train N-grams and use them during beam search
```

Next, we will split TinyVox into train/dev/test sets:

```shell
python utils/split_tinyvox.py --data ../tinyvox/TinyVox
```

To train a model, you can run:

```shell
unset WANDB_SILENT
unset WANDB_MODE
python main.py --gpu 1 --num_workers 8 --network_name WavLM --train True --num_proc 8 --lr 2e-2 --wandb_project probing
# Debug mode without wandb logging:
export WANDB_SILENT=true    
export WANDB_MODE=disabled
python main.py --gpu 1 --num_workers 0 --network_name WavLM --train True --num_proc 8 --lr 2e-2 --dev_run --wandb_project dev_run
python main.py --gpu 1 --num_workers 8 --network_name WavLM --train True --num_proc 8 --lr 2e-2 --wandb_project probing


python main.py --gpu 1 --num_workers 8 --network_name Wav2Vec2XLSR --train True --freeze_transformer False --freeze True --num_proc 8 --lr 1e-4 --dataset_path /scratch2/mlavechin/tinyvox/TinyVox --inventory_path  /scratch2/mlavechin/tinyvox/TinyVox/unique_phonemes.json --early_stopping False --log_freq_audio 1 --wandb_project wav2vec2xlsr  --batch_size 16 --accumulate_grad_batches 4 --scheduler TriStage --warmup_steps 35000 --total_training_steps 100000 --max_epochs 18 --use_vad
 ```

Test:

```shell
python main.py --train False --language ru --subset ru  --network_name WavLM --best_model_run WavLM_ru_tf_freezed
```

To evaluate pretrained phoneme recognizers, you can run:

```shell
python evaluate_pretrained.py --dataset_path /scratch2/mlavechin/tinyvox/TinyVox --network_name Wav2Vec2 --pretrained_name wav2vec_children_ASR/save_100h_MyST_Providence --batch_size 64 --save_details --split test --use_vad
```


# Dependencies for Jialu Li's model

```shell
conda install git-lfs
git clone https://huggingface.co/lijialudew/wav2vec_children_ASR
cd wav2vec_children_ASR
git lfs pull
pip install speechbrain
```

