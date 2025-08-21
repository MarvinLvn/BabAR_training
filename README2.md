# Installation

To install the dependencies, you can run: 

```shell
conda env create -f conda_environment.yml
conda activate phorec
pip install -r requirements.txt
pip install pytorch_lightning==1.5.10 lightning-bolts==0.5.0
pip install transformers --upgrade
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
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


python main.py --gpu 1 --num_workers 8 --network_name WavLM --train True --freeze_transformer True --num_proc 8 --lr 2e-2 --dataset_path /scratch2/mlavechin/tinyvox/TinyVox_debug --inventory_path /scratch2/mlavechin/tinyvox/TinyVox_debug/unique_phonemes.json --early_stopping False --max_epochs 200 --log_freq_audio 10 --wandb_project overfit```

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