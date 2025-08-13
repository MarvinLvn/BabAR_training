# Installation

```shell
conda env create -f conda_environment.yml
conda activate phorec
pip install -r requirements.txt
pip install pytorch_lightning==1.5.10 lightning-bolts==0.5.0
pip install transformers --upgrade
pip install torch==1.13.1+cu117 torchvision==0.14.1+cu117 torchaudio==0.13.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
```

Split:

```shell
python utils/split_tinyvox.py --data ../tinyvox/TinyVox
```

Train:

```shell
python main.py --gpu 1 --num_workers 2 --network_name WavLM --train True --num_proc 1 --lr 2e-2 --dev_run
```

Test:

```shell
python main.py --train False --language ru --subset ru  --network_name WavLM --best_model_run WavLM_ru_tf_freezed
```

# For Jialu Li's model

```shell
conda install git-lfs
git clone https://huggingface.co/lijialudew/wav2vec_children_ASR
cd wav2vec_children_ASR
git lfs pull
pip install speechbrain
```