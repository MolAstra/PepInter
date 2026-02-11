# PepInter

This is the official implementation for the paper titled 'XXX'.

![](./assets/f1.png)

## Setups

```bash
git clone https://github.com/MolAstra/PepInter.git
cd PepInter

mamba create -n pepinter python=3.12 -y
mamba activate pepinter

pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

pip install -e .
pip install -e ".[serve]"  # [Optinal] for sever deployment
```

## Datasets and Model Checkpoints

- The prediction code is tested on GPU A800 with `python==3.12` and `torch==2.7.1+cu126`.
- The raw PepPBA dataset can be found at `datasets` dir.
- The trained model ckpt can be found at [zenodo](https://zenodo.org/records/XXXX)

```bash
# affinity, cls
pepinter predict \
    --input_path example_input.csv \
    --output_path ./pred_cls.csv \
    --task "cls" \
    --model_path ./ckpt/pepinter_cls_step741516.ckpt \
    --seed 42 \
    --gpus 2
```

## Webserver

- [PepInter Webserver](https://molastra.com/projects/pepinter)

## Training

- the training framework is depend on the project named `astra` which will be released in comming months
  The training framework is built upon an internal project named Astra,
  which will be released in the coming months. [astra-open](https://github.com/zhaisilong/astra-open.git)
- More details will be provided soon.
- If you have any question please email to `zhaisilong@outlook.com`

## Citaion

```bash

```
