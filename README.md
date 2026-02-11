# PepInter

This is the official implementation for the paper titled 'XXX'.

![](./assets/f1.png)

## Setups

```bash
git clone https://github.com/MolAstra/PepInter.git
cd PepInter

conda create -n pepinter python=3.12 -y
conda activate pepinter

pip install -e .
```

## Datasets and Model Checkpoints

- The prediction code is tested on GPU A800 with `python==3.12` and `torch==2.7.1+cu126`.
- The raw PepPBA dataset can be found at `datasets` dir.
- The trained model ckpt can be found at [zenodo](https://zenodo.org/records/XXXX)

```bash
pepinter predict --input_path input.csv \
    --output_path output.csv \
    --model_path pepinter_step=882-val_loss=0.51.ckpt \
    --seed 42 \
    --device cuda:0
```

## Webserver

- [PepInter Webserver](https://molastra.com/projects/pepinter)

## Training

- the training framework is depend on the project named `astra` which will be released in comming months
  The training framework is built upon an internal project named Astra,
  which will be released in the coming months. [astra-open](https://github.com/zhaisilong/astra-open.git)

More details will be provided soon.
