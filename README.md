## Test-Time Training with Masked Autoencoders<br><sub>Official PyTorch Implementation</sub>

### [Paper](https://arxiv.org/abs/2209.07522) | [Project Page](https://yossigandelsman.github.io/ttt_mae/index.html)

[Yossi Gandelsman*](https://yossigandelsman.github.io/), [Yu Sun*](https://yueatsprograms.github.io/), [Xinlei Chen](https://xinleic.xyz/) and [Alexei A. Efros](https://people.eecs.berkeley.edu/~efros/)

![Teaser](images/teaser.png)

### Setup
We provide an [`environment.yml`](environment.yml) file that can be used to create a Conda environment:

```bash
conda env create -f environment.yml
conda activate ttt
```

### Training MAE
To train a model on the main task, please use the code base from [Masked Autoencoders Are Scalable Vision Learners](https://github.com/facebookresearch/mae).
We provided a self-contained code for training here as well. Please run:

```bash
TIME=$(date +%s%3N)
DATA_PATH='...'
OUTPUT_DIR='...'
python -m torch.distributed.launch --nproc_per_node=8 main_pretrain.py \
        --data_path ${DATA_PATH} \
        --model mae_vit_large_patch16 \
        --input_size 224 \
        --batch_size 64 \
        --mask_ratio 0.75 \
        --warmup_epochs 5 \
        --epochs 90 \
        --blr 1e-4 \
        --save_ckpt_freq 30 \
        --output_dir ${OUTPUT_DIR}  \
        --dist_url "file://$OUTPUT_DIR/$TIME"
```

Alternatively, you can use a pretrained large VIT model from [here](https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large_full.pth):

```bash
mkdir checkpoints
cd checkpoints
wget https://dl.fbaipublicfiles.com/mae/pretrain/mae_pretrain_vit_large_full.pth
```


### BibTeX

bibtex```
@article{maettt, 
        title={Test-Time Training with Masked Autoencoders},
        author={Gandelsman, Yossi and Sun, Yu and Chen, Xinlei and Efros, Alexei A.},
        year={2022},
        journal={arXiv preprint arXiv:2209.07522}
}
```