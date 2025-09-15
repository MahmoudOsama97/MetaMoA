# MetaMoA: Top-Down Dynamic Guidance for Parameter-Efficient Domain Generalization

by Mahmoud Soliman, Ahmed Redwan, Omar Abdelaziz, Mohamed Sami Shehata

### Prerequisites 

Dataset Preparation

```bash
python -m domainbed.scripts.download --data_dir=/my/datasets/path
```

Environment Setup

```bash
conda create -n MoA python=3.9.12
conda activate MoA
pip install -r requirements.txt
```

### Training
We use [OpenCLIP ViT-B/16](https://huggingface.co/laion/CLIP-ViT-B-16-laion2B-s34B-b88K) for all experiments. The pretrained model can be loaded from timm. You can use the following command to get the model.


```bash
python train_all.py MetaMoA_full_router_reg_weight \
    --data_dir data \
    --algorithm MetaMoA \
    --dataset OfficeHome \
    --model nf_vitbase_moek_every_qkv_new \
    --seed 1 \
    --steps 10000 \
    --l_aux \
    --use_router_reg True \
    --router_reg_weight 0.2 \
    --meta_lr 1e-5 \
    --meta_update_freq 1 \
    --diversity_weight 0.01 \
    --router_type mlp \
    --hidden_dim 256 \
```

### Results

<p align="center">
    <img src="./assets/results.png" width="70%" />
</p>

### Acknowledgements

This code is heavily based on [MoA](https://github.com/cvlab-kaist/MoA), [MIRO](https://github.com/kakaobrain/miro), [SWAD](https://github.com/khanrc/swad) and [DomainBed](https://github.com/facebookresearch/DomainBed). Also, the LoRA implementation is based on [LoRA](https://github.com/cloneofsimo/LoRA). We also used the official implementation of [KAdaptation](https://github.com/eric-ai-lab/PEViT), and the [Cosine Router](https://arxiv.org/abs/2206.04046) using [this](https://github.com/Luodian/Generalizable-Mixture-of-Experts) github. We highly appreciate the authors for their great work.

### Citation
If you found this code useful, please consider citing our paper.

