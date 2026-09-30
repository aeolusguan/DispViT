# DispViT: Direct Stereo Disparity Regression with a Single-Stream Vision Transformer

Official implementation of **DispViT**, published at [ICLR 2026](https://openreview.net/forum?id=c21yqwf02V).

## Released checkpoint

**DispViT ViT-G** is currently the only released checkpoint.

| Model card | #Params |
| --- | --- |
| [aeolusguan/dispvit-vitg](https://huggingface.co/aeolusguan/dispvit-vitg) | 1.26B |

The checkpoint includes the model weights and architecture configuration,
which `DispViT.from_pretrained()` loads automatically.

## TODO

- [ ] Release a ViT-G model self-distilled on real-world data.
- [ ] Release the full model with NMRF refinement.
- [ ] Release checkpoints for smaller model variants.

## Installation

Use Python 3.10 or newer. Clone the repository and enter its root directory:

```bash
git clone https://github.com/aeolusguan/DispViT.git
cd DispViT
```

Install PyTorch 2.0 or newer and a matching torchvision build for your CUDA
environment using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Then install the dependencies needed for evaluation:

```bash
python -m pip install 'hydra-core>=1.3,<1.4' huggingface-hub numpy Pillow opencv-python imageio iopath wcmatch
```

The default evaluation uses BF16 automatic mixed precision on a CUDA GPU.

## Loading the model

Load the released checkpoint directly from Hugging Face:

```python
from dispvit.models.dispvit import DispViT

model = DispViT.from_pretrained("aeolusguan/dispvit-vitg").cuda().eval()
```

A downloaded checkpoint can be loaded with the same API:

```python
model = DispViT.from_pretrained("/path/to/model_final.pt").cuda().eval()
```

## Zero-shot evaluation

Place the evaluation datasets under `datasets/` in the repository root.
The [evaluation configuration](config/zero_shot_eval.yaml) covers KITTI 2012,
KITTI 2015, ETH3D, and Middlebury. Run the following command from the repository root:

```bash
torchrun --standalone --nproc_per_node=1 launch.py --config-name zero_shot_eval \
  pretrained=aeolusguan/dispvit-vitg
```

To use a local checkpoint:

```bash
torchrun --standalone --nproc_per_node=1 launch.py --config-name zero_shot_eval \
  pretrained=/path/to/model_final.pt
```

Evaluation metrics are printed to the terminal.

### Results

All error rates are reported as percentages; lower is better.

| Model | KITTI 2012 (D1 ↓) | KITTI 2015 (D1 ↓) | ETH3D (BP-1 ↓) | Middlebury (BP-2 ↓) |
| --- | --- | --- | --- | --- |
| dispvit-vitg | 3.03 | 3.30 | 1.00 | 3.82 |

## Acknowledgements

This repository includes code adapted from [VGGT](https://github.com/facebookresearch/vggt).
We thank the authors for making their code publicly available.

## Citation

If you find DispViT useful in your research, please cite our [ICLR 2026 paper](https://openreview.net/forum?id=c21yqwf02V):

```bibtex
@inproceedings{guan2026dispvit,
  title = {{DispViT}: Direct Stereo Disparity Regression with a Single-Stream Vision Transformer},
  author = {Guan, Tongfan and Guo, Jiaxin and Huang, Tianyu and Dong, Jinhu and Wang, Chen and Liu, Yun-Hui},
  booktitle = {International Conference on Learning Representations (ICLR)},
  year = {2026},
  url = {https://openreview.net/forum?id=c21yqwf02V}
}
```
