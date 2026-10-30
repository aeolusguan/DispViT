# DispViT: Direct Stereo Disparity Regression with a Single-Stream Vision Transformer

Official implementation of **DispViT**, published at [ICLR 2026](https://openreview.net/forum?id=c21yqwf02V).

**DispViT+** adds learned disparity refinement to DispViT.

## Released checkpoints

| Model card | #Params | License |
| --- | --- | --- |
| [aeolusguan/dispvit-vitg](https://huggingface.co/aeolusguan/dispvit-vitg) | 1.26B | [Apache 2.0](LICENSE) |
| [aeolusguan/dispvit-plus-vitg](https://huggingface.co/aeolusguan/dispvit-plus-vitg) | 1.28B | [Apache 2.0](LICENSE) |

Released checkpoints include the weights and architecture needed for loading.

## TODO

- [ ] Release a ViT-G model self-distilled on real-world data.
- [x] Release a DispViT+ checkpoint, including learned refinement.
- [ ] Release checkpoints for smaller model variants.

## Installation

Use Python 3.10 or newer. Clone the repository and enter its root directory:

```bash
git clone https://github.com/aeolusguan/DispViT.git
cd DispViT
```

Install PyTorch 2.0 or newer and a matching torchvision build for your CUDA
environment using the [PyTorch installation instructions](https://pytorch.org/get-started/locally/).
Then install the dependencies in [requirements-eval.txt](requirements-eval.txt):

```bash
python -m pip install -r requirements-eval.txt
```

The default evaluation uses BF16 automatic mixed precision on a CUDA GPU.

## Loading the model

Load the released checkpoint directly from Hugging Face:

```python
from dispvit.models.dispvit import DispViT

model = DispViT.from_pretrained("aeolusguan/dispvit-vitg").cuda().eval()
```

### DispViT+

Load DispViT+ with its refinement module:

```python
from dispvit.models.dispvit_plus import DispViTPlus

model = DispViTPlus.from_pretrained("aeolusguan/dispvit-plus-vitg").cuda().eval()
```

Both methods also accept a local checkpoint path in place of the Hugging Face ID.

## Zero-shot evaluation

Zero-shot evaluation uses a standalone evaluator and its own
[configuration](config/zero_shot_eval.yaml), covering KITTI 2012, KITTI 2015,
ETH3D, and Middlebury. Place only the selected benchmarks under `datasets/` in
the repository root. Run the following command from the repository root:

```bash
python evaluate.py pretrained=aeolusguan/dispvit-vitg
```

DispViT is the default model preset. For DispViT+, select its preset and
released checkpoint:

```bash
python evaluate.py model=dispvit_plus \
  pretrained=aeolusguan/dispvit-plus-vitg
```

To use a local checkpoint:

```bash
python evaluate.py pretrained=/path/to/model_final.pt
```

To select individual benchmarks, override the dataset list:

```bash
python evaluate.py pretrained=aeolusguan/dispvit-vitg \
  'data.val_datasets=[{eth3d_nonocc:null}]'
```

Use `torchrun --standalone --nproc_per_node=4 evaluate.py ...` for evaluation
across four GPUs.

Evaluation metrics are printed to the terminal.

### Results

All error rates are reported as percentages; lower is better.

| Model | KITTI 2012 (D1 ↓) | KITTI 2015 (D1 ↓) | ETH3D (BP-1 ↓) | Middlebury (BP-2 ↓) |
| --- | --- | --- | --- | --- |
| dispvit-vitg | 3.03 | 3.30 | 1.00 | 3.82 |
| dispvit-plus-vitg | 2.67 | 3.00 | 0.52 | 2.26 |

## License

Original DispViT code and contributions are copyright (c) 2026 CUHK T-Stone
Robotics Institute and are released under the [Apache License 2.0](LICENSE).
Attribution information is provided in [NOTICE](NOTICE).

When redistributing this code or derivative works, include a copy of the
license, retain applicable copyright and attribution notices, mark modified
files, and reproduce applicable notices from `NOTICE` as required by Section 4
of the license.

Third-party code retains its original copyright notices and license terms.

For research that uses or builds on DispViT, please cite the paper in
[Citation](#citation).

### Checkpoint licenses

Project-created checkpoint weights are licensed under the [Apache License 2.0](LICENSE), subject
to applicable upstream component license terms.
When distributing a checkpoint separately, include `LICENSE`, `NOTICE`, and
applicable upstream attribution notices alongside the checkpoint file.
For each release on Hugging Face, declare the applicable license in the model
card's YAML metadata, using `license: apache-2.0` for Apache 2.0 releases.

Third-party pretrained components retain their original license terms,
including when incorporated into a DispViT or DispViT+ checkpoint.
[DINOv2 code and model weights](https://github.com/facebookresearch/dinov2#license)
are released under Apache 2.0. The
[Depth Anything V2 model licenses](https://github.com/DepthAnything/Depth-Anything-V2#license)
are CC BY-NC 4.0 for the Base/Large checkpoints used during training. Those terms
continue to apply to checkpoints incorporating the corresponding weights.
Each checkpoint's release documentation or model card should identify its
pretrained components and applicable license terms.

## Acknowledgements

We use pretrained checkpoints from [DINOv2](https://github.com/facebookresearch/dinov2)
and [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) during
training. See [Checkpoint licenses](#checkpoint-licenses) for their applicable terms.

This repository includes code adapted from [VGGT](https://github.com/facebookresearch/vggt).
We thank the authors for making their code and pretrained weights publicly available.

## Citation

If you find DispViT useful in your research, please cite our [ICLR 2026 paper](https://openreview.net/forum?id=c21yqwf02V):

```bibtex
@inproceedings{guan2026dispvit,
  title={DispViT: Direct Stereo Disparity Regression with a Single-Stream Vision Transformer},
  author={Guan, Tongfan and Guo, Jiaxin and Huang, Tianyu and Dong, Jinhu and Wang, Chen and Liu, Yun-Hui},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026}
}
```
