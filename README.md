# DispViT: Direct Stereo Disparity Regression with a Single-Stream Vision Transformer

Official implementation of **DispViT**, published at [ICLR 2026](https://openreview.net/forum?id=c21yqwf02V).

**DispViT+** (DispViT Plus) is the full model with learned refinement, implemented
as [`DispViTPlus`](dispvit/models/dispvit_plus.py). Its training configs use the
`dispvit_plus` prefix, for example `--config-name dispvit_plus_all_syn`.

## Released checkpoints

Released models include the DispViT regressor and the full DispViT+ model with
learned refinement.

| Model card | #Params | License |
| --- | --- | --- |
| [aeolusguan/dispvit-vitg](https://huggingface.co/aeolusguan/dispvit-vitg) | 1.26B | [Apache 2.0](LICENSE) |
| [aeolusguan/dispvit-plus-vitg](https://huggingface.co/aeolusguan/dispvit-plus-vitg) | 1.28B | [Apache 2.0](LICENSE) |

Each checkpoint includes the model weights and architecture configuration.
Load the regressor with `DispViT.from_pretrained()` and the full model with
`DispViTPlus.from_pretrained()`.

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
Then install the dependencies needed for evaluation:

```bash
python -m pip install 'hydra-core>=1.3,<1.4' huggingface-hub einops numpy Pillow opencv-python imageio iopath wcmatch
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

### DispViT+

A DispViT+ release checkpoint includes both the DispViT and refinement weights,
plus the complete architecture configuration:

```python
from dispvit.models.dispvit_plus import DispViTPlus

model = DispViTPlus.from_pretrained("aeolusguan/dispvit-plus-vitg").cuda().eval()
```

A local directory containing `model_final.pt` or a Hugging Face repository ID
can be passed to the same API. Loading restores the complete model without
separate backbone, extractor, or DispViT checkpoint downloads.

Regressor initialization is controlled by its own config. For refinement
training with a pretrained DispViT, configure it directly:

```yaml
regressor:
  _target_: dispvit.models.dispvit.DispViT.from_pretrained
  pretrained_model_name_or_path: aeolusguan/dispvit-vitg
```

`DispViTPlus` instantiates and freezes this regressor. Full-model exports store
its architecture and weights, so deployment does not depend on the original
regressor checkpoint location.

Training's checkpoint exporter records the architecture from the model config.
`from_pretrained()` passes this config directly to the constructor, with optional
`model_kwargs` overrides. For an older checkpoint without `model_config`, supply
the matching constructor settings with pretrained initialization disabled:

```python
from omegaconf import OmegaConf

config = OmegaConf.load("config/dispvit_plus_all_syn.yaml").model
config.regressor = {
    "_target_": "dispvit.models.dispvit.DispViT",
    "encoder_type": "vitg",
    "groups": 8,
    "init_weights": False,
}
config.extractor._init_weights = False
model = DispViTPlus.from_pretrained(
    "/path/to/older_checkpoint.pt",
    model_kwargs=OmegaConf.to_container(config, resolve=True),
).cuda().eval()
```

Use the config matching the checkpoint's backbone, extractor, and refinement
settings. To export it for subsequent loading without `model_kwargs`:

```python
import torch
from training.train_utils.checkpoint import export_model_config

torch.save(
    {"model": model.state_dict(), "model_config": export_model_config(config, model)},
    "/path/to/model_final.pt",
)
```

## Zero-shot evaluation

Zero-shot evaluation uses a standalone evaluator and its own
[configuration](config/zero_shot_eval.yaml), covering KITTI 2012, KITTI 2015,
ETH3D, and Middlebury. Place only the selected benchmarks under `datasets/` in
the repository root. SceneFlow is needed only if you select `things` for
evaluation. Run the following command from the repository root:

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
[Citation](#citation). This citation request is separate from the license
conditions.

### Checkpoint licenses

Project-created checkpoint weights in all current and future DispViT and
DispViT+ releases are licensed under the [Apache License 2.0](LICENSE), subject
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
