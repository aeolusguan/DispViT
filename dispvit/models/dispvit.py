import importlib
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dpt_head import DPTHead


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


def shift_along_width(x, shifts):
    out = []
    _, _, _, W = x.shape
    for shift in shifts:
        # pad left side with shift zeros, then crop right
        out.append(F.pad(x, (shift, 0, 0, 0))[:, :, :, :W])
    return out


class BlendConv(nn.Module):
    def __init__(self, proj, groups):
        super().__init__()
        self.proj = proj
        self.proj_asym = nn.Conv2d(
            in_channels=3*groups,
            out_channels=proj.out_channels // 2,
            kernel_size=proj.kernel_size,
            stride=proj.stride,
            groups=groups,
            bias=False,
        )
        # initialize with zero
        nn.init.zeros_(self.proj_asym.weight)
    
    def forward(self, img):
        # first 3 channels: original image
        x1 = self.proj(img[:, :3].contiguous())
        # last channels: right-shifted versions of the original image
        x2 = self.proj_asym(img[:, 3:].contiguous())
        return x1 + torch.cat((torch.zeros_like(x2), x2), dim=1)


class DispViT(nn.Module):
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }

    def __init__(self, encoder_type="vitl", groups=8, init_weights=True):
        super().__init__()

        self.intermediate_layer_idx = {
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }

        self.encoder_type = encoder_type
        self.groups = groups

        dinov2_model = getattr(importlib.import_module("..layers.dinov2", __package__), f"dinov2_{encoder_type}14")
        encoder = dinov2_model(pretrained=init_weights)

        self.pretrained = encoder

        self.depth_head = DPTHead(encoder.embed_dim, patch_size=encoder.patch_size, output_dim=128,
                                  features=self.model_configs[encoder_type]["features"], 
                                  hidden_dims=[128, 128],
                                  out_channels=self.model_configs[encoder_type]["out_channels"])
        
        # load depth anything weights
        encoder_map = {
            "vits": "Small",
            "vitb": "Base",
            "vitl": "Large",
        }
        url = f"https://huggingface.co/depth-anything/Depth-Anything-V2-{encoder_map[encoder_type]}/resolve/main/depth_anything_v2_{encoder_type}.pth"
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", weights_only=True)
        self.load_state_dict(checkpoint, strict=False)

        # Reuse the pretrained Conv2d weights of patch embed layer and make it work with mixed input channels
        self.__build_patch_embed__(self.pretrained.patch_embed, groups=groups)

        # Register normalization constants as buffers
        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 3, 1, 1), persistent=False)
        
    def __build_patch_embed__(self, patch_embed, groups):
        new_proj = BlendConv(patch_embed.proj, groups)
        patch_embed.proj = new_proj

    def forward(self, batch):
        # Normalize images
        img1 = (batch["img1"] / 255.0 - self._resnet_mean) / self._resnet_std
        img2 = (batch["img2"] / 255.0 - self._resnet_mean) / self._resnet_std

        padder = None
        if not self.training:
            padder = InputPadder(img1.shape, mode="nmrf", divis_by=self.pretrained.patch_size)
            img1, img2 = padder.pad(img1, img2)

        B, C_in, H, W = img1.shape

        if C_in != 3:
            raise ValueError(f"Expected 3 input channels, got {C_in}")
        
        # Shift img2 along width
        shift_unit = 192 // self.groups
        shifts = [i * shift_unit for i in range(self.groups)]
        img = torch.cat([img1] + shift_along_width(img2, shifts), dim=1)
        
        patch_h, patch_w = H // self.pretrained.patch_size, W // self.pretrained.patch_size

        features = self.pretrained.get_intermediate_layers(img, self.intermediate_layer_idx[self.encoder_type], return_class_token=True)

        disp, disp_logits, feature = self.prediction_head(features, patch_h, patch_w)
        if padder is not None:
            disp = padder.unpad(disp.unsqueeze(1)).squeeze(1)
            disp_logits = padder.unpad(disp_logits)
            feature = padder.unpad(feature)
        out =  {"disp": disp, "disp_logits": disp_logits, "feature": feature}
        if self.training:
            out["gram_feats"] = features
        return out

    def prediction_head(self, x, patch_h, patch_w):
        soft_argmax_threshold = 7
        softmax_temperature = 0.5
        disp_logits, feature = self.depth_head(x, patch_h, patch_w)
        argmax_w = disp_logits.argmax(
            dim=1, keepdim=True
        )
        index = torch.arange(disp_logits.shape[1], device=disp_logits.device).view(1, -1, 1, 1)
        mask = (torch.abs(argmax_w - index) <= soft_argmax_threshold).float()
        probs = F.softmax(disp_logits * softmax_temperature, dim=1) * mask
        probs = probs / probs.sum(dim=1, keepdim=True)

        ticks = torch.linspace(0, 381, 128, dtype=torch.float32, device=disp_logits.device).view(1, -1, 1, 1)
        disp = torch.sum(probs * ticks, dim=1)
        return disp, disp_logits, feature


class InputPadder:
    """ Pads images such that dimensions are divisible by given factor """

    def __init__(self, dims, mode='sintel', divis_by=8):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
        pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        elif mode == 'nmrf':
            self._pad = [0, pad_wd, pad_ht, 0]
        else:
            raise ValueError(f"Non recognized mode '{mode}'")

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]