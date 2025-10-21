import timm
import torch
import torch.nn as nn

import os.path as osp
import sys
import math

sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))
from dino_utils import trunc_normal_


def build_vitencoder(
    img_size=224, channel=3, n_classes=1, drop_path_rate=0, model_name="gigapath"):
    print('image size: ', img_size)
    if model_name == "gigapath":
        print('loading gigapath!')
        encoder = timm.create_model(
            "hf_hub:prov-gigapath/prov-gigapath", 
            drop_path_rate=drop_path_rate, pretrained=True)
        model = WrapDINOv2(encoder, img_size, channel)
    elif model_name == "dinov1":
        print('loading facebook dinov1!')
        encoder = torch.hub.load('facebookresearch/dino:main', 'dino_vits16')
        model = WrapDINOv1(encoder, img_size, channel, n_classes)
    elif model_name == "dinov2":
        print('loading facebook dinov2!')
        encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        model = WrapDINOv2(encoder, img_size, channel)
    elif model_name == "chvit":
        print('loading channel vit trained on jumpcp!')
        # encoder = torch.hub.load('insitro/ChannelViT', 'cpjump_cellpaint_channelvit_small_p8_with_hcs_supervised', pretrained=True)
        encoder = torch.hub.load('insitro/ChannelViT', 'cpjump_cellpaint_bf_channelvit_small_p8_with_hcs_supervised', pretrained=True)
        model = encoder
        # model = WrapDINOv1(encoder, img_size, channel)
        for param in model.parameters():
            param.requires_grad = False
    model.embed_dim = encoder.embed_dim
    return model


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class ChPatchEmbed(nn.Module):
    """ Image to Channel-based Patch Embedding
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.embed_dim = embed_dim

        self.proj_in = nn.Conv3d(
            1, embed_dim, 
            kernel_size=(1, patch_size, patch_size), 
            stride=(1, patch_size, patch_size)
        )

        #self.channel_embed = nn.parameter.Parameter(
        #    torch.zeros(in_chans, embed_dim, 1, 1)
        #)

        self.channel_cls_token = nn.parameter.Parameter(
            torch.zeros(embed_dim, 1, 1)
        )
        trunc_normal_(self.channel_cls_token, std=0.02)
        #trunc_normal_(self.channel_embed, std=0.02)

        self.proj_out = nn.Conv2d(in_chans*embed_dim, embed_dim, 3, 1, 1)

    def forward(self, x, cls_mask=None):
        B, C, H, W = x.shape
        x = self.proj_in(x.unsqueeze(1)).transpose(1, 2) # [B, C, emb, patchNH, patchNW]

        if cls_mask == None:
            cls_mask = C - 1 # take the last channel as "cls" token
        batch_idx = torch.arange(B)
        x[batch_idx, cls_mask] += self.channel_cls_token
        #x[batch_idx, cls_mask] += self.channel_embed[cls_mask]

        x = self.proj_out(x.flatten(start_dim=1, end_dim=2)) # [B, emb, patchNH, patchNW]
        x = x.flatten(2).transpose(1, 2) # [B, num_patches, emb]
        return x


class WrapDINOv2(nn.Module):
    def __init__(self, encoder, img_size=224, channel=3):
        super().__init__()
        encoder.patch_embed = PatchEmbed(
            img_size=img_size, 
            in_chans=channel, 
            embed_dim=encoder.embed_dim, 
            patch_size=encoder.patch_embed.patch_size[0])
        self.encoder = encoder

    def forward(self, x):
        x = self.encoder.forward_features(x)
        return self.encoder.head(x['x_norm_clstoken'])


class WrapDINOv1(nn.Module):
    def __init__(self, encoder, img_size=224, channel=3, n_classes=1):
        super().__init__()
        self.patch_embed = PatchEmbed(
            img_size=img_size, 
            in_chans=channel, 
            embed_dim=encoder.embed_dim, 
            patch_size=encoder.patch_embed.patch_size)
        self.embed_dim = encoder.embed_dim
        self.cls_token = encoder.cls_token
        self.pos_embed = encoder.pos_embed
        self.pos_drop = encoder.pos_drop
        self.blocks = encoder.blocks
        self.norm = encoder.norm
        self.head = nn.Linear(encoder.embed_dim, n_classes) if n_classes > 0 else nn.Identity()

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)  # patch linear embedding

        # add the [CLS] token to the embed patch tokens
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # add positional encoding to each token
        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x[:, 0]

    def get_last_selfattention(self, x):
        x = self.prepare_tokens(x)
        for i, blk in enumerate(self.blocks):
            if i < len(self.blocks) - 1:
                x = blk(x)
            else:
                # return attention of the last block
                return blk(x, return_attention=True)

    def get_intermediate_layers(self, x, n=1):
        x = self.prepare_tokens(x)
        # we return the output tokens from the `n` last blocks
        output = []
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if len(self.blocks) - i <= n:
                output.append(self.norm(x))
        return output


class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, use_bn=False, norm_last_layer=True, nlayers=3, hidden_dim=2048, bottleneck_dim=256):
        super().__init__()
        nlayers = max(nlayers, 1)
        if nlayers == 1:
            self.mlp = nn.Linear(in_dim, bottleneck_dim)
        else:
            layers = [nn.Linear(in_dim, hidden_dim)]
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
            for _ in range(nlayers - 2):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                if use_bn:
                    layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.GELU())
            layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            self.mlp = nn.Sequential(*layers)
        self.apply(self._init_weights)
        self.last_layer = nn.utils.parametrizations.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight.data.fill_(1)
        if norm_last_layer:
            self.last_layer.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x



