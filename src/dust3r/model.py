import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from copy import deepcopy
from functools import partial
from typing import Optional, Tuple, List, Any
from dataclasses import dataclass
from transformers import PretrainedConfig
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutput
from transformers.file_utils import ModelOutput
import time

# Add safe globals for PyTorch 2.6+ compatibility
try:
    from omegaconf import DictConfig
    torch.serialization.add_safe_globals([DictConfig])
except ImportError:
    pass
from dust3r.utils.misc import (
    fill_default_args,
    freeze_all_params,
    is_symmetrized,
    interleave,
    transpose_to_landscape,
)
from dust3r.heads import head_factory
from dust3r.utils.camera import PoseEncoder
from dust3r.patch_embed import get_patch_embed
import dust3r.utils.path_to_croco  # noqa: F401
from models.croco import CroCoNet, CrocoConfig  # noqa
from dust3r.blocks import (
    Block,
    DecoderBlock,
    Mlp,
    Attention,
    CrossAttention,
    DropPath,
)  # noqa

inf = float("inf")
from accelerate.logging import get_logger

from einops import rearrange
from dust3r.utils.device import to_cpu, to_gpu
printer = get_logger(__name__, log_level="DEBUG")


@dataclass
class ARCroco3DStereoOutput(ModelOutput):
    """
    Custom output class for ARCroco3DStereo.
    """

    ress: Optional[List[Any]] = None
    views: Optional[List[Any]] = None


def strip_module(state_dict):
    """
    Removes the 'module.' prefix from the keys of a state_dict.
    Args:
        state_dict (dict): The original state_dict with possible 'module.' prefixes.
    Returns:
        OrderedDict: A new state_dict with 'module.' prefixes removed.
    """
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] if k.startswith("module.") else k
        new_state_dict[name] = v
    return new_state_dict


def load_model(model_path, device, verbose=True):
    if verbose:
        print("... loading model from", model_path)
    # Use weights_only=False for compatibility with older checkpoints containing omegaconf
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    args = ckpt["args"].model.replace(
        "ManyAR_PatchEmbed", "PatchEmbedDust3R"
    )  # ManyAR only for aspect ratio not consistent
    if "landscape_only" not in args:
        args = args[:-2] + ", landscape_only=False))"
    else:
        args = args.replace(" ", "").replace(
            "landscape_only=True", "landscape_only=False"
        )
    assert "landscape_only=False" in args
    if verbose:
        print(f"instantiating : {args}")
    net = eval(args)
    s = net.load_state_dict(ckpt["model"], strict=False)
    if verbose:
        print(s)
    return net.to(device)


SUPPORTED_MODEL_UPDATE_TYPES = {"cut3r", "ttt3r", "recal3r"}


def canonicalize_model_update_type(model_update_type):
    model_update_type = (model_update_type or "cut3r").strip()
    if model_update_type not in SUPPORTED_MODEL_UPDATE_TYPES:
        raise ValueError(
            "Unsupported model_update_type "
            f"{model_update_type!r}. Expected one of "
            f"{sorted(SUPPORTED_MODEL_UPDATE_TYPES)}."
        )
    return model_update_type


def default_update_pressure_decay():
    return 0.95


def default_beta_base(model_update_type):
    model_update_type = canonicalize_model_update_type(model_update_type)
    return 0.1 if model_update_type == "recal3r" else 0.0


class ARCroco3DStereoConfig(PretrainedConfig):
    model_type = "arcroco_3d_stereo"

    def __init__(
        self,
        output_mode="pts3d",
        head_type="linear",  # or dpt
        depth_mode=("exp", -float("inf"), float("inf")),
        conf_mode=("exp", 1, float("inf")),
        pose_mode=("exp", -float("inf"), float("inf")),
        freeze="none",
        landscape_only=True,
        patch_embed_cls="PatchEmbedDust3R",
        ray_enc_depth=2,
        state_size=324,
        local_mem_size=256,
        state_pe="2d",
        state_dec_num_heads=16,
        depth_head=False,
        rgb_head=False,
        pose_conf_head=False,
        pose_head=False,
        model_update_type="cut3r",
        entropy_eps=1e-12,
        beta_base=None,
        entropy_head_reduce="mean",
        uncertainty_clamp_max=1.0,
        decay=None,
        **croco_kwargs,
    ):
        super().__init__()
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.freeze = freeze
        self.landscape_only = landscape_only
        self.patch_embed_cls = patch_embed_cls
        self.ray_enc_depth = ray_enc_depth
        self.state_size = state_size
        self.state_pe = state_pe
        self.state_dec_num_heads = state_dec_num_heads
        self.local_mem_size = local_mem_size
        self.depth_head = depth_head
        self.rgb_head = rgb_head
        self.pose_conf_head = pose_conf_head
        self.pose_head = pose_head
        self.model_update_type = canonicalize_model_update_type(model_update_type)
        self.entropy_eps = entropy_eps
        self.beta_base = (
            default_beta_base(self.model_update_type)
            if beta_base is None
            else beta_base
        )
        self.entropy_head_reduce = entropy_head_reduce
        self.uncertainty_clamp_max = uncertainty_clamp_max
        self.decay = default_update_pressure_decay() if decay is None else decay
        self.croco_kwargs = croco_kwargs


class LocalMemory(nn.Module):
    def __init__(
        self,
        size,
        k_dim,
        v_dim,
        num_heads,
        depth=2,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        norm_mem=True,
        rope=None,
    ) -> None:
        super().__init__()
        self.v_dim = v_dim
        self.proj_q = nn.Linear(k_dim, v_dim)
        self.masked_token = nn.Parameter(
            torch.randn(1, 1, v_dim) * 0.2, requires_grad=True
        ) # [1, 1, 768] pose mask token
        self.mem = nn.Parameter(
            torch.randn(1, size, 2 * v_dim) * 0.2, requires_grad=True
        ) # [1, 256, 1536] pose mem
        self.write_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )
        self.read_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    2 * v_dim,
                    num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    attn_drop=attn_drop,
                    drop=drop,
                    drop_path=drop_path,
                    act_layer=act_layer,
                    norm_mem=norm_mem,
                    rope=rope,
                )
                for _ in range(depth)
            ]
        )

    def update_mem(self, mem, feat_k, feat_v, return_attn=False):
        """
        mem_k: [B, size, C]
        mem_v: [B, size, C]
        feat_k: [B, 1, C] global_img_feat
        feat_v: [B, 1, C] out_pose_feat
        """
        feat_k = self.proj_q(feat_k)  # [B, 1, C]
        feat = torch.cat([feat_k, feat_v], dim=-1)

        attention_maps = []
        for blk in self.write_blocks:
            mem, _, self_attn, cross_attn = blk(mem, feat, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return mem

    def inquire(self, query, mem, return_attn=False):
        x = self.proj_q(query)  # [B, 1, C]
        x = torch.cat([x, self.masked_token.expand(x.shape[0], -1, -1)], dim=-1) # [1, 1, 768 global_img_feat_i + 768 masked_token(pose)]
        attention_maps = []
        for blk in self.read_blocks:
            x, _, self_attn, cross_attn = blk(x, mem, None, None, return_attn=return_attn)
            attention_maps.append((self_attn, cross_attn))
        return x[..., -self.v_dim :]


class ARCroco3DStereo(CroCoNet):
    config_class = ARCroco3DStereoConfig
    base_model_prefix = "arcroco3dstereo"
    supports_gradient_checkpointing = True

    def __init__(self, config: ARCroco3DStereoConfig):
        self.gradient_checkpointing = False
        self.fixed_input_length = True
        config.model_update_type = canonicalize_model_update_type(
            getattr(config, "model_update_type", "cut3r")
        )
        config.croco_kwargs = fill_default_args(
            config.croco_kwargs, CrocoConfig.__init__
        )
        self.config = config
        self.patch_embed_cls = config.patch_embed_cls
        self.croco_args = config.croco_kwargs
        croco_cfg = CrocoConfig(**self.croco_args)
        super().__init__(croco_cfg)
        self.enc_blocks_ray_map = nn.ModuleList(
            [
                Block(
                    self.enc_embed_dim,
                    16,
                    4,
                    qkv_bias=True,
                    norm_layer=partial(nn.LayerNorm, eps=1e-6),
                    rope=self.rope,
                )
                for _ in range(config.ray_enc_depth)
            ]
        )
        self.enc_norm_ray_map = nn.LayerNorm(self.enc_embed_dim, eps=1e-6)
        self.dec_num_heads = self.croco_args["dec_num_heads"]
        self.pose_head_flag = config.pose_head
        if self.pose_head_flag:
            self.pose_token = nn.Parameter(
                torch.randn(1, 1, self.dec_embed_dim) * 0.02, requires_grad=True
            ) # [1, 1, 768]
            self.pose_retriever = LocalMemory(
                size=config.local_mem_size,
                k_dim=self.enc_embed_dim,
                v_dim=self.dec_embed_dim,
                num_heads=self.dec_num_heads,
                mlp_ratio=4,
                qkv_bias=True,
                attn_drop=0.0,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                rope=None,
            )
        self.register_tokens = nn.Embedding(config.state_size, self.enc_embed_dim) # init state tokens [768, 1024]
        self.state_size = config.state_size
        self.state_pe = config.state_pe
        self.masked_img_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self.masked_ray_map_token = nn.Parameter(
            torch.randn(1, self.enc_embed_dim) * 0.02, requires_grad=True
        )
        self._set_state_decoder(
            self.enc_embed_dim,
            self.dec_embed_dim,
            config.state_dec_num_heads,
            self.dec_depth,
            self.croco_args.get("mlp_ratio", None),
            self.croco_args.get("norm_layer", None),
            self.croco_args.get("norm_im2_in_dec", None),
        )
        self.set_downstream_head(
            config.output_mode,
            config.head_type,
            config.landscape_only,
            config.depth_mode,
            config.conf_mode,
            config.pose_mode,
            config.depth_head,
            config.rgb_head,
            config.pose_conf_head,
            config.pose_head,
            **self.croco_args,
        )
        self.set_freeze(config.freeze)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kw):
        if os.path.isfile(pretrained_model_name_or_path):
            return load_model(pretrained_model_name_or_path, device="cpu")
        else:
            try:
                model = super(ARCroco3DStereo, cls).from_pretrained(
                    pretrained_model_name_or_path, **kw
                )
            except TypeError as e:
                raise Exception(
                    f"tried to load {pretrained_model_name_or_path} from huggingface, but failed"
                )
            return model

    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768):
        self.patch_embed = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=3
        )
        self.patch_embed_ray_map = get_patch_embed(
            self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans=6
        )

    def _set_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm = norm_layer(dec_embed_dim)

    def _set_state_decoder(
        self,
        enc_embed_dim,
        dec_embed_dim,
        dec_num_heads,
        dec_depth,
        mlp_ratio,
        norm_layer,
        norm_im2_in_dec,
    ):
        self.dec_depth_state = dec_depth
        self.dec_embed_dim_state = dec_embed_dim
        self.decoder_embed_state = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        self.dec_blocks_state = nn.ModuleList(
            [
                DecoderBlock(
                    dec_embed_dim,
                    dec_num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    norm_mem=norm_im2_in_dec,
                    rope=self.rope,
                )
                for i in range(dec_depth)
            ]
        )
        self.dec_norm_state = norm_layer(dec_embed_dim)

    def load_state_dict(self, ckpt, **kw):
        if all(k.startswith("module") for k in ckpt):
            ckpt = strip_module(ckpt)
        new_ckpt = dict(ckpt)
        if not any(k.startswith("dec_blocks_state") for k in ckpt):
            for key, value in ckpt.items():
                if key.startswith("dec_blocks"):
                    new_ckpt[key.replace("dec_blocks", "dec_blocks_state")] = value
        try:
            return super().load_state_dict(new_ckpt, **kw)
        except:
            try:
                new_new_ckpt = {
                    k: v
                    for k, v in new_ckpt.items()
                    if not k.startswith("dec_blocks")
                    and not k.startswith("dec_norm")
                    and not k.startswith("decoder_embed")
                }
                return super().load_state_dict(new_new_ckpt, **kw)
            except:
                new_new_ckpt = {}
                for key in new_ckpt:
                    if key in self.state_dict():
                        if new_ckpt[key].size() == self.state_dict()[key].size():
                            new_new_ckpt[key] = new_ckpt[key]
                        else:
                            printer.info(
                                f"Skipping '{key}': size mismatch (ckpt: {new_ckpt[key].size()}, model: {self.state_dict()[key].size()})"
                            )
                    else:
                        printer.info(f"Skipping '{key}': not found in model")
                return super().load_state_dict(new_new_ckpt, **kw)

    def set_freeze(self, freeze):  # this is for use by downstream models
        self.freeze = freeze
        to_be_frozen = {
            "none": [],
            "mask": [self.mask_token] if hasattr(self, "mask_token") else [],
            "encoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
            ],
            "encoder_and_head": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.downstream_head,
            ],
            "encoder_and_decoder": [
                self.patch_embed,
                self.patch_embed_ray_map,
                self.masked_img_token,
                self.masked_ray_map_token,
                self.enc_blocks,
                self.enc_blocks_ray_map,
                self.enc_norm,
                self.enc_norm_ray_map,
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
                self.register_tokens,
                self.decoder_embed_state,
                self.decoder_embed,
                self.dec_norm,
                self.dec_norm_state,
            ],
            "decoder": [
                self.dec_blocks,
                self.dec_blocks_state,
                self.pose_retriever,
                self.pose_token,
            ],
        }
        freeze_all_params(to_be_frozen[freeze])

    def _set_prediction_head(self, *args, **kwargs):
        """No prediction head"""
        return

    def set_downstream_head(
        self,
        output_mode,
        head_type,
        landscape_only,
        depth_mode,
        conf_mode,
        pose_mode,
        depth_head,
        rgb_head,
        pose_conf_head,
        pose_head,
        patch_size,
        img_size,
        **kw,
    ):
        assert (
            img_size[0] % patch_size == 0 and img_size[1] % patch_size == 0
        ), f"{img_size=} must be multiple of {patch_size=}"
        self.output_mode = output_mode
        self.head_type = head_type
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode
        self.pose_mode = pose_mode
        self.downstream_head = head_factory(
            head_type,
            output_mode,
            self,
            has_conf=bool(conf_mode),
            has_depth=bool(depth_head),
            has_rgb=bool(rgb_head),
            has_pose_conf=bool(pose_conf_head),
            has_pose=bool(pose_head),
        )
        self.head = transpose_to_landscape(
            self.downstream_head, activate=landscape_only
        )

    def _encode_image(self, image, true_shape):
        x, pos = self.patch_embed(image, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm(x)
        return [x], pos, None

    def _encode_ray_map(self, ray_map, true_shape):
        x, pos = self.patch_embed_ray_map(ray_map, true_shape=true_shape)
        assert self.enc_pos_embed is None
        for blk in self.enc_blocks_ray_map:
            if self.gradient_checkpointing and self.training:
                x = checkpoint(blk, x, pos, use_reentrant=False)
            else:
                x = blk(x, pos)
        x = self.enc_norm_ray_map(x)
        return [x], pos, None

    def _encode_state(self, image_tokens, image_pos):
        batch_size = image_tokens.shape[0]
        state_feat = self.register_tokens(
            torch.arange(self.state_size, device=image_pos.device)
        ) # [768, 1024]
        if self.state_pe == "1d":
            state_pos = (
                torch.tensor(
                    [[i, i] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )  # .long()
        elif self.state_pe == "2d":
            width = int(self.state_size**0.5)
            width = width + 1 if width % 2 == 1 else width
            state_pos = (
                torch.tensor(
                    [[i // width, i % width] for i in range(self.state_size)],
                    dtype=image_pos.dtype,
                    device=image_pos.device,
                )[None]
                .expand(batch_size, -1, -1)
                .contiguous()
            )
        elif self.state_pe == "none":
            state_pos = None
        state_feat = state_feat[None].expand(batch_size, -1, -1)
        return state_feat, state_pos, None

    def _encode_views(self, views, img_mask=None, ray_mask=None):
        device = views[0]["img"].device
        batch_size = views[0]["img"].shape[0]
        given = True
        if img_mask is None and ray_mask is None:
            given = False
        if not given:
            img_mask = torch.stack(
                [view["img_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
            ray_mask = torch.stack(
                [view["ray_mask"] for view in views], dim=0
            )  # Shape: (num_views, batch_size)
        imgs = torch.stack(
            [view["img"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, C, H, W)
        ray_maps = torch.stack(
            [view["ray_map"] for view in views], dim=0
        )  # Shape: (num_views, batch_size, H, W, C)
        shapes = []
        for view in views:
            if "true_shape" in view:
                shapes.append(view["true_shape"])
            else:
                shape = torch.tensor(view["img"].shape[-2:], device=device)
                shapes.append(shape.unsqueeze(0).repeat(batch_size, 1))
        shapes = torch.stack(shapes, dim=0).to(
            imgs.device
        )  # Shape: (num_views, batch_size, 2)
        imgs = imgs.view(
            -1, *imgs.shape[2:]
        )  # Shape: (num_views * batch_size, C, H, W)
        ray_maps = ray_maps.view(
            -1, *ray_maps.shape[2:]
        )  # Shape: (num_views * batch_size, H, W, C)
        shapes = shapes.view(-1, 2)  # Shape: (num_views * batch_size, 2)
        img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
        ray_masks_flat = ray_mask.view(-1)
        selected_imgs = imgs[img_masks_flat]
        selected_shapes = shapes[img_masks_flat]
        if selected_imgs.size(0) > 0:
            img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
        else:
            raise NotImplementedError
        full_out = [
            torch.zeros(
                len(views) * batch_size, *img_out[0].shape[1:], device=img_out[0].device
            )
            for _ in range(len(img_out))
        ]
        full_pos = torch.zeros(
            len(views) * batch_size,
            *img_pos.shape[1:],
            device=img_pos.device,
            dtype=img_pos.dtype,
        )
        for i in range(len(img_out)):
            full_out[i][img_masks_flat] += img_out[i]
            full_out[i][~img_masks_flat] += self.masked_img_token
        full_pos[img_masks_flat] += img_pos
        ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
        selected_ray_maps = ray_maps[ray_masks_flat]
        selected_shapes_ray = shapes[ray_masks_flat]
        if selected_ray_maps.size(0) > 0:
            ray_out, ray_pos, _ = self._encode_ray_map(
                selected_ray_maps, selected_shapes_ray
            )
            assert len(ray_out) == len(full_out), f"{len(ray_out)}, {len(full_out)}"
            for i in range(len(ray_out)):
                full_out[i][ray_masks_flat] += ray_out[i]
                full_out[i][~ray_masks_flat] += self.masked_ray_map_token
            full_pos[ray_masks_flat] += (
                ray_pos * (~img_masks_flat[ray_masks_flat][:, None, None]).long()
            )
        else:
            raymaps = torch.zeros(
                1, 6, imgs[0].shape[-2], imgs[0].shape[-1], device=img_out[0].device
            )
            ray_mask_flat = torch.zeros_like(img_masks_flat)
            ray_mask_flat[:1] = True
            ray_out, ray_pos, _ = self._encode_ray_map(raymaps, shapes[ray_mask_flat])
            for i in range(len(ray_out)):
                full_out[i][ray_mask_flat] += ray_out[i] * 0.0
                full_out[i][~ray_mask_flat] += self.masked_ray_map_token * 0.0
        return (
            shapes.chunk(len(views), dim=0),
            [out.chunk(len(views), dim=0) for out in full_out],
            full_pos.chunk(len(views), dim=0),
        )

    def _decoder(self, f_state, pos_state, f_img, pos_img, f_pose, pos_pose, return_attn):
        final_output = [(f_state, f_img)]  # before projection
        assert f_state.shape[-1] == self.dec_embed_dim
        f_img = self.decoder_embed(f_img) # Linear: [1, 576, 1024] -> [1, 576, 768]
        if self.pose_head_flag:
            assert f_pose is not None and pos_pose is not None
            f_img = torch.cat([f_pose, f_img], dim=1) # [1, 1 + 576, 768]
            pos_img = torch.cat([pos_pose, pos_img], dim=1) # [1, 1 + 576, 2]
        final_output.append((f_state, f_img))
        attention_maps = []
        for blk_state, blk_img in zip(self.dec_blocks_state, self.dec_blocks):
            if (
                self.gradient_checkpointing
                and self.training
                and torch.is_grad_enabled()
            ):
                f_state, _, self_attn_state, cross_attn_state = checkpoint(
                    blk_state,
                    *final_output[-1][::+1],
                    pos_state,
                    pos_img,
                    return_attn,
                    use_reentrant=not self.fixed_input_length,
                )
                f_img, _, self_attn_img, cross_attn_img = checkpoint(
                    blk_img,
                    *final_output[-1][::-1],
                    pos_img,
                    pos_state,
                    return_attn,
                    use_reentrant=not self.fixed_input_length,
                )
            else:
                f_state, _, self_attn_state, cross_attn_state = blk_state(*final_output[-1][::+1], pos_state, pos_img, return_attn=return_attn)
                f_img, _, self_attn_img, cross_attn_img = blk_img(*final_output[-1][::-1], pos_img, pos_state, return_attn=return_attn)
            final_output.append((f_state, f_img))
            attention_maps.append((self_attn_state, cross_attn_state, self_attn_img, cross_attn_img))
        del final_output[1]  # duplicate with final_output[0]
        final_output[-1] = (
            self.dec_norm_state(final_output[-1][0]),
            self.dec_norm(final_output[-1][1]),
        )
        return zip(*final_output), zip(*attention_maps)

    def _downstream_head(self, decout, img_shape, **kwargs):
        B, S, D = decout[-1].shape
        head = getattr(self, f"head")
        return head(decout, img_shape, **kwargs)

    def _init_state(self, image_tokens, image_pos):
        """
        Current Version: input the first frame img feature and pose to initialize the state feature and pose
        # [1, 768, 768] [1, 768, 2]
        """
        state_feat, state_pos, _ = self._encode_state(image_tokens, image_pos)
        state_feat = self.decoder_embed_state(state_feat) # Linear: [1, 768, 1024] -> [1, 768, 768]
        return state_feat, state_pos

    def _recurrent_rollout(
        self,
        state_feat,
        state_pos,
        current_feat,
        current_pos,
        pose_feat,
        pose_pos,
        init_state_feat,
        img_mask=None,
        reset_mask=None,
        update=None,
        return_attn=False,
    ):
        (new_state_feat, dec), (self_attn_state, cross_attn_state, self_attn_img, cross_attn_img) = self._decoder(
            state_feat, state_pos, current_feat, current_pos, pose_feat, pose_pos, return_attn
        )
        new_state_feat = new_state_feat[-1]
        return new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img

    def _get_img_level_feat(self, feat):
        return torch.mean(feat, dim=1, keepdim=True)

    def _uses_recal3r(self):
        return getattr(self.config, "model_update_type", None) == "recal3r"

    def _uses_update_pressure_update(self):
        return self._uses_recal3r()

    def _entropy_eps(self):
        return float(
            getattr(self, "entropy_eps", getattr(self.config, "entropy_eps", 1e-12))
        )

    def _beta_base(self):
        beta_base = getattr(self, "beta_base", getattr(self.config, "beta_base", None))
        if beta_base is None:
            beta_base = default_beta_base(
                getattr(self.config, "model_update_type", "cut3r")
            )
        return float(beta_base)

    def _entropy_head_reduce(self):
        return getattr(
            self,
            "entropy_head_reduce",
            getattr(self.config, "entropy_head_reduce", "mean"),
        )

    def _uncertainty_clamp_max(self):
        clamp_max = float(
            getattr(
                self,
                "uncertainty_clamp_max",
                getattr(self.config, "uncertainty_clamp_max", 1.0),
            )
        )
        return max(0.0, min(1.0, clamp_max))

    def _update_pressure_decay(self):
        if not self._uses_recal3r():
            raise RuntimeError("token heat decay requested outside recal3r mode")
        model_update_type = getattr(
            self, "model_update_type", getattr(self.config, "model_update_type", None)
        )
        decay = getattr(self, "decay", None)
        if decay is None:
            decay = getattr(self.config, "decay", None)
        if decay is None:
            decay = getattr(
                self.config,
                "heat_decay",
                default_update_pressure_decay(),
            )
        decay = float(decay)
        return max(0.0, min(1.0, decay))


    def enable_u_calibration_trace(self, final_state=None, oracle_window=None):
        self._u_calibration_trace = {
            "u": [],
            "err": [],
            "delta_norm": [],
            "frame_idx": [],
            "frame_u_mean": [],
            "frame_u_min": [],
            "frame_u_max": [],
            "frame_h_mean": [],
            "frame_h_min": [],
            "frame_h_max": [],
            "frame_step": [],
        }
        self._u_calibration_pending_u = None
        self._u_calibration_pending_h = None
        self._u_calibration_last_state = None
        self._u_calibration_final_state = final_state
        self._u_calibration_oracle_window = (
            max(1, int(oracle_window)) if oracle_window is not None else None
        )
        self._u_calibration_queue = []

    def disable_u_calibration_trace(self):
        for attr in (
            "_u_calibration_trace",
            "_u_calibration_pending_u",
            "_u_calibration_pending_h",
            "_u_calibration_last_state",
            "_u_calibration_final_state",
            "_u_calibration_oracle_window",
            "_u_calibration_queue",
        ):
            if hasattr(self, attr):
                delattr(self, attr)

    def get_u_calibration_trace(self):
        return getattr(self, "_u_calibration_trace", None)

    def get_u_calibration_last_state(self):
        return getattr(self, "_u_calibration_last_state", None)

    def _stash_u_calibration_stats(self, R, h_m):
        if hasattr(self, "_u_calibration_trace"):
            self._u_calibration_pending_u = R.detach()
            self._u_calibration_pending_h = h_m.detach()

    def _init_recal3r_reference_state(self, state_feat):
        if not self._uses_recal3r():
            return
        self.recal3r_state0 = state_feat.detach().clone()
        self._recal3r_sequence_age = torch.zeros(
            state_feat.shape[0], device=state_feat.device, dtype=torch.long
        )

    def _advance_recal3r_sequence_age(self, state_feat=None):
        del state_feat
        if not self._uses_recal3r():
            return
        if hasattr(self, "_recal3r_sequence_age"):
            self._recal3r_sequence_age = self._recal3r_sequence_age + 1

    def _reset_recal3r_reference_state_if_needed(self, reset_mask, init_state_feat):
        if not self._uses_recal3r() or reset_mask is None:
            return
        if not hasattr(self, "recal3r_state0"):
            self._init_recal3r_reference_state(init_state_feat)
            return

        if reset_mask.dim() == 3 and reset_mask.shape[-1] == 1:
            reset_mask = reset_mask.squeeze(-1)
        if reset_mask.dim() > 1:
            reset_mask = reset_mask.squeeze(-1)
        reset_mask = reset_mask.reshape(-1).to(
            device=init_state_feat.device, dtype=torch.bool
        )
        if reset_mask.shape[0] != init_state_feat.shape[0]:
            raise AssertionError(
                "recal3r reset mask batch mismatch: "
                f"reset_mask_batch={reset_mask.shape[0]} "
                f"state_batch={init_state_feat.shape[0]}"
            )
        if not bool(reset_mask.any().item()):
            return

        reset_mask_ref = reset_mask[:, None, None]
        init_snapshot = init_state_feat.detach().clone()
        self.recal3r_state0 = torch.where(
            reset_mask_ref, init_snapshot, self.recal3r_state0
        )
        if hasattr(self, "_recal3r_sequence_age"):
            self._recal3r_sequence_age = torch.where(
                reset_mask,
                torch.zeros_like(self._recal3r_sequence_age),
                self._recal3r_sequence_age,
            )

    def _maybe_record_u_calibration_step(self, frame_idx, state_prev, state_post):
        if not hasattr(self, "_u_calibration_trace"):
            return

        self._u_calibration_last_state = state_post.detach().cpu()
        pending_u = getattr(self, "_u_calibration_pending_u", None)
        pending_h = getattr(self, "_u_calibration_pending_h", None)
        final_state = getattr(self, "_u_calibration_final_state", None)
        oracle_window = getattr(self, "_u_calibration_oracle_window", None)
        if pending_u is None or pending_h is None:
            self._u_calibration_pending_u = None
            self._u_calibration_pending_h = None
            return

        delta_norm = torch.norm(state_post - state_prev, dim=-1)
        trace = self._u_calibration_trace
        trace["frame_u_mean"].append(pending_u.detach().mean().reshape(1).float().cpu())
        trace["frame_u_min"].append(pending_u.detach().min().reshape(1).float().cpu())
        trace["frame_u_max"].append(pending_u.detach().max().reshape(1).float().cpu())
        trace["frame_h_mean"].append(pending_h.detach().mean().reshape(1).float().cpu())
        trace["frame_h_min"].append(pending_h.detach().min().reshape(1).float().cpu())
        trace["frame_h_max"].append(pending_h.detach().max().reshape(1).float().cpu())
        trace["frame_step"].append(
            torch.full((1,), int(frame_idx), dtype=torch.int32)
        )

        if oracle_window is not None:
            queue = self._u_calibration_queue
            queue.append(
                {
                    "u": pending_u.detach(),
                    "delta_norm": delta_norm.detach(),
                    "post_state": state_post.detach(),
                    "frame_idx": int(frame_idx),
                }
            )
            if len(queue) >= oracle_window:
                future_state = state_post.detach()
                entry = queue.pop(0)
                err = torch.norm(entry["post_state"] - future_state, dim=-1)
                trace["u"].append(entry["u"].reshape(-1).float().cpu())
                trace["err"].append(err.detach().reshape(-1).float().cpu())
                trace["delta_norm"].append(
                    entry["delta_norm"].reshape(-1).float().cpu()
                )
                trace["frame_idx"].append(
                    torch.full(
                        (entry["u"].numel(),),
                        int(entry["frame_idx"]),
                        dtype=torch.int32,
                    )
                )
        elif final_state is not None:
            final_state = final_state.to(device=state_post.device, dtype=state_post.dtype)
            err = torch.norm(state_post - final_state, dim=-1)
            trace["u"].append(pending_u.detach().reshape(-1).float().cpu())
            trace["err"].append(err.detach().reshape(-1).float().cpu())
            trace["delta_norm"].append(delta_norm.detach().reshape(-1).float().cpu())
            trace["frame_idx"].append(
                torch.full((pending_u.numel(),), int(frame_idx), dtype=torch.int32)
            )

        self._u_calibration_pending_u = None
        self._u_calibration_pending_h = None

    def _reset_update_pressure_if_needed(self, reset_mask):
        if not self._uses_update_pressure_update():
            return
        if isinstance(reset_mask, torch.Tensor):
            should_reset = bool(reset_mask.detach().any().cpu().item())
        else:
            should_reset = bool(reset_mask)
        if should_reset and hasattr(self, "update_pressure"):
            del self.update_pressure

    def _compute_state_update_mask(self, update_mask, raw_cross_attn_state):
        state_attn_tensor = torch.stack(raw_cross_attn_state, dim=0)
        state_query_img_key = state_attn_tensor.mean(dim=(0, 2, 4))
        beta_t = torch.sigmoid(state_query_img_key)[..., None]
        return update_mask * beta_t, beta_t, beta_t

    def _align_token_stat_with_beta(self, stat, beta_like):
        """
        Align row-wise token statistics (e.g. R, rho) to beta tensor layout.
        Expected output shape matches beta_like, typically [B, N_state, 1].
        """
        aligned = stat
        if beta_like.dim() == 3:
            if aligned.dim() == 1:
                aligned = aligned.unsqueeze(0).unsqueeze(-1)
            elif aligned.dim() == 2:
                aligned = aligned.unsqueeze(-1)
        elif beta_like.dim() == 2 and aligned.dim() == 1:
            aligned = aligned.unsqueeze(0)
        return aligned

    def _smooth_beta_toward_frame_mean(self, beta_m, alpha=None):
        if alpha is None:
            alpha = 0.5
        if beta_m.dim() <= 1:
            beta_mean = beta_m.mean(dim=0, keepdim=True)
        else:
            beta_mean = beta_m.mean(dim=1, keepdim=True)
        return alpha * beta_m + (1.0 - alpha) * beta_mean

    def _compute_recal3r_u(self, gamma_m):
        v_m = 4.0 * gamma_m * (1.0 - gamma_m)
        return 1.0 - gamma_m * (1.0 - v_m)

    def _compute_row_entropy_norm(self, attn_matrix):
        """
        Compute row-wise normalized entropy h_m in [0, 1] using the same
        head-reduction convention as the uncertainty computation.
        """
        eps = self._entropy_eps()
        head_reduce = self._entropy_head_reduce()

        if attn_matrix.dim() == 4:
            B, H, N, P = attn_matrix.shape
            attn_flat = attn_matrix.reshape(B * H * N, P)
        elif attn_matrix.dim() == 3:
            H, N, P = attn_matrix.shape
            attn_flat = attn_matrix.reshape(H * N, P)
        elif attn_matrix.dim() == 2:
            _, P = attn_matrix.shape
            attn_flat = attn_matrix
        else:
            raise ValueError(f"Unsupported attention shape: {attn_matrix.shape}")

        attn_flat = attn_flat.clamp_min(eps)
        entropy = -torch.sum(attn_flat * torch.log(attn_flat), dim=-1)
        h_max = torch.log(
            torch.tensor(P, dtype=entropy.dtype, device=entropy.device)
        ).clamp_min(eps)
        h_norm = (entropy / h_max).clamp(0.0, 1.0)

        if attn_matrix.dim() == 4:
            h_norm = h_norm.reshape(B, H, N)
            if head_reduce == "mean":
                h_norm = h_norm.mean(dim=1)
            elif head_reduce == "max":
                h_norm = h_norm.max(dim=1).values
            elif head_reduce == "min":
                h_norm = h_norm.min(dim=1).values
            else:
                raise ValueError(f"unknown head_reduce: {head_reduce}")
            return h_norm.squeeze(0) if B == 1 else h_norm

        if attn_matrix.dim() == 3:
            h_norm = h_norm.reshape(H, N)
            if head_reduce == "mean":
                return h_norm.mean(dim=0)
            if head_reduce == "max":
                return h_norm.max(dim=0).values
            if head_reduce == "min":
                return h_norm.min(dim=0).values
            raise ValueError(f"unknown head_reduce: {head_reduce}")

        return h_norm

    def _compute_empirical_quantile_norm(self, token_stat):
        """
        Rank-normalize per-frame token statistics to an empirical uniform [0, 1].
        """
        if token_stat.dim() == 1:
            token_stat = token_stat.unsqueeze(0)
            squeeze_batch = True
        elif token_stat.dim() == 2:
            squeeze_batch = False
        else:
            raise ValueError(f"Unsupported token stat shape: {token_stat.shape}")

        num_tokens = token_stat.shape[-1]
        if num_tokens <= 1:
            quantile = torch.zeros_like(token_stat)
        else:
            sort_idx = torch.argsort(token_stat, dim=-1)
            ranks = torch.argsort(sort_idx, dim=-1).to(token_stat.dtype)
            quantile = ranks / float(num_tokens - 1)

        return quantile.squeeze(0) if squeeze_batch else quantile

    def _state_attention_weights(self, raw_cross_attn_state):
        attn_logits = torch.stack(raw_cross_attn_state, dim=0)
        if self.pose_head_flag and attn_logits.shape[-1] > 1:
            attn_logits = attn_logits[..., 1:]
        attn_weights = torch.softmax(attn_logits, dim=-1)
        if attn_weights.dim() == 5:
            num_blocks, batch_size, num_heads, num_state, num_obs = attn_weights.shape
            attn_weights = attn_weights.permute(1, 0, 2, 3, 4).reshape(
                batch_size, num_blocks * num_heads, num_state, num_obs
            )
        elif attn_weights.dim() == 4:
            num_blocks, num_heads, num_state, num_obs = attn_weights.shape
            attn_weights = attn_weights.reshape(
                1, num_blocks * num_heads, num_state, num_obs
            )
        return attn_weights

    def _compute_recal3r_prior(self, prev_state_feat):
        if not self._uses_recal3r():
            raise RuntimeError("recal3r prior requested outside recal3r mode")
        if not hasattr(self, "recal3r_state0"):
            self._init_recal3r_reference_state(prev_state_feat)

        state_prev = prev_state_feat.detach()
        drift_reference = self.recal3r_state0.detach()
        if state_prev.shape != drift_reference.shape:
            raise AssertionError(
                "recal3r state reference mismatch: "
                f"prev_state_shape={tuple(state_prev.shape)} "
                f"reference_shape={tuple(drift_reference.shape)}"
            )

        drift_m = torch.norm(
            state_prev.float() - drift_reference.float(), p=2, dim=-1
        )
        num_tokens = max(1, drift_m.shape[-1])
        sort_idx = torch.argsort(drift_m, dim=-1)
        ranks = torch.argsort(sort_idx, dim=-1).to(drift_m.dtype)
        rank_norm = (ranks + 1.0) / float(num_tokens)
        pi_0 = 1.0 - rank_norm

        sequence_age = getattr(self, "_recal3r_sequence_age", None)
        if sequence_age is None or sequence_age.shape[0] != drift_m.shape[0]:
            sequence_age = torch.zeros(
                drift_m.shape[0], device=drift_m.device, dtype=torch.long
            )
            self._recal3r_sequence_age = sequence_age
        first_frame_mask = sequence_age == 0
        if bool(first_frame_mask.any().item()):
            pi_0[first_frame_mask] = 0.5

        if drift_m.shape[0] == 1:
            return pi_0.squeeze(0), drift_m.squeeze(0)
        return pi_0, drift_m

    def _compute_residual_score(self, dec):
        with torch.no_grad():
            f_img_input_proj = self.decoder_embed(dec[0])
            f_img_output = dec[-1][:, 1:]

            diff = f_img_input_proj - f_img_output
            raw_recon_err = torch.norm(diff, p=2, dim=-1).mean()
            input_norm = torch.norm(f_img_input_proj, p=2, dim=-1).mean() + 1e-6
            relative_err = raw_recon_err / input_norm
            residual_score = torch.sigmoid(relative_err - 0.2)

        return residual_score, relative_err

    def _compute_recal3r_update_mask(
        self,
        update_mask,
        raw_cross_attn_state,
        dec,
        prev_state_feat=None,
    ):
        _, beta_t, _ = self._compute_state_update_mask(update_mask, raw_cross_attn_state)
        alignment_gate = beta_t

        with torch.no_grad():
            if (
                not hasattr(self, "update_pressure")
                or self.update_pressure.shape != alignment_gate.shape
            ):
                self.update_pressure = torch.zeros_like(alignment_gate)
            decay = self._update_pressure_decay()
            self.update_pressure = (
                decay * self.update_pressure + (1.0 - decay) * alignment_gate.detach()
            )
            attenuation = torch.exp(-self.update_pressure)
            residual_score_scalar, _ = self._compute_residual_score(dec)

            if prev_state_feat is None:
                raise RuntimeError("recal3r requires prev_state_feat for drift prior")

            attn_weights = self._state_attention_weights(raw_cross_attn_state)
            pi_0, _ = self._compute_recal3r_prior(prev_state_feat)
            h_m = self._compute_empirical_quantile_norm(
                self._compute_row_entropy_norm(attn_weights)
            )
            beta_trust = alignment_gate * residual_score_scalar * attenuation
            gamma_num = pi_0 * (1.0 - h_m)
            gamma_bad = (1.0 - pi_0) * h_m
            gamma_den = gamma_num + gamma_bad + 1e-8
            gamma_m = gamma_num / gamma_den
            R = self._compute_recal3r_u(gamma_m)

            if beta_trust.dim() > gamma_m.dim():
                h_m = self._align_token_stat_with_beta(h_m, beta_trust)
                gamma_m = self._align_token_stat_with_beta(gamma_m, beta_trust)
                R = self._align_token_stat_with_beta(R, beta_trust)

            beta_base = torch.full_like(beta_trust, self._beta_base())
            beta_final = (1.0 - R) * beta_trust + R * beta_base
            beta_final = self._smooth_beta_toward_frame_mean(beta_final)
            beta_t = beta_final.clamp(min=0.0, max=1.0)

        self._stash_u_calibration_stats(
            R.squeeze(-1) if R.dim() > 2 else R,
            h_m.squeeze(-1) if h_m.dim() > 2 else h_m,
        )
        return update_mask * beta_t

    # tbptt training encoder: Truncated Backpropagation Through Time
    def _forward_encoder(self, views):
        shape, feat_ls, pos = self._encode_views(views)
        feat = feat_ls[-1]
        state_feat, state_pos = self._init_state(feat[0], pos[0])
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1)
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        return (feat, pos, shape), (
            init_state_feat,
            init_mem,
            state_feat,
            state_pos,
            mem,
        )

    # tbptt training decoder step: Truncated Backpropagation Through Time
    def _forward_decoder_step(
        self,
        views,
        i,
        feat_i,
        pos_i,
        shape_i,
        init_state_feat,
        init_mem,
        state_feat,
        state_pos,
        mem,
    ):
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            if i == 0:
                pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
            else:
                pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=views[i]["img_mask"],
            reset_mask=views[i]["reset"],
            update=views[i].get("update", None),
            return_attn=False,
        )
        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape_i, pos=pos_i)
        img_mask = views[i]["img_mask"]
        update = views[i].get("update", None)
        if update is not None:
            update_mask = img_mask & update  # if don't update, then whatever img_mask
        else:
            update_mask = img_mask
        update_mask = update_mask[:, None, None].float()
        state_feat = new_state_feat * update_mask + state_feat * (
            1 - update_mask
        )  # update global state
        mem = new_mem * update_mask + mem * (1 - update_mask)  # then update local state
        reset_mask = views[i]["reset"]
        if reset_mask is not None:
            reset_mask = reset_mask[:, None, None].float()
            state_feat = init_state_feat * reset_mask + state_feat * (1 - reset_mask)
            mem = init_mem * reset_mask + mem * (1 - reset_mask)
        return res, (state_feat, mem)

    # training and testing
    def _forward_impl(self, views, ret_state=False):
        self.config.model_update_type = canonicalize_model_update_type(
            getattr(self.config, "model_update_type", "cut3r")
        )
        model_update_type = self.config.model_update_type
        # [B, C, H, W] -> [B, H/16*W/16, 1024]
        shape, feat_ls, pos = self._encode_views(views) # [15, 3, 288, 512] -> feat [15, 576, 1024], pos [15, 576, 2]
        feat = feat_ls[-1]
        if self._uses_update_pressure_update() and hasattr(self, "update_pressure"):
            del self.update_pressure
        state_feat, state_pos = self._init_state(feat[0], pos[0]) # init state feat [1, 768, 768], state_pos [1, 768, 2]
        self._init_recal3r_reference_state(state_feat)
        mem = self.pose_retriever.mem.expand(feat[0].shape[0], -1, -1) # [1, 256, 1536] init pose mem
        init_state_feat = state_feat.clone()
        init_mem = mem.clone()
        all_state_args = [(state_feat, state_pos, init_state_feat, mem, init_mem)]
        ress = []
        for i in range(len(views)):
            feat_i = feat[i]
            pos_i = pos[i]
            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i) # avg pool: [1, 576, 1024] -> [1, 1, 1024]
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1) # [1, 1, 768] init pose token
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem) 
                    # [1, 1, 768] use [global_img_feat_i, masked_token(pose)] as query, cross-attend mem, get pose_feat_i
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                ) # [1, 1, 2]
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat, # [1, 768, 768]
                state_pos, # [1, 768, 2]
                feat_i, # [1, 576, 1024]
                pos_i, # [1, 576, 2]
                pose_feat_i, # [1, 1, 768] coarse pose token from pose_retriever
                pose_pos_i, # [1, 1, 2]
                init_state_feat,
                img_mask=views[i]["img_mask"],
                reset_mask=views[i]["reset"],
                update=views[i].get("update", None),
                return_attn=True,
            ) # [1, 768, 768]
            out_pose_feat_i = dec[-1][:, 0:1] # [1, 1, 768] refined pose token from dust3r
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            ) # [1, 256, 1536] use mem as query, cross-attend [global_img_feat_i, out_pose_feat_i], get new_mem
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(), # [1, 576, 1024]
                dec[self.dec_depth * 2 // 4][:, 1:].float(), # [1, 576, 768]
                dec[self.dec_depth * 3 // 4][:, 1:].float(), # [1, 576, 768]
                dec[self.dec_depth].float(), # [1, 1 + 576, 768]
            ]
            res = self._downstream_head(head_input, shape[i], pos=pos_i)
            ress.append(res)
            img_mask = views[i]["img_mask"]
            update = views[i].get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            # update with learning rate
            raw_cross_attn_state = cross_attn_state
            if i  == 0:
                update_mask1 = update_mask
            else:
                if model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif model_update_type == "ttt3r":
                    update_mask1, _, _ = self._compute_state_update_mask(
                        update_mask, raw_cross_attn_state
                    )
                else:
                    update_mask1 = self._compute_recal3r_update_mask(
                        update_mask, raw_cross_attn_state, dec, prev_state_feat=state_feat
                    )

            update_mask2 = update_mask
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state
            ress[-1] = res
            self._advance_recal3r_sequence_age(state_feat)
            reset_mask = views[i]["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                self._reset_recal3r_reference_state_if_needed(
                    reset_mask, init_state_feat
                )
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
                self._reset_update_pressure_if_needed(reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward(self, views, ret_state=False):
        if ret_state:
            ress, views, state_args = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views), state_args
        else:
            ress, views = self._forward_impl(views, ret_state=ret_state)
            return ARCroco3DStereoOutput(ress=ress, views=views)

    # testing: generate rgb xyz condition on raymap
    def inference_step(
        self, view, state_feat, state_pos, init_state_feat, mem, init_mem
    ):
        batch_size = view["img"].shape[0]
        raymaps = []
        shapes = []
        for j in range(batch_size):
            assert view["ray_mask"][j]
            raymap = view["ray_map"][[j]].permute(0, 3, 1, 2)
            raymaps.append(raymap)
            shapes.append(
                view.get(
                    "true_shape",
                    torch.tensor(view["ray_map"].shape[-2:])[None].repeat(
                        view["ray_map"].shape[0], 1
                    ),
                )[[j]]
            )

        raymaps = torch.cat(raymaps, dim=0)
        shape = torch.cat(shapes, dim=0).to(raymaps.device)
        feat_ls, pos, _ = self._encode_ray_map(raymaps, shapes) # [1, 6, 384, 512] -> feat [1, 768, 1024], pos [1, 768, 2]

        feat_i = feat_ls[-1]
        pos_i = pos
        if self.pose_head_flag:
            global_img_feat_i = self._get_img_level_feat(feat_i)
            pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
            pose_pos_i = -torch.ones(
                feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
            )
        else:
            pose_feat_i = None
            pose_pos_i = None
        new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
            state_feat,
            state_pos,
            feat_i,
            pos_i,
            pose_feat_i,
            pose_pos_i,
            init_state_feat,
            img_mask=view["img_mask"],
            reset_mask=view["reset"],
            update=view.get("update", None),
            return_attn=False,
        )

        out_pose_feat_i = dec[-1][:, 0:1]
        new_mem = self.pose_retriever.update_mem(
            mem, global_img_feat_i, out_pose_feat_i
        )
        assert len(dec) == self.dec_depth + 1
        head_input = [
            dec[0].float(),
            dec[self.dec_depth * 2 // 4][:, 1:].float(),
            dec[self.dec_depth * 3 // 4][:, 1:].float(),
            dec[self.dec_depth].float(),
        ]
        res = self._downstream_head(head_input, shape, pos=pos_i)
        return res, view

    # recurrent testing
    def forward_recurrent(self, views, device, ret_state=False):
        ress = []
        all_state_args = []
        for i, view in enumerate(views):
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()
                all_state_args.append(
                    (state_feat, state_pos, init_state_feat, mem, init_mem)
                )

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=False,
            )
            out_pose_feat_i = dec[-1][:, 0:1]
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )
            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            ress.append(res)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()
            state_feat = new_state_feat * update_mask + state_feat * (
                1 - update_mask
            )  # update global state
            mem = new_mem * update_mask + mem * (
                1 - update_mask
            )  # then update local state
            reset_mask = view["reset"]
            if reset_mask is not None:
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)
            all_state_args.append(
                (state_feat, state_pos, init_state_feat, mem, init_mem)
            )
        if ret_state:
            return ress, views, all_state_args
        return ress, views

    def forward_recurrent_lighter(self, views, device='cuda', ret_state=False):
        self.config.model_update_type = canonicalize_model_update_type(
            getattr(self.config, "model_update_type", "cut3r")
        )
        model_update_type = self.config.model_update_type
        ress = []
        all_state_args = []
        reset_mask = False
        if self._uses_update_pressure_update() and hasattr(self, "update_pressure"):
            del self.update_pressure
        for i, _view in enumerate(views):
            view = to_gpu(_view, device)
            device = view["img"].device
            batch_size = view["img"].shape[0]
            img_mask = view["img_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            ray_mask = view["ray_mask"].reshape(
                -1, batch_size
            )  # Shape: (1, batch_size)
            imgs = view["img"].unsqueeze(0)  # Shape: (1, batch_size, C, H, W)
            ray_maps = view["ray_map"].unsqueeze(
                0
            )  # Shape: (num_views, batch_size, H, W, C)
            shapes = (
                view["true_shape"].unsqueeze(0)
                if "true_shape" in view
                else torch.tensor(view["img"].shape[-2:], device=device)
                .unsqueeze(0)
                .repeat(batch_size, 1)
                .unsqueeze(0)
            )  # Shape: (num_views, batch_size, 2)
            imgs = imgs.view(
                -1, *imgs.shape[2:]
            )  # Shape: (num_views * batch_size, C, H, W)
            ray_maps = ray_maps.view(
                -1, *ray_maps.shape[2:]
            )  # Shape: (num_views * batch_size, H, W, C)
            shapes = shapes.view(-1, 2).to(
                imgs.device
            )  # Shape: (num_views * batch_size, 2)
            img_masks_flat = img_mask.view(-1)  # Shape: (num_views * batch_size)
            ray_masks_flat = ray_mask.view(-1)
            selected_imgs = imgs[img_masks_flat]
            selected_shapes = shapes[img_masks_flat]
            if selected_imgs.size(0) > 0:
                img_out, img_pos, _ = self._encode_image(selected_imgs, selected_shapes)
            else:
                img_out, img_pos = None, None
            ray_maps = ray_maps.permute(0, 3, 1, 2)  # Change shape to (N, C, H, W)
            selected_ray_maps = ray_maps[ray_masks_flat]
            selected_shapes_ray = shapes[ray_masks_flat]
            if selected_ray_maps.size(0) > 0:
                ray_out, ray_pos, _ = self._encode_ray_map(
                    selected_ray_maps, selected_shapes_ray
                )
            else:
                ray_out, ray_pos = None, None

            shape = shapes
            if img_out is not None and ray_out is None:
                feat_i = img_out[-1]
                pos_i = img_pos
            elif img_out is None and ray_out is not None:
                feat_i = ray_out[-1]
                pos_i = ray_pos
            elif img_out is not None and ray_out is not None:
                feat_i = img_out[-1] + ray_out[-1]
                pos_i = img_pos
            else:
                raise NotImplementedError

            if i == 0:
                state_feat, state_pos = self._init_state(feat_i, pos_i)
                self._init_recal3r_reference_state(state_feat)
                mem = self.pose_retriever.mem.expand(feat_i.shape[0], -1, -1)
                init_state_feat = state_feat.clone()
                init_mem = mem.clone()

            if self.pose_head_flag:
                global_img_feat_i = self._get_img_level_feat(feat_i)
                if i == 0 or reset_mask:
                    pose_feat_i = self.pose_token.expand(feat_i.shape[0], -1, -1)
                else:
                    pose_feat_i = self.pose_retriever.inquire(global_img_feat_i, mem)
                pose_pos_i = -torch.ones(
                    feat_i.shape[0], 1, 2, device=feat_i.device, dtype=pos_i.dtype
                )
            else:
                pose_feat_i = None
                pose_pos_i = None
            new_state_feat, dec, self_attn_state, cross_attn_state, self_attn_img, cross_attn_img = self._recurrent_rollout(
                state_feat,
                state_pos,
                feat_i,
                pos_i,
                pose_feat_i,
                pose_pos_i,
                init_state_feat,
                img_mask=view["img_mask"],
                reset_mask=view["reset"],
                update=view.get("update", None),
                return_attn=True,
            )
            out_pose_feat_i = dec[-1][:, 0:1]

            # update mem
            new_mem = self.pose_retriever.update_mem(
                mem, global_img_feat_i, out_pose_feat_i
            )

            assert len(dec) == self.dec_depth + 1
            head_input = [
                dec[0].float(),
                dec[self.dec_depth * 2 // 4][:, 1:].float(),
                dec[self.dec_depth * 3 // 4][:, 1:].float(),
                dec[self.dec_depth].float(),
            ]
            res = self._downstream_head(head_input, shape, pos=pos_i)
            img_mask = view["img_mask"]
            update = view.get("update", None)
            if update is not None:
                update_mask = (
                    img_mask & update
                )  # if don't update, then whatever img_mask
            else:
                update_mask = img_mask
            update_mask = update_mask[:, None, None].float()

            # update with learning rate
            raw_cross_attn_state = cross_attn_state
            prev_state_feat = state_feat
            if i  == 0 or reset_mask:
                update_mask1 = update_mask
            else:
                if model_update_type == "cut3r":
                    update_mask1 = update_mask
                elif model_update_type == "ttt3r":
                    update_mask1, _, _ = self._compute_state_update_mask(
                        update_mask, raw_cross_attn_state
                    )
                else:
                    update_mask1 = self._compute_recal3r_update_mask(
                        update_mask, raw_cross_attn_state, dec, prev_state_feat=state_feat
                    )

            update_mask2 = update_mask
            state_feat = new_state_feat * update_mask1 + state_feat * (
                1 - update_mask1
            )  # update global state
            mem = new_mem * update_mask2 + mem * (
                1 - update_mask2
            )  # then update local state
            self._advance_recal3r_sequence_age(state_feat)
            self._maybe_record_u_calibration_step(i, prev_state_feat, state_feat)
            res_cpu = to_cpu(res)
            ress.append(res_cpu)

            reset_mask = view["reset"]
            if reset_mask is not None:
                self._reset_update_pressure_if_needed(reset_mask)
                self._reset_recal3r_reference_state_if_needed(
                    reset_mask, init_state_feat
                )
                reset_mask = reset_mask[:, None, None].float()
                state_feat = init_state_feat * reset_mask + state_feat * (
                    1 - reset_mask
                )
                mem = init_mem * reset_mask + mem * (1 - reset_mask)

        if ret_state:
            return ress, views, all_state_args
        return ress, views

if __name__ == "__main__":
    print(ARCroco3DStereo.mro())
    cfg = ARCroco3DStereoConfig(
        state_size=256,
        pos_embed="RoPE100",
        rgb_head=True,
        pose_head=True,
        img_size=(224, 224),
        head_type="linear",
        output_mode="pts3d+pose",
        depth_mode=("exp", -inf, inf),
        conf_mode=("exp", 1, inf),
        pose_mode=("exp", -inf, inf),
        enc_embed_dim=1024,
        enc_depth=24,
        enc_num_heads=16,
        dec_embed_dim=768,
        dec_depth=12,
        dec_num_heads=12,
    )
    ARCroco3DStereo(cfg)
