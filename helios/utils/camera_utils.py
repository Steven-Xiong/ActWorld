"""Camera-pose utilities for Helios plücker ray conditioning.

The geometric helpers in this module are a verbatim port from
``lingbot-world/wan/utils/cam_utils.py``: the signatures, coordinate
conventions and numerical behaviour all match bit-for-bit so that
experimental results stay comparable with the original lingbot pipeline.

The only additions are:

* ``sample_c2w_to_latent_frames``: a small helper that picks one c2w per
  latent frame when the source tensor is sampled at pixel FPS. This
  sidesteps Wan-VAE's non-uniform temporal compression (33 pixel frames
  → 9 latent frames).
* ``PluckerFiLM``: a shared FiLM head (~165M params at ``dim=5120``) that
  is instantiated **once** at the root model and reused across every
  transformer block. See ``docs/plan_plucker_conditioning.md`` for the
  design rationale.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation, Slerp


# ---------------------------------------------------------------------
# Lingbot-world cam_utils verbatim port (functions below are byte-for-byte
# equivalent to wan/utils/cam_utils.py up to import path changes).
# ---------------------------------------------------------------------


def interpolate_camera_poses(
    src_indices: np.ndarray,
    src_rot_mat: np.ndarray,
    src_trans_vec: np.ndarray,
    tgt_indices: np.ndarray,
) -> torch.Tensor:
    """Interpolate sparse (R, t) keyframes onto a dense index grid."""
    interp_func_trans = interp1d(
        src_indices,
        src_trans_vec,
        axis=0,
        kind="linear",
        bounds_error=False,
        fill_value="extrapolate",
    )
    interpolated_trans_vec = interp_func_trans(tgt_indices)

    src_quat_vec = Rotation.from_matrix(src_rot_mat)
    quats = src_quat_vec.as_quat().copy()
    for i in range(1, len(quats)):
        if np.dot(quats[i], quats[i - 1]) < 0:
            quats[i] = -quats[i]
    src_quat_vec = Rotation.from_quat(quats)
    slerp_func_rot = Slerp(src_indices, src_quat_vec)
    interpolated_rot_quat = slerp_func_rot(tgt_indices)
    interpolated_rot_mat = interpolated_rot_quat.as_matrix()

    poses = np.zeros((len(tgt_indices), 4, 4))
    poses[:, :3, :3] = interpolated_rot_mat
    poses[:, :3, 3] = interpolated_trans_vec
    poses[:, 3, 3] = 1.0
    return torch.from_numpy(poses).float()


def SE3_inverse(T: torch.Tensor) -> torch.Tensor:
    """Batched SE(3) inverse: (B,4,4) -> (B,4,4)."""
    Rot = T[:, :3, :3]
    trans = T[:, :3, 3:]
    R_inv = Rot.transpose(-1, -2)
    t_inv = -torch.bmm(R_inv, trans)
    T_inv = torch.eye(4, device=T.device, dtype=T.dtype)[None, :, :].repeat(T.shape[0], 1, 1)
    T_inv[:, :3, :3] = R_inv
    T_inv[:, :3, 3:] = t_inv
    return T_inv


def compute_relative_poses(
    c2ws_mat: torch.Tensor,
    framewise: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    """Normalise a (F,4,4) c2w sequence so frame 0 is identity and
    translations fit in roughly unit magnitude.

    ``normalize_trans=True`` divides by the max translation norm across
    the sequence (camctrl2 convention), which is critical for
    VIPE-estimated poses that have arbitrary absolute scale.
    """
    ref_w2cs = SE3_inverse(c2ws_mat[0:1])
    relative_poses = torch.matmul(ref_w2cs, c2ws_mat)
    relative_poses[0] = torch.eye(4, device=c2ws_mat.device, dtype=c2ws_mat.dtype)
    if framewise:
        relative_poses_framewise = torch.bmm(SE3_inverse(relative_poses[:-1]), relative_poses[1:])
        relative_poses[1:] = relative_poses_framewise
    if normalize_trans:
        translations = relative_poses[:, :3, 3]
        max_norm = torch.norm(translations, dim=-1).max()
        if max_norm > 0:
            relative_poses[:, :3, 3] = translations / max_norm
    return relative_poses


@torch.no_grad()
def create_meshgrid(
    n_frames: int,
    height: int,
    width: int,
    bias: float = 0.5,
    device="cuda",
    dtype=torch.float32,
) -> torch.Tensor:
    """Return a pixel-centre meshgrid replicated across ``n_frames``.

    Output shape: (n_frames, height*width, 2) with [x, y] in pixel units
    (shifted by +bias so integer indices become pixel centres).
    """
    x_range = torch.arange(width, device=device, dtype=dtype)
    y_range = torch.arange(height, device=device, dtype=dtype)
    grid_y, grid_x = torch.meshgrid(y_range, x_range, indexing="ij")
    grid_xy = torch.stack([grid_x, grid_y], dim=-1).view([-1, 2]) + bias
    grid_xy = grid_xy[None, ...].repeat(n_frames, 1, 1)
    return grid_xy


def get_plucker_embeddings(
    c2ws_mat: torch.Tensor,
    Ks: torch.Tensor,
    height: int,
    width: int,
    only_rays_d: bool = False,
):
    """Compute the Plücker embedding (rays_o, rays_d) at every pixel.

    Args:
        c2ws_mat: (F, 4, 4) per-frame camera-to-world matrices.
        Ks:       (F, 4)    per-frame intrinsics [fx, fy, cx, cy] in
                             pixel units of the target (H, W) grid.
        height, width: spatial resolution to sample rays at.
        only_rays_d: if True return only the 3D ray direction.

    Returns:
        (F, H, W, 6) when ``only_rays_d`` is False (rays_o || rays_d)
        (F, H, W, 3) when ``only_rays_d`` is True.
    """
    n_frames = c2ws_mat.shape[0]
    grid_xy = create_meshgrid(
        n_frames, height, width, device=c2ws_mat.device, dtype=c2ws_mat.dtype
    )  # (F, H*W, 2)
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)  # each (F, 1)

    i = grid_xy[..., 0]
    j = grid_xy[..., 1]
    zs = torch.ones_like(i)
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs

    directions = torch.stack([xs, ys, zs], dim=-1)  # (F, H*W, 3)
    directions = directions / directions.norm(dim=-1, keepdim=True)

    rays_d = directions @ c2ws_mat[:, :3, :3].transpose(-1, -2)  # (F, H*W, 3)
    if only_rays_d:
        plucker_embeddings = rays_d
        plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 3])
    else:
        rays_o = c2ws_mat[:, :3, 3]  # (F, 3)
        rays_o = rays_o[:, None, :].expand_as(rays_d)
        plucker_embeddings = torch.cat([rays_o, rays_d], dim=-1)  # (F, H*W, 6)
        plucker_embeddings = plucker_embeddings.view([n_frames, height, width, 6])
    return plucker_embeddings


def get_Ks_transformed(
    Ks: torch.Tensor,
    height_org: int,
    width_org: int,
    height_resize: int,
    width_resize: int,
    height_final: int,
    width_final: int,
):
    """Rescale (fx, fy, cx, cy) to follow a resize + centre-crop to target.

    Identical convention to lingbot: pixels are first resized from
    (height_org, width_org) to (height_resize, width_resize), then centre
    cropped to (height_final, width_final). When the training pipeline
    resizes directly to the target (no crop), set
    ``height_resize == height_final`` and ``width_resize == width_final``.
    """
    fx, fy, cx, cy = Ks.chunk(4, dim=-1)

    scale_x = width_resize / width_org
    scale_y = height_resize / height_org

    fx_resize = fx * scale_x
    fy_resize = fy * scale_y
    cx_resize = cx * scale_x
    cy_resize = cy * scale_y

    crop_offset_x = (width_resize - width_final) / 2
    crop_offset_y = (height_resize - height_final) / 2

    cx_final = cx_resize - crop_offset_x
    cy_final = cy_resize - crop_offset_y

    Ks_transformed = torch.zeros_like(Ks)
    Ks_transformed[:, 0:1] = fx_resize
    Ks_transformed[:, 1:2] = fy_resize
    Ks_transformed[:, 2:3] = cx_final
    Ks_transformed[:, 3:4] = cy_final

    return Ks_transformed


# ---------------------------------------------------------------------
# Helios-specific helpers
# ---------------------------------------------------------------------


def sample_c2w_to_latent_frames(
    c2ws_pixel: torch.Tensor,
    latent_frame_count: int,
) -> torch.Tensor:
    """Downsample a per-pixel-frame c2w tensor to a per-latent-frame one.

    Wan VAE compresses the time axis non-uniformly (e.g. 33 pixel frames
    → 9 latent frames: latent 0 covers pixel 0 only, latent ``k>0``
    covers pixels ``4k-3..4k``). We pick the midpoint pixel frame of each
    latent group, which matches how `camera_poses_chunk` is already
    defined in the .pt schema (mid frame of each 33-frame chunk).

    Args:
        c2ws_pixel: (T_pixel, 4, 4) or (B, T_pixel, 4, 4)
        latent_frame_count: target latent frame count (e.g. 9).

    Returns:
        Same rank as input with the time dimension replaced by
        ``latent_frame_count``.
    """
    if c2ws_pixel.ndim == 3:
        T_pixel = c2ws_pixel.shape[0]
        stride = (T_pixel - 1) / max(latent_frame_count - 1, 1) if latent_frame_count > 1 else 1
        indices = [min(round(i * stride), T_pixel - 1) for i in range(latent_frame_count)]
        return c2ws_pixel[indices]
    elif c2ws_pixel.ndim == 4:
        T_pixel = c2ws_pixel.shape[1]
        stride = (T_pixel - 1) / max(latent_frame_count - 1, 1) if latent_frame_count > 1 else 1
        indices = [min(round(i * stride), T_pixel - 1) for i in range(latent_frame_count)]
        return c2ws_pixel[:, indices]
    raise ValueError(f"expected (T,4,4) or (B,T,4,4), got {tuple(c2ws_pixel.shape)}")


# ---------------------------------------------------------------------
# PluckerFiLM: shared head used by HeliosTransformer3DModel
# ---------------------------------------------------------------------


class PluckerFiLM(nn.Module):
    """Shared Plücker FiLM head applied at every transformer block.

    Design (see docs/plan_plucker_conditioning.md):

      * A single instance lives on the root model. The transformer blocks
        hold no cam parameters; they simply call `modulate(x, emb)` with
        a reference to this module plus the pre-computed emb.
      * `encode(plucker_raw)` is called once per forward to produce the
        token-level embedding; `modulate(x, emb)` is called per block.
      * Internal structure matches lingbot-world's `WanAttentionBlock`
        cam head layout exactly (4 × Linear(D, D) for injector/scale/shift
        + 2 × Linear(D, D) residual + 1 Linear(in_dim, D) pixel-grid
        patch embedding), so per-module arithmetic stays aligned.
      * Params at D=5120 ≈ 165M (1.2% of a 14B DiT).

    One Helios-specific tweak: ``cam_scale_layer`` and
    ``cam_shift_layer`` WEIGHTS are zero-initialised (biases are zero too
    per lingbot). This guarantees that at step 0
    ``modulate(x, emb) = (1 + 0)*x + 0 = x``, so enabling `use_plucker`
    on top of a trained stage_1_post checkpoint does not perturb the
    baseline loss.
    """

    def __init__(
        self,
        inner_dim: int = 5120,
        patch_size=(1, 2, 2),
        only_rays_d: bool = False,
        vae_space_down: int = 8,
    ) -> None:
        super().__init__()
        self.inner_dim = inner_dim
        self.patch_size = tuple(patch_size)
        self.only_rays_d = bool(only_rays_d)
        self.vae_space_down = int(vae_space_down)

        in_c = 3 if self.only_rays_d else 6
        pt, ph, pw = self.patch_size
        sd = self.vae_space_down
        # Pixel-grid plucker folded into channels: each transformer token
        # absorbs a (1, 8*ph, 8*pw) pixel block × pt latent frames.
        in_dim = in_c * (sd * sd) * pt * ph * pw  # 1536 for 6D, (1,2,2), sd=8

        # ----- Global encoding (called once per forward) -----
        self.patch_embedding_wancamctrl = nn.Linear(in_dim, inner_dim)
        self.c2ws_hidden_states_layer1 = nn.Linear(inner_dim, inner_dim)
        self.c2ws_hidden_states_layer2 = nn.Linear(inner_dim, inner_dim)

        # ----- Shared per-block FiLM head (called N times per forward) -----
        self.cam_injector_layer1 = nn.Linear(inner_dim, inner_dim)
        self.cam_injector_layer2 = nn.Linear(inner_dim, inner_dim)
        self.cam_scale_layer = nn.Linear(inner_dim, inner_dim)
        self.cam_shift_layer = nn.Linear(inner_dim, inner_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        # Lingbot init: Xavier + zeros bias for every cam Linear.
        for m in (
            self.patch_embedding_wancamctrl,
            self.c2ws_hidden_states_layer1,
            self.c2ws_hidden_states_layer2,
            self.cam_injector_layer1,
            self.cam_injector_layer2,
            self.cam_scale_layer,
            self.cam_shift_layer,
        ):
            nn.init.xavier_uniform_(m.weight)
            nn.init.zeros_(m.bias)
        # Helios tweak: force step-0 identity on the FiLM output heads.
        nn.init.zeros_(self.cam_scale_layer.weight)
        nn.init.zeros_(self.cam_shift_layer.weight)

    # ------------------------------------------------------------------
    def encode(self, plucker: torch.Tensor) -> torch.Tensor:
        """Pixel-grid plucker -> (B, N_tok, D) token embedding.

        Args:
            plucker: (B, C, T_lat, H_pixel, W_pixel) where
              C = 3 or 6, H_pixel = H_lat*vae_space_down,
              W_pixel = W_lat*vae_space_down.

        Returns:
            (B, N_tok, D) ready to be consumed by ``modulate``. The token
            order matches patch_embedding's (T_lat, H_lat/ph, W_lat/pw)
            flattening so 1:1 alignment with current-chunk hidden states
            is preserved.
        """
        pt, ph, pw = self.patch_size
        sd = self.vae_space_down
        # (B, C, (f pt), (h sph), (w spw)) -> (B, f*h*w, C*pt*sph*spw)
        x = rearrange(
            plucker,
            "b c (f pt) (h sph) (w spw) -> b (f h w) (c pt sph spw)",
            pt=pt,
            sph=ph * sd,
            spw=pw * sd,
        )
        x = self.patch_embedding_wancamctrl(x)
        residual = self.c2ws_hidden_states_layer2(
            F.silu(self.c2ws_hidden_states_layer1(x))
        )
        return x + residual

    def modulate(self, x: torch.Tensor, plucker_emb: torch.Tensor) -> torch.Tensor:
        """Apply FiLM to ``x`` using the cached ``plucker_emb``.

        Args:
            x:           (B, N_cur, D) slice of hidden states (current-chunk tokens).
            plucker_emb: (B, N_cur, D) output of ``encode``.

        Returns:
            (B, N_cur, D) with ``x = (1 + cam_scale) * x + cam_shift``.
            At step 0 (zero-init on the scale/shift heads) this reduces
            to identity.
        """
        cam_feat = self.cam_injector_layer2(
            F.silu(self.cam_injector_layer1(plucker_emb))
        )
        cam_feat = cam_feat + plucker_emb
        cam_scale = self.cam_scale_layer(cam_feat)
        cam_shift = self.cam_shift_layer(cam_feat)
        return (1.0 + cam_scale) * x + cam_shift


# ---------------------------------------------------------------------
# Helper used by transformer.forward to turn batch poses + intrinsics
# into a (B, C, T_lat, H_pixel, W_pixel) plucker tensor.
# ---------------------------------------------------------------------


def build_plucker_tensor_for_current_chunk(
    camera_poses: torch.Tensor,        # (B, T_pixel, 4, 4)  per-pixel-frame c2w
    camera_intrinsics: torch.Tensor,   # (B, 4) OR (B, T_pixel, 4)   [fx,fy,cx,cy]
    latent_frame_count: int,           # T_lat, e.g. 9
    h_pixel: int,                      # training pixel height, e.g. 384
    w_pixel: int,                      # training pixel width,  e.g. 640
    only_rays_d: bool = False,
    normalize_trans: bool = True,
) -> torch.Tensor:
    """Build per-sample plücker tensors and stack them into a batch.

    Applies per-chunk ``compute_relative_poses`` so that frame 0 is the
    canonical origin and translations are normalised. Returns a tensor
    shaped (B, 3|6, T_lat, H_pixel, W_pixel) ready for
    ``PluckerFiLM.encode``.
    """
    B = camera_poses.shape[0]
    dtype = camera_poses.dtype
    device = camera_poses.device

    # Downsample pixel-frame c2w to latent-frame c2w.
    c2ws_lat = sample_c2w_to_latent_frames(camera_poses, latent_frame_count)  # (B, T_lat, 4, 4)

    out_list = []
    for b in range(B):
        c2w_b = compute_relative_poses(
            c2ws_lat[b].to(torch.float32), framewise=False, normalize_trans=normalize_trans
        )
        # Intrinsics: (4,) video-level or (T_pixel, 4) per-frame -> (T_lat, 4)
        Ks = camera_intrinsics[b]
        if Ks.ndim == 1:
            Ks_lat = Ks.unsqueeze(0).expand(latent_frame_count, -1).contiguous().to(torch.float32)
        elif Ks.ndim == 2:
            # (T_pixel, 4) -> (T_lat, 4). Must use the SAME midpoint rule as
            # sample_c2w_to_latent_frames so Ks_lat[k] and c2w_b[k] describe
            # the same pixel frame. sample_c2w_to_latent_frames doesn't accept
            # this 2D layout (its ndim=3 branch treats dim-0 as T), so we
            # inline the index calculation here.
            T_pixel = Ks.shape[0]
            if latent_frame_count > 1:
                stride = (T_pixel - 1) / max(latent_frame_count - 1, 1)
                indices = [min(round(i * stride), T_pixel - 1) for i in range(latent_frame_count)]
            else:
                indices = [0]
            Ks_lat = Ks[indices].to(torch.float32)
        else:
            raise ValueError(f"camera_intrinsics[b] must be rank 1 or 2, got {Ks.ndim}")

        plucker_b = get_plucker_embeddings(
            c2w_b.to(device), Ks_lat.to(device), h_pixel, w_pixel, only_rays_d=only_rays_d
        )
        out_list.append(plucker_b)

    plucker = torch.stack(out_list, dim=0)  # (B, T_lat, H, W, C)
    plucker = plucker.permute(0, 4, 1, 2, 3).contiguous().to(dtype)  # (B, C, T_lat, H, W)
    return plucker
