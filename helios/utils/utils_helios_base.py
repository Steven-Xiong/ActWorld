import random

import torch
import torch.nn.functional as F


# Process-local one-shot probes for the Dual-Memory paths. Each module
# fires its log line once on first activation so training output makes
# it obvious that the path actually triggered (vs being silently no-op'd
# by an empty annotation list, AssertionError fallback, etc.).
_EAFR_RUNTIME_LOGGED: dict = {}
from accelerate.logging import get_logger
from einops import rearrange

from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3, free_memory

from .utils_base import apply_schedule_shift, get_config_value
from .utils_recycle_batch import apply_error_injection, process_and_update_error_buffers


logger = get_logger(__name__)


# ======================================== flow loss ========================================


def _flow_loss(
    args,
    accelerator,
    lr_scheduler,
    transformer,
    prompt_embeds,
    prompt_attention_masks,
    noisy_model_input_list,
    sigmas_list,
    timesteps_list,
    targets_list,
    indices_hidden_states,
    latents_history_short,
    indices_latents_history_short,
    latents_history_mid,
    indices_latents_history_mid,
    latents_history_long,
    indices_latents_history_long,
    recycle_vars,
    global_step,
    noise_scheduler_copy,
    use_clean_input,
    action_id=None,
    action_vec=None,
    camera_poses=None,
    camera_intrinsics=None,
    action_memory_context=None,
):
    assert len(noisy_model_input_list) == len(sigmas_list) == len(timesteps_list) == len(targets_list)

    for noisy_model_input, sigmas, timesteps, target in zip(
        noisy_model_input_list, sigmas_list, timesteps_list, targets_list
    ):
        # ----- w/o mini batch ------
        model_pred = transformer(
            hidden_states=noisy_model_input,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            indices_hidden_states=indices_hidden_states,  # torch.Size([2, 9])
            indices_latents_history_short=indices_latents_history_short,  # torch.Size([2, 2])
            indices_latents_history_mid=indices_latents_history_mid,  # torch.Size([2, 2])
            indices_latents_history_long=indices_latents_history_long,  # torch.Size([2, 16])
            latents_history_short=latents_history_short,  # torch.Size([2, 16, 2, 60, 104])
            latents_history_mid=latents_history_mid,  # torch.Size([2, 16, 2, 60, 104])
            latents_history_long=latents_history_long,  # torch.Size([2, 16, 16, 60, 104])
            action_id=action_id,
            action_vec=action_vec,
            camera_poses=camera_poses,
            camera_intrinsics=camera_intrinsics,
            # Event / Object bank tokens and ACHA's action class are produced
            # inside transformer.forward from this context; without it those
            # modules never enter the graph and receive no gradient.
            action_memory_context=action_memory_context,
            return_dict=False,
        )[0]

        # Compute regular loss.
        if isinstance(model_pred, list):
            loss_list = []
            for cur_model_pred, cur_target, cur_sigmas in zip(model_pred, target, sigmas):
                cur_weighting = compute_loss_weighting_for_sd3(
                    weighting_scheme=args.training_config.weighting_scheme, sigmas=cur_sigmas
                )
                loss = torch.mean(
                    (cur_weighting.float() * (cur_model_pred.float() - cur_target.float()) ** 2).reshape(
                        cur_target.shape[0], -1
                    ),
                    1,
                ).mean()
                loss_list.append(loss)
            loss = torch.stack(loss_list, dim=0).mean()
            del loss_list
        else:
            # these weighting schemes use a uniform timestep sampling
            # and instead post-weight the loss
            weighting = compute_loss_weighting_for_sd3(
                weighting_scheme=args.training_config.weighting_scheme, sigmas=sigmas
            )

            loss = torch.mean(
                (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                1,
            ).mean()

        # loss = loss * (batch_size / total_sample_count)
        assert loss.requires_grad, f"Loss should have gradient! Got {loss.requires_grad}"
        assert loss.grad_fn is not None, "Loss should have grad_fn!"
        accelerator.backward(loss)

        if args.training_config.use_error_recycling:
            if isinstance(model_pred, list):
                with torch.no_grad():
                    for cur_model_pred, cur_target, cur_timesteps, cur_noisy_model_input in zip(
                        model_pred, target, timesteps, noisy_model_input
                    ):
                        process_and_update_error_buffers(
                            args,
                            recycle_vars,
                            accelerator,
                            global_step,
                            noise_scheduler_copy,
                            cur_model_pred,
                            cur_target,
                            cur_timesteps,
                            cur_noisy_model_input,
                            use_clean_input,
                        )
            else:
                with torch.no_grad():
                    process_and_update_error_buffers(
                        args,
                        recycle_vars,
                        accelerator,
                        global_step,
                        noise_scheduler_copy,
                        model_pred,
                        target,
                        timesteps,
                        noisy_model_input,
                        use_clean_input,
                    )

    # Check if the gradient of each model parameter contains NaN/Inf.
    # Previously this logged one line per bad param (~400 lines per bad step).
    # Now: collect counts + a short sample of names, and flag the step so the
    # caller can skip optimizer.step() to avoid corrupting AdamW moments.
    nan_grad_names = []
    inf_grad_names = []
    for name, param in transformer.named_parameters():
        if param.grad is None:
            continue
        g = param.grad
        if torch.isnan(g).any():
            nan_grad_names.append(name)
        elif torch.isinf(g).any():
            inf_grad_names.append(name)
    has_bad_grad = bool(nan_grad_names or inf_grad_names)
    if has_bad_grad:
        _head_nan = ", ".join(nan_grad_names[:3])
        _head_inf = ", ".join(inf_grad_names[:3])
        logger.error(
            f"[grad-sanity] bad grads detected: "
            f"NaN count={len(nan_grad_names)} (first 3: [{_head_nan}]), "
            f"Inf count={len(inf_grad_names)} (first 3: [{_head_inf}]). "
            f"Caller should skip optimizer.step() and zero grads."
        )
        # Check loss / model_pred / target magnitudes to help root-cause
        try:
            _mp_str = (
                f"min={model_pred.float().min().item():.3e} "
                f"max={model_pred.float().max().item():.3e}"
                if not isinstance(model_pred, list) else "<list>"
            )
            _tg_str = (
                f"min={target.float().min().item():.3e} "
                f"max={target.float().max().item():.3e}"
                if not isinstance(target, list) else "<list>"
            )
            logger.error(
                f"[grad-sanity] loss={loss.detach().float().item():.3e}  "
                f"model_pred {_mp_str}  target {_tg_str}"
            )
        except Exception:
            pass

    grad_norm = None
    if accelerator.sync_gradients:
        params_to_clip = transformer.parameters()
        grad_norm = accelerator.clip_grad_norm_(params_to_clip, args.training_config.max_grad_norm)

    # Log every group's LR. When use_plucker enables the multi-group split,
    # the plucker_zero_init group lives at index >=2 and was previously
    # invisible — making it impossible to tell whether *plucker_lr_multiplier
    # had actually taken effect.
    #
    # Single-group case  -> {"lr": ...}                          (legacy key)
    # Multi-group case   -> {"lr0": ..., "lr1": ..., ...}        (compact keys
    #     so tqdm's postfix doesn't truncate them on narrow terminals; keys
    #     stay short and consistent across wandb / progress_bar / train.log).
    _group_lrs = lr_scheduler.get_last_lr()
    logs = {
        "loss": loss.detach().item(),
        "has_bad_grad": has_bad_grad,  # caller reads this to skip optimizer.step()
    }
    if len(_group_lrs) == 1:
        logs["lr"] = _group_lrs[0]
    else:
        for _i, _lr in enumerate(_group_lrs):
            logs[f"lr{_i}"] = _lr
    if grad_norm is not None:
        logs["grad_norm"] = grad_norm.item() if hasattr(grad_norm, "item") else grad_norm

    del noisy_model_input_list
    del sigmas_list
    del timesteps_list
    del targets_list
    del noisy_model_input
    del timesteps
    del prompt_embeds
    del prompt_attention_masks
    del indices_hidden_states
    del latents_history_short
    del indices_latents_history_short
    del latents_history_mid
    del indices_latents_history_mid
    del latents_history_long
    del indices_latents_history_long
    del model_pred
    del target
    del loss
    free_memory()

    return logs


# ======================================== easy anti-drifting ========================================


def downsample_corrupt(model_input, downsample_min_corrupt_ratio, downsample_max_corrupt_ratio):
    corrupt_ratio = random.uniform(downsample_min_corrupt_ratio, downsample_max_corrupt_ratio)

    is_5d = model_input.ndim == 5

    if is_5d:
        B, C, T, H, W = model_input.shape
        model_input = model_input.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    else:
        B, C, H, W = model_input.shape

    h0, w0 = model_input.shape[-2:]

    h1 = max(1, int(round(h0 * corrupt_ratio)))
    w1 = max(1, int(round(w0 * corrupt_ratio)))

    model_input = F.interpolate(model_input, size=(h1, w1), mode="bilinear", align_corners=False, antialias=True)

    model_input = F.interpolate(model_input, size=(h0, w0), mode="bilinear", align_corners=False, antialias=True)

    if is_5d:
        model_input = model_input.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)

    return model_input


def get_corrupt_noise_sigma(model_input, batch_size, corrupt_ratio=1 / 3, num_frames=None, is_frame_independent=False):
    if is_frame_independent:
        noise_sigma_shape = (batch_size, 1, num_frames)
    else:
        noise_sigma_shape = (batch_size,)
    noise_sigma = (
        torch.rand(size=noise_sigma_shape, device=model_input.device, dtype=model_input.dtype) * corrupt_ratio
    )
    while len(noise_sigma.shape) < model_input.ndim:
        noise_sigma = noise_sigma.unsqueeze(-1)
    return noise_sigma


def corrupt_model_input(
    model_input,
    # choose mode
    corrupt_mode="noise",  # "noise" | "downsample" | "random"
    noise_mode_prob=0.9,  # when corrupt_mode="random", select the probability of noise (select downsample for the remaining probability).
    # for noise
    is_frame_independent=False,
    is_chunk_independent=False,
    noise_corrupt_ratio=1 / 3,
    noise_corrupt_clean_prob=0.1,
    # for downsample
    downsample_min_corrupt_ratio=0.9,
    downsample_max_corrupt_ratio=1.0,
):
    assert not (is_frame_independent and is_chunk_independent), (
        "is_frame_independent and is_chunk_independent cannot both be True"
    )
    assert corrupt_mode in ("noise", "downsample", "random"), (
        f"corrupt_mode must be 'noise', 'downsample', or 'random', got '{corrupt_mode}'"
    )

    # ==================== choose mode ====================
    if corrupt_mode == "random":
        mode = "noise" if random.random() < noise_mode_prob else "downsample"
    else:
        mode = corrupt_mode

    # ==================== downsample branch ====================
    if mode == "downsample":
        model_input = downsample_corrupt(
            model_input=model_input,
            downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
            downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
        )
        return model_input

    # ==================== noise branch ====================
    clean_random = random.random()
    if clean_random < noise_corrupt_clean_prob:
        return model_input

    noise_sigma = get_corrupt_noise_sigma(
        model_input=model_input,
        batch_size=model_input.shape[0],
        corrupt_ratio=noise_corrupt_ratio,
        num_frames=model_input.shape[2],
        is_frame_independent=is_frame_independent,
    )

    model_input = noise_sigma * torch.randn_like(model_input) + (1 - noise_sigma) * model_input

    return model_input


def corrupt_history_latents(
    latents_history_short,
    latents_history_mid,
    latents_history_long,
    latent_window_size,
    is_keep_x0=True,
    # choose mode
    corrupt_mode="noise",  # "noise" | "downsample" | "random"
    noise_mode_prob=0.9,  # when corrupt_mode="random", select the probability of noise (select downsample for the remaining probability).
    # for noise
    is_frame_independent=False,
    is_chunk_independent=False,
    corrupt_ratio_1x=1 / 3,
    corrupt_ratio_2x=1 / 3,
    corrupt_ratio_4x=1 / 3,
    noise_corrupt_clean_prob=0.1,
    # for downsample
    downsample_min_corrupt_ratio=0.9,
    downsample_max_corrupt_ratio=1.0,
):
    assert not (is_frame_independent and is_chunk_independent), (
        "is_frame_independent and is_chunk_independent cannot both be True"
    )
    assert corrupt_mode in ("noise", "downsample", "random"), (
        f"corrupt_mode must be 'noise', 'downsample', or 'random', got '{corrupt_mode}'"
    )

    clean_random = random.random()
    if clean_random < noise_corrupt_clean_prob:
        return latents_history_short, latents_history_mid, latents_history_long
    
    # ==================== choose mode ====================
    if corrupt_mode == "random":
        mode = "noise" if random.random() < noise_mode_prob else "downsample"
    else:
        mode = corrupt_mode

    # ==================== noise branch ====================
    if mode == "noise":
        batch_size = latents_history_short.shape[0]
        if not is_frame_independent and not is_chunk_independent:
            noise_sigma = get_corrupt_noise_sigma(
                model_input=latents_history_short, batch_size=batch_size, corrupt_ratio=corrupt_ratio_1x
            )

    len_4x = latents_history_long.shape[2]
    len_2x = latents_history_mid.shape[2]
    len_1x = latents_history_short.shape[2]

    hist_seq_len = len_4x + len_2x + len_1x
    hist_seq_len_copy = hist_seq_len

    ori_len_1x = len_1x
    if is_keep_x0:
        len_1x -= 1
        hist_seq_len -= 1
        begin_num = 1
    else:
        begin_num = 0

    max_windows = hist_seq_len // latent_window_size
    tail_num = hist_seq_len % latent_window_size

    assert hist_seq_len_copy == tail_num + max_windows * latent_window_size + begin_num

    tail_latents_history = None
    begin_latents_history = None

    if tail_num != 0:
        tail_latents_history = latents_history_long[:, :, :tail_num, :, :]
        latents_history_long = latents_history_long[:, :, tail_num:, :, :]
        if tail_latents_history.sum() != 0:
            if mode == "downsample":
                tail_latents_history = downsample_corrupt(
                    model_input=tail_latents_history,
                    downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                    downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                )
            else:
                noise_sigma = get_corrupt_noise_sigma(
                    model_input=latents_history_short,
                    batch_size=batch_size,
                    corrupt_ratio=corrupt_ratio_4x,
                    num_frames=tail_latents_history.shape[2],
                    is_frame_independent=is_frame_independent,
                )
                tail_latents_history = (
                    noise_sigma * torch.randn_like(tail_latents_history) + (1 - noise_sigma) * tail_latents_history
                )

    if begin_num != 0:
        begin_latents_history = latents_history_short[:, :, :begin_num, :, :]
        latents_history_short = latents_history_short[:, :, begin_num:, :, :]
        if begin_latents_history.sum() != 0:
            if mode == "downsample":
                begin_latents_history = downsample_corrupt(
                    model_input=begin_latents_history,
                    downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                    downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                )
            else:
                noise_sigma = get_corrupt_noise_sigma(
                    model_input=latents_history_short,
                    batch_size=batch_size,
                    corrupt_ratio=corrupt_ratio_1x,
                    num_frames=begin_latents_history.shape[2],
                    is_frame_independent=is_frame_independent,
                )
                begin_latents_history = (
                    noise_sigma * torch.randn_like(begin_latents_history) + (1 - noise_sigma) * begin_latents_history
                )

    mid_latents_history = torch.cat([latents_history_long, latents_history_mid, latents_history_short], dim=2)
    window_num = mid_latents_history.shape[2] // latent_window_size
    assert mid_latents_history.shape[2] % latent_window_size == 0, (
        f"mid length {mid_latents_history.shape[2]} not divisible by window size {latent_window_size}"
    )

    seq_begin = 0
    for idx in range(window_num):
        seq_end = seq_begin + latent_window_size
        if mid_latents_history[:, :, seq_begin:seq_end, :, :].sum() != 0:
            if idx == window_num - 1:
                len_2x_end = seq_begin + len_2x
                if mode == "downsample":
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = downsample_corrupt(
                        model_input=mid_latents_history[:, :, seq_begin:len_2x_end, :, :],
                        downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                        downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                    )
                else:
                    noise_sigma_4x = get_corrupt_noise_sigma(
                        model_input=latents_history_short,
                        batch_size=batch_size,
                        corrupt_ratio=corrupt_ratio_4x,
                        num_frames=len_2x,
                        is_frame_independent=is_frame_independent,
                    )
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = (
                        noise_sigma_4x * torch.randn_like(mid_latents_history[:, :, seq_begin:len_2x_end, :, :])
                        + (1 - noise_sigma_4x) * mid_latents_history[:, :, seq_begin:len_2x_end, :, :]
                    )

                remaining_frames = seq_end - len_2x_end
                if mode == "downsample":
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = downsample_corrupt(
                        model_input=mid_latents_history[:, :, len_2x_end:seq_end, :, :],
                        downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                        downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                    )
                else:
                    noise_sigma_2x = get_corrupt_noise_sigma(
                        model_input=latents_history_short,
                        batch_size=batch_size,
                        corrupt_ratio=corrupt_ratio_2x,
                        num_frames=remaining_frames,
                        is_frame_independent=is_frame_independent,
                    )
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = (
                        noise_sigma_2x * torch.randn_like(mid_latents_history[:, :, len_2x_end:seq_end, :, :])
                        + (1 - noise_sigma_2x) * mid_latents_history[:, :, len_2x_end:seq_end, :, :]
                    )
            else:
                if mode == "downsample":
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = downsample_corrupt(
                        model_input=mid_latents_history[:, :, seq_begin:seq_end, :, :],
                        downsample_min_corrupt_ratio=downsample_min_corrupt_ratio,
                        downsample_max_corrupt_ratio=downsample_max_corrupt_ratio,
                    )
                else:
                    noise_sigma = get_corrupt_noise_sigma(
                        model_input=latents_history_short,
                        batch_size=batch_size,
                        corrupt_ratio=corrupt_ratio_4x,
                        num_frames=latent_window_size,
                        is_frame_independent=is_frame_independent,
                    )
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = (
                        noise_sigma * torch.randn_like(mid_latents_history[:, :, seq_begin:seq_end, :, :])
                        + (1 - noise_sigma) * mid_latents_history[:, :, seq_begin:seq_end, :, :]
                    )
        seq_begin = seq_end

    recovers = []
    if tail_latents_history is not None:
        recovers.append(tail_latents_history)
    recovers.append(mid_latents_history[:, :, :-len_1x, :, :])
    if begin_latents_history is not None:
        recovers.append(begin_latents_history)
    recovers.append(mid_latents_history[:, :, -len_1x:, :, :])
    mid_latents_history = torch.cat(recovers, dim=2)

    # Split and update back to original tensors
    latents_4x_recovered, latents_2x_recovered, latents_history_short_recovered = mid_latents_history.split(
        [len_4x, len_2x, ori_len_1x], dim=2
    )

    return (
        latents_history_short_recovered,
        latents_2x_recovered,
        latents_4x_recovered,
    )


def add_saturation_to_history_latents(
    latents_history_short,
    latents_history_mid,
    latents_history_long,
    latent_window_size,
    is_keep_x0=False,
    saturation_ratio_min=0.7,
    saturation_ratio_max=2.0,
    saturation_clean_prob=0.2,
):
    # clean_random = random.random()
    # if clean_random < saturation_clean_prob:
    #     return latents_history_short, latents_history_mid, latents_history_long

    def get_saturation(x1, saturation_ratio_min, saturation_ratio_max):
        if random.random() < 0.5:
            sat_factor = random.uniform(saturation_ratio_min, 1.0 - 1e-3)
        else:
            sat_factor = random.uniform(1.0 + 1e-3, saturation_ratio_max)
        latent_mean = torch.mean(x1, dim=1, keepdim=True)
        x1_saturated = (x1 - latent_mean) * sat_factor + latent_mean
        return x1_saturated

    len_4x = latents_history_long.shape[2]
    len_2x = latents_history_mid.shape[2]
    len_1x = latents_history_short.shape[2]

    hist_seq_len = len_4x + len_2x + len_1x
    hist_seq_len_copy = hist_seq_len

    ori_len_1x = len_1x
    if is_keep_x0:
        len_1x -= 1
        hist_seq_len -= 1
        begin_num = 1
    else:
        begin_num = 0

    max_windows = hist_seq_len // latent_window_size
    tail_num = hist_seq_len % latent_window_size

    assert hist_seq_len_copy == tail_num + max_windows * latent_window_size + begin_num

    tail_latents_history = None
    begin_latents_history = None

    if tail_num != 0:
        tail_latents_history = latents_history_long[:, :, :tail_num, :, :]
        latents_history_long = latents_history_long[:, :, tail_num:, :, :]
        if tail_latents_history.sum() != 0:
            if random.random() < saturation_clean_prob:
                tail_latents_history = tail_latents_history
            else:
                tail_latents_history = get_saturation(
                    tail_latents_history,
                    saturation_ratio_min=saturation_ratio_min,
                    saturation_ratio_max=saturation_ratio_max,
                )

    if begin_num != 0:
        begin_latents_history = latents_history_short[:, :, :begin_num, :, :]
        latents_history_short = latents_history_short[:, :, begin_num:, :, :]
        # if begin_latents_history.sum() != 0:
        #     begin_latents_history = get_saturation(
        #         begin_latents_history,
        #         saturation_ratio_min=saturation_ratio_min,
        #         saturation_ratio_max=saturation_ratio_max,
        #     )

    mid_latents_history = torch.cat([latents_history_long, latents_history_mid, latents_history_short], dim=2)
    window_num = mid_latents_history.shape[2] // latent_window_size
    assert mid_latents_history.shape[2] % latent_window_size == 0, (
        f"mid length {mid_latents_history.shape[2]} not divisible by window size {latent_window_size}"
    )

    seq_begin = 0
    for idx in range(window_num):
        seq_end = seq_begin + latent_window_size
        if mid_latents_history[:, :, seq_begin:seq_end, :, :].sum() != 0:
            if idx == window_num - 1:
                len_2x_end = seq_begin + len_2x
                if random.random() < saturation_clean_prob:
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = mid_latents_history[
                        :, :, seq_begin:len_2x_end, :, :
                    ]
                else:
                    mid_latents_history[:, :, seq_begin:len_2x_end, :, :] = get_saturation(
                        mid_latents_history[:, :, seq_begin:len_2x_end, :, :],
                        saturation_ratio_min=saturation_ratio_min,
                        saturation_ratio_max=saturation_ratio_max,
                    )

                if random.random() < saturation_clean_prob:
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = mid_latents_history[
                        :, :, len_2x_end:seq_end, :, :
                    ]
                else:
                    mid_latents_history[:, :, len_2x_end:seq_end, :, :] = get_saturation(
                        mid_latents_history[:, :, len_2x_end:seq_end, :, :],
                        saturation_ratio_min=saturation_ratio_min,
                        saturation_ratio_max=saturation_ratio_max,
                    )
            else:
                if random.random() < saturation_clean_prob:
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = mid_latents_history[
                        :, :, seq_begin:seq_end, :, :
                    ]
                else:
                    mid_latents_history[:, :, seq_begin:seq_end, :, :] = get_saturation(
                        mid_latents_history[:, :, seq_begin:seq_end, :, :],
                        saturation_ratio_min=saturation_ratio_min,
                        saturation_ratio_max=saturation_ratio_max,
                    )

        seq_begin = seq_end

    recovers = []
    if tail_latents_history is not None:
        recovers.append(tail_latents_history)
    recovers.append(mid_latents_history[:, :, :-len_1x, :, :])
    if begin_latents_history is not None:
        recovers.append(begin_latents_history)
    recovers.append(mid_latents_history[:, :, -len_1x:, :, :])
    mid_latents_history = torch.cat(recovers, dim=2)

    # Split and update back to original tensors
    latents_4x_recovered, latents_2x_recovered, latents_history_short_recovered = mid_latents_history.split(
        [len_4x, len_2x, ori_len_1x], dim=2
    )

    return (
        latents_history_short_recovered,
        latents_2x_recovered,
        latents_4x_recovered,
    )


# ======================================== prepare stage1 training ========================================


def prepare_stage1_clean_input_from_latents(
    history_latents,  # VAE latents, (B, C_latent, F_latent, H_latent, W_latent)
    target_latents,
    x0_latents=None,
    latent_window_size: int = 9,
    history_sizes: list = [16, 2, 1],
    is_random_drop: bool = False,
    random_drop_i2v_ratio: float = 0,
    random_drop_v2v_ratio: float = 0,
    random_drop_t2v_ratio: float = 0,
    is_keep_x0: bool = True,
    dtype=torch.bfloat16,
    device="cpu",
    action_memory_context: dict | None = None,  # Dual-Memory Helios (branch idea1)
):
    if is_keep_x0:
        latents_prefix = x0_latents.to(device, dtype=dtype)
    else:
        assert x0_latents is None

    history_sizes = sorted(history_sizes, reverse=True)  # From big to small
    history_window_size = sum(history_sizes)
    total_window_size = history_window_size + latent_window_size
    assert total_window_size == history_latents.shape[2] + target_latents.shape[2], (
        f"total_window_size mismatch: expected {total_window_size}"
        f"(history={history_latents.shape[2]} + target={target_latents.shape[2]}), "
        f"but got {history_latents.shape[2] + target_latents.shape[2]}"
    )

    # ------------------------------------------------------------------
    # Dual-Memory Helios: optional Event-Aware Frame Re-assignment (EAFR).
    # When action_memory_context carries eafr_enabled=True together with
    # per-sample chunk annotations, we reorder the 19-frame history window
    # so that interaction-relevant frames land in the short bucket while
    # low-importance navigation frames are routed to the long bucket. The
    # original temporal position of every frame is preserved in `indices`
    # so that RoPE continues to use the real timestamps.
    # ------------------------------------------------------------------
    eafr_enabled = bool(action_memory_context is not None and action_memory_context.get("eafr_enabled", False))

    indices = (
        torch.arange(0, sum([1, *history_sizes, latent_window_size])).unsqueeze(0).expand(target_latents.shape[0], -1)
    )
    (
        indices_prefix,
        indices_latents_history_long,
        indices_latents_history_mid,
        indices_latents_history_1x,
        indices_hidden_states,
    ) = indices.split([1, *history_sizes, latent_window_size], dim=1)
    indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=1)

    if eafr_enabled:
        from helios.utils.action_utils import (
            action_class_to_idx,
            event_aware_split,
            event_aware_split_soft,
            event_aware_split_with_dedup,
            frame_to_chunk_map,
        )

        choice_idx_list = action_memory_context.get("choice_idx", [])
        chunk_has_interaction_list = action_memory_context.get("chunk_has_interaction", [])
        chunk_phase_list = action_memory_context.get("chunk_phase", [])
        has_semantic_labels_list = action_memory_context.get("has_semantic_labels", [])
        alpha = float(action_memory_context.get("eafr_alpha", 1.0))
        beta = float(action_memory_context.get("eafr_beta", 0.5))
        tau = float(action_memory_context.get("eafr_tau", 8.0))
        eafr_learnable = bool(action_memory_context.get("eafr_learnable", False))
        scorer = action_memory_context.get("eafr_scorer", None)
        action_class_list = action_memory_context.get("action_class", [])
        # Phase 1b: redundancy penalty. Independent of eafr_learnable; takes
        # precedence only when learnable is off (combining a learnable scorer
        # with greedy dedup is a future extension that needs careful gradient
        # routing).
        # eafr_dedup_metric: "latent" (always works in training; cosine on VAE
        #                              latents, free) | "clip" (semantics-aware,
        #                              requires precomputed clip_embeds in batch
        #                              -- otherwise we silently fall back to
        #                              "latent" so training never blocks).
        eafr_dedup_enabled = bool(action_memory_context.get("eafr_dedup_enabled", False))
        eafr_dedup_lambda = float(action_memory_context.get("eafr_dedup_lambda", 0.3))
        eafr_dedup_metric = str(action_memory_context.get("eafr_dedup_metric", "latent"))
        # Per-batch precomputed CLIP sim matrix (shape (B, T_hist, T_hist))
        # is the only way "clip" actually works at training time -- decoding
        # 19 latent frames per step is too expensive. Set by a future
        # tools/precompute_clip_embeds.py pass; absent for now -> falls back
        # to latent silently.
        eafr_clip_sim_batch = action_memory_context.get("clip_sim_per_batch", None)

        B = history_latents.shape[0]
        T_hist = history_latents.shape[2]
        reordered = []
        new_indices_long_rows = []
        new_indices_mid_rows = []
        new_indices_short_rows = []
        soft_weights_rows = []
        # indices order in this layout: [prefix(1), long(T_long), mid(T_mid), 1x(1), current(latent_window)]
        offset_long = 1
        offset_mid = offset_long + history_sizes[0]
        offset_short = offset_mid + history_sizes[1]
        for b in range(B):
            has_labels = (
                bool(has_semantic_labels_list[b])
                if b < len(has_semantic_labels_list)
                else False
            )
            if not has_labels:
                # Unlabeled sample (e.g. game data): keep the baseline chronological split.
                reordered.append(history_latents[b])
                new_indices_long_rows.append(indices_latents_history_long[b])
                new_indices_mid_rows.append(indices_latents_history_mid[b])
                new_indices_short_rows.append(indices_latents_history_1x[b])
                soft_weights_rows.append(None)
                continue

            choice_idx_b = int(choice_idx_list[b]) if b < len(choice_idx_list) else 0
            ftc = frame_to_chunk_map(choice_idx_b, T_hist, latent_window_size)
            hi = chunk_has_interaction_list[b] if b < len(chunk_has_interaction_list) else []
            ph = chunk_phase_list[b] if b < len(chunk_phase_list) else []
            try:
                if eafr_learnable and scorer is not None:
                    # Same action-class id mapping as event_memory.forward.
                    _ac_to_idx_local = action_class_to_idx

                    num_classes = getattr(scorer, "num_action_classes", 64)
                    cur_ac = _ac_to_idx_local(
                        action_class_list[b] if b < len(action_class_list) else None,
                        num_classes,
                    )
                    # Treat every history chunk as having the same action_class
                    # signature as the current chunk (the .pt carries video-level
                    # labels). This is the cleanest default; a future refinement
                    # could expose per-chunk action_class.
                    hist_ac = [cur_ac for _ in range(max(1, len(hi)))]
                    long_idx, mid_idx, short_idx, soft_w = event_aware_split_soft(
                        num_history_frames=T_hist,
                        history_sizes=history_sizes,
                        frame_to_chunk=ftc,
                        chunk_has_interaction=hi,
                        chunk_phase=ph,
                        chunk_action_class_idx=hist_ac,
                        current_action_class_idx=cur_ac,
                        scorer=scorer,
                        alpha=alpha,
                        beta=beta,
                        tau=tau,
                        device=history_latents.device,
                        dtype=torch.float32,
                    )
                elif eafr_dedup_enabled:
                    # Phase 1b: greedy fill with redundancy penalty.
                    # Resolve similarity backend: "clip" with cached sim_mat
                    # > "latent" cosine (cheap default).
                    _sim_b = None
                    if (
                        eafr_dedup_metric == "clip"
                        and eafr_clip_sim_batch is not None
                        and b < len(eafr_clip_sim_batch)
                        and eafr_clip_sim_batch[b] is not None
                    ):
                        _sim_b = eafr_clip_sim_batch[b].to(history_latents.device)
                    long_idx, mid_idx, short_idx = event_aware_split_with_dedup(
                        history_latents=history_latents[b, :, :T_hist],
                        sim_mat=_sim_b,
                        history_sizes=history_sizes,
                        frame_to_chunk=ftc,
                        chunk_has_interaction=hi,
                        chunk_phase=ph,
                        dedup_lambda=eafr_dedup_lambda,
                        alpha=alpha,
                        beta=beta,
                        tau=tau,
                    )
                    soft_w = None
                else:
                    long_idx, mid_idx, short_idx = event_aware_split(
                        num_history_frames=T_hist,
                        history_sizes=history_sizes,
                        frame_to_chunk=ftc,
                        chunk_has_interaction=hi,
                        chunk_phase=ph,
                        alpha=alpha,
                        beta=beta,
                        tau=tau,
                    )
                    soft_w = None
            except (AssertionError, IndexError):
                reordered.append(history_latents[b])
                new_indices_long_rows.append(indices_latents_history_long[b])
                new_indices_mid_rows.append(indices_latents_history_mid[b])
                new_indices_short_rows.append(indices_latents_history_1x[b])
                soft_weights_rows.append(None)
                continue

            perm = long_idx + mid_idx + short_idx
            frame_sel = history_latents[b, :, perm, :, :]

            if soft_w is not None:
                w = soft_w.index_select(0, torch.tensor(perm, device=soft_w.device, dtype=torch.long))
                # w: (T_hist,) -> broadcast over (C, T, H, W).
                frame_sel = frame_sel * w.view(1, -1, 1, 1).to(frame_sel.dtype).to(frame_sel.device)

            reordered.append(frame_sel)
            new_indices_long_rows.append(
                torch.tensor([offset_long + i for i in long_idx], device=indices.device)
            )
            new_indices_mid_rows.append(
                torch.tensor([offset_long + i for i in mid_idx], device=indices.device)
            )
            new_indices_short_rows.append(
                torch.tensor([offset_long + i for i in short_idx], device=indices.device)
            )
            soft_weights_rows.append(soft_w)

        history_latents = torch.stack(reordered, dim=0)
        indices_latents_history_long = torch.stack(new_indices_long_rows, dim=0)
        indices_latents_history_mid = torch.stack(new_indices_mid_rows, dim=0)
        indices_latents_history_1x = torch.stack(new_indices_short_rows, dim=0)
        indices_latents_history_short = torch.cat([indices_prefix, indices_latents_history_1x], dim=1)

        # One-shot probe: confirm EAFR actually re-ordered frames once
        # per process (not just gated on; the per-sample fallback above
        # may have skipped the re-order due to AssertionError/IndexError).
        if not _EAFR_RUNTIME_LOGGED.get("seen"):
            _orig_long = list(range(0, history_sizes[0]))
            _new_long = new_indices_long_rows[0].tolist() if new_indices_long_rows else []
            _new_long_norm = [i - offset_long for i in _new_long]
            _moved = sum(1 for a, b in zip(_orig_long, _new_long_norm) if a != b)
            print(
                f"[EAFR runtime] FIRED: history_sizes={list(history_sizes)}, "
                f"long_bucket_perm={_new_long_norm} "
                f"(moved {_moved}/{history_sizes[0]} frames vs time-only baseline), "
                f"dedup={eafr_dedup_enabled} learnable={eafr_learnable}"
            )
            _EAFR_RUNTIME_LOGGED["seen"] = True

    latents_history_long, latents_history_mid, latents_history_1x = history_latents.split(history_sizes, dim=2)

    if is_random_drop:
        if random_drop_t2v_ratio != 0 and torch.rand(1).item() <= random_drop_t2v_ratio:
                if is_keep_x0:
                    latents_prefix = torch.zeros_like(
                        latents_prefix, device=latents_history_1x.device, dtype=latents_history_1x.dtype
                    )
                latents_history_1x = torch.zeros_like(
                    latents_history_1x,
                    device=latents_history_1x.device,
                    dtype=latents_history_1x.dtype,
                )
                latents_history_mid = torch.zeros_like(
                    latents_history_mid,
                    device=latents_history_1x.device,
                    dtype=latents_history_1x.dtype,
                )
                latents_history_long = torch.zeros_like(
                    latents_history_long,
                    device=latents_history_1x.device,
                    dtype=latents_history_1x.dtype,
                )
        else:
            len_4x = latents_history_long.shape[2]
            len_2x = latents_history_mid.shape[2]
            len_1x = latents_history_1x.shape[2]
            hist_seq_len = len_4x + len_2x + len_1x

            total_drop = 0
            is_drop_triggered = False

            if random_drop_i2v_ratio != 0 and torch.rand(1).item() <= random_drop_i2v_ratio:
                total_drop = max(0, hist_seq_len - 1)
                is_drop_triggered = True
            elif random_drop_v2v_ratio != 0 and torch.rand(1).item() <= random_drop_v2v_ratio:
                max_windows = hist_seq_len // latent_window_size
                tail_num = hist_seq_len % latent_window_size
                total_drop = tail_num
                if max_windows > 0:
                    drop_windows = random.randint(0, max_windows)
                    total_drop += drop_windows * latent_window_size
                is_drop_triggered = True

            if is_drop_triggered and total_drop > 0:
                remaining_drop = total_drop
                if remaining_drop > 0 and len_4x > 0:
                    drop_4x = min(remaining_drop, len_4x)
                    latents_history_long[:, :, :drop_4x, :, :] = 0
                    remaining_drop -= drop_4x
                if remaining_drop > 0 and len_2x > 0:
                    drop_2x = min(remaining_drop, len_2x)
                    latents_history_mid[:, :, :drop_2x, :, :] = 0
                    remaining_drop -= drop_2x
                if remaining_drop > 0 and len_1x > 0:
                    drop_1x = min(remaining_drop, len_1x)
                    latents_history_1x[:, :, :drop_1x, :, :] = 0

    if is_keep_x0:
        latents_history_short = torch.cat([latents_prefix, latents_history_1x], dim=2)
    else:
        latents_history_short = latents_history_1x

    return (
        target_latents,
        indices_hidden_states,
        indices_latents_history_short,
        indices_latents_history_mid,
        indices_latents_history_long,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
    )


def prepare_stage1_noise_input(
    args,
    model_input,
    noise_scheduler,
    recycle_vars=None,
    latents_history_short=None,
    latents_history_mid=None,
    latents_history_long=None,
    latent_window_size=9,
    is_keep_x0=True,
    return_list=True,
):
    # Sample noise that we'll add to the latents
    noise = torch.randn_like(model_input)
    bsz = model_input.shape[0]

    use_clean_input = False
    noise_w_error = noise
    model_input_w_error = model_input

    # Sample a random timestep for each image
    # for weighting schemes where we sample timesteps non-uniformly
    u = compute_density_for_timestep_sampling(
        weighting_scheme=args.training_config.weighting_scheme,
        batch_size=bsz,
        logit_mean=args.training_config.logit_mean,
        logit_std=args.training_config.logit_std,
        mode_scale=args.training_config.mode_scale,
    )
    indices = (u * noise_scheduler.config.num_train_timesteps).long()

    noise_scheduler.temp_sigmas = noise_scheduler.sigmas
    noise_scheduler.temp_timesteps = noise_scheduler.timesteps
    if args.training_config.use_dynamic_shifting:
        noise_scheduler.temp_sigmas = apply_schedule_shift(
            noise_scheduler.sigmas,
            noise,
            base_seq_len=args.training_config.base_seq_len,
            max_seq_len=args.training_config.max_seq_len,
            base_shift=args.training_config.base_shift,
            max_shift=args.training_config.max_shift,
        )  # torch.Size([2, 1, 1, 1, 1])

        noise_scheduler.temp_timesteps = noise_scheduler.temp_sigmas * 1000.0  # rescale to [0, 1000.0)
        while noise_scheduler.temp_timesteps.ndim > 1:
            noise_scheduler.temp_timesteps = noise_scheduler.temp_timesteps.squeeze(-1)

    timesteps = noise_scheduler.temp_timesteps[indices].to(
        device=model_input.device, non_blocking=True
    )  # torch.Size([2]), torch.float32

    # Add noise according to flow matching.
    # zt = (1 - texp) * x + texp * z1
    sigmas = noise_scheduler.temp_sigmas[indices].flatten()
    while len(sigmas.shape) < model_input.ndim:
        sigmas = sigmas.unsqueeze(-1)

    sigmas = sigmas.to(model_input.device, dtype=model_input.dtype)

    if args.training_config.use_error_recycling:
        (
            model_input_w_error,
            noise_w_error,
            latents_history_long,
            latents_history_mid,
            latents_history_short,
            use_clean_input,
        ) = apply_error_injection(
            args,
            recycle_vars,
            model_input,
            noise,
            timesteps,
            latents_history_long,
            latents_history_mid,
            latents_history_short,
            model_input_w_error,
            noise_w_error,
            is_keep_x0,
            latent_window_size,
        )

    if args.training_config.corrupt_history and latents_history_short is not None:
        latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            latent_window_size,
            is_keep_x0=True,
            # choose mode
            corrupt_mode=args.training_config.corrupt_mode_history,
            noise_mode_prob=args.training_config.corrupt_mode_prob_history,
            # for noise
            is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
            is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
            corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
            corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
            corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
            noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,
            # for downsample
            downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
            downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
        )

    if args.training_config.corrupt_model_input:
        model_input_w_error = corrupt_model_input(
            model_input_w_error,
            # choose mode
            corrupt_mode=args.training_config.corrupt_mode_model_input,
            noise_mode_prob=args.training_config.corrupt_mode_prob_model_input,
            # for noise
            is_frame_independent=args.training_config.is_frame_independent_corrupt_model_input,
            is_chunk_independent=args.training_config.is_chunk_independent_corrupt_model_input,
            noise_corrupt_ratio=args.training_config.noise_corrupt_ratio_model_input,
            noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_model_input,
            # for downsample
            downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_model_input,
            downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_model_input,
        )

    # Get flow-matching target
    noisy_model_input = (1.0 - sigmas) * model_input_w_error + sigmas * noise_w_error
    target = noise_w_error - model_input

    noisy_model_input_list = [noisy_model_input] if return_list else noisy_model_input
    sigmas_list = [sigmas] if return_list else sigmas
    timesteps_list = [timesteps] if return_list else timesteps
    targets_list = [target] if return_list else target

    return (
        noisy_model_input_list,
        sigmas_list,
        timesteps_list,
        targets_list,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
        use_clean_input,
    )


# ======================================== prepare stage2 training ========================================


def prepare_stage2_clean_input(
    args,
    scheduler,
    latents,  # [b c t h w]
    pyramid_stage_num=3,
    stage2_sample_ratios=[1, 1, 1],
):
    assert pyramid_stage_num == len(stage2_sample_ratios)

    # Get clen pyramid latent list
    pyramid_latent_list = []
    pyramid_latent_list.append(latents)
    num_frames, height, width = latents.shape[-3], latents.shape[-2], latents.shape[-1]
    for _ in range(pyramid_stage_num - 1):
        height //= 2
        width //= 2
        latents = rearrange(latents, "b c t h w -> (b t) c h w")
        latents = torch.nn.functional.interpolate(latents, size=(height, width), mode="bilinear")
        latents = rearrange(latents, "(b t) c h w -> b c t h w", t=num_frames)
        pyramid_latent_list.append(latents)
    pyramid_latent_list = list(reversed(pyramid_latent_list))

    # Get pyramid noise list
    noise = torch.randn_like(pyramid_latent_list[-1])
    device = noise.device
    dtype = pyramid_latent_list[-1].dtype
    latent_frame_num = noise.shape[2]
    input_video_num = noise.shape[0]

    height, width = noise.shape[-2], noise.shape[-1]
    noise_list = [noise]
    cur_noise = noise
    for i_s in range(pyramid_stage_num - 1):
        height //= 2
        width //= 2
        cur_noise = rearrange(cur_noise, "b c t h w -> (b t) c h w")
        cur_noise = F.interpolate(cur_noise, size=(height, width), mode="bilinear") * 2
        cur_noise = rearrange(cur_noise, "(b t) c h w -> b c t h w", t=latent_frame_num)
        noise_list.append(cur_noise)
    noise_list = list(reversed(noise_list))  # make sure from low res to high res

    # Get pyramid target list
    # To calculate the batchsize
    bsz = input_video_num

    # from low resolution to high resolution
    noisy_latents_list = []
    sigmas_list = []
    targets_list = []
    timesteps_list = []
    training_steps = scheduler.config.num_train_timesteps
    for i_s, cur_sample_ratio in zip(range(pyramid_stage_num), stage2_sample_ratios):
        clean_latent = pyramid_latent_list[i_s]  # [bs, c, t, h, w]
        last_clean_latent = None if i_s == 0 else pyramid_latent_list[i_s - 1]
        start_sigma = scheduler.start_sigmas[i_s]
        end_sigma = scheduler.end_sigmas[i_s]

        if i_s == 0:
            start_point = noise_list[i_s]
        else:
            # Get the upsampled latent
            last_clean_latent = rearrange(last_clean_latent, "b c t h w -> (b t) c h w")
            last_clean_latent = F.interpolate(
                last_clean_latent,
                size=(
                    last_clean_latent.shape[-2] * 2,
                    last_clean_latent.shape[-1] * 2,
                ),
                mode="nearest",
            )
            last_clean_latent = rearrange(last_clean_latent, "(b t) c h w -> b c t h w", t=latent_frame_num)
            start_point = start_sigma * noise_list[i_s] + (1 - start_sigma) * last_clean_latent

        if i_s == pyramid_stage_num - 1:
            end_point = clean_latent
        else:
            end_point = end_sigma * noise_list[i_s] + (1 - end_sigma) * clean_latent

        for _ in range(cur_sample_ratio):
            # Sample a random timestep for each image
            # for weighting schemes where we sample timesteps non-uniformly
            u = compute_density_for_timestep_sampling(
                weighting_scheme=get_config_value(args, "weighting_scheme"),
                batch_size=bsz,
                logit_mean=get_config_value(args, "logit_mean"),
                logit_std=get_config_value(args, "logit_std"),
                mode_scale=get_config_value(args, "mode_scale"),
            )
            indices = (u * training_steps).long()  # Totally 1000 training steps per stage
            indices = indices.clamp(0, training_steps - 1)
            timesteps = scheduler.timesteps_per_stage[i_s][indices].to(device=device)

            # Add noise according to flow matching.
            # zt = (1 - texp) * x + texp * z1
            sigmas = scheduler.sigmas_per_stage[i_s][indices].to(device=device)
            while len(sigmas.shape) < start_point.ndim:
                sigmas = sigmas.unsqueeze(-1)

            if get_config_value(args, "use_dynamic_shifting"):
                temp_sigmas = apply_schedule_shift(
                    sigmas,
                    start_point,
                    base_seq_len=get_config_value(args, "base_seq_len"),
                    max_seq_len=get_config_value(args, "max_seq_len"),
                    base_shift=get_config_value(args, "base_shift"),
                    max_shift=get_config_value(args, "max_shift"),
                )  # torch.Size([2, 1, 1, 1, 1])
                temp_timesteps = scheduler.timesteps_per_stage[i_s].min() + temp_sigmas * (
                    scheduler.timesteps_per_stage[i_s].max() - scheduler.timesteps_per_stage[i_s].min()
                )
                while temp_timesteps.ndim > 1:
                    temp_timesteps = temp_timesteps.squeeze(-1)

                sigmas = temp_sigmas
                timesteps = temp_timesteps

            if args.training_config.corrupt_model_input:
                end_point = corrupt_model_input(
                    end_point,
                    # choose mode
                    corrupt_mode=args.training_config.corrupt_mode_model_input,
                    noise_mode_prob=args.training_config.corrupt_mode_prob_model_input,
                    # for noise
                    is_frame_independent=args.training_config.is_frame_independent_corrupt_model_input,
                    is_chunk_independent=args.training_config.is_chunk_independent_corrupt_model_input,
                    noise_corrupt_ratio=args.training_config.noise_corrupt_ratio_model_input,
                    noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_model_input,
                    # for downsample
                    downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_model_input,
                    downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_model_input,
                )

            noisy_latents = sigmas * start_point + (1 - sigmas) * end_point

            # [stage1_latent, stage2_latent, ..., stagen_latent]
            noisy_latents_list.append(noisy_latents.to(dtype))
            sigmas_list.append(sigmas.to(dtype))
            timesteps_list.append(timesteps)
            targets_list.append(start_point - end_point)  # The standard rectified flow matching objective

    return noisy_latents_list, sigmas_list, timesteps_list, targets_list


def prepare_stage2_noise_input(
    args,
    scheduler,
    latents,  # [b c t h w]
    pyramid_stage_num=3,
    stage2_sample_ratios=[1, 1, 1],
    latents_history_short=None,
    latents_history_mid=None,
    latents_history_long=None,
    latent_window_size=9,
    return_list=True,
    is_navit_pyramid=False,
    is_efficient_sample=False,
):
    noisy_model_input_list, sigmas_list, timesteps_list, targets_list = prepare_stage2_clean_input(
        args=args,
        scheduler=scheduler,
        latents=latents,
        pyramid_stage_num=pyramid_stage_num,
        stage2_sample_ratios=stage2_sample_ratios,
    )

    if args.training_config.corrupt_history and latents_history_short is not None:
        latents_history_short, latents_history_mid, latents_history_long = corrupt_history_latents(
            latents_history_short,
            latents_history_mid,
            latents_history_long,
            latent_window_size,
            is_keep_x0=True,
            # choose mode
            corrupt_mode=args.training_config.corrupt_mode_history,
            noise_mode_prob=args.training_config.corrupt_mode_prob_history,
            # for noise
            is_frame_independent=args.training_config.is_frame_independent_corrupt_history,
            is_chunk_independent=args.training_config.is_chunk_independent_corrupt_history,
            corrupt_ratio_1x=args.training_config.noise_corrupt_ratio_history_short,
            corrupt_ratio_2x=args.training_config.noise_corrupt_ratio_history_mid,
            corrupt_ratio_4x=args.training_config.noise_corrupt_ratio_history_long,
            noise_corrupt_clean_prob=args.training_config.noise_corrupt_clean_prob_history,
            # for downsample
            downsample_min_corrupt_ratio=args.training_config.downsample_min_corrupt_ratio_history,
            downsample_max_corrupt_ratio=args.training_config.downsample_max_corrupt_ratio_history,
        )

    if is_navit_pyramid:
        return (
            [noisy_model_input_list],
            [sigmas_list],
            [timesteps_list],
            [targets_list],
            latents_history_short,
            latents_history_mid,
            latents_history_long,
        )

    if is_efficient_sample:
        temp_list = list(range(len(noisy_model_input_list)))
        random_index = random.choice(temp_list)

        noisy_model_input = noisy_model_input_list[random_index]
        sigmas = sigmas_list[random_index]
        timesteps = timesteps_list[random_index]
        targets = targets_list[random_index]

        base_results = (noisy_model_input, sigmas, timesteps, targets)
        additional_results = (latents_history_short, latents_history_mid, latents_history_long)

        if return_list:
            return tuple([item] for item in base_results) + additional_results
        else:
            return base_results + additional_results

    return (
        noisy_model_input_list,
        sigmas_list,
        timesteps_list,
        targets_list,
        latents_history_short,
        latents_history_mid,
        latents_history_long,
    )
