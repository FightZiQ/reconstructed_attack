import torch
from pytorch_wavelets import DTCWTForward


def cw_ssim(img_batch, ref_batch, scales=5, skip_scales=None, K=1e-6, reduction="mean"):
    """Batched complex wavelet structural similarity.

    As in Zhou Wang and Eero P. Simoncelli, "TRANSLATION INSENSITIVE IMAGE SIMILARITY IN COMPLEX WAVELET DOMAIN"
    Ok, not quite, this implementation computes no local SSIM and neither averaging over local patches and uses only
    the existing wavelet structure to provide a similar scale-invariant decomposition.

    skip_scales can be a list like [True, False, False, False] marking levels to be skipped.
    K is a small fudge factor.
    """

    '''try:
        from pytorch_wavelets import DTCWTForward
    except ModuleNotFoundError:
        warnings.warn(
            "To utilize wavelet SSIM, install pytorch wavelets from https://github.com/fbcotter/pytorch_wavelets."
        )
        return torch.as_tensor(float("NaN")), torch.as_tensor(float("NaN"))'''

    # 1) Compute wavelets:
    setup = dict(device=img_batch.device, dtype=img_batch.dtype)
    if skip_scales is not None:
        include_scale = [~s for s in skip_scales]
        total_scales = scales - sum(skip_scales)
    else:
        include_scale = True
        total_scales = scales
    xfm = DTCWTForward(J=scales, biort="near_sym_b", qshift="qshift_b", include_scale=include_scale).to(**setup)
    img_coefficients = xfm(img_batch)
    ref_coefficients = xfm(ref_batch)

    # 2) Multiscale complex SSIM:
    ssim = 0
    for xs, ys in zip(img_coefficients[1], ref_coefficients[1]):
        if len(xs) > 0:
            xc = torch.view_as_complex(xs)
            yc = torch.view_as_complex(ys)

            conj_product = (xc * yc.conj()).sum(dim=2).abs()
            square_img = (xc * xc.conj()).abs().sum(dim=2)
            square_ref = (yc * yc.conj()).abs().sum(dim=2)

            ssim_val = (2 * conj_product + K) / (square_img + square_ref + K)
            ssim += ssim_val.mean(dim=[1, 2, 3])
    ssim = ssim / total_scales
    return ssim.mean().item(), ssim.max().item()


def psnr_compute(img_batch, ref_batch, batched=False, factor=1.0, clip=False):
    """Standard PSNR."""
    if clip:
        img_batch = torch.clamp(img_batch, 0, 1)

    if batched:
        mse = ((img_batch.detach() - ref_batch) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(factor ** 2 / mse)
        elif not torch.isfinite(mse):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
    else:
        B = img_batch.shape[0]
        mse_per_example = ((img_batch.detach() - ref_batch) ** 2).view(B, -1).mean(dim=1)
        if any(mse_per_example == 0):
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
        elif not all(torch.isfinite(mse_per_example)):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            psnr_per_example = 10 * torch.log10(factor ** 2 / mse_per_example)
            return psnr_per_example.mean().item(), psnr_per_example.max().item()


def total_variation(batch_figures, beta=2, device='cuda'):
    batch_size, channel, width, height = batch_figures.shape

    h_diff = torch.pow(batch_figures[:, :, 1:, :] - batch_figures[:, :, :-1, :], 2)
    h_pad = torch.zeros(batch_size, channel, 1, height).to(device)
    # pad h_diff to size -> batch_size * channel * width * height
    h_diff = torch.cat([h_diff, h_pad], dim=2)

    w_diff = torch.pow(batch_figures[:, :, :, 1:] - batch_figures[:, :, :, :-1], 2)
    w_pad = torch.zeros(batch_size, channel, width, 1).to(device)
    w_diff = torch.cat([w_diff, w_pad], dim=3)

    return torch.sum(torch.pow(h_diff + w_diff, beta/2))


def mse_compute(img_batch, ref_batch, batched=False, factor=1.0, clip=False):
    if clip:
        img_batch = torch.clamp(img_batch, 0, 1)

    if batched:
        mse = ((img_batch.detach() - ref_batch) ** 2).mean()
        if mse > 0 and torch.isfinite(mse):
            return 10 * torch.log10(factor ** 2 / mse)
        elif not torch.isfinite(mse):
            return [torch.tensor(float("nan"), device=img_batch.device)] * 2
        else:
            return [torch.tensor(float("inf"), device=img_batch.device)] * 2
    else:
        B = img_batch.shape[0]
        mse_per_example = ((img_batch.detach() - ref_batch) ** 2).view(B, -1).mean(dim=1)
        return mse_per_example.mean()


def tv_element_wise(batch_figures, beta=2, device='cuda'):
    """tv metrics, but computed in each channel"""
    batch_size, channel, width, height = batch_figures.shape

    h_diff = torch.pow(batch_figures[:, :, 1:, :] - batch_figures[:, :, :-1, :], 2)
    h_pad = torch.zeros(batch_size, channel, 1, height).to(device)
    # pad h_diff to size -> batch_size * channel * width * height
    h_diff = torch.cat([h_diff, h_pad], dim=2)

    w_diff = torch.pow(batch_figures[:, :, :, 1:] - batch_figures[:, :, :, :-1], 2)
    w_pad = torch.zeros(batch_size, channel, width, 1).to(device)
    w_diff = torch.cat([w_diff, w_pad], dim=3)

    return torch.sum(torch.pow(h_diff + w_diff, beta / 2).reshape(batch_size, channel, -1), dim=2)
