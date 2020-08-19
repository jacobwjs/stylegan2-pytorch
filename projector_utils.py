import argparse
import math
import os

import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm.auto import tqdm

# import lpips
from model import Generator



def noise_regularize(noises):
    loss = 0

    for noise in noises:
        size = noise.shape[2]

        while True:
            loss = (
                loss
                + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2)
                + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            )

            if size <= 8:
                break

            noise = noise.reshape([-1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([3, 5])
            size //= 2

    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()

        noise.data.add_(-mean).div_(std)


def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp


def latent_noise(latent, strength):
    noise = torch.randn_like(latent) * strength

    return latent + noise


def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )




def run_projections(args, g_ema, percept, device='cuda:0',
                    save_modulo=None):
    loss_prev_step = 1000.0

    # Transform for generated images, which is needed for
    # perceptual model.
    #
    resize = min(args.size, 256)
    transform = transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(resize),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )

    imgs_real = []
    for imgfile in args.files:
        img = transform(Image.open(imgfile).convert("RGB"))
        imgs_real.append(img)

    imgs_real = torch.stack(imgs_real, 0).to(device)

    n_mean_latent = 10000
    with torch.no_grad():
        noise_sample = torch.randn(n_mean_latent, 512, device=device)
        latent_out = g_ema.style(noise_sample)

        latent_mean = latent_out.mean(0)
        latent_std = ((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5

    batch_size = imgs_real.shape[0]

    noises_single = g_ema.make_noise()
    noises = []
    for noise in noises_single:
        noises.append(noise.repeat(batch_size, 1, 1, 1).normal_())

    if args.w_plus:
        #     latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
        latent_in = latent_mean.detach().clone().view(1, 1, -1).repeat(batch_size, g_ema.n_latent, 1) # w_plus
        print("\t...using w_plus in projection")
        print("\t", latent_in.shape)
    else:
        latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(batch_size, 1)
        print("\t...using w in projection")
        print("\t", latent_in.shape)

    latent_in.requires_grad = True

    for noise in noises:
        noise.requires_grad = True

    optimizer = optim.Adam([latent_in] + noises, lr=args.lr)

    # pbar = tqdm(range(args.step))
    pbar = tqdm(range(args.step), ncols=1000)

    latent_path = []

    for i in pbar:
        t = i / args.step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2
        latent_n = latent_noise(latent_in, noise_strength.item())

        imgs_gen, _ = g_ema([latent_n], input_is_latent=True, noise=noises)

        batch, channel, height, width = imgs_gen.shape

        if height > 256:
            factor = height // 256

            imgs_gen = imgs_gen.reshape(
                batch, channel, height // factor, factor, width // factor, factor
            )
            imgs_gen = imgs_gen.mean([3, 5])

        p_loss = percept(imgs_gen, imgs_real).sum()
        n_loss = noise_regularize(noises)
        mse_loss = F.mse_loss(imgs_gen, imgs_real)

        loss = p_loss + args.noise_regularize * n_loss + args.mse * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        noise_normalize_(noises)

        if save_modulo is not None:
            # if (i + 1) % 100 == 0:
            if (i + 1) % (args.step / save_modulo) == 0:
                print("step = %d, saving latent_in" % (i))
                latent_path.append(latent_in.detach().clone())


        pbar.set_description(
            (
                f"perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};"
                f" mse: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )


    result_dict = {}
    for i, input_path in enumerate(args.files):
        noise_single = []
        for noise in noises:
            noise_single.append(noise[i : i + 1].detach())

        file_name = os.path.split(input_path)[-1]
        result_dict[file_name] = {
            "img": imgs_gen[i],
            "latent": latent_in[i].unsqueeze(0), # Add in batch dimension
            "noise": noise_single,
        }

    # Return the latents learned during optimization if true, otherwise return
    # the final result.
    #
    if save_modulo is not None:
        return result_dict, latent_path
    else:
        return result_dict
