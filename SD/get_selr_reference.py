from contextlib import nullcontext
# import gradio as gr
import numpy as np
import torch
import os
import PIL
from einops import rearrange
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from omegaconf import OmegaConf
from PIL import Image
from torch import autocast
from torchvision import transforms

# from scripts.image_variations import load_model_from_config
from ldm.models.diffusion.ddpm import SimpleUpscaleDiffusion
from ldm.util import instantiate_from_config

def make_unc(model, n_samples, all_conds):
    uc_tmp = model.get_unconditional_conditioning(n_samples, [""])
    uc = dict()
    for k in all_conds:
        if k == "c_crossattn":
            assert isinstance(all_conds[k], list) and len(all_conds[k]) == 1
            uc[k] = [uc_tmp]
        elif k == "c_adm":  # todo: only run with text-based guidance?
            assert isinstance(all_conds[k], torch.Tensor)
            uc[k] = torch.ones_like(all_conds[k]) * model.low_scale_model.max_noise_level
        elif isinstance(all_conds[k], list):
            uc[k] = [all_conds[k][i] for i in range(len(all_conds[k]))]
        else:
            uc[k] = all_conds[k]
    return uc


@torch.no_grad()
def sample_model(model, sampler, prompt, input_im, precision, use_ema, h, w, ddim_steps, n_samples, scale, ddim_eta):

    precision_scope = autocast if precision=="autocast" else nullcontext
    ema = model.ema_scope if use_ema else nullcontext
    with precision_scope("cuda"):
        with ema():
            c = model.get_learned_conditioning(n_samples * [prompt])
            shape = [4, h // 8, w // 8]
            x_low = input_im.tile(n_samples,1,1,1)
            x_low = x_low.to(memory_format=torch.contiguous_format).half()
            if isinstance(model, SimpleUpscaleDiffusion):
                zx = model.get_first_stage_encoding(model.encode_first_stage(x_low))
                all_conds = {"c_concat": [zx], "c_crossattn": [c]}
            else:
                zx = model.low_scale_model.model.encode(x_low).sample()
                zx = zx * model.low_scale_model.scale_factor
                noise_level = torch.tensor([0]).tile(n_samples).to(input_im.device)
                all_conds = {"c_concat": [zx], "c_crossattn": [c], "c_adm": noise_level}

            uc = None
            if scale != 1.0:
                uc = make_unc(model, n_samples, all_conds)

            samples_ddim, _ = sampler.sample(S=ddim_steps,
                                            conditioning=all_conds,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            )

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            return torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

def load_img(path):
    image = Image.open(path).convert("RGB")
    return image

def load_model_from_config(config, ckpt, device, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location=device)
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model

def main_begin(
    input_im,
    output_path,
    output_name,
    target_res,
    pre_size,
    prompt,
    scale,
    seed,
    plms=True,
    ddim_steps=50,
    n_samples=1,
    ddim_eta=1.0,
    precision="autocast",
    ):

    # Using the pruned ckpt so ema weights are moved to the normal weights
    use_ema=False

    torch.manual_seed(seed)

    input_im = transforms.ToTensor()(input_im).unsqueeze(0).to(device)
    if pre_size is not None:
        input_im = transforms.Resize((pre_size, pre_size))(input_im)
    input_im = transforms.Resize((target_res, target_res))(input_im)
    input_im = input_im*2-1

    h = w = target_res

    if plms:
        sampler = PLMSSampler(model)
        ddim_eta = 0.0
    else:
        sampler = DDIMSampler(model)

    x_samples_ddim = sample_model(
        model=model,
        sampler=sampler,
        prompt=prompt,
        input_im=input_im,
        precision=precision,
        use_ema=use_ema,
        h=h, w=w,
        ddim_steps = ddim_steps,
        n_samples=n_samples,
        scale=scale,
        ddim_eta=ddim_eta,
    )
    x_sample = 255. * rearrange(x_samples_ddim[0].cpu().numpy(), 'c h w -> h w c')

    return Image.fromarray(x_sample.astype(np.uint8)).save(output_path + output_name)

device_idx = 0
device = f"cuda:{device_idx}"


if __name__ == '__main__':
    from huggingface_hub import hf_hub_download

    # load model
    config = '/models/sd_sr_config.yaml'
    ckpt = '/models/sd_sr_pretrain_model.ckpt'
    config = OmegaConf.load(config)
    model = load_model_from_config(config, ckpt, device=device)

    # Load decoder
    decoder_path = hf_hub_download(repo_id="stabilityai/sd-vae-ft-mse-original", filename="vae-ft-mse-840000-ema-pruned.ckpt")
    decoder = torch.load(decoder_path, map_location='cpu')["state_dict"]
    model.first_stage_model.load_state_dict(decoder, strict=False)
    model.half()

    # prompt
    default_prompt = "high quality high resolution uhd 4k image"

    # load data
    data_path = ''
    save_path = ''
    datanames = os.listdir(data_path)
    data_list = []
    for i in datanames:
        data_list.append(i)
    data_list.sort()

    for i in range(0, len(data_list)):
        input_im = load_img(data_path + data_list[i])
        # process
        main_begin(input_im=input_im, output_path = save_path, output_name = data_list[i], target_res=512, pre_size=256, prompt=default_prompt,scale=1,seed=0)

