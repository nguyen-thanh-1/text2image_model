# conditional_ddpm.py
import os
import math
import argparse
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, utils
from tqdm import tqdm
import random

# optional CLIP text encoder via transformers
try:
    from transformers import CLIPTokenizer, CLIPTextModel
    HAS_TRANSFORMERS = True
except Exception:
    HAS_TRANSFORMERS = False

# ---------------------------
# Utilities / Beta schedules
# ---------------------------
class BetaSchedule:
    @staticmethod
    def linear(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
        return torch.linspace(beta_start, beta_end, timesteps)

    @staticmethod
    def cosine(timesteps: int, s: float = 0.008) -> torch.Tensor:
        steps = timesteps
        t = torch.linspace(0, steps, steps + 1) / steps
        alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return betas.clamp(max=0.999)

# ---------------------------
# Small, replaceable U-Net block (add conditioning support)
# ---------------------------
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, cond_dim=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.act = nn.SiLU()
        self.norm1 = nn.BatchNorm2d(out_ch)
        self.norm2 = nn.BatchNorm2d(out_ch)
        if cond_dim is not None:
            self.cond_proj = nn.Linear(cond_dim, out_ch)
        else:
            self.cond_proj = None
        if in_ch != out_ch:
            self.res_conv = nn.Conv2d(in_ch, out_ch, 1)
        else:
            self.res_conv = None

    def forward(self, x, emb=None):
        h = self.conv1(x)
        h = self.norm1(h)
        h = self.act(h)
        if self.cond_proj is not None and emb is not None:
            he = self.cond_proj(emb).unsqueeze(-1).unsqueeze(-1)
            h = h + he
        h = self.conv2(h)
        h = self.norm2(h)
        out = self.act(h)
        if self.res_conv is not None:
            x = self.res_conv(x)
        return out + x

class SimpleUNet(nn.Module):
    """U-Net small. Now accepts optional `cond` embedding (text+time combined)."""
    def __init__(self, in_ch=3, base_ch=64, time_emb_dim=128, cond_dim=None):
        super().__init__()
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        # If cond_dim provided, project cond -> same dim as time_emb and we'll sum them
        self.cond_proj = nn.Linear(cond_dim, time_emb_dim) if cond_dim is not None else None

        # encoding
        self.enc1 = ResidualBlock(in_ch, base_ch, cond_dim=time_emb_dim)
        self.enc2 = ResidualBlock(base_ch, base_ch * 2, cond_dim=time_emb_dim)
        self.enc3 = ResidualBlock(base_ch * 2, base_ch * 4, cond_dim=time_emb_dim)
        # decoding
        self.dec3 = ResidualBlock(base_ch * 4, base_ch * 2, cond_dim=time_emb_dim)
        self.dec2 = ResidualBlock(base_ch * 2, base_ch, cond_dim=time_emb_dim)
        self.out_conv = nn.Conv2d(base_ch, in_ch, 1)
        self.down = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, t, cond_emb=None):
        # t: (B,) long tensor
        t_emb = self._time_embedding(t, x.device)
        t_emb = self.time_mlp(t_emb)  # (B, time_emb_dim)
        if cond_emb is not None and self.cond_proj is not None:
            c = self.cond_proj(cond_emb)
            t_emb = t_emb + c
        # pass t_emb as the shared conditioning embedding to residual blocks
        e1 = self.enc1(x, t_emb)
        e2 = self.enc2(self.down(e1), t_emb)
        e3 = self.enc3(self.down(e2), t_emb)
        d3 = self.dec3(e3, t_emb)
        d2 = self.dec2(self.up(d3) + e2, t_emb)
        out = self.out_conv(self.up(d2) + e1)
        return out

    @staticmethod
    def _time_embedding(t, device, dim=128):
        half = dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if dim % 2 == 1:
            emb = F.pad(emb, (0,1,0,0))
        return emb

# ---------------------------
# Text encoder (CLIP if available else a simple trainable encoder)
# ---------------------------
class SimpleTextEncoder(nn.Module):
    """Fallback text encoder: tokenizes by whitespace, hashes tokens into a small vocab, average pools embeddings."""
    def __init__(self, emb_dim=512, vocab_size=10000, token_max_len=32):
        super().__init__()
        self.vocab_size = vocab_size
        self.token_max_len = token_max_len
        self.token_emb = nn.Embedding(vocab_size, emb_dim)
        self.proj = nn.Sequential(nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim))

    def forward(self, texts: List[str]):
        # texts -> batch of strings
        device = next(self.parameters()).device
        B = len(texts)
        toks = torch.zeros((B, self.token_max_len), dtype=torch.long, device=device)
        for i, s in enumerate(texts):
            parts = s.lower().split()[:self.token_max_len]
            for j, p in enumerate(parts):
                idx = (hash(p) % self.vocab_size)
                toks[i, j] = idx
        emb = self.token_emb(toks)  # (B, L, D)
        emb = emb.mean(dim=1)  # simple avg pooling
        return self.proj(emb)

class TextEncoderWrapper:
    """Unified wrapper: uses CLIP (transformers) if available, else SimpleTextEncoder."""
    def __init__(self, device, out_dim=512):
        self.device = device
        self.out_dim = out_dim
        if HAS_TRANSFORMERS:
            # load CLIP text encoder
            self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            self.model.eval()
            # project to out_dim if needed
            if self.model.config.hidden_size != out_dim:
                self.proj = nn.Linear(self.model.config.hidden_size, out_dim).to(device)
            else:
                self.proj = nn.Identity()
            self.use_clip = True
        else:
            self.model = SimpleTextEncoder(emb_dim=out_dim).to(device)
            self.tok = None
            self.proj = nn.Identity()
            self.use_clip = False

    @torch.no_grad()
    def encode(self, texts: List[str]):
        if self.use_clip:
            inputs = self.tok(texts, padding=True, truncation=True, return_tensors="pt").to(self.device)
            out = self.model(**inputs)
            pooled = out.last_hidden_state[:, 0, :]  # CLIP default pooling
            return self.proj(pooled)
        else:
            return self.model(texts)  # already on device

# ---------------------------
# GaussianDiffusion class (support cond embeddings + classifier-free guidance)
# ---------------------------
class GaussianDiffusion:
    def __init__(self, model: nn.Module, timesteps: int = 1000, beta_schedule: str = 'linear'):
        self.model = model
        self.timesteps = timesteps
        if beta_schedule == 'linear':
            betas = BetaSchedule.linear(timesteps)
        elif beta_schedule == 'cosine':
            betas = BetaSchedule.cosine(timesteps)
        else:
            raise ValueError('Unknown beta schedule')

        self.register_buffer = {}
        self.betas = betas
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.tensor([1.], dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], dim=0)

        self._set_buffer('betas', betas)
        self._set_buffer('alphas', alphas)
        self._set_buffer('alphas_cumprod', alphas_cumprod)
        self._set_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        self._set_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self._set_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - alphas_cumprod))
        self._set_buffer('sqrt_recip_alphas', torch.sqrt(1.0 / alphas))
        self._set_buffer('posterior_variance', betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod))

    def _set_buffer(self, name, value):
        self.register_buffer[name] = value

    def to(self, device):
        for k, v in self.register_buffer.items():
            if isinstance(v, torch.Tensor):
                self.register_buffer[k] = v.to(device)
        self.model.to(device)
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.register_buffer['sqrt_alphas_cumprod'][t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.register_buffer['sqrt_one_minus_alphas_cumprod'][t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def predict_eps_from_xstart(self, x_t, t, x0):
        return (x_t - self.register_buffer['sqrt_alphas_cumprod'][t].view(-1,1,1,1) * x0) / self.register_buffer['sqrt_one_minus_alphas_cumprod'][t].view(-1,1,1,1)

    def p_mean_variance(self, x_t, t, cond_emb=None, guidance_scale=1.0):
        # compute eps prediction, with optional classifier-free guidance
        # model signature: model(x_t, t, cond_emb) where cond_emb can be None
        if guidance_scale == 1.0 or cond_emb is None:
            eps_pred = self.model(x_t, t, cond_emb)
        else:
            # classifier-free guidance: get eps_uncond and eps_cond
            eps_cond = self.model(x_t, t, cond_emb)
            eps_uncond = self.model(x_t, t, None)
            eps_pred = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

        betas_t = self.register_buffer['betas'][t].view(-1,1,1,1)
        sqrt_recip_alphas_t = self.register_buffer['sqrt_recip_alphas'][t].view(-1,1,1,1)
        sqrt_one_minus_alphas_cumprod_t = self.register_buffer['sqrt_one_minus_alphas_cumprod'][t].view(-1,1,1,1)

        model_mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_alphas_cumprod_t * eps_pred)
        posterior_variance_t = self.register_buffer['posterior_variance'][t].view(-1,1,1,1)
        return model_mean, posterior_variance_t, eps_pred

    @torch.no_grad()
    def p_sample(self, x_t, t, cond_emb=None, guidance_scale=1.0):
        model_mean, model_var, eps_pred = self.p_mean_variance(x_t, t, cond_emb, guidance_scale)
        noise = torch.randn_like(x_t)
        nonzero_mask = (t != 0).float().view(-1,1,1,1)
        return model_mean + nonzero_mask * torch.sqrt(model_var) * noise

    @torch.no_grad()
    def sample(self, batch_size: int, shape, device, cond_emb=None, guidance_scale=1.0, progress=True):
        x = torch.randn((batch_size,) + shape, device=device)
        for t_ in tqdm(reversed(range(self.timesteps)), disable=not progress):
            t = torch.full((batch_size,), t_, device=device, dtype=torch.long)
            x = self.p_sample(x, t, cond_emb=cond_emb, guidance_scale=guidance_scale)
        return x

    def loss(self, x0, device, cond_emb=None, cf_drop_prob=0.1):
        B = x0.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=device).long()
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        # classifier-free training: with prob cf_drop_prob, drop cond (set to None)
        if cond_emb is not None and cf_drop_prob > 0:
            mask = (torch.rand(B, device=device) > cf_drop_prob).float().unsqueeze(1)
            # mask shape (B,1) to zero-out cond for some examples
            cond_used = cond_emb * mask
        else:
            cond_used = cond_emb
        eps_pred = self.model(x_t, t, cond_used)
        return F.mse_loss(eps_pred, noise)

# ---------------------------
# Trainer
# ---------------------------
class Trainer:
    def __init__(self, diffusion: GaussianDiffusion, dataset: Dataset, device, text_encoder: TextEncoderWrapper,
                 batch_size=64, lr=2e-4, ckpt_dir='checkpoints', cf_drop_prob=0.1):
        self.diffusion = diffusion
        self.device = device
        self.batch_size = batch_size
        self.dl = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
        self.opt = torch.optim.Adam(diffusion.model.parameters(), lr=lr)
        self.ckpt_dir = ckpt_dir
        os.makedirs(ckpt_dir, exist_ok=True)
        self.text_encoder = text_encoder
        self.cf_drop_prob = cf_drop_prob

    def save(self, step:int):
        path = os.path.join(self.ckpt_dir, f'cddpm_step{step}.pth')
        to_save = {
            'model_state': self.diffusion.model.state_dict(),
            'opt_state': self.opt.state_dict(),
            'step': step,
            'betas': self.diffusion.register_buffer['betas'].cpu()
        }
        torch.save(to_save, path)
        torch.save(to_save, os.path.join(self.ckpt_dir, 'cddpm_latest.pth'))
        print('Saved checkpoint to', path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.diffusion.model.load_state_dict(ckpt['model_state'])
        self.opt.load_state_dict(ckpt['opt_state'])
        print('Loaded checkpoint', path)

    def train(self, epochs=10, log_interval=100):
        self.diffusion.to(self.device)
        global_step = 0
        pbar = range(epochs)
        for epoch in pbar:
            loop = tqdm(self.dl)
            for batch in loop:
                # support dataset returning (img, caption) or (img, label) or just img
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                    caption = batch[1] if len(batch) > 1 else None
                else:
                    x = batch
                    caption = None
                x = x.to(self.device)
                if caption is None:
                    # as fallback: use '': empty captions
                    captions = [''] * x.size(0)
                else:
                    # ensure list of strings
                    captions = [c if isinstance(c, str) else str(c) for c in caption]
                with torch.no_grad():
                    cond_emb = self.text_encoder.encode(captions).to(self.device)
                loss = self.diffusion.loss(x, self.device, cond_emb=cond_emb, cf_drop_prob=self.cf_drop_prob)
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

                if global_step % log_interval == 0:
                    loop.set_description(f"Epoch {epoch} step {global_step} loss {loss.item():.4f}")
                global_step += 1
            # save at end of epoch
            self.save(global_step)

# ---------------------------
# Dataloader helper: if dataset has no captions, create dummy captions from class names
# ---------------------------
class CIFAR10WithCaptions(datasets.CIFAR10):
    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        # create a simple caption from class name
        caption = self.classes[label]
        return img, caption

def get_dataloader(name, image_size, batch_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5]*3, [0.5]*3),
    ])
    if name.lower() == 'cifar10':
        ds = CIFAR10WithCaptions(root='data', download=True, transform=transform)
    else:
        raise ValueError('Dataset not supported in demo (add your dataset that returns (img, caption))')
    return ds

# ---------------------------
# CLI / Example usage
# ---------------------------
# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--mode', choices=['train','sample'], default='train')
#     parser.add_argument('--timesteps', type=int, default=1000)
#     parser.add_argument('--image_size', type=int, default=32)
#     parser.add_argument('--batch_size', type=int, default=128)
#     parser.add_argument('--epochs', type=int, default=10)
#     parser.add_argument('--lr', type=float, default=2e-4)
#     parser.add_argument('--base_ch', type=int, default=64)
#     parser.add_argument('--ckpt', type=str, default=None)
#     parser.add_argument('--ckpt_dir', type=str, default='checkpoints')
#     parser.add_argument('--n_samples', type=int, default=16)
#     parser.add_argument('--out_dir', type=str, default='out')
#     parser.add_argument('--dataset', type=str, default='CIFAR10')
#     parser.add_argument('--beta_schedule', type=str, default='linear')
#     parser.add_argument('--text_dim', type=int, default=512)
#     parser.add_argument('--guidance_scale', type=float, default=3.0)
#     args = parser.parse_args()

#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#     # build model: pass cond_dim = text_dim so UNet can accept conditioning
#     model = SimpleUNet(in_ch=3, base_ch=args.base_ch, time_emb_dim=128, cond_dim=args.text_dim)
#     diffusion = GaussianDiffusion(model, timesteps=args.timesteps, beta_schedule=args.beta_schedule)
#     diffusion.to(device)

#     text_encoder = TextEncoderWrapper(device, out_dim=args.text_dim)

#     if args.mode == 'train':
#         ds = get_dataloader(args.dataset, args.image_size, args.batch_size)
#         trainer = Trainer(diffusion, ds, device, text_encoder, batch_size=args.batch_size, lr=args.lr, ckpt_dir=args.ckpt_dir)
#         if args.ckpt is not None:
#             trainer.load(args.ckpt)
#         trainer.train(epochs=args.epochs)

#     elif args.mode == 'sample':
#         if args.ckpt is None:
#             raise ValueError('Provide --ckpt for sampling')
#         os.makedirs(args.out_dir, exist_ok=True)
#         ckpt = torch.load(args.ckpt, map_location=device)
#         model.load_state_dict(ckpt['model_state'])
#         diffusion.to(device)

#         # texts to condition on
#         prompts = ["a photo of a cat", "a photo of a car"]  # example; will be repeated to reach n_samples
#         # repeat prompts to batch size n_samples
#         prompts = (prompts * ((args.n_samples // len(prompts)) + 1))[:args.n_samples]
#         cond_emb = text_encoder.encode(prompts).to(device)

#         samples = diffusion.sample(args.n_samples, (3, args.image_size, args.image_size), device, cond_emb=cond_emb, guidance_scale=args.guidance_scale)
#         samples = (samples.clamp(-1,1) + 1) / 2
#         utils.save_image(samples, os.path.join(args.out_dir, 'samples.png'), nrow=int(math.sqrt(args.n_samples)))
#         print('Saved samples to', os.path.join(args.out_dir, 'samples.png'))

# if __name__ == '__main__':
#     main()
