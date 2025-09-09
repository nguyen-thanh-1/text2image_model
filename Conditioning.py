"""
Purpose:
- Extend Module 1 UNet/DDPM core to accept text conditioning via cross-attention.
- Integrate a CLIP text encoder (HuggingFace Transformers) as the default text encoder.
- Implement classifier-free guidance support (conditioning dropout during training and guidance during sampling).
- Provide training loop skeleton that consumes (image, caption) pairs.

Notes:
- This module builds on the UNet and GaussianDiffusion classes from Module1. You can import them
  if both files are in the same package. For convenience this file is written to be runnable
  after `module1_ddpm.py` is placed in the same folder.
- The code favors clarity and modifiability. For large-scale use, replace tokenizer/encoder
  with a frozen/pretrained CLIP text encoder checkpoint for best quality.

Requirements:
  pip install torch torchvision transformers sentencepiece

"""

from typing import Optional, Tuple
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, utils

# We import module1 components. Make sure module1_ddpm.py is in the same dir or installed.
from Diffusion_core import UNet, GaussianDiffusion, linear_beta_schedule, cosine_beta_schedule

# HuggingFace CLIP text encoder
from transformers import CLIPTokenizerFast, CLIPTextModel


# -------------------- Cross-Attention Block --------------------

class CrossAttention(nn.Module):
    """Multi-head cross-attention where queries come from image features and keys/values from text embeddings.
    Input shapes:
      - x: (B, C, H, W)  -> queries (flatten spatial)
      - context: (B, L, D) -> keys/values
    """
    def __init__(self, channels: int, context_dim: int, num_heads: int = 8):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.scale = (channels // num_heads) ** -0.5

        # projectors
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(context_dim, channels, bias=False)
        self.to_v = nn.Linear(context_dim, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W
        x_flat = x.view(B, C, N).permute(0, 2, 1)  # (B, N, C)
        x_flat = self.norm(x_flat)

        q = self.to_q(x_flat)  # (B, N, C)
        k = self.to_k(context)  # (B, L, C)
        v = self.to_v(context)  # (B, L, C)

        # reshape for heads
        q = q.view(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, N, Cph)
        k = k.view(B, k.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)  # (B, heads, L, Cph)
        v = v.view(B, v.shape[1], self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, heads, N, L)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)  # (B, heads, N, Cph)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, N, C)  # (B, N, C)

        out = self.to_out(out)  # (B, N, C)
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out


# -------------------- ResidualBlock with Cross-Attention --------------------

class ResidualBlockWithCA(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, text_emb_dim: int, dropout: float = 0.0,
                 use_cross_attention: bool = True, ca_heads: int = 8):
        super().__init__()
        self.res = nn.Sequential(
            nn.GroupNorm(8, in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        )
        self.res2 = nn.Sequential(
            nn.GroupNorm(8, out_ch),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
        )
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch))
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attn = CrossAttention(out_ch, text_emb_dim, num_heads=ca_heads)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.res(x)
        # add time embedding
        h = h + self.time_mlp(t_emb)[:, :, None, None]
        h = self.res2(h)
        if self.use_cross_attention and context is not None:
            h = h + self.cross_attn(h, context)
        return h + self.skip(x)


# -------------------- Conditional UNet --------------------

class ConditionalUNet(UNet):
    """Extend the UNet from module1 to include cross-attention conditioning.
    We override only what's necessary: use ResidualBlockWithCA in place of ResidualBlock,
    and pass context through forward().
    """
    def __init__(self, text_emb_dim: int = 768, *args, **kwargs):
        # Reuse UNet init but we will rebuild parts that use ResidualBlock
        super().__init__(*args, **kwargs)
        # Reconstruct key modules to instances with cross-attention.
        # Note: For simplicity, we re-create the architecture here rather than monkey-patch.
        # A production refactor would modularize block factories.
        # We'll create a new smaller UNet here that mirrors shapes from base UNet.

        # Basic config
        in_channels = kwargs.get('in_channels', 3)
        base_ch = kwargs.get('base_ch', 64)
        ch_mults = kwargs.get('ch_mults', (1, 2, 4))
        num_res_blocks = kwargs.get('num_res_blocks', 2)
        time_emb_dim = kwargs.get('time_emb_dim', 256)

        # Recreate time_mlp with same signature to ensure compatibility
        self.time_mlp = nn.Sequential(self.time_mlp[0], self.time_mlp[1], self.time_mlp[2]) if hasattr(self, 'time_mlp') else None
        # We'll rebuild encoder/decoder using ResidualBlockWithCA
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)
        in_ch = base_ch
        self.downs = nn.ModuleList()
        self.skips_channels = []
        for mult in ch_mults:
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlockWithCA(in_ch, out_ch, time_emb_dim, text_emb_dim))
                in_ch = out_ch
                self.skips_channels.append(in_ch)
            self.downs.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1))

        self.mid_block1 = ResidualBlockWithCA(in_ch, in_ch, time_emb_dim, text_emb_dim)
        self.mid_attn = nn.Identity()  # keep earlier attention if desired
        self.mid_block2 = ResidualBlockWithCA(in_ch, in_ch, time_emb_dim, text_emb_dim)

        # up blocks
        self.ups = nn.ModuleList()
        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            self.ups.append(nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1))
            for _ in range(num_res_blocks):
                skip_ch = self.skips_channels.pop()
                self.ups.append(ResidualBlockWithCA(in_ch + skip_ch, out_ch, time_emb_dim, text_emb_dim))
                in_ch = out_ch

        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        t_emb = self.time_mlp(t) if self.time_mlp is not None else None
        hs = []
        h = self.init_conv(x)
        for layer in self.downs:
            if isinstance(layer, ResidualBlockWithCA):
                h = layer(h, t_emb, context)
                hs.append(h)
            else:
                h = layer(h)
        h = self.mid_block1(h, t_emb, context)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb, context)
        for layer in self.ups:
            if isinstance(layer, nn.ConvTranspose2d):
                h = layer(h)
            else:
                skip = hs.pop()
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb, context)
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h


# -------------------- Text Encoder Wrapper --------------------

class CLIPTextEmbedder(nn.Module):
    """Wrapper around HuggingFace CLIP text encoder to produce token embeddings for cross-attention.
    We return the sequence output (not pooled), shape (B, L, D).
    """
    def __init__(self, pretrained: str = 'openai/clip-vit-base-patch32', device: Optional[torch.device] = None):
        super().__init__()
        self.device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
        self.tokenizer = CLIPTokenizerFast.from_pretrained(pretrained)
        self.model = CLIPTextModel.from_pretrained(pretrained).to(self.device)
        self.text_emb_dim = self.model.config.hidden_size
        self.model.eval()  # we will not finetune text encoder by default

    @torch.no_grad()
    def encode(self, texts: list, max_length: int = 77) -> Tuple[torch.Tensor, torch.Tensor]:
        # returns (tokens_tensor, embeddings) where embeddings: (B, L, D)
        toks = self.tokenizer(texts, padding='longest', truncation=True, max_length=max_length, return_tensors='pt')
        input_ids = toks['input_ids'].to(self.device)
        attention_mask = toks['attention_mask'].to(self.device)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=False)
        seq_emb = out.last_hidden_state  # (B, L, D)
        return input_ids, seq_emb


# -------------------- Classifier-free Guidance helpers --------------------

def maybe_mask_context(seq_emb: torch.Tensor, keep_prob: float = 0.9) -> Optional[torch.Tensor]:
    """With probability (1-keep_prob) return None (unconditioned). For classifier-free guidance we
    train on some fraction of unconditioned samples by providing empty context.
    Note: here we expect seq_emb already as (B, L, D). We'll return None for all in a batch when dropping.
    For more fine-grained dropout, you can mix conditioned/unconditioned within batch.
    """
    if torch.rand(1).item() < (1.0 - keep_prob):
        return None
    return seq_emb


# -------------------- Conditional Diffusion wrapper --------------------

class ConditionalGaussianDiffusion(GaussianDiffusion):
    def __init__(self, model: nn.Module, text_embedder: CLIPTextEmbedder, **kwargs):
        super().__init__(model=model, **kwargs)
        self.text_embedder = text_embedder

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        # Model signature: model(x_t, t, context)
        eps_pred = self.model(x_t, t, context)
        x0_pred = (self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1) * x_t -
                   self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * eps_pred)
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)
        mean_pred = (
            self.posterior_mean_coeff1(t).view(-1, 1, 1, 1) * x0_pred +
            self.posterior_mean_coeff2(t).view(-1, 1, 1, 1) * x_t
        )
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        return mean_pred, var

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        mean, var = self.p_mean_variance(x_t, t, context)
        if t[0] == 0:
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    def sample_with_guidance(self, batch_size: int, context_texts: list, guidance_scale: float = 5.0, device: Optional[torch.device] = None):
        device = device or self.device
        # encode texts (B, L, D)
        _, context = self.text_embedder.encode(context_texts)
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)
            # classifier-free guidance: run conditioned and unconditioned model and combine

            # conditioned eps
            eps_cond = self.model(x, t, context)
            # unconditioned eps (context dropped)
            eps_uncond = self.model(x, t, None)
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # compute x_{t-1} from eps
            x0_pred = (self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1) * x -
                       self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * eps)
            x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

            coef1 = self.posterior_mean_coeff1(t).view(-1, 1, 1, 1)
            coef2 = self.posterior_mean_coeff2(t).view(-1, 1, 1, 1)
            mean = coef1 * x0_pred + coef2 * x
            var = self.posterior_variance[t].view(-1, 1, 1, 1)

            if i == 0:
                x = mean
            else:
                x = mean + torch.sqrt(var) * torch.randn_like(x)
        return x

    def loss_fn(self, x_start: torch.Tensor, captions: list, cond_drop_prob: float = 0.1) -> torch.Tensor:
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        # encode captions (we do not grad text encoder here by default)
        _, seq_emb = self.text_embedder.encode(captions)

        # apply classifier-free dropout per-batch-element: randomly set some contexts to None
        # We'll create a mask of which elements keep context
        keep_mask = (torch.rand(B, device=self.device) > cond_drop_prob).to(torch.bool)
        contexts = [seq_emb[i:i+1] if keep_mask[i] else None for i in range(B)]
        # To pass to model we need either a batched context tensor or None. We'll pass the full seq_emb
        # but modify model to accept None as unconditioned - here we simply set context to seq_emb when keep_mask all True
        # For simplicity, if any unconditioned elements exist, we create two passes (inefficient) and compute loss per element.

        eps_pred = torch.zeros_like(noise)
        if keep_mask.all():
            eps_pred = self.model(x_t, t, seq_emb)
        elif (~keep_mask).all():
            eps_pred = self.model(x_t, t, None)
        else:
            # mixed batch -> compute in two pieces
            idx_keep = keep_mask.nonzero(as_tuple=False).squeeze(1)
            idx_drop = (~keep_mask).nonzero(as_tuple=False).squeeze(1)
            if idx_keep.numel() > 0:
                eps_pred[idx_keep] = self.model(x_t[idx_keep], t[idx_keep], seq_emb[idx_keep])
            if idx_drop.numel() > 0:
                eps_pred[idx_drop] = self.model(x_t[idx_drop], t[idx_drop], None)

        return F.mse_loss(eps_pred, noise)


# -------------------- Example dataset for (image, caption) --------------------

class DummyImageCaptionDataset(Dataset):
    """Small dummy dataset to test conditioning pipeline. Replace with COCO/LAION loader in production."""
    def __init__(self, image_size: int = 32, num_samples: int = 1000):
        self.tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])
        self.num = num_samples
    def __len__(self):
        
        return self.num

    def __getitem__(self, idx):
        # return a random noise image plus a simple caption (toy example)
        img = torch.randn(3, 32, 32) # original (3, 32, 32)
        img = torch.clamp(img, -1.0, 1.0)
        caption = f"A random pattern {idx % 10}"
        return img, caption


# -------------------- Training loop for Module 2 --------------------

def train_module2(model: nn.Module, diffusion: ConditionalGaussianDiffusion, dataset: Dataset, epochs: int = 5,
                  batch_size: int = 16, lr: float = 2e-4, save_path: str = './checkpoints_module2'):
    device = diffusion.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        running_loss = 0.0
        for i, (x, captions) in enumerate(dataloader):
            x = x.to(device)
            loss = diffusion.loss_fn(x, captions, cond_drop_prob=0.1)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running_loss += loss.item()
            if (i + 1) % 50 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {i+1}/{len(dataloader)} | Loss: {running_loss / 50:.5f}")
                running_loss = 0.0
        # checkpoint
        ckpt = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': opt.state_dict(), 'epoch': epoch}
        torch.save(ckpt, os.path.join(save_path, f'ckpt_epoch_{epoch}.pt'))
        # sample with guidance from a few prompts
        model.eval()
        with torch.no_grad():
            prompts = ["A colorful abstract pattern", "A simple line drawing", "A noisy texture"]
            samples = diffusion.sample_with_guidance(batch_size=3, context_texts=prompts, guidance_scale=4.0)
            grid = (samples + 1.0) / 2.0
            utils.save_image(grid, os.path.join(save_path, f'samples_epoch_{epoch}.png'), nrow=3)
        model.train()
    print('Module 2 training finished')
