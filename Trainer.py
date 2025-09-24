import os
import copy
import time
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_

from Diffusion_core import UNet, GaussianDiffusion
from Conditioning import ConditionalUNet, CLIPTextEmbedder, ConditionalGaussianDiffusion
from Guidance_Sampling import DDIMSampler
from Latent_Diffusion import LatentDiffusion, SimpleVAE
from Evaluation_Logging import TrainingLogger, Evaluator, Metrics

# -------------------
# Dataset
# -------------------
class TextImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, captions, transform=None, tokenizer=None):
        self.images = images
        self.captions = captions
        self.transform = transform
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, cap = self.images[idx], self.captions[idx]
        if self.transform:
            img = self.transform(img)
        if self.tokenizer:
            cap = self.tokenizer(cap)
        return img, cap


# -------------------
# Trainer
# -------------------
class Trainer:
    def __init__(self, model, diffusion, vae, optimizer, sampler, dataloader, evaluator, logger,
                 ema_decay=0.999, device="cuda"):
        self.model = model.to(device)
        self.diffusion = diffusion
        self.vae = vae.to(device)
        self.optimizer = optimizer
        self.sampler = sampler
        self.dataloader = dataloader
        self.evaluator = evaluator
        self.logger = logger
        self.device = device

        # EMA
        # self.ema_model = type(model)().to(device)
        self.ema_model = copy.deepcopy(model).to(device)
        self.ema_model.load_state_dict(model.state_dict())
        self.ema_decay = ema_decay

        self.global_step = 0

    def update_ema(self):
        with torch.no_grad():
            for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
                p_ema.data = self.ema_decay * p_ema.data + (1. - self.ema_decay) * p.data

    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        for batch in self.dataloader:
            imgs, caps = batch

            # move images to device
            imgs = imgs.to(self.device)

            # do NOT blindly call caps.to(self.device) because caps can be list[str]
            # if caps is a tensor (token ids), move it; otherwise leave as list of strings
            if isinstance(caps, torch.Tensor):
                caps_device = caps.to(self.device)
            else:
                caps_device = caps  # keep list[str] or other format; diffusion.loss_fn handles it

            self.optimizer.zero_grad()

            # Use the diffusion's loss_fn interface.
            # Conditional case: ConditionalGaussianDiffusion.loss_fn(x, captions, ...)
            # Unconditional: GaussianDiffusion.loss_fn(x)
            if isinstance(self.diffusion, ConditionalGaussianDiffusion):
                loss = self.diffusion.loss_fn(imgs, caps_device)  # returns scalar tensor
            else:
                loss = self.diffusion.loss_fn(imgs)

            loss.backward()
            clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.update_ema()

            self.logger.log_losses({"train": loss.item()}, self.global_step)

            # periodic sampling / eval: call sampler.sample with the right signature
            if self.global_step % 1000 == 0:
                with torch.no_grad():
                    # If diffusion is conditional and has a text_embedder, encode to context
                    if isinstance(self.diffusion, ConditionalGaussianDiffusion):
                        # If captions are strings:
                        sample_captions = caps_device[:4] if not isinstance(caps_device, torch.Tensor) else None
                        if sample_captions is not None:
                            # encode -> (tokens, seq_emb)
                            _, context = self.diffusion.text_embedder.encode(list(sample_captions))
                            context = context.to(self.device)
                            samples = self.sampler.sample(batch_size=context.shape[0],
                                                          model=self.ema_model,
                                                          context=context)
                        else:
                            # captions are token tensors already (or incompatible) -> fallback to unconditional sampling
                            samples = self.sampler.sample(batch_size=min(4, imgs.shape[0]),
                                                          model=self.ema_model,
                                                          context=None)
                    else:
                        # unconditional sampler usage
                        samples = self.sampler.sample(batch_size=min(4, imgs.shape[0]),
                                                      model=self.ema_model,
                                                      context=None)

                self.logger.log_images(samples, self.global_step)

                results = self.evaluator.evaluate(samples)
                self.logger.log_metrics(results, self.global_step)
            
            total_loss += loss.item()

            self.global_step += 1
        avg_loss = total_loss / len(self.dataloader)
        return avg_loss


    def save_checkpoint(self, path="checkpoint.pt", epoch=None):
        torch.save({
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "opt": self.optimizer.state_dict(),
            "step": self.global_step,
            "epoch": epoch if epoch is not None else 0
        }, path)

    def load_checkpoint(self, path="checkpoint.pt"):
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.ema_model.load_state_dict(ckpt["ema"])
        self.optimizer.load_state_dict(ckpt["opt"])
        self.global_step = ckpt["step"]
        start_epoch = ckpt.get("epoch", 0)
        print(f"Loaded checkpoint from {path}, resume at epoch {start_epoch}")
        return start_epoch

def benchmark_text_encoder(text_embedder, batch_size=8, seq_len=77, n_iters=50, device=None):
    """
    Đo hiệu suất CLIPTextEmbedder bằng cách tính tokens/second.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    dummy_texts = ["a cat"] * batch_size

    # Warm-up (loại bỏ overhead lần đầu)
    for _ in range(5):
        _ = text_embedder.encode(dummy_texts)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()

    total_tokens = 0
    for _ in range(n_iters):
        _, seq_emb = text_embedder.encode(dummy_texts, max_length=seq_len)
        total_tokens += batch_size * seq_emb.shape[1]  # B * L

    if device.type == 'cuda':
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    tokens_per_sec = total_tokens / elapsed
    print(f"ext encoder throughput: {tokens_per_sec:.2f} tokens/s "
          f"(batch={batch_size}, seq_len={seq_len}, iters={n_iters})")
    return tokens_per_sec

def benchmark_iterations(model, diffusion, batch_size=8, image_size=32, n_iters=50, device=None):
    """
    Đo số iterations (forward pass) trên UNet+diffusion mỗi giây.
    """
    import time
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # fake input
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    t = torch.randint(0, diffusion.timesteps, (batch_size,), device=device)
    _, context = diffusion.text_embedder.encode(["benchmark"] * batch_size)

    # Warm-up
    for _ in range(5):
        _ = diffusion.model(x, t, context)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    for _ in range(n_iters):
        _ = diffusion.model(x, t, context)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    it_per_sec = n_iters / elapsed
    print(f"Diffusion throughput: {it_per_sec:.2f} iterations/s "
          f"(batch={batch_size}, image_size={image_size}, iters={n_iters})")
    return it_per_sec


def benchmark_images(sampler, model, text_embedder, batch_size=4, prompt="cat", n_iters=10, device=None):
    """
    Đo số images/s khi sampling với sampler (DDIM/DPMSolver...).
    """
    import time
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, context = text_embedder.encode([prompt] * batch_size)

    # Warm-up
    with torch.no_grad():
        _ = sampler.sample(batch_size=batch_size, model=model, context=context)

    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.time()

    with torch.no_grad():
        for _ in range(n_iters):
            _ = sampler.sample(batch_size=batch_size, model=model, context=context)

    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    elapsed = end - start
    img_per_sec = (batch_size * n_iters) / elapsed
    print(f"Sampling throughput: {img_per_sec:.2f} images/s "
          f"(batch={batch_size}, n_iters={n_iters})")
    return img_per_sec