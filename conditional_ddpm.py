# conditional_ddpm_fixed.py
import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image

from Conditioning import ConditionalUNet, CLIPTextEmbedder, ConditionalGaussianDiffusion
from Latent_Diffusion import SimpleVAE
from Guidance_Sampling import DDIMSampler, SamplerConfig
from Trainer import TextImageDataset, Trainer, benchmark_text_encoder, benchmark_iterations, benchmark_images
from Evaluation_Logging import Evaluator, TrainingLogger, Metrics

def training(
    root_dir="dataset",
    ckpt_resume_path = 'ckpt_last.pt',
    image_size=32,
    batch_size=8,
    add_epochs = 100,
    epochs=100,
    lr=1e-4,
    device=None,
):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # -------------------------
    # Dataset (ImageFolder -> TextImageDataset)
    # -------------------------
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),               # [0,1]
        transforms.Lambda(lambda t: 2.0 * t - 1.0)  # -> [-1,1] expected by models
    ])

    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Dataset folder '{root_dir}' not found.")

    img_folder = datasets.ImageFolder(root=root_dir, transform=transform)
    if len(img_folder) == 0:
        raise RuntimeError(f"No images found in '{root_dir}' (expect structure root/class_x/*.jpg).")

    # Convert ImageFolder into lists (images already transformed -> tensors)
    images = []
    captions = []
    for img_tensor, label in img_folder:
        images.append(img_tensor)            # tensor in [-1,1]
        cls_name = img_folder.classes[label]
        # Here we assign a simple caption; replace with actual captions if available
        captions.append(cls_name)

    dataset = TextImageDataset(images, captions)  # uses dataset items as (img_tensor, caption)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # -------------------------
    # Model / Diffusion / VAE
    # -------------------------
    model = ConditionalUNet(in_channels=3, base_ch=64, ch_mults=(1,2,4), num_res_blocks=2)
    model.to(device)

    text_embedder = CLIPTextEmbedder(device=device)

    diffusion = ConditionalGaussianDiffusion(
        model=model,
        text_embedder=text_embedder,
        image_size=image_size,
        channels=3,
        timesteps=200,
        device=device
    )

    vae = SimpleVAE(in_channels=3, base_ch=64, z_channels=4)
    vae.to(device)

    # -------------------------
    # Sampler
    # -------------------------
    sampler_cfg = SamplerConfig(steps=50, device=device)
    sampler = DDIMSampler(diffusion, sampler_cfg)

    # -------------------------
    # Metrics / Logger / Evaluator
    # -------------------------
    metrics = Metrics(device=device)

    # build a small validation set (real images) for metrics (denormalize to [0,1])
    n_val = min(16, len(images))
    real_images = torch.stack([images[i] for i in range(n_val)], dim=0)  # [-1,1]
    real_images = (real_images + 1.0) / 2.0
    real_images = real_images.to(device)

    prompts_for_eval = ["cat"] * real_images.shape[0]
    evaluator = Evaluator(metrics, prompts_for_eval, real_images)

    logger = TrainingLogger(logdir="logs", use_wandb=False)

    # -------------------------
    # Optimizer & Trainer
    # -------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        vae=vae,
        optimizer=optimizer,
        sampler=sampler,
        dataloader=dataloader,
        evaluator=evaluator,
        logger=logger,
        device=device
    )

    # -------------------------
    # Training loop
    # -------------------------
    
    ckpt_dir = "checkpoints_largedataset"
    os.makedirs(ckpt_dir, exist_ok=True)

    resume_path = os.path.join(ckpt_dir, ckpt_resume_path) # ckpt_last.pt
    start_epoch = epochs
    
    if os.path.exists(resume_path):
        start_epoch = trainer.load_checkpoint(resume_path)
        print(f"Resume training from epoch {start_epoch}")
    else:
        print("Start training from scratch")

    end_epoch = start_epoch + add_epochs

    # --- Training loop ---
    for epoch in range(start_epoch, end_epoch + 1):
        print(f"=== Epoch {epoch+1}/{end_epoch} ===")
        avg_loss = trainer.train_one_epoch(epoch)

        # Lưu checkpoint mỗi 25 epoch
        if (epoch + 1) % 50 == 0 and (epoch + 1) < end_epoch:
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch+1}.pt")
            trainer.save_checkpoint(ckpt_path, epoch=epoch+1)
            print(f'Loss: {avg_loss}')
        elif (epoch + 1) == end_epoch:
            # Luôn lưu checkpoint cuối
            ckpt_path = os.path.join(ckpt_dir, "ckpt_last.pt")
            trainer.save_checkpoint(ckpt_path, epoch=epoch+1)
            print(f'Loss: {avg_loss}')
    print(start_epoch)
    print(end_epoch)
    # # -------------------------
    # # Sampling (generate 1 image)
    # # -------------------------
    # prompt = ["cat"]
    # _, context = text_embedder.encode(prompt)   # returns (tokens, seq_emb)
    # # context is already on device in encode(); safe to pass directly
    # with torch.no_grad():
    #     samples = sampler.sample(batch_size=1, model=trainer.ema_model, context=context)

    # # samples expected in [-1,1] -> convert to [0,1]
    # out = (samples + 1.0) / 2.0
    # out = torch.clamp(out, 0.0, 1.0)
    # save_image(out, f"sample_cat_epoch{end_epoch}.png")
    # print("Saved sample_cat.png")
def inference(
    ckpt_path="checkpoints/ckpt_last.pt",
    prompt="cat",
    out_file="sample.png",
    image_size=32,
    device=None,
):
    device = torch.device(device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")

    # -------------------------
    # Model / Diffusion / VAE
    # -------------------------
    model = ConditionalUNet(in_channels=3, base_ch=64, ch_mults=(1,2,4), num_res_blocks=2).to(device)
    text_embedder = CLIPTextEmbedder(device=device)
    benchmark_text_encoder(text_embedder, batch_size=8, seq_len=77, n_iters=100, device=device)
    diffusion = ConditionalGaussianDiffusion(
        model=model,
        text_embedder=text_embedder,
        image_size=image_size,
        channels=3,
        timesteps=200,
        device=device
    )
    vae = SimpleVAE(in_channels=3, base_ch=64, z_channels=4).to(device)

    sampler_cfg = SamplerConfig(steps=50, device=device)
    sampler = DDIMSampler(diffusion, sampler_cfg)

    # -------------------------
    # Load checkpoint
    # -------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)  # dummy optimizer, chỉ để khớp cấu trúc
    trainer = Trainer(
        model=model,
        diffusion=diffusion,
        vae=vae,
        optimizer=optimizer,
        sampler=sampler,
        dataloader=None,   # không cần dataloader khi inference
        evaluator=None,
        logger=None,
        device=device
    )

    if os.path.exists(ckpt_path):
        trainer.load_checkpoint(ckpt_path)
        print(f"Loaded model from {ckpt_path}")

        # --- Benchmark hiệu suất ---
        benchmark_iterations(model, diffusion, batch_size=8, image_size=image_size, n_iters=50, device=device)
        benchmark_images(sampler, trainer.ema_model, text_embedder, batch_size=4, prompt=prompt, n_iters=10, device=device)
    else:
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")

    # -------------------------
    # Generate sample
    # -------------------------
    _, context = text_embedder.encode([prompt])   # returns (tokens, seq_emb)
    with torch.no_grad():
        samples = sampler.sample(batch_size=1, model=trainer.ema_model, context=context)

    # [-1,1] → [0,1]
    out = (samples + 1.0) / 2.0
    out = torch.clamp(out, 0.0, 1.0)
    save_image(out, out_file)
    print(f"Saved {out_file}")


if __name__ == "__main__":
    mode = 'inference'  # đổi thành "train" nếu muốn train lại

    if mode == "train":
        training(ckpt_resume_path='ckpt_epoch400.pt',add_epochs=100)
    elif mode == "inference":
        inference(ckpt_path=r"checkpoints_largedataset\ckpt_last.pt", prompt="cat stands on the beach", out_file="cat_and_beach.png")