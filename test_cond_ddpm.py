import torch
from torchvision.utils import save_image
from Conditional_DDPMs import (
    SimpleUNet as UNetModel,
    GaussianDiffusion as Diffusion,
    TextEncoderWrapper,
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Thông số phải khớp với lúc train
# TEXT_DIM = 512
# IMAGE_SIZE = 64
# TIMESTEPS = 1000

# # 1) Load model
# model = UNetModel(
#     in_ch=3,
#     base_ch=64,
#     time_emb_dim=128,
#     cond_dim=TEXT_DIM   # cần thiết cho conditional
# ).to(device)

# ckpt = torch.load("checkpoints/cddpm_latest.pth", map_location=device)
# model.load_state_dict(ckpt["model_state"])
# model.eval()

# # 2) Diffusion + Text encoder
# diffusion = Diffusion(model, timesteps=TIMESTEPS, beta_schedule="linear").to(device)
# text_encoder = TextEncoderWrapper(device=device, out_dim=TEXT_DIM)

# # 3) Prompt để generate ảnh
# prompts = ["a photo of a cat"] * 16   # lặp lại để khớp batch_size
# cond_emb = text_encoder.encode(prompts).to(device)

# # 4) Sampling
# with torch.no_grad():
#     sampled_images = diffusion.sample(
#         batch_size=16,
#         shape=(3, IMAGE_SIZE, IMAGE_SIZE),
#         device=device,
#         cond_emb=cond_emb,
#         guidance_scale=3.0,   # bật classifier-free guidance
#     )

# # 5) Lưu ảnh
# save_image((sampled_images.clamp(-1, 1) + 1) / 2, "generated_cond.png", nrow=4)
# print("Saved to generated_cond.png")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===== 1. Load model =====
TEXT_DIM = 512
IMAGE_SIZE = 64
TIMESTEPS = 1000

model = UNetModel(
    in_ch=3,
    base_ch=64,
    time_emb_dim=128,
    cond_dim=TEXT_DIM
).to(device)

ckpt = torch.load("checkpoints/cddpm_latest.pth", map_location=device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# ===== 2. Diffusion + Text Encoder =====
diffusion = Diffusion(model, timesteps=TIMESTEPS, beta_schedule="linear").to(device)
text_encoder = TextEncoderWrapper(device=device, out_dim=TEXT_DIM)

# ===== 3. Prompt =====
prompt = "hat"
cond_emb = text_encoder.encode([prompt]).to(device)   # batch_size=1

# ===== 4. Sampling =====
with torch.no_grad():
    sampled_image = diffusion.sample(
        batch_size=1,
        shape=(3, IMAGE_SIZE, IMAGE_SIZE),
        device=device,
        cond_emb=cond_emb,
        guidance_scale=3.0   # tăng/giảm để ảnh bám prompt hơn
    )

# ===== 5. Save =====
save_image((sampled_image.clamp(-1, 1) + 1) / 2, "generated_cat_beach_hat.png")
print("Saved to generated_cat_beach_hat.png")