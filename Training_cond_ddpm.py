import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# >>> thay bằng module chứa phiên bản conditional <<<
# nếu bạn đặt code trong 'conditional_ddpm.py'
from Conditional_DDPMs import (
    Trainer,                    # Trainer đã hỗ trợ captions + classifier-free
    SimpleUNet as UNetModel,    # UNet có cond_dim
    GaussianDiffusion as Diffusion,
    TextEncoderWrapper          # CLIP nếu có, fallback encoder nếu không
)

# 1) Dataset (ImageFolder): thư mục con = class name -> dùng làm caption
#   CapturedImages/
#     ├─ cat/   img1.jpg, ...
#     └─ car/   img2.jpg, ...
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3),  # scale ảnh về [-1, 1]
])

dataset = datasets.ImageFolder("animal and item", transform=transform)
# Trainer conditional đã tự xử lý các trường hợp (img, label) -> caption = class name

# 2) Model + Diffusion + Text Encoder
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TEXT_DIM = 512  # kích thước embedding text
model = UNetModel(
    in_ch=3,
    base_ch=64,
    time_emb_dim=128,
    cond_dim=TEXT_DIM          # rất quan trọng: bật chế độ conditional
).to(device)

diffusion = Diffusion(model, timesteps=1000, beta_schedule="linear").to(device)

# Text encoder: tự dùng CLIP nếu bạn đã 'pip install transformers', nếu không sẽ fallback encoder trainable
text_encoder = TextEncoderWrapper(device=device, out_dim=TEXT_DIM)

# 3) Trainer (Windows friendly: num_workers=0)
trainer = Trainer(
    diffusion=diffusion,
    dataset=dataset,
    device=device,
    text_encoder=text_encoder,
    batch_size=64,
    lr=2e-4,
    ckpt_dir="checkpoints",
    cf_drop_prob=0.1   # classifier-free guidance drop prob (nên 0.1~0.2)
)

# ép DataLoader dùng single-process để tránh lỗi spawn/pickle trên Windows
trainer.dl = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=True)

# 4) Train
trainer.train(epochs=200)
