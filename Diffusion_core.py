"""
Module 1 — Forward/Reverse Diffusion Core (Unconditional DDPM)

Features included:
- Noise schedules (linear, cosine)
- Forward q_sample (adding noise)
- UNet-like denoiser with Time Embedding
- Training loss (simple MSE on predicted noise)
- Sampling loop (ancestral DDPM sampling + optional DDIM stub)
- Checkpointing and basic training loop skeleton

"""

from typing import Optional, Tuple
import math
import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

# -------------------- Utilities: Noise Schedules --------------------
'''
Noise schedule: Quy định cách nhiễu Beta(t) thay đổi theo timestep, nó sẽ giúp tạo ra dạy noise schedule để xây dựng các biên liên quan như alpha(t) = 1 - beta(t), alpha_mũ(t) = PI(t)(s=1)alpha(s)
'''
def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 2e-2) -> torch.Tensor:
    """Linear schedule from beta_start to beta_end over timesteps."""
    return torch.linspace(beta_start, beta_end, timesteps)
'''
Đầu vào:
    timesteps: số bước khuếch tán T (DDPM gốc dùng T = 1000)
    beta_start: giá trị Beta(1), tức lượng nhiễu thêm ở bước đầu, thường rất it để ảnh ban đầu chưa bị phá nhiều
    beta_end: giá trị Beta(T), tức lượng nhiễu ở bước cuối cùng, thường lớn hơn để đảm bảo x(T)~N(0,I)

Đâu ra:
    torch.tensor có shape(timesteps,): chứa dãy Beta(t) từ t=1 -> t=T, được nội suy tuyến tính từ Beta_start -> Beta_end (công thức cho nội suy tuyến tính để tính Beta(t) = Beta_start + [(t-1) / (T-1)](Beta_end - Beta_start)

EX:  linear_beta_schedule(5, 0.1, 0.5) --> output: tensor([0.1000, 0.2000, 0.3000, 0.4000, 0.5000])
'''

def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule from Nichol & Dhariwal (improved).
    Returns beta_t for t in 0..timesteps-1 as a torch.Tensor.
    """
    '''
    cosine schedule cải tiến hơn so với linear schedule 
    Alpha_mũ(t) giảm theo hàm cos^2, tức ban đầu giảm chậm, sau đó nhanh lên --> giữ lại nhiều thông tin ở giai đoạn đầu, training ổn hơn
    Nghiên cứu cho thấy cosine schedule giúp model tọa ảnh rỏ nét hơn vả chất lượng cao hơn
    '''
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps) # tạo x = [0, 1,...,T]: tạo dãy giá trị t từ 1 đến T 
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0] 
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.999) # đảm bảo Beta(t) thuộc [10^-4, 0.999], tránh quá nhỏ hoặc quá lớn gây bất ổn số học
'''
Đầu vào: 
    timesteps: số bước diffusion T
    s: offset nhỏ để tránh cos^2 = 0 hoàn toàn, giúp ổn định training

Đầu ra: 
    betas: torch.Tensor với shape(timestep,)
    Đây là dãy Beta(t) được suy ra từ cosine schedule  
'''

# -------------------- Time Embedding --------------------
'''
timestep embedding + x(t) sẽ là đầu vào cho UNet 
embedding này đưa qua MLP và trông vào residula blocks hoặc attention blocks
Nhờ đó mô hình biết được đang xử lý timestep nào
Nếu ko có embedding thì UNet không biết sự khác biệt giữa x10 và x500 là gì
'''
class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for timesteps.
    Input: (batch,) long tensor of timesteps
    Output: (batch, dim)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,)
        device = t.device
        half_dim = self.dim // 2
        emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / (half_dim - 1)))
        emb = t[:, None].float() * emb[None, :] # nhân t với vector tần số ( shape(B, haft_dim) )
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # láy sin, cos --> tạo embedding tuần hoàn. Shape(B, dim) hoặc shape(B, 2*half_dim)
        if self.dim % 2 == 1:  # zero pad
            emb = F.pad(emb, (0, 1)) # nếu dim số lẻ, pad thêm 0 để giữ shape(B, dim)
        return emb  # (B, dim)

'''
input:
    t: timestep trong quá trình diffusion, mỗi phần tử là một giá trị t thuộc [0,T] ( shape(B,) )
output:
    tensor shape(B, dim)

Đây là timestep embedding vector mà mô hình sẽ dùng như một tín hiệu phụ thêm vào mạng UNet

EX: SinusoidalPosEmb(8)(torch.tensor([1, 2])) --> output: tensor shape (2, 8)
'''
# -------------------- Basic Blocks for UNet --------------------

class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, dropout: float = 0.0):
        super().__init__()
        # chuẩn hóa và convolution
        self.norm1 = nn.GroupNorm(8, in_ch) # ổn định khi batch nhỏ
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1) # giữ kích thức không gian, thay đổi số kênh
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Time embedding projection
        self.time_mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, out_ch)) # từ vector t(emb) (output từ SinusoidalPosEmb + MLP), chiếu sang không gian cùng số kênh out_ch
        self.nin_shortcut = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity() # đảm bảo match kích thước khi cộng
        self.dropout = nn.Dropout(dropout) # Giúp regularization và tăng khả năng học hiểu mạnh hơn

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        # norm -> activation -> conv1
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        # Norm -> activation -> dropout -> conv2
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.nin_shortcut(x)

'''
Công thức: cho input feature x và time embedding e(t):
    h1 = Conv1(SiLU(GroupNorm(x)))
    h2 = h1 + W(t)e(t)
    h3 = Conv2(Dropout(SiLU(GroupNorm(h2))))
    y = h3 + Shortcut(x)

Vài trò của UNet backbone:
    xử lý thông tin ảnh (qua conv layers) vừa kết hợp thông tin thời gian (qua time embedding)
    model học denoising ở nhiều mức độ mà không bị mất gradient 
'''

class AttentionBlock(nn.Module):
    """Simple self-attention block over channels (like SAGAN/BigGAN style but adapted).
    Input shape: (B, C, H, W) (batch, channels, height, width) -> flatten to (B, H*W, C) for attention.
    """
    '''
    Thực hiện multi-hêad self-attention trên spatial dimensions (H*W), coi ảnh như một chuỗi gồm H*W tokens, mỗi token có kích thức C
    sao khi attention, reshape về lại ảnh ban đầu đều hợp nhất với pipeline convolution
    '''
    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv1d(channels, channels, 1)
        self.k = nn.Conv1d(channels, channels, 1)
        self.v = nn.Conv1d(channels, channels, 1)
        self.proj_out = nn.Conv1d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        x_in = x
        x = self.norm(x)
        x = x.view(B, C, H * W)  # (B, C, N)

        q = self.q(x)  # (B, C, N)
        k = self.k(x)
        v = self.v(x)

        # reshape for multihead: (B, heads, C_per_head, N)
        C_per_head = C // self.num_heads
        q = q.view(B, self.num_heads, C_per_head, -1)
        k = k.view(B, self.num_heads, C_per_head, -1)
        v = v.view(B, self.num_heads, C_per_head, -1)

        # scaled dot-product attention
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) * (1.0 / math.sqrt(C_per_head))
        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.contiguous().view(B, C, -1)
        out = self.proj_out(out)
        out = out.view(B, C, H, W)

        return x_in + out


# -------------------- UNet Denoiser --------------------

class UNet(nn.Module):
    """Simplified UNet with down/up blocks, residual blocks and attention in middle.

    Input: noisy image x_t, timestep t
    Output: predicted noise eps_theta(x_t, t)

    Shapes (example for 3-channel RGB images):
        x: (B, 3, H, W)
        t: (B,) long
    
    self-attention: Cơ chế giúp model hiểu các phần tử trong cùng dữ liệu
                    trong trường hợp này là ảnh thì mỗi pixel sẽ không chỉ lưu thông tin của nó mà còn biết về những pixel khác bằng việc tính độ tương đồng của một pixel với những pixel khác bằng hmaf Softmax
    skip-connections: là quá trình nối tắt đặc trưng giữa encoder sang decoder trong UNet
                    Khi encoder (downsampling nhiều lần), mô hình sẽ có được ngữ cảnh thông qua đặc trưng nhưng lại mất chi tiết cục bộ. Skip-connetions sẽ truyền trực tiếp feature maps từ encoder sang decoder ở cùng mức phân giải
   """
    def __init__(self, in_channels: int = 3, base_ch: int = 64, ch_mults=(1, 2, 4, 8),
                 num_res_blocks: int = 2, time_emb_dim: int = 256):
        super().__init__()
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(time_emb_dim), nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(), nn.Linear(time_emb_dim, time_emb_dim))

        # initial conv
        self.init_conv = nn.Conv2d(in_channels, base_ch, kernel_size=3, padding=1)

        # down blocks
        in_ch = base_ch
        self.downs = nn.ModuleList()
        self.skips_channels = []
        for mult in ch_mults:
            out_ch = base_ch * mult
            for _ in range(num_res_blocks):
                self.downs.append(ResidualBlock(in_ch, out_ch, time_emb_dim))
                in_ch = out_ch
                self.skips_channels.append(in_ch)
            # downsample
            self.downs.append(nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=2, padding=1))

        # middle
        self.mid_block1 = ResidualBlock(in_ch, in_ch, time_emb_dim)
        self.mid_attn = AttentionBlock(in_ch)
        self.mid_block2 = ResidualBlock(in_ch, in_ch, time_emb_dim)

        # up blocks
        self.ups = nn.ModuleList()
        for mult in reversed(ch_mults):
            out_ch = base_ch * mult
            # upsample
            self.ups.append(nn.ConvTranspose2d(in_ch, in_ch, kernel_size=4, stride=2, padding=1))
            for _ in range(num_res_blocks):
                # skip connection will concatenate, so input channels = in_ch + skip_ch
                skip_ch = self.skips_channels.pop()
                self.ups.append(ResidualBlock(in_ch + skip_ch, out_ch, time_emb_dim))
                in_ch = out_ch

        # final convs
        self.out_norm = nn.GroupNorm(8, in_ch)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(in_ch, in_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return predicted noise (epsilon) with same shape as x."""
        # x: (B, C, H, W). t: (B,)
        t_emb = self.time_mlp(t)  # (B, time_emb_dim)
        hs = []

        h = self.init_conv(x)
        # Down
        i = 0
        for layer in self.downs:
            if isinstance(layer, ResidualBlock):
                h = layer(h, t_emb)
                hs.append(h)
            else:  # downsample conv
                h = layer(h)

        # Middle
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Up
        for layer in self.ups:
            if isinstance(layer, nn.ConvTranspose2d):
                h = layer(h)
            else:  # ResidualBlock expecting concatenated skip
                skip = hs.pop()
                # concat along channels
                h = torch.cat([h, skip], dim=1)
                h = layer(h, t_emb)

        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        return h

'''
Quy trình UNet: Khử nhiễu ảnh
        Input: ảnh nhiễu tại timestep x(t). Thông tin timestep t sẽ giúp mô hình biết được đang xử lý ở bước nào thông qua encode bằng positional embedding
        Encode (Downsampling path): ảnh sẽ qua Conv + norm + activation / Mỗi tầng sẽ giảm kích thước không gian (H*W) và tăng kênh đặc trưng C / đầu ra sẽ giàu thông tin ngữ nghĩa nhưng mất chi tiết cục bộ
        Bottleneck: Tầng sâu nhất, nén nhất (nới chứa nhiều thông tin ngữ nghĩa nhất )

'''

'''
Encode ảnh bằng downsampleing path
Them thông tin thời gian vào từng block
Capture global context ở middle attention block
Decode lại bằng upsampling path + skip connections
--> dự đoán nhiễu ϵ(theta)(x(t),t)
'''
# -------------------- Gaussian Diffusion Utilities --------------------

@dataclass
class DiffusionHyperparams:
    timesteps: int = 1000
    beta_schedule: str = 'linear'  # 'linear' or 'cosine'


class GaussianDiffusion:
    def __init__(self, model: nn.Module, image_size: int = 32, channels: int = 3,
                 timesteps: int = 1000, beta_schedule: str = 'linear', device: Optional[torch.device] = None):
        self.model = model
        self.image_size = image_size
        self.channels = channels
        self.timesteps = timesteps
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # betas
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError('Unknown beta schedule')

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        # register as buffers so they move with .to(device) and saved in checkpoints
        self.register_buffer = lambda name, val: setattr(self, name, val.to(self.device))

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1.0 - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1.0 / alphas_cumprod - 1))

        # posterior variance
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer('posterior_variance', posterior_variance)

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Sample from q(x_t | x_0) by adding noise according to schedule.
        x_start: (B, C, H, W)
        t: (B,) long (0..T-1)
        returns x_t
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def predict_eps_from_xstart(self, x_t: torch.Tensor, t: torch.Tensor, x0_pred: torch.Tensor) -> torch.Tensor:
        return (x_t - self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1) * x0_pred) / self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

    def p_mean_variance(self, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Given x_t, predict mean and variance for p(x_{t-1} | x_t)
        Model outputs predicted noise eps_theta(x_t, t)
        """
        eps_pred = self.model(x_t, t)
        # predict x0
        x0_pred = (self.sqrt_recip_alphas_cumprod[t].view(-1, 1, 1, 1) * x_t -
                   self.sqrt_recipm1_alphas_cumprod[t].view(-1, 1, 1, 1) * eps_pred)

        # clamp x0 for numerical stability
        x0_pred = torch.clamp(x0_pred, -1.0, 1.0)

        mean_pred = (
            self.posterior_mean_coeff1(t).view(-1, 1, 1, 1) * x0_pred +
            self.posterior_mean_coeff2(t).view(-1, 1, 1, 1) * x_t
        )
        var = self.posterior_variance[t].view(-1, 1, 1, 1)
        return mean_pred, var

    def posterior_mean_coeff1(self, t: torch.Tensor) -> torch.Tensor:
        # coef for x0 in posterior mean
        return (self.betas[t] * torch.sqrt(self.alphas_cumprod_prev[t]) / (1.0 - self.alphas_cumprod[t]))

    def posterior_mean_coeff2(self, t: torch.Tensor) -> torch.Tensor:
        # coef for x_t in posterior mean
        return ((1.0 - self.alphas_cumprod_prev[t]) * torch.sqrt(self.alphas[t]) / (1.0 - self.alphas_cumprod[t]))

    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Sample x_{t-1} from p(x_{t-1} | x_t) using model predictions.
        x_t: (B, C, H, W)
        t: (B,) long
        returns x_{t-1}
        """
        mean, var = self.p_mean_variance(x_t, t)
        if t[0] == 0:
            # if t==0, we return mean (no noise)
            return mean
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    def sample(self, batch_size: int = 16, device: Optional[torch.device] = None) -> torch.Tensor:
        device = device or self.device
        shape = (batch_size, self.channels, self.image_size, self.image_size)
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((batch_size,), i, dtype=torch.long, device=device)
            x = self.p_sample(x, t)
        return x

    def loss_fn(self, x_start: torch.Tensor) -> torch.Tensor:
        """Compute training loss for a batch x_start.
        Steps:
          - sample t ~ Uniform({0..T-1})
          - sample noise
          - compute x_t = q_sample(x_start, t, noise)
          - predict eps_pred = model(x_t, t)
          - loss = MSE(noise, eps_pred)
        """
        B = x_start.shape[0]
        t = torch.randint(0, self.timesteps, (B,), device=self.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        eps_pred = self.model(x_t, t)
        return F.mse_loss(eps_pred, noise)


# -------------------- Training Loop Skeleton --------------------

def get_dataloader(dataset_name: str = 'CIFAR10', image_size: int = 32, batch_size: int = 32, data_root: str = './data') -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),  # in [0,1]
        transforms.Normalize([0.5], [0.5])  # scale to [-1,1]
    ])

    if dataset_name == 'CIFAR10':
        ds = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    else:
        raise NotImplementedError

    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)


def train(model: nn.Module, diffusion: GaussianDiffusion, epochs: int = 10, batch_size: int = 32, lr: float = 2e-4,
          save_path: str = './checkpoints', dataset_name: str = 'CIFAR10'):
    device = diffusion.device
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    dataloader = get_dataloader(dataset_name=dataset_name, image_size=diffusion.image_size, batch_size=batch_size)
    os.makedirs(save_path, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (x, _) in enumerate(dataloader):
            x = x.to(device)
            loss = diffusion.loss_fn(x)
            opt.zero_grad()
            loss.backward()
            opt.step()

            running_loss += loss.item()
            if (i + 1) % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Step {i+1}/{len(dataloader)} | Loss: {running_loss / 100:.5f}")
                running_loss = 0.0

        # Save checkpoint and sample
        ckpt = {
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'epoch': epoch,
            'timesteps': diffusion.timesteps,
        }
        torch.save(ckpt, os.path.join(save_path, f'ckpt_epoch_{epoch}.pt'))

        # sample some images for visual monitoring (small batch)
        model.eval()
        with torch.no_grad():
            samples = diffusion.sample(batch_size=16)
            # samples in [-1,1], convert to [0,1]
            grid = (samples + 1.0) / 2.0
            utils.save_image(grid, os.path.join(save_path, f'samples_epoch_{epoch}.png'), nrow=4)

    print('Training finished')
