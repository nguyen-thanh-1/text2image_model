import torch
import Diffusion_core as module1
import Conditioning as module2
import torch.nn as nn
# HuggingFace CLIP text encoder
from transformers import CLIPTokenizerFast, CLIPTextModel

from Conditioning import ConditionalGaussianDiffusion
from Guidance_Sampling import SamplerConfig, EMA, sample_images, make_sampler

class DummyModel(nn.Module):
    """Mô phỏng UNet: đầu ra có cùng shape với input (noise prediction)."""
    def __init__(self, channels=3):
        super().__init__()
        self.channels = channels

    def forward(self, x, t, context=None):
        # Trả về noise giả lập (giống shape x)
        return torch.randn_like(x)

class DummyTextEncoder:
    def __init__(self, dim=16, length=5):
        self.dim = dim
        self.length = length
    def encode(self, texts):
        batch = len(texts)
        embeddings = torch.randn(batch, self.length, self.dim)
        return None, embeddings

# -----------------------------
# Dummy Conditional Diffusion
# -----------------------------
class DummyConditionalDiffusion(ConditionalGaussianDiffusion):
    def __init__(self, timesteps=20, image_size=32, channels=3, device="cpu"):
        # Fake GaussianDiffusion base attributes
        self.timesteps = timesteps
        self.image_size = image_size
        self.channels = channels
        self.device = device
        # Linear betas
        betas = torch.linspace(1e-4, 0.02, timesteps)
        alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device)


# -----------------------------
# Test Functions
# -----------------------------
def test_ddim_sampler():
    device = "cpu"
    model = DummyModel().to(device)
    text_encoder = DummyTextEncoder()
    diffusion = DummyConditionalDiffusion(device=device)

    cfg = SamplerConfig(sampler="ddim", steps=5, guidance_scale=1.0, device=device)

    # Test EMA apply/restore
    ema = EMA(model)
    ema.apply_shadow()
    ema.restore()

    prompts = ["a red car", "a blue cat"]

    samples = sample_images(
        model=model,
        diffusion=diffusion,
        text_encoder=text_encoder,
        prompts=prompts,
        batch_size=2,
        sampler_config=cfg,
        ema=ema
    )

    print("DDIM samples shape:", samples.shape)
    assert samples.shape == (2, 3, diffusion.image_size, diffusion.image_size)


def test_pndm_sampler():
    device = "cpu"
    model = DummyModel().to(device)
    text_encoder = DummyTextEncoder()
    diffusion = DummyConditionalDiffusion(device=device)

    cfg = SamplerConfig(sampler="pndm", steps=5, guidance_scale=1.0, device=device)
    sampler = make_sampler(diffusion, cfg)
    samples = sampler.sample(batch_size=2, model=model, context=None)

    print("PNDM samples shape:", samples.shape)
    assert samples.shape == (2, 3, diffusion.image_size, diffusion.image_size)

if __name__ == '__main__':
        
    import torch.multiprocessing

    torch.multiprocessing.freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    #------------------Test module 1
    # unet = module1.UNet(in_channels=3, base_ch=64, ch_mults=(1, 2, 3), num_res_blocks=1, time_emb_dim=128)
    # diffusion = module1.GaussianDiffusion(model=unet, image_size=32, channels=3, timesteps=200, beta_schedule='linear', device=device)
    # module1.train(unet, diffusion, epochs=3, batch_size=64, lr=2e-4, save_path='./module1_out', dataset_name='CIFAR10')

    #------------------Test module 2
    # text_embedder = module2.CLIPTextEmbedder(pretrained='openai/clip-vit-base-patch32', device=device)
    # cond_unet = module2.ConditionalUNet(in_channels=3, base_ch=64, ch_mults=(1, 2, 2), num_res_blocks=1, text_emb_dim=text_embedder.text_emb_dim, time_emb_dim=128)
    # diffusion = module2.ConditionalGaussianDiffusion(model=cond_unet, text_embedder=text_embedder, image_size=32, channels=3, timesteps=200, beta_schedule='linear', device=device)
    # dataset = module2.DummyImageCaptionDataset(image_size=32, num_samples=2000)
    # module2.train_module2(cond_unet, diffusion, dataset, epochs=3, batch_size=32, lr=2e-4)

    #------------------Test module 3
    test_ddim_sampler()
    test_pndm_sampler()
    print("All Module 3 sampler tests passed!")