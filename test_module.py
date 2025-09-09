import torch
import Diffusion_core as module1
import Conditioning as module2
# HuggingFace CLIP text encoder
from transformers import CLIPTokenizerFast, CLIPTextModel

if __name__ == '__main__':
        
    import torch.multiprocessing

    torch.multiprocessing.freeze_support()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # unet = module1.UNet(in_channels=3, base_ch=64, ch_mults=(1, 2, 3), num_res_blocks=1, time_emb_dim=128)
    # diffusion = module1.GaussianDiffusion(model=unet, image_size=32, channels=3, timesteps=200, beta_schedule='linear', device=device)
    # module1.train(unet, diffusion, epochs=3, batch_size=64, lr=2e-4, save_path='./module1_out', dataset_name='CIFAR10')

    text_embedder = module2.CLIPTextEmbedder(pretrained='openai/clip-vit-base-patch32', device=device)
    cond_unet = module2.ConditionalUNet(in_channels=3, base_ch=64, ch_mults=(1, 2, 2), num_res_blocks=1, text_emb_dim=text_embedder.text_emb_dim, time_emb_dim=128)
    diffusion = module2.ConditionalGaussianDiffusion(model=cond_unet, text_embedder=text_embedder, image_size=32, channels=3, timesteps=200, beta_schedule='linear', device=device)
    dataset = module2.DummyImageCaptionDataset(image_size=32, num_samples=2000)
    module2.train_module2(cond_unet, diffusion, dataset, epochs=3, batch_size=32, lr=2e-4)

