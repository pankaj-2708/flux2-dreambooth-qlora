import argparse
import gc
import logging
import os
import matplotlib

matplotlib.use("Agg")  # non-interactive backend for headless environments
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset,load_from_disk
from diffusers import AutoModel, AutoencoderKLFlux2
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training ,PeftModel

from torchvision import transforms
from tqdm import tqdm


def empty_cache():
    """Free unused GPU memory."""
    return torch.cuda.empty_cache()


class custom_dataset(torch.utils.data.Dataset):
    """
    Wraps a HuggingFace dataset that already contains pre-encoded
    ``transformer_input`` tensors alongside raw images.
    """

    def __init__(self, dataset, height: int = 768, width: int = 512):
        self.dataset = dataset
        self.height = height
        self.width = width
        self.transform = transforms.Compose(
            [
                transforms.Resize(
                    (self.height, self.width),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.RandomCrop((self.height, self.width)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.transform(self.dataset[idx]["image"])
        text_hidden_state = torch.tensor(self.dataset[idx]["transformer_input"])
        return img, text_hidden_state


class fineTunningFlux2Klein:
    def __init__(
        self,
        dataset,
        collate_fn,
        batch_size=1,
        epochs=10,
        lr=1e-4,
        img_height=768,
        img_width=512,
        enable_gradient_checkpointing=True,
        save_dir="./checkpoints"
    ):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model_id = "black-forest-labs/FLUX.2-klein-9B"
        self.quantization_config = {
            "load_in_4bit": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16,
        }
        self.enable_gradient_checkpointing = True
        self.H = img_height
        self.W = img_width

        # 8x reduction in size
        self.vae_scale_factor = 8

        # shapes at which transformer operate
        self.patched_height = int(self.H / self.vae_scale_factor / 2)
        self.patched_width = int(self.W / self.vae_scale_factor / 2)
        self.patched_channels = 32 * 4

        self.dataloader = self._create_dataloader()
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.save_dir + "/images", exist_ok=True)
        os.makedirs(self.save_dir + "/model", exist_ok=True)

        self.transformer, self.vae = self._load_models(
            self.model_id, self.quantization_config
        )
        for param in self.transformer.parameters():
            param.requires_grad = False
        for param in self.vae.parameters():
            param.requires_grad = False

        self.lora_config = config = {
            "r": 4,
            "lora_alpha": 8,
            "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
            "lora_dropout": 0.05,
            "bias": "none",
            "task_type": None,
        }
        self.total_parameter = sum(
            [param.numel() for param in self.transformer.parameters()]
            + [param.numel() for param in self.vae.parameters()]
        )   

        self.loss_history=[]

    def _create_dataloader(self):
        return torch.utils.data.DataLoader(
            custom_dataset(self.dataset),
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

    def _load_models(self, model_id, quantization_config):
        transformer = AutoModel.from_pretrained(
            model_id,
            subfolder="transformer",
            quantization_config=DiffusersBitsAndBytesConfig(**quantization_config),
            device_map="auto",
        )
        vae = AutoencoderKLFlux2.from_pretrained(
            model_id, subfolder="vae", device_map="auto", torch_dtype=torch.bfloat16
        )

        return transformer, vae

    def _add_lora_adapter(self, model, lora_config):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        if self.enable_gradient_checkpointing:
            model.enable_gradient_checkpointing()

        model.requires_grad_(False)
        self.vae.requires_grad_(False)

        config = LoraConfig(**lora_config)
        model = get_peft_model(model, config)

        return model

    def _prepare_for_fine_tunning(self):
        self.transformer = self._add_lora_adapter(self.transformer, self.lora_config)
        self.total_trainable_params = sum(
            [
                param.numel()
                for param in self.transformer.parameters()
                if param.requires_grad
            ]
        )
        print(
            f"Trainable: {self.total_trainable_params} / {self.total_parameter} = {self.total_trainable_params/self.total_parameter:.6f}"
        )

        self.optimizer = torch.optim.AdamW(
            params=[
                param for param in self.transformer.parameters() if param.requires_grad
            ],
            lr=self.lr,
        )

    def _loss_fn(self, vel_pred, vel_true):
        return torch.mean((vel_pred - vel_true) ** 2)

    def _prepare_text_ids(self, x, t_coord=None):
        B, L, _ = x.shape
        out_ids = []

        for i in range(B):
            t = torch.arange(1) if t_coord is None else t_coord[i]
            h = torch.arange(1)
            w = torch.arange(1)
            l = torch.arange(L)

            coords = torch.cartesian_prod(t, h, w, l)
            out_ids.append(coords)

        return torch.stack(out_ids)

    def _patchify_latent(self, latents):
        # B,C,H,W => B,C4,H/2,W/2
        B, C, H, W = latents.shape
        return (
            latents.reshape(B, C, H // 2, 2, W // 2, 2)
            .permute(0, 1, 3, 5, 2, 4)
            .reshape(B, C * 4, H // 2, W // 2)
        )

    def _pack_latents(self, latents):
        # B,C,H,W => B,H*W,C
        B, C, H, W = latents.shape
        return latents.permute(0, 2, 3, 1).reshape(B, H * W, C)

    def _prepare_latent_ids(self, latents):  # (B, C, H, W)
        batch_size, _, height, width = latents.shape

        t = torch.arange(1)  # [0] - time dimension
        h = torch.arange(height)
        w = torch.arange(width)
        l = torch.arange(1)  # [0] - layer dimension

        # Create position IDs: (H*W, 4)
        latent_ids = torch.cartesian_prod(t, h, w, l)

        # Expand to batch: (B, H*W, 4)
        latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

        return latent_ids

    # org image shape => B,3,H,W
    # vae encoded shape => B,32,H/8,W/8
    # patchified shape => B,128,H/16,W/16
    # packed shape => B,H/16*W/16,128  => transformer operate at this shape

    def _unpack_latents(self, latents):
        # B,N,C -> B,C,H,W
        B, N, C = latents.shape
        return latents.permute(0, 2, 1).reshape(
            B, self.patched_channels, self.patched_height, self.patched_width
        )

    def _unpatchify_latent(self, latents):
        # B,C4,h,w => B,C,H,W
        B, C4, h, w = latents.shape
        return (
            latents.reshape(B, C4 // 4, 2, 2, h, w)
            .permute(0, 1, 4, 2, 5, 3)
            .reshape(B, C4 // 4, 2 * h, 2 * w)
        )

    def _encode_image(self, x0):
        # x0 is clean and x1 is noise
        x0_latent = self.vae.encode(x0).latent_dist.mode()
        x0_patched = self._patchify_latent(x0_latent)

        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            x0.device, x0.dtype
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(x0.device, x0.dtype)

        x0_norm = (x0_patched - latents_bn_mean) / latents_bn_std

        return x0_norm

    def train(self, start_epoch=1,resume_training=False):
        if not resume_training:
            self._prepare_for_fine_tunning()

        for i in range(start_epoch, self.epochs + 1):
            print(f"Running Epoch : {i}/{self.epochs}")
            tot_loss = 0.0

            for batch in tqdm(self.dataloader, desc="Training"):
                timestamps = (
                    torch.rand((self.batch_size,))
                    .reshape((-1, 1, 1))
                    .to(self.device)
                    .to(torch.bfloat16)
                )

                original_image, text_encoding = batch

                text_encoding=text_encoding.to(self.device)
                x0 = original_image.to(torch.bfloat16).to(self.device)
                x0_latent = self._encode_image(x0)
                latent_ids = self._prepare_latent_ids(x0_latent).to(self.device)
                x0_latent = self._pack_latents(x0_latent)

                x1_latent = torch.randn_like(
                    x0_latent, device=self.device, dtype=torch.bfloat16
                )
                xt_latent = (1 - timestamps) * x0_latent + timestamps * x1_latent

                text_ids = self._prepare_text_ids(text_encoding).to(self.device)

                vel_pred = self.transformer(
                    hidden_states=xt_latent,
                    encoder_hidden_states=text_encoding,
                    timestep=timestamps.reshape((-1)).to(self.device, torch.bfloat16),
                    img_ids=latent_ids,
                    txt_ids=text_ids,
                    return_dict=False,
                    guidance=None,
                )[0]

                vel_true = x1_latent - x0_latent

                self.optimizer.zero_grad()
                loss = self._loss_fn(vel_pred, vel_true)
                loss.backward()
                self.optimizer.step()
                tot_loss += loss.item()


            self.plot_batch(epoch=i)

            print(
                f"Epoch {i}/{self.epochs} Average Loss: {tot_loss/len(self.dataloader):.2f}"
            )
            self.loss_history.append(tot_loss/len(self.dataloader))
            self._save_checkpoint(save_dir=f"{self.save_dir}/model", epoch=i)

        plt.plot(self.loss_history)
        plt.savefig(f"{self.save_dir}/loss_history.png")
        plt.close()

    def _save_checkpoint(self, save_dir, epoch):
        self.transformer.save_pretrained(f"{save_dir}/transformer")
        torch.save(
            {
                "completed_epoch": epoch,
                "optimizer_state_dict": self.optimizer.state_dict(),
                "loss_history": self.loss_history,
            },
            f"{save_dir}/checkpoint.pth",
        )

    def resume_training(self, checkpoint_dir):
        checkpoint = torch.load(checkpoint_dir+"/checkpoint.pth")

        self.transformer = prepare_model_for_kbit_training(self.transformer, use_gradient_checkpointing=False)
        if self.enable_gradient_checkpointing:
            self.transformer.enable_gradient_checkpointing()

        self.transformer = PeftModel.from_pretrained(self.transformer,f"{checkpoint_dir}/transformer",is_trainable=True)
        
        self.start_epoch = checkpoint["completed_epoch"]
        self.total_trainable_params = sum(
            [
                param.numel()
                for param in self.transformer.parameters()
                if param.requires_grad
            ]
        )
        print(
            f"Trainable: {self.total_trainable_params} / {self.total_parameter} = {self.total_trainable_params/self.total_parameter:.6f}"
        )

        self.optimizer = torch.optim.AdamW(
            params=[
                param for param in self.transformer.parameters() if param.requires_grad
            ],
            lr=self.lr,
        )

        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.loss_history = checkpoint["loss_history"]
        self.train(start_epoch=self.start_epoch + 1,resume_training=True)


    def plot_batch(
        self, n_step=4, inverse_transform=transforms.Normalize([-1.0], [2.0]), epoch=-1
    ):

        self.transformer.eval()
        self.vae.eval()

        original_image, text_encoding = next(iter(self.dataloader))
        text_ids = self._prepare_text_ids(text_encoding)

        # sampling latents
        latents = (
            torch.randn(
                (1, self.patched_channels, self.patched_height, self.patched_width)
            )
            .to(torch.bfloat16)
            .to(self.device)
        )
        latent_ids = self._prepare_latent_ids(latents).to(self.device)
        latents = self._pack_latents(latents)

        text_ids = self._prepare_text_ids(text_encoding).to(self.device)

        timestamps = torch.linspace(1.0, 0.0, n_step + 1).to(self.device)
        prev_t = torch.tensor(1.00)
        for t in timestamps[1:]:

            with torch.no_grad():
                vel_pred = self.transformer(
                    hidden_states=latents.to(self.transformer.device).to(
                        torch.bfloat16
                    ),
                    encoder_hidden_states=text_encoding.to(self.transformer.device).to(
                        torch.bfloat16
                    ),
                    timestep=prev_t.reshape((-1)).to(self.device, torch.bfloat16),
                    img_ids=latent_ids,
                    txt_ids=text_ids,
                    return_dict=False,
                    guidance=None,
                )[0]

                latents = latents - (prev_t - t) * vel_pred
                prev_t = t

        # denormalize
        latents = self._unpack_latents(latents)
        latents_bn_mean = self.vae.bn.running_mean.view(1, -1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents_bn_std = torch.sqrt(
            self.vae.bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps
        ).to(latents.device, latents.dtype)
        latents = latents * latents_bn_std + latents_bn_mean
        latents = self._unpatchify_latent(latents)

        generated_image = self.vae.decode(latents, return_dict=False)[0][0]

        plt.subplot(1, 2, 1)
        plt.imshow(
            inverse_transform(generated_image).clamp(0,1)
            .permute(1, 2, 0)
            .detach()
            .to("cpu")
            .to(torch.float32)
            .numpy()
        )
        plt.title("Generated Image")

        plt.subplot(1, 2, 2)
        plt.imshow(
            inverse_transform(original_image[0]).clamp(0,1)
            .permute(1, 2, 0)
            .detach()
            .to("cpu")
            .to(torch.float32)
            .numpy()
        )
        plt.title("Original Image")

        if epoch != -1:
            save_path = f"{self.save_dir}/images/{epoch}.png"
            plt.savefig(save_path, dpi=300)
        plt.show()
        plt.close()

        self.transformer.train()
        self.vae.train()


def custom_collate(inps):
    imgs = torch.stack([inp[0] for inp in inps])
    text_hidden_state = torch.stack([inp[1] for inp in inps])

    return imgs, text_hidden_state

def parse_args():

    parser=argparse.ArgumentParser()
    parser.add_argument("--hf_token",type=str)
    parser.add_argument("--cache_dir",type=str)
    parser.add_argument("--epochs",type=int,default=10)
    return parser.parse_args()

    

if __name__=="__main__":
    args = parse_args()
    if args.cache_dir:
        os.environ["HF_HOME"] = args.cache_dir
        os.environ["TRANSFORMERS_CACHE"] = args.cache_dir
    if args.hf_token:
        os.environ["HF_TOKEN"] = args.hf_token


    if os.environ["SM_CHANNEL_TRAIN"]:
        dataset_path=os.environ["SM_CHANNEL_TRAIN"]
        dataset = load_from_disk(dataset_path)
    else:
        dataset = load_dataset("Pankaj121212/alphonse_mucha_encoded_dataset")["train"]

    if os.environ["SM_MODEL_DIR"]:
        save_dir=os.environ["SM_MODEL_DIR"]
    else:
        save_dir=""

    print(f"Save directory is ",save_dir)
        
    ft=fineTunningFlux2Klein(dataset,custom_collate,save_dir=save_dir,epochs=args.epochs)
    ft.train()