import cv2 as cv
from datetime import datetime
import rioxarray as rio
import torch
import torch.nn as nn
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
import torch.nn.functional as F
import pickle
import random
from pathlib import Path
from tqdm import tqdm
import einops
import albumentations as A
import torchvision
import numpy as np
import math
from torchvision import transforms
import matplotlib.pyplot as plt
from torchmetrics import Accuracy, F1Score, JaccardIndex, Precision, Recall
from kornia.enhance import denormalize
import matplotlib as mpl
import gc

def free_gpu_memory():
    torch.cuda.empty_cache()
    gc.collect()


def read_sar_timeseries_data(path):
  '''
    Helper function to read SAR timeseries for a given sample
  '''
  for current_path in path.glob('*'):
      if "xml" not in current_path.name:
          if current_path.name.startswith("MS1_IVV"):
              # Get master ivv channel
              flood_vv = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)
          elif current_path.name.startswith("MS1_IVH"):
              # Get master ivh channel
              flood_vh = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)
              post_date = current_path.name.split("_")[-1][:-4]
              post_date = datetime.strptime(post_date, "%Y%m%d")

          elif current_path.name.startswith("SL1_IVV"):
              # Get slave1 vv channel
              sec1_vv = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)

          elif current_path.name.startswith("SL1_IVH"):
              # Get sl1 vh channel
              sec1_vh = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)
              pre1_date = current_path.name.split("_")[-1][:-4]
              pre1_date = datetime.strptime(pre1_date, "%Y%m%d")

          elif current_path.name.startswith("SL2_IVV"):
              # Get sl2 vv channel
              sec2_vv = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)

          elif current_path.name.startswith("SL2_IVH"):
              # Get sl2 vh channel
              sec2_vh = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)
              pre2_date = current_path.name.split("_")[-1][:-4]
              pre2_date = datetime.strptime(pre2_date, "%Y%m%d")

          elif current_path.name.startswith("MK0_MLU"):
              # Get mask of flooded/perm water pixels
              mask = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)

          elif current_path.name.startswith("MK0_DEM"):
              # Get DEM
              dem = rio.open_rasterio(current_path)
              nans = dem.isnull()
              dem = dem.to_numpy()

  return flood_vv, flood_vh, sec1_vv, sec1_vh, sec2_vv, sec2_vh, dem, mask, post_date, pre1_date, pre2_date


def get_1d_sincos_pos_embed_from_grid_torch(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    Taken from: https://github.com/sustainlab-group/SatMAE/blob/main/util/pos_embed.py#L66
    """
    assert embed_dim % 2 == 0
    omega = torch.arange(embed_dim // 2, dtype=float, device=pos.device)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = torch.sin(out) # (M, D/2)
    emb_cos = torch.cos(out) # (M, D/2)

    emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
    return emb.double()


# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


# classes


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                        FeedForward(dim, mlp_dim, dropout=dropout),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    def __init__(
        self,
        *,
        image_size,
        patch_size,
        num_classes,
        dim,
        depth,
        heads,
        mlp_dim,
        pool="cls",
        channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        temporal_embedding = False,
        input_length=None,
    ):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)
        self.temp_pos_dimension = dim
        assert (
            image_height % patch_height == 0 and image_width % patch_width == 0
        ), "Image dimensions must be divisible by the patch size."

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {"cls", "mean"}, "pool type must be either cls (cls token) or mean (mean pooling)"

        self.to_patch_embedding = nn.Sequential(
            Rearrange("b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim),
        )
        if temporal_embedding:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        else:
            self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.temporal_embedding = temporal_embedding
        if temporal_embedding:
            assert input_length is not None, "input_length must be provided if temporal_embedding is True"
            print("Temporal embedding is enabled")

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

    def forward(self, img, temporal_info=None):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, "1 1 d -> b 1 d", b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, : (n + 1)]
        if self.temporal_embedding:
            [post_year, post_month, post_day, pre1_year, pre1_month, pre1_day, pre2_year, pre2_month, pre2_day] = temporal_info
            post_embed = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(self.temp_pos_dimension//3, post_year).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.temp_pos_dimension//3, post_month).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.temp_pos_dimension//3, post_day).float()], dim=1).float()
            pre1_embed = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(self.temp_pos_dimension//3, pre1_year).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.temp_pos_dimension//3, pre1_month).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.temp_pos_dimension//3, pre1_day).float()], dim=1).float()
            pre2_embed = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(self.temp_pos_dimension//3, pre2_year).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.temp_pos_dimension//3, pre2_month).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.temp_pos_dimension//3, pre2_day).float()], dim=1).float()
            post_embed = repeat(post_embed, 'b d -> b p d', p=n).to(x.device)
            pre1_embed = repeat(pre1_embed, 'b d -> b p d', p=n).to(x.device)
            pre2_embed = repeat(pre2_embed, 'b d -> b p d', p=n).to(x.device)
            x[:, 1:] += post_embed
            x[:, 1:] += pre1_embed
            x[:, 1:] += pre2_embed

        x = self.dropout(x)

        x = self.transformer(x)

        if self.pool == "mean":
            x = x.mean(dim=1)
        else:
            return x[:, 1:]

        x = self.to_latent(x)
        return self.mlp_head(x)


class MAE(nn.Module):
    def __init__(
        self,
        *,
        encoder,
        decoder_dim,
        masking_ratio=0.75,
        decoder_depth=1,
        decoder_heads=8,
        decoder_dim_head=64,
        temporal_embedding = False,
        input_length=None
    ):
        super().__init__()
        assert (
            masking_ratio > 0 and masking_ratio < 1
        ), "masking ratio must be kept between 0 and 1"
        self.masking_ratio = masking_ratio

        # extract some hyperparameters and functions from encoder (vision transformer to be trained)

        self.encoder = encoder
        self.temporal_embedding = temporal_embedding
        self.input_length = input_length
        num_patches, encoder_dim = encoder.pos_embedding.shape[-2:]

        self.to_patch = encoder.to_patch_embedding[0]
        self.patch_to_emb = nn.Sequential(*encoder.to_patch_embedding[1:])

        pixel_values_per_patch = encoder.to_patch_embedding[2].weight.shape[-1]

        # decoder parameters
        self.decoder_dim = decoder_dim
        self.enc_to_dec = (
            nn.Linear(encoder_dim, decoder_dim)
            if encoder_dim != decoder_dim
            else nn.Identity()
        )
        self.mask_token = nn.Parameter(torch.randn(decoder_dim))
        self.decoder = Transformer(
            dim=decoder_dim,
            depth=decoder_depth,
            heads=decoder_heads,
            dim_head=decoder_dim_head,
            mlp_dim=decoder_dim * 4,
        )
        self.decoder_pos_emb = nn.Embedding(num_patches, decoder_dim)
        self.to_pixels = nn.Linear(decoder_dim, pixel_values_per_patch)

    def forward(self, img,temporal_info=None):
        device = img.device

        # get patches

        patches = self.to_patch(img)
        batch, num_patches, *_ = patches.shape

        # patch to encoder tokens and add positions

        tokens = self.patch_to_emb(patches)
        if self.encoder.pool == "cls":
            tokens += self.encoder.pos_embedding[:, 1 : (num_patches + 1)]
        elif self.encoder.pool == "mean":
            tokens += self.encoder.pos_embedding.to(device, dtype=tokens.dtype)

        if self.temporal_embedding:
            [post_year, post_month, post_day, pre1_year, pre1_month, pre1_day, pre2_year, pre2_month, pre2_day] = temporal_info
            post_embed = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(self.encoder.temp_pos_dimension//3, post_year).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.encoder.temp_pos_dimension//3, post_month).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.encoder.temp_pos_dimension//3, post_day).float()], dim=1).float()
            pre1_embed = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(self.encoder.temp_pos_dimension//3, pre1_year).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.encoder.temp_pos_dimension//3, pre1_month).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.encoder.temp_pos_dimension//3, pre1_day).float()], dim=1).float()
            pre2_embed = torch.cat([get_1d_sincos_pos_embed_from_grid_torch(self.encoder.temp_pos_dimension//3, pre2_year).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.encoder.temp_pos_dimension//3, pre2_month).float(),
                                    get_1d_sincos_pos_embed_from_grid_torch(self.encoder.temp_pos_dimension//3, pre2_day).float()], dim=1).float()
            post_embed = repeat(post_embed, 'b d -> b p d', p=num_patches).to(device)
            pre1_embed = repeat(pre1_embed, 'b d -> b p d', p=num_patches).to(device)
            pre2_embed = repeat(pre2_embed, 'b d -> b p d', p=num_patches).to(device)

            tokens += post_embed
            tokens += pre1_embed
            tokens += pre2_embed
        # calculate of patches needed to be masked, and get random indices, dividing it up for mask vs unmasked

        num_masked = int(self.masking_ratio * num_patches)
        rand_indices = torch.rand(batch, num_patches, device=device).argsort(dim=-1)
        masked_indices, unmasked_indices = (
            rand_indices[:, :num_masked],
            rand_indices[:, num_masked:],
        )

        # get the unmasked tokens to be encoded

        batch_range = torch.arange(batch, device=device)[:, None]
        tokens = tokens[batch_range, unmasked_indices]

        # get the patches to be masked for the final reconstruction loss

        masked_patches = patches[batch_range, masked_indices]

        # attend with vision transformer

        encoded_tokens = self.encoder.transformer(tokens)

        # project encoder to decoder dimensions, if they are not equal - the paper says you can get away with a smaller dimension for decoder

        decoder_tokens = self.enc_to_dec(encoded_tokens)

        # reapply decoder position embedding to unmasked tokens

        unmasked_decoder_tokens = decoder_tokens + self.decoder_pos_emb(
            unmasked_indices
        )

        # repeat mask tokens for number of masked, and add the positions using the masked indices derived above

        mask_tokens = repeat(self.mask_token, "d -> b n d", b=batch, n=num_masked)
        mask_tokens = mask_tokens + self.decoder_pos_emb(masked_indices)

        # concat the masked tokens to the decoder tokens and attend with decoder

        decoder_tokens = torch.zeros(
            batch, num_patches, self.decoder_dim, device=device
        )
        decoder_tokens[batch_range, unmasked_indices] = unmasked_decoder_tokens
        decoder_tokens[batch_range, masked_indices] = mask_tokens
        decoded_tokens = self.decoder(decoder_tokens)

        # splice out the mask tokens and project to pixel values

        mask_tokens = decoded_tokens[batch_range, masked_indices]
        pred_pixel_values = self.to_pixels(mask_tokens)

        # calculate reconstruction loss

        recon_loss = F.mse_loss(pred_pixel_values, masked_patches)
        return recon_loss


class SSLDataset(torch.utils.data.Dataset):
    def __init__(self, configs=None):
        # Added the 'size' parameter to RandomResizedCrop
        resized_crop = A.augmentations.RandomResizedCrop(height=224, width=224, size=(224,224), p=1.0, scale=(0.2, 1.0), interpolation=3)
        flip = A.augmentations.HorizontalFlip(p=0.5)
        self.augmentations = A.Compose([resized_crop, flip])
        self.root_path = configs["root_path"]
        self.configs = configs
        self.samples = []
        if not Path("ssl_samples.pkl").is_file():
            for folder_dir in tqdm(self.root_path.glob('*')):
                if int(folder_dir.name) in self.configs["test_acts"]:
                    print("Skipping test event:", folder_dir)
                    continue
                for subfolder_dir in folder_dir.glob('*'):
                    if ".gpkg" in subfolder_dir.name or ".gkpg" in subfolder_dir.name:
                        continue
                    for hashes_dir in subfolder_dir.glob('*'):
                        for hash_folder_dir in hashes_dir.glob('*'):
                            if hash_folder_dir.is_file():
                                self.samples.append(hashes_dir)
                                break
                            else:
                                self.samples.append(hash_folder_dir)
            with open("ssl_samples.pkl", "wb") as file:
                pickle.dump(self.samples, file)
        else:
            with open("ssl_samples.pkl", "rb") as file:
                self.samples = pickle.load(file)
        random.Random(999).shuffle(self.samples)
        self.num_examples = len(self.samples)

    def __len__(self):
        return self.num_examples

    def concat(self, image1, image2):
        image1_exp = np.expand_dims(image1, 0)  # vv
        image2_exp = np.expand_dims(image2, 0)  # vh

        if set(self.configs["channels"]) == set(["vv", "vh", "vh/vv"]):
            eps = 1e-7
            image = np.vstack((image1_exp, image2_exp, image2_exp / (image1_exp + eps)))  # vv, vh, vh/vv
        elif set(self.configs["channels"]) == set(["vv", "vh"]):
            image = np.vstack((image1_exp, image2_exp))  # vv, vh
        elif self.configs["channels"] == ["vh"]:
            image = image2_exp  # vh

        image = torch.from_numpy(image).float()

        if self.configs["clamp_input"] is not None:
            image = torch.clamp(image, min=0.0, max=self.configs["clamp_input"])
            image = torch.nan_to_num(image, self.configs["clamp_input"])
        else:
            image = torch.nan_to_num(image, 200)
        return image

    def __getitem__(self, index):
        path = self.samples[index]

        for current_path in path.glob('*'):
            if "xml" not in current_path.name:
                if current_path.name.startswith("MS1_IVV"):
                    # Get master ivv channel
                    flood_vv = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)

                    if flood_vv is None:
                        print(current_path)

                elif current_path.name.startswith("MS1_IVH"):
                    # Get master ivh channel
                    flood_vh = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)
                    post_date = current_path.name.split("_")[-1][:-4]
                    post_date = datetime.strptime(post_date, "%Y%m%d")

                elif current_path.name.startswith("SL1_IVV"):
                    # Get slave1 vv channel
                    sec1_vv = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)

                elif current_path.name.startswith("SL1_IVH"):
                    # Get sl1 vh channel
                    sec1_vh = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)
                    pre1_date = current_path.name.split("_")[-1][:-4]
                    pre1_date = datetime.strptime(pre1_date, "%Y%m%d")

                elif current_path.name.startswith("SL2_IVV"):
                    # Get sl2 vv channel
                    sec2_vv = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)

                elif current_path.name.startswith("SL2_IVH"):
                    # Get sl2 vh channel
                    sec2_vh = cv.imread(str(current_path), cv.IMREAD_ANYDEPTH)
                    pre2_date = current_path.name.split("_")[-1][:-4]
                    pre2_date = datetime.strptime(pre2_date, "%Y%m%d")

        # Concat channels
        flood = self.concat(flood_vv, flood_vh)
        pre_event_1 = self.concat(sec1_vv, sec1_vh)
        pre_event_2 = self.concat(sec2_vv, sec2_vh)

        # Hardcoded mean and std for all of Kuro Siwo (labeled + unlabeled part)
        mean = torch.tensor([0.0953, 0.0264])
        std = torch.tensor([0.0427, 0.0215])

        normalize = torchvision.transforms.Normalize(mean, std)
        flood = normalize(flood)
        pre_event_1 = normalize(pre_event_1)
        pre_event_2 = normalize(pre_event_2)

        image = torch.cat((flood, pre_event_1, pre_event_2), dim=0)
        image = einops.rearrange(image, "c h w -> h w c").numpy()
        transform = self.augmentations(image=image)
        image = transform["image"]
        image = einops.rearrange(image, "h w c -> c h w")
        image = torch.from_numpy(image)

        post_year = post_date.year
        post_month = post_date.month
        post_day = post_date.day
        pre1_year = pre1_date.year
        pre1_month = pre1_date.month
        pre1_day = pre1_date.day
        pre2_year = pre2_date.year
        pre2_month = pre2_date.month
        pre2_day = pre2_date.day

        return (
            image,
            flood,
            pre_event_1,
            pre_event_2,
            [post_year, post_month, post_day, pre1_year, pre1_month, pre1_day, pre2_year, pre2_month, pre2_day],
        )
    

def adjust_learning_rate(optimizer, epoch, configs):
    """Decay the learning rate with half-cycle cosine after warmup"""
    if epoch < configs["warmup_epochs"]:
        lr = configs["lr"] * epoch / configs["warmup_epochs"]
    else:
        lr = configs["min_lr"] + (configs["lr"] - configs["min_lr"]) * 0.5 * (
            1.0
            + math.cos(
                math.pi
                * (epoch - configs["warmup_epochs"])
                / (configs["epochs"] - configs["warmup_epochs"])
            )
        )
    for param_group in optimizer.param_groups:
        if "lr_scale" in param_group:
            param_group["lr"] = lr * param_group["lr_scale"]
        else:
            param_group["lr"] = lr
    return lr


def get_current_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        return param_group["lr"]


def train_epoch(loader, mae, optimizer, epoch, configs, scaler):
    mae.train()
    configs["num_steps_per_epoch"] = (
        configs["num_samples_per_epoch"] // configs["batch_size"]
    )
    num_steps_per_epoch = configs["num_steps_per_epoch"]

    device = configs["device"]
    disable = False

    # Set up gradient accumulation
    if configs["accumulate_gradients"] is not None:
        batches_to_accumulate = configs["accumulate_gradients"]

    running_loss = 0.0
    number_of_batches = 0.0
    data_loading_time = 0
    loader_iter = loader.__iter__()
    for idx in tqdm(range(num_steps_per_epoch), disable=disable):

        batch = loader_iter.__next__()

        if (
            configs["accumulate_gradients"] is None
            or (idx + 1) % batches_to_accumulate == 0
            or (idx + 1) == num_steps_per_epoch
        ):
            optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
            if (
                configs["accumulate_gradients"] is None
                or (idx + 1) % batches_to_accumulate == 0
                or (idx + 1) == num_steps_per_epoch
            ):
                # we use a per iteration (instead of per epoch) lr scheduler as done in official MAE implementation
                adjust_learning_rate(
                    optimizer, idx / num_steps_per_epoch + epoch, configs
                )

            image, flood, pre_event_1, pre_event_2, dates = batch

            image = image.to(device, non_blocking=True)

            loss = mae(image,dates)

        running_loss += loss.item()
        number_of_batches += 1

        if idx % 100 == 0:

            log_dict = {
                "Epoch": epoch,
                "Iteration": idx,
                "train loss": running_loss / number_of_batches,
            }
            running_loss = 0.0
            number_of_batches = 0.0

            log_dict["Current Learning Rate"] = get_current_learning_rate(optimizer)

            print(log_dict)

        # Scale loss according to gradient accumulation
        if configs["accumulate_gradients"] is not None:
            loss = loss / batches_to_accumulate

        # If gradient accumulation is enabled, update weights every batches_to_accumulate iterations.
        if (
            configs["accumulate_gradients"] is None
            or (idx + 1) % batches_to_accumulate == 0
            or (idx + 1) == num_steps_per_epoch
        ):
            if configs["mixed_precision"]:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()


def train_mae(configs,mae_configs):
    print("=" * 20)
    print("Initializing MAE")
    print("=" * 20)
    configs.update(mae_configs)
    if configs["mixed_precision"]:
        # Creates a GradScaler once at the beginning of training.
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    train_dataset = SSLDataset(configs=configs)

    loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=configs["batch_size"],
        shuffle=False,
        num_workers=configs["num_workers"],
        pin_memory=True,
        drop_last=False,
    )

    # Calculate effective batch size
    if configs["accumulate_gradients"] is None:
        accumulated_batches = 1
    else:
        accumulated_batches = configs["accumulate_gradients"]

    configs["lr"] = configs["learning_rate"]

    # Scale learning rate
    configs["lr"] = configs["lr"] * accumulated_batches
    print("=" * 20)
    print("Scaled Learning Rate: ", configs["lr"])
    print("=" * 20)

    num_channels = len(configs['channels']) * len(configs['inputs'])
    configs['num_channels']=num_channels
    v = ViT(
        image_size=configs["image_size"],
        patch_size=configs["patch_size"],
        channels=configs["num_channels"],
        num_classes=configs["num_classes"],
        dim=configs["dim"],
        depth=configs["depth"],
        heads=configs["heads"],
        mlp_dim=configs["mlp_dim"],
        temporal_embedding=configs["temporal_embedding"],
        input_length=len(configs["inputs"]),
    )
    model = MAE(
        encoder=v,
        masking_ratio=configs[
            "masked_ratio"
        ],  # the paper recommended 75% masked patches
        decoder_dim=configs["decoder_dim"],  # paper showed good results with just 512
        decoder_depth=configs["decoder_depth"],  # anywhere from 1 to 8
        decoder_heads=configs["decoder_heads"],  # attention heads for decoder
        temporal_embedding=configs["temporal_embedding"],
        input_length=len(configs["inputs"]),
    )
    model.to(configs["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=configs["lr"])

    if "start_epoch" not in configs or configs["start_epoch"] is None:
        start_epoch = 0
    else:
        start_epoch = configs["start_epoch"]
    for epoch in range(start_epoch, configs["epochs"]):
        train_epoch(loader, model, optimizer, epoch, configs, scaler)
        if epoch % 1 == 0:

            torch.save(
                model.state_dict(),
                    configs["checkpoint_path"] / f"mae_{epoch}.pt"
            )
            torch.save(
                model.encoder,
                    configs["checkpoint_path"] / f"vit_{epoch}.pt"
            )

    torch.save(
        model.encoder.state_dict(),
            configs["checkpoint_path"] / f"mae_vit_{configs['epochs']}.pt"
    )
    torch.save(
        model.encoder,
            configs["checkpoint_path"] / f"trained_vit_{configs['epochs']}.pt"
    )


class Decoder(nn.Module):
    def __init__(self, input_size, output_channels):
        super(Decoder, self).__init__()

        # Deconvolutional layers
        self.deconv1 = nn.ConvTranspose2d(input_size, 512, kernel_size=4, stride=2, padding=1)
        self.deconv21 = nn.ConvTranspose2d(512, 1024, kernel_size=4, stride=2, padding=1)

        self.relu = nn.ReLU()
        self.up = nn.Upsample(scale_factor=2)

        self.deconv2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(
            512, output_channels, kernel_size=4, stride=2, padding=1
        )

    def forward(self, x):
        x = self.deconv1(x)
        x = self.relu(x)

        x = self.deconv21(x)
        x = self.relu(x)

        #x = self.up(x)

        x = self.deconv2(x)

        x = self.relu(x)

        x = self.deconv3(x)

        return x

class FinetunerSegmentation(nn.Module):
    def __init__(self, encoder, configs=None, pool=False):
        super().__init__()
        self.configs = configs
        self.model = encoder
        self.model.pool = pool
        self.pool = pool
        if not self.pool:

          self.head = Decoder(
                    encoder.mlp_head.in_features, configs["num_classes"]
                )

        else:
            self.head = nn.Linear(
                encoder.mlp_head.in_features,
                configs["num_classes"] * configs["image_size"] * configs["image_size"],
            )
        self.model.mlp_head = nn.Identity()

    def forward(self, x,temporal_info=None):
        x = x
        img_size = 224
        GS = img_size // self.configs["finetuning_patch_size"]
        x = self.model(x, temporal_info=temporal_info)

        if self.pool == False:
            x = einops.rearrange(x, "b (h w) c -> b (c) h w", h=GS, w=GS)
            if not self.configs["decoder"]:
                upsample = nn.Upsample(size=(img_size, img_size), mode="bilinear")

                x = upsample(x)

        x = self.head(x)
        return x
    

def get_grids(pickle_path):
	'''
	Load the pickle file containing the flood events.
	'''
	if not pickle_path.exists():
		print("Pickle file not found! ", pickle_path)
		exit(2)
	with open(pickle_path, "rb") as file:
		grid_dict = pickle.load(file)
	return grid_dict


class Dataset(torch.utils.data.Dataset):
	def __init__(self, mode="train", configs=None):
		self.train_acts = configs["train_acts"]
		self.val_acts = configs["val_acts"]
		self.test_acts = configs["test_acts"]
		self.mode = mode
		self.configs = configs
		self.root_path = Path(self.configs["root_path"])
		self.non_valids = []

		# Keep some statistics per climatic zone and activation
		self.clz_stats = {1: 0, 2: 0, 3: 0}
		self.act_stats = {}

		 # Initialize valid activations and pickle paths
		if self.mode == "train":
			self.valid_acts = self.train_acts
			self.pickle_path = configs["train_pickle"]
		elif self.mode == "val":
			self.valid_acts = self.val_acts
			self.pickle_path = configs["test_pickle"]
		else:
			self.valid_acts = self.test_acts
			self.pickle_path = configs["test_pickle"]

		self.negative_grids = None
		total_grids = {}

		# Load the pickle files
		self.grids = get_grids(pickle_path=Path('pickle') / self.pickle_path)
		total_grids = self.grids

		all_activations = []
		all_activations.extend(self.train_acts)
		all_activations.extend(self.val_acts)
		all_activations.extend(self.test_acts)

		# Create a list containing metadata for all activations
		self.records = []
		for key in total_grids:
			activation = total_grids[key]["info"]["actid"]

			# If the activation is not in valid train/val/test activations, discard it
			if activation not in all_activations and activation not in self.non_valids:
				self.non_valids.append(activation)
				continue

			record = {}
			record["id"] = key
			record["path"] = total_grids[key]["path"]

			record["info"] = total_grids[key]["info"]
			record["clz"] = total_grids[key]["clz"]
			aoi = total_grids[key]["info"]["aoiid"]
			record["activation"] = activation

			# Update stats for the activations of this mode
			if activation in self.valid_acts:
				self.clz_stats[record["clz"]] += 1
				if activation in self.act_stats:
					self.act_stats[activation] += 1
				else:
					self.act_stats[activation] = 1

				self.records.append(record)

		print("Samples per Climatic zone for mode: ", self.mode)
		print(self.clz_stats)
		print("Samples per Activation for mode: ", self.mode)
		print(self.act_stats)

		self.num_examples = len(self.records)
		self.activations = set([record["activation"] for record in self.records])


	def __len__(self):
		return self.num_examples


	def concat(self, image1, image2):
		'''
		Selects particular bands per image, removes NaN values and clamps values if necessary.
		'''
		image1_exp = np.expand_dims(image1, 0)  # vv
		image2_exp = np.expand_dims(image2, 0)  # vh

		# Stack the channels
		if set(self.configs["channels"]) == set(["vv", "vh", "vh/vv"]):
			eps = 1e-7
			image = np.vstack((image1_exp, image2_exp, image2_exp / (image1_exp + eps)))  # vv, vh, vh/vv
		elif set(self.configs["channels"]) == set(["vv", "vh"]):
			image = np.vstack((image1_exp, image2_exp))  # vv, vh
		elif self.configs["channels"] == ["vh"]:
			image = image2_exp  # vh

		image = torch.from_numpy(image).float()

		# Convert NaN values to number
		if self.configs["clamp_input"] is not None:
			image = torch.clamp(image, min=0.0, max=self.configs["clamp_input"])
			image = torch.nan_to_num(image, self.configs["clamp_input"])
		else:
			image = torch.nan_to_num(image, 200)
		return image

	def scale_img(self, img):
		'''
		Normalizes the given image
		'''
		means = self.configs["data_mean"]
		stds = self.configs["data_std"]

		return means, stds, transforms.Normalize(means, stds)(img)


	def __getitem__(self, index):
		sample = self.records[index]

		path = sample["path"]
		path = self.root_path / path
		clz = sample["clz"]
		activation = sample["activation"]
		mask = None

		for file in path.glob('*'):
			if "xml" not in file.name:
				if file.name.startswith("MK0_MLU"):
					# Get mask of flooded/perm water pixels
					mask = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
				elif file.name.startswith("MK0_MNA"):
					# Get mask of valid pixels
					valid_mask = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
				elif file.name.startswith("MS1_IVV"):
					# Get master ivv channel
					flood_vv = cv.imread(str(file), cv.IMREAD_ANYDEPTH)

				elif file.name.startswith("MS1_IVH"):
					# Get master ivh channel
					flood_vh = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
					post_date = file.name.split("_")[-1][:-4]
					post_date = datetime.strptime(post_date, "%Y%m%d")

				elif file.name.startswith("SL1_IVV"):
					# Get slave1 vv channel
					sec1_vv = cv.imread(str(file), cv.IMREAD_ANYDEPTH)

				elif file.name.startswith("SL1_IVH"):
					# Get sl1 vh channel
					sec1_vh = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
					pre1_date = file.name.split("_")[-1][:-4]
					pre1_date = datetime.strptime(pre1_date, "%Y%m%d")

				elif file.name.startswith("SL2_IVV"):
					# Get sl2 vv channel
					sec2_vv = cv.imread(str(file), cv.IMREAD_ANYDEPTH)

				elif file.name.startswith("SL2_IVH"):
					# Get sl2 vh channel
					sec2_vh = cv.imread(str(file), cv.IMREAD_ANYDEPTH)
					pre2_date = file.name.split("_")[-1][:-4]
					pre2_date = datetime.strptime(pre2_date, "%Y%m%d")

				elif file.name.startswith("MK0_DEM"):
					if self.configs['dem'] and not self.configs['slope']:
						# Get DEM
						dem = rio.open_rasterio(file)
						nans = dem.isnull()
						if nans.any():
							dem = dem.rio.interpolate_na()
							nans = dem.isnull()

						nodata = dem.rio.nodata
						dem = dem.to_numpy()
						if not self.configs["dem"] and self.configs["slope"]:
							print(
								"To return the slope the DEM option must be enabled. Validate the config file!"
							)
							exit(2)

						if self.configs["scale_input"]:
							normalization = transforms.Normalize(
								mean=self.configs["dem_mean"],
								std=self.configs["dem_std"],
							)
							dem = normalization(torch.from_numpy(dem))
					elif self.configs['dem'] and self.configs['slope']:
						# Get slope
						slope_path = Path(self.configs['slope_path']) / sample['path']
						dem = rio.open_rasterio(list(slope_path.glob('*'))[0]).to_numpy()

						if self.configs["scale_input"]:
							normalization = transforms.Normalize(
								mean=self.configs["slope_mean"],
								std=self.configs["slope_std"],
							)
							dem = normalization(torch.from_numpy(dem))

		# Concat channels
		flood = self.concat(flood_vv, flood_vh)
		pre_event_1 = self.concat(sec1_vv, sec1_vh)
		pre_event_2 = self.concat(sec2_vv, sec2_vh)

		if mask is None:
			mask = np.zeros((224, 224))

		mask = torch.from_numpy(mask).long()

		valid_mask = torch.from_numpy(valid_mask)

		mask = mask.long()

		# Scale images if necessary
		if self.configs["scale_input"]:
			valid_mask = valid_mask == 1
			flood_scale_var_1, flood_scale_var_2, flood = self.scale_img(flood)
			pre1_scale_var_1, pre1_scale_var_2, pre_event_1 = self.scale_img(pre_event_1)
			pre2_scale_var_1, pre2_scale_var_2, pre_event_2 = self.scale_img(pre_event_2)

		if not self.configs["dem"]:
			if self.configs["scale_input"]:
				if "temporal_embedding" in self.configs and self.configs["temporal_embedding"]:
						post_year = post_date.year
						post_month = post_date.month
						post_day = post_date.day
						pre1_year = pre1_date.year
						pre1_month = pre1_date.month
						pre1_day = pre1_date.day
						pre2_year = pre2_date.year
						pre2_month = pre2_date.month
						pre2_day = pre2_date.day
						return (
										flood_scale_var_1,
										flood_scale_var_2,
										flood,
										mask,
										pre1_scale_var_1,
										pre1_scale_var_2,
										pre_event_1,
										pre2_scale_var_1,
										pre2_scale_var_2,
										pre_event_2,
										clz,
										activation,
										[post_year, post_month, post_day, pre1_year, pre1_month, pre1_day, pre2_year, pre2_month, pre2_day]
								)
				return (
					flood_scale_var_1,
					flood_scale_var_2,
					flood,
					mask,
					pre1_scale_var_1,
					pre1_scale_var_2,
					pre_event_1,
					pre2_scale_var_1,
					pre2_scale_var_2,
					pre_event_2,
					clz,
					activation,
				)
			else:
				return flood, mask, pre_event_1, pre_event_2, clz, activation
		else:
			if self.configs["scale_input"]:
				return (
					flood_scale_var_1,
					flood_scale_var_2,
					flood,
					mask,
					pre1_scale_var_1,
					pre1_scale_var_2,
					pre_event_1,
					pre2_scale_var_1,
					pre2_scale_var_2,
					pre_event_2,
					dem,
					clz,
					activation,
				)
			else:
				return flood, mask, pre_event_1, pre_event_2, dem, clz, activation
               

CLASS_LABELS = {0: "No water", 1: "Permanent Waters", 2: "Floods", 3: "Invalid pixels"}

def train_semantic_segmentation(model, train_loader, val_loader, test_loader, configs, model_configs):

  #Define metrics
  accuracy, fscore, precision, recall, iou = create_metrics(configs)

  # Define loss function
  criterion = nn.CrossEntropyLoss(ignore_index=3).to(configs["device"])

  # Define optimizer
  optimizer = torch.optim.Adam(model.parameters(), lr=model_configs["learning_rate"])

  # Define LR scheduling
  lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(train_loader))

  model.to(configs["device"])
  best_val = 0.0
  best_stats = {}

  if configs["mixed_precision"]:
    # Creates a GradScaler once at the beginning of training.
    scaler = torch.cuda.amp.GradScaler()

  for epoch in range(configs["epochs"]):
    model.train()

    train_loss = 0.0

    for index, batch in tqdm(enumerate(train_loader), total=len(train_loader), desc="Epoch " + str(epoch)):
      optimizer.zero_grad()
      with torch.cuda.amp.autocast(enabled=configs["mixed_precision"]):
        if configs["scale_input"] is not None:
          if not configs["dem"]:
            if "temporal_embedding" in configs and configs["temporal_embedding"]:
              (
                  image_scale_var_1,
                  image_scale_var_2,
                  image,
                  mask,
                  pre_scale_var_1,
                  pre_scale_var_2,
                  pre_event,
                  pre2_scale_var_1,
                  pre2_scale_var_2,
                  pre_event_2,
                  clz,
                  activation,
                  temporal_info,
                ) = batch
            else:
              (
                image_scale_var_1,
                image_scale_var_2,
                image,
                mask,
                pre_scale_var_1,
                pre_scale_var_2,
                pre_event,
                pre2_scale_var_1,
                pre2_scale_var_2,
                pre_event_2,
                clz,
                activation,
              ) = batch
          else:
            (
              image_scale_var_1,
              image_scale_var_2,
              image,
              mask,
              pre_scale_var_1,
              pre_scale_var_2,
              pre_event,
              pre2_scale_var_1,
              pre2_scale_var_2,
              pre_event_2,
              dem,
              clz,
              activation,
            ) = batch
        else:
          if not configs["dem"]:
            image, mask, pre_event, pre_event_2, clz, activation = batch
          else:
            (
              image,
              mask,
              pre_event,
              pre_event_2,
              dem,
              clz,
              activation,
            ) = batch

        image = image.to(configs["device"])
        mask = mask.to(configs["device"])

        # Define input according to the provided configurations
        if configs["dem"]:
          dem = dem.to(configs["device"])
          image = torch.cat((image, dem), dim=1)

        if configs["inputs"] == ["post_event"]:
          output = model(image)
        elif set(configs["inputs"]) == set(["pre_event_1", "post_event"]):
          pre_event = pre_event.to(configs["device"])
          image = torch.cat((image, pre_event), dim=1)
          output = model(image)
        elif set(configs["inputs"]) == set(["pre_event_2", "post_event"]):
          pre_event_2 = pre_event_2.to(configs["device"])
          image = torch.cat((image, pre_event_2), dim=1)
          output = model(image)
        elif set(configs["inputs"]) == set(["pre_event_1", "pre_event_2", "post_event"]):
          pre_event = pre_event.to(configs["device"])
          image = torch.cat((image, pre_event), dim=1)
          image = torch.cat((image, pre_event_2.to(configs["device"])), dim=1)
          if "temporal_embedding" in configs and configs["temporal_embedding"]:
            output = model(image, temporal_info)
          else:
            output = model(image)
        else:
          print('Invalid configuration for "inputs". Exiting...')
          exit(1)

        predictions = output.argmax(1)
        loss = criterion(output, mask)

        train_loss += loss.item() * image.size(0)

      if configs["mixed_precision"]:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
      else:
        loss.backward()
        optimizer.step()

      acc = accuracy(predictions, mask)
      score = fscore(predictions, mask)
      prec = precision(predictions, mask)
      rec = recall(predictions, mask)
      ious = iou(predictions, mask)
      mean_iou = (ious[0] + ious[1] + ious[2]) / 3

      if index % configs["print_frequency"] == 0:
          print(f"Epoch: {epoch}")
          print(f"Iteration: {index}")
          print(f"Train Loss: {loss.item()}")
          print(f"Train Accuracy ({CLASS_LABELS[0]}): {100 * acc[0].item()}")
          print(f"Train Accuracy ({CLASS_LABELS[1]}): {100 * acc[1].item()}")
          print(f"Train Accuracy ({CLASS_LABELS[2]}): {100 * acc[2].item()}")
          print(f"Train F-Score ({CLASS_LABELS[0]}): {100 * score[0].item()}")
          print(f"Train F-Score ({CLASS_LABELS[1]}): {100 * score[1].item()}")
          print(f"Train F-Score ({CLASS_LABELS[2]}): {100 * score[2].item()}")
          print(
            f"Train Precision ({CLASS_LABELS[0]}): {100 * prec[0].item()}"
          )
          print(
            f"Train Precision ({CLASS_LABELS[1]}): {100 * prec[1].item()}"
          )
          print(
            f"Train Precision ({CLASS_LABELS[2]}): {100 * prec[2].item()}"
          )
          print(f"Train Recall ({CLASS_LABELS[0]}): {100 * rec[0].item()}")
          print(f"Train Recall ({CLASS_LABELS[1]}): {100 * rec[1].item()}")
          print(f"Train Recall ({CLASS_LABELS[2]}): {100 * rec[2].item()}")
          print(f"Train IoU ({CLASS_LABELS[0]}): {100 * ious[0].item()}")
          print(f"Train IoU ({CLASS_LABELS[1]}): {100 * ious[1].item()}")
          print(f"Train IoU ({CLASS_LABELS[2]}): {100 * ious[2].item()}")
          print(f"Train MeanIoU: {mean_iou * 100}")
          print(f"lr: {lr_scheduler.get_last_lr()[0]}")

    # Update LR scheduler
    lr_scheduler.step()

    # Evaluate on validation set
    model.eval()
    val_acc, val_score, miou = eval_semantic_segmentation(
      model,
      val_loader,
      settype="Val",
      configs=configs,
      model_configs=model_configs,
    )

    if miou > best_val:
      print(f"Epoch: {epoch}")
      print(f"New best validation mIOU: {miou}")
      print(f"Saving model to: {Path(configs['checkpoint_path']) / 'best_segmentation.pt'}")
      best_val = miou
      best_stats["miou"] = best_val
      best_stats["epoch"] = epoch
      torch.save(model, Path(configs["checkpoint_path"]) / "best_segmentation.pt")
    torch.save(model, Path(configs["checkpoint_path"]) / "last_segmentation.pt")


def eval_semantic_segmentation(model, loader, configs=None, settype="Test", model_configs=None):
  # Initialize metrics
  accuracy, fscore, precision, recall, iou = create_metrics(configs)

  # Define loss function
  criterion = nn.CrossEntropyLoss(ignore_index=3).to(configs["device"])

  model.to(configs["device"])

  total_loss = 0.0

  for index, batch in tqdm(enumerate(loader), total=len(loader)):
    with torch.cuda.amp.autocast(enabled=False):
      with torch.no_grad():
        if configs["scale_input"]:
          if not configs["dem"]:
            if "temporal_embedding" in configs and configs["temporal_embedding"]:
              (
                image_scale_var_1,
                image_scale_var_2,
                image,
                mask,
                pre_scale_var_1,
                pre_scale_var_2,
                pre_event,
                pre2_scale_var_1,
                pre2_scale_var_2,
                pre_event_2,
                clz,
                activ,
                temporal_info
              ) = batch
            else:
              (
                image_scale_var_1,
                image_scale_var_2,
                image,
                mask,
                pre_scale_var_1,
                pre_scale_var_2,
                pre_event,
                pre2_scale_var_1,
                pre2_scale_var_2,
                pre_event_2,
                clz,
                activ,
            ) = batch
          else:
            (
              image_scale_var_1,
              image_scale_var_2,
              image,
              mask,
              pre_scale_var_1,
              pre_scale_var_2,
              pre_event,
              pre2_scale_var_1,
              pre2_scale_var_2,
              pre_event_2,
              dem,
              clz,
              activ,
          ) = batch
        else:
          if not configs["dem"]:
            image, mask, pre_event, pre_event_2, clz, activ = batch
          else:
            image, mask, pre_event, pre_event_2, dem, clz, activ = batch

        image = image.to(configs["device"])
        mask = mask.to(configs["device"])


        #Define inputs
        if configs["dem"]:
          dem = dem.to(configs["device"])
          image = torch.cat((image, dem), dim=1)

        if configs["inputs"] == ["post_event"]:
          output = model(image)
        elif set(configs["inputs"]) == set(["pre_event_1", "post_event"]):
          pre_event = pre_event.to(configs["device"])
          image = torch.cat((image, pre_event), dim=1)
          output = model(image)
        elif set(configs["inputs"]) == set(["pre_event_2", "post_event"]):
          pre_event_2 = pre_event_2.to(configs["device"])
          image = torch.cat((image, pre_event_2), dim=1)
          output = model(image)
        elif set(configs["inputs"]) == set(["pre_event_1", "pre_event_2", "post_event"]):
          pre_event = pre_event.to(configs["device"])
          image = torch.cat((image, pre_event), dim=1)
          image = torch.cat((image, pre_event_2.to(configs["device"])), dim=1)
          if "temporal_embedding" in configs and configs["temporal_embedding"]:
            output = model(image, temporal_info)
          else:
            output = model(image)
        else:
          print('Invalid configuration for "inputs". Exiting...')
          exit(1)

        loss = criterion(output, mask)
        total_loss += loss.item() * image.size(0)
        predictions = output.argmax(1)

        accuracy(predictions, mask)
        fscore(predictions, mask)
        precision(predictions, mask)
        recall(predictions, mask)
        iou(predictions, mask)

  # Calculate average loss over an epoch
  val_loss = total_loss / len(loader)

  acc = accuracy.compute()
  score = fscore.compute()
  prec = precision.compute()
  rec = recall.compute()
  ious = iou.compute()
  mean_iou = ious[:3].mean()

  print(f'\n{"="*20}')
  print(f"{settype} Loss: {val_loss}")
  print(f"{settype} Accuracy ({CLASS_LABELS[0]}): {100 * acc[0].item()}")
  print(f"{settype} Accuracy ({CLASS_LABELS[1]}): {100 * acc[1].item()}")
  print(f"{settype} Accuracy ({CLASS_LABELS[2]}): {100 * acc[2].item()}")
  print(f"{settype} F-Score ({CLASS_LABELS[0]}): {100 * score[0].item()}")
  print(f"{settype} F-Score ({CLASS_LABELS[1]}): {100 * score[1].item()}")
  print(f"{settype} F-Score ({CLASS_LABELS[2]}): {100 * score[2].item()}")
  print(f"{settype} Precision ({CLASS_LABELS[0]}): {100 * prec[0].item()}")
  print(f"{settype} Precision ({CLASS_LABELS[1]}): {100 * prec[1].item()}")
  print(f"{settype} Precision ({CLASS_LABELS[2]}): {100 * prec[2].item()}")
  print(f"{settype} Recall ({CLASS_LABELS[0]}): {100 * rec[0].item()}")
  print(f"{settype} Recall ({CLASS_LABELS[1]}): {100 * rec[1].item()}")
  print(f"{settype} Recall ({CLASS_LABELS[2]}): {100 * rec[2].item()}")
  print(f"{settype} IoU ({CLASS_LABELS[0]}): {100 * ious[0].item()}")
  print(f"{settype} IoU ({CLASS_LABELS[1]}): {100 * ious[1].item()}")
  print(f"{settype} IoU ({CLASS_LABELS[2]}): {100 * ious[2].item()}")
  print(f"{settype} MeanIoU: {mean_iou * 100}")
  print(f'\n{"="*20}')

  return 100 * acc, 100 * score[:3].mean(), 100 * mean_iou


def visualize_predictions(test_dataset,model,configs):
  cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', [(0, 0, 0, 10), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529, 1.0), (0.8647058823529412, 0.30980392156862746, 0.45882352941176474, 1.0), (0.00000, 1.00000, 0.60000, 1.0)], 4)
  configs['device'] = "cpu"
  model.to(configs['device'])
  test_loader = torch.utils.data.DataLoader(
      test_dataset,
      batch_size=1,
      shuffle=False,
      num_workers=1,
      pin_memory=True,
      drop_last=False,
  )
  for i,batch in enumerate(test_loader):

    if configs['scale_input'] is not None:
        if not configs['dem']:
          if "temporal_embedding" in configs and configs["temporal_embedding"]:
              (
                  image_scale_var_1,
                  image_scale_var_2,
                  image,
                  mask,
                  pre_scale_var_1,
                  pre_scale_var_2,
                  pre_event,
                  pre2_scale_var_1,
                  pre2_scale_var_2,
                  pre_event_2,
                  clz,
                  activation,
                  temporal_info,
                ) = batch
          else:
            image_scale_var_1, image_scale_var_2, image, mask, pre_scale_var_1, \
            pre_scale_var_2, pre_event, pre2_scale_var_1, pre2_scale_var_2, pre_event_2, clz, activ = batch
        else:
            image_scale_var_1, image_scale_var_2, image, mask, pre_scale_var_1, \
            pre_scale_var_2, pre_event, pre2_scale_var_1, pre2_scale_var_2, pre_event_2, dem, clz, activ = batch

    else:
        image, mask, pre_event, pre_event_2, dem, clz, activ = batch
        pre_event = pre_event
    if mask.sum()==0:
      continue
    flood = image.clone()
    image = image.to(configs['device'])

    if configs['dem']:
        dem = dem.to(configs['device'])
    if set(configs['inputs']) == set(['pre_event_1', 'post_event']):
        pre_event = pre_event.to(configs['device'])
        image = torch.cat((image,pre_event),dim=1)
    elif set(configs['inputs']) == set(['pre_event_2', 'post_event']):
        pre_event_2 = pre_event_2.to(configs['device'])
        image = torch.cat((image,pre_event_2),dim=1)

    elif set(configs['inputs']) == set(['pre_event_1', 'pre_event_2', 'post_event']):
        pre_event = pre_event.to(configs['device'])
        image = torch.cat((image, pre_event), dim=1)
        image  = torch.cat((image,pre_event_2.to(configs['device'])),dim=1)

    if "temporal_embedding" in configs and configs["temporal_embedding"]:
        output = model(image, temporal_info)
    else:
        output = model(image)
    predictions = output.argmax(1)

    flood = denormalize(flood,torch.tensor(configs['data_mean']),torch.tensor(configs['data_std']))
    pre_event = denormalize(pre_event,torch.tensor(configs['data_mean']),torch.tensor(configs['data_std']))
    pre_event_2 = denormalize(pre_event_2,torch.tensor(configs['data_mean']),torch.tensor(configs['data_std']))

    post_vv = flood.squeeze()[0]
    post_vh = flood.squeeze()[1]
    mask = mask.squeeze()
    pre_event_vv = pre_event.squeeze()[0]
    pre_event_vh = pre_event.squeeze()[1]
    pre_event_2_vv = pre_event_2.squeeze()[0]
    pre_event_2_vh = pre_event_2.squeeze()[1]

    if configs['dem']:
       dem = dem.squeeze()

    fig, axes = plt.subplots(1, 8, figsize=(30, 15), num=1, clear=True)

    axes[0].imshow(pre_event_vv,cmap='gray')

    axes[0].set_title('Pre event 1 (VV)',fontsize=28)

    axes[1].imshow(pre_event_vh,cmap='gray')

    axes[1].set_title('Pre event 1 (VH)',fontsize=28)

    axes[ 2].imshow(pre_event_2_vv,cmap='gray')

    axes[2].set_title('Pre event 2 (VV)',fontsize=28)

    axes[3].imshow(pre_event_2_vh,cmap='gray',)

    axes[3].set_title('Pre event 2 (VH)',fontsize=28)

    axes[ 4].imshow(post_vv,cmap='gray')

    axes[4].set_title('Post event 3 (VV)',fontsize=28)

    axes[ 5].imshow(post_vh,cmap='gray')

    axes[5].set_title('Post event 3 (VH)',fontsize=28)

    mask_ax = axes[6].imshow(mask, vmin=0, vmax=3, cmap=cmap)
    axes[6].set_title('Ground truth',fontsize=28)
    axes[7].imshow(predictions.squeeze(0),vmin=0,vmax=3,cmap=cmap)
    axes[7].set_title('Predictions',fontsize=28)

    mask = mask.numpy()

    # Remove all axis labels
    for irow in range(1):
        for icol in range(8):
            axes[icol].set_xticks([])
            axes[icol].set_yticks([])
            axes[icol].spines['top'].set_visible(False)
            axes[ icol].spines['right'].set_visible(False)
            axes[ icol].spines['bottom'].set_visible(False)
            axes[ icol].spines['left'].set_visible(False)


    fig.tight_layout()
    plt.show()
    break
  

def create_metrics(configs):
    accuracy = Accuracy(
      task="multiclass",
      average="none",
      multidim_average="global",
      num_classes=configs["num_classes"] + 1,
      ignore_index=3,
    ).to(configs["device"])

    fscore = F1Score(
      task="multiclass",
      num_classes=configs["num_classes"] + 1,
      average="none",
      multidim_average="global",
      ignore_index=3,
    ).to(configs["device"])

    precision = Precision(
      task="multiclass",
      average="none",
      num_classes=configs["num_classes"] + 1,
      multidim_average="global",
      ignore_index=3,
    ).to(configs["device"])

    recall = Recall(
      task="multiclass",
      average="none",
      num_classes=configs["num_classes"] + 1,
      multidim_average="global",
      ignore_index=3,
    ).to(configs["device"])

    iou = JaccardIndex(
      task="multiclass",
      num_classes=configs["num_classes"] + 1,
      average="none",
      ignore_index=3,
    ).to(configs["device"])
    return accuracy, fscore, precision, recall, iou