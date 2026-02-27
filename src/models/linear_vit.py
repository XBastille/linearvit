import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    """
    ReLU-based linear attention from L2ViT.

    Complexity: O(N * C^2 / h) instead of O(N^2 * C).
    Uses phi(x) = ReLU(x) as kernel feature map to guarantee non-negative attention.
    Includes learnable temperature and clamped denominator for stable training.
    """

    def __init__(self, dim, num_heads=4, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) * self.scale)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)
        self.act = nn.ReLU()

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, h, N, d)
        q, k, v = qkv.unbind(0)
        # Apply ReLU kernel (non-negative property from L2ViT paper)
        q = self.act(q)
        k = self.act(k)
        # KV-first formulation: O(N * d^2)
        # denom = Q @ (K^T @ 1) -- normalization
        denom = torch.clamp(
            torch.einsum("bhnd,bhnd->bhn", q, k.sum(dim=2, keepdim=True).expand_as(q)),
            min=1e2,
        ).unsqueeze(-1)
        # attn_out = Q @ (K^T @ V) * temperature / denom
        kv = torch.einsum("bhnd,bhne->bhde", k, v)  # (B, h, d, d)

        kv = kv * self.temperature.unsqueeze(
            0
        )  # (1, h, 1, 1) broadcast with (B, h, d, d)

        out = torch.einsum("bhnd,bhde->bhne", q, kv)  # (B, h, N, d)
        out = out / denom
        out = out.transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class LocalConcentrationModule(nn.Module):
    """
    Local Concentration Module (LCM) from L2ViT.
    Two depthwise conv layers to concentrate dispersive linear attention on local features.
    """

    def __init__(self, dim, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2

        self.conv1 = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )

        self.act = nn.GELU()
        self.bn = nn.BatchNorm2d(dim)

        self.conv2 = nn.Conv2d(
            dim, dim, kernel_size=kernel_size, padding=padding, groups=dim
        )

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.conv1(x)
        x = self.act(x)
        x = self.bn(x)
        x = self.conv2(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MLP(nn.Module):
    """Standard 2-layer FFN with GELU activation."""

    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LinearViTBlock(nn.Module):
    """
    Transformer block with linear attention and local concentration module.

    Structure:
      x -> LayerNorm -> LinearAttention -> residual
        -> LCM -> residual
        -> LayerNorm -> MLP -> residual
    """

    def __init__(
        self, dim, num_heads=4, mlp_ratio=4.0, drop=0.0, drop_path=0.0, lcm_kernel=7
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = LinearAttention(dim, num_heads=num_heads, proj_drop=drop)
        self.norm_lcm = nn.LayerNorm(dim)
        self.lcm = LocalConcentrationModule(dim, kernel_size=lcm_kernel)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio=mlp_ratio, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x, H, W, skip_lcm=False):
        # Linear attention + residual
        x = x + self.drop_path(self.attn(self.norm1(x)))
        # LCM + residual (skip when tokens don't form a full spatial grid, e.g. MAE)
        if not skip_lcm:
            x = x + self.drop_path(self.lcm(self.norm_lcm(x), H, W))
        # MLP + residual
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class DropPath(nn.Module):
    """Stochastic depth for regularization."""

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x

        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = torch.bernoulli(torch.full(shape, keep, device=x.device, dtype=x.dtype))
        return x / keep * mask


class PatchEmbedding(nn.Module):
    """Convert image into patch token sequence via convolution."""

    def __init__(self, in_channels=3, embed_dim=256, patch_size=8):
        super().__init__()
        self.patch_size = patch_size

        self.proj = nn.Conv2d(
            in_channels, embed_dim, kernel_size=patch_size, stride=patch_size
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, D, H/P, W/P)
        H, W = x.shape[2], x.shape[3]
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = self.norm(x)
        return x, H, W


class LinearViT(nn.Module):
    """
    Linear Attention Vision Transformer with dual heads for
    mass regression and classification.

    Architecture:
      Image -> PatchEmbedding -> + PosEmbed -> [LinearViTBlock] x depth
      -> LayerNorm -> GlobalAvgPool -> cls_head, reg_head
    """

    def __init__(
        self,
        in_channels=3,
        img_size=125,
        patch_size=8,
        embed_dim=256,
        depth=6,
        num_heads=4,
        mlp_ratio=4.0,
        num_classes=2,
        drop_rate=0.1,
        drop_path_rate=0.1,
        lcm_kernel=7,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        num_patches = (img_size // patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.pos_drop = nn.Dropout(drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]

        self.blocks = nn.ModuleList(
            [
                LinearViTBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    drop_path=dpr[i],
                    lcm_kernel=lcm_kernel,
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.reg_head = nn.Linear(embed_dim, 1)
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.apply(self._init_module)

    def _init_module(self, m):

        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)

            if m.bias is not None:
                nn.init.zeros_(m.bias)

        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward_features(self, x):
        """Extract feature representation from image."""
        x, H, W = self.patch_embed(x)  # (B, N, D)
        if x.shape[1] != self.pos_embed.shape[1]:
            pos = self._interpolate_pos(x.shape[1], H, W)

        else:
            pos = self.pos_embed

        x = self.pos_drop(x + pos)

        for blk in self.blocks:
            x = blk(x, H, W)

        x = self.norm(x)
        x = x.mean(dim=1)  # (B, D)
        return x

    def _interpolate_pos(self, num_tokens, H, W):
        """Interpolate positional embeddings for different image sizes."""
        N = self.pos_embed.shape[1]
        if num_tokens == N:
            return self.pos_embed

        old_size = int(math.sqrt(N))
        pos = self.pos_embed.reshape(1, old_size, old_size, -1).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(H, W), mode="bilinear", align_corners=False)
        pos = pos.permute(0, 2, 3, 1).reshape(1, H * W, -1)
        return pos

    def forward_mae(self, x, mask_ratio=0.75):
        """
        MAE-style forward: mask patches, encode only visible ones.

        Args:
            x: (B, C, H_img, W_img) input images
            mask_ratio: fraction of patches to mask (0.75 = 75% masked)

        Returns:
            visible_tokens: (B, N_vis, D) encoded visible patch tokens
            mask: (B, N) boolean mask (True = masked)
            ids_restore: (B, N) indices to restore original order
            H, W: patch grid dimensions
        """
        tokens, H, W = self.patch_embed(x)  # (B, N, D)
        B, N, D = tokens.shape
        if N != self.pos_embed.shape[1]:
            pos = self._interpolate_pos(N, H, W)

        else:
            pos = self.pos_embed

        tokens = tokens + pos
        N_vis = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=tokens.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :N_vis]  # (B, N_vis)
        visible_tokens = torch.gather(
            tokens, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D)
        )  # (B, N_vis, D)

        # Create mask: True = masked, False = visible
        mask = torch.ones(B, N, device=tokens.device, dtype=torch.bool)
        mask.scatter_(1, ids_keep, False)

        # Encode visible tokens only (skip LCM since tokens aren't a full spatial grid)
        visible_tokens = self.pos_drop(visible_tokens)

        for blk in self.blocks:
            visible_tokens = blk(visible_tokens, H, W, skip_lcm=True)

        visible_tokens = self.norm(visible_tokens)

        return visible_tokens, mask, ids_restore, H, W

    def forward(self, x):
        """
        Forward pass returning both classification logits and mass prediction.

        Returns:
            cls_logits: (B, num_classes)
            mass_pred: (B, 1)
        """
        features = self.forward_features(x)
        cls_logits = self.cls_head(features)
        mass_pred = self.reg_head(features)
        return cls_logits, mass_pred

    def forward_contrastive(self, x, proj_head):
        """
        Forward pass for contrastive pretraining (SimCLR).

        Runs the backbone and returns L2-normalized projection head output.

        Args:
            x: (B, C, H, W) input images
            proj_head: SimCLRProjectionHead module

        Returns:
            z: (B, proj_dim) L2-normalized projections
        """
        features = self.forward_features(x)  # (B, D)
        z = proj_head(features)  # (B, proj_dim)
        z = F.normalize(z, dim=-1)
        return z


class SimCLRProjectionHead(nn.Module):
    """
    SimCLR projection head: 2-layer MLP that maps backbone features
    to a lower-dimensional space for contrastive learning.
    Used only during pretraining, discarded during finetuning.
    """

    def __init__(self, embed_dim=256, hidden_dim=256, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


class PretrainDecoder(nn.Module):
    """
    MAE-style decoder for reconstruction pretraining.
    Takes visible tokens + mask info, reconstructs full image.
    """

    def __init__(
        self,
        embed_dim=256,
        decoder_embed_dim=128,
        decoder_depth=2,
        decoder_num_heads=4,
        patch_size=8,
        in_channels=3,
        num_patches=625,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.num_patches = num_patches
        pixel_dim = patch_size * patch_size * in_channels
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        nn.init.normal_(self.mask_token, std=0.02)

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, decoder_embed_dim)
        )

        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        self.decoder_blocks = nn.ModuleList(

            [
                nn.TransformerEncoderLayer(
                    d_model=decoder_embed_dim,
                    nhead=decoder_num_heads,
                    dim_feedforward=decoder_embed_dim * 4,
                    dropout=0.0,
                    activation="gelu",
                    batch_first=True,
                    norm_first=True,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, pixel_dim)

    def forward(self, visible_tokens, ids_restore, H, W):
        """
        Reconstruct full image from visible tokens.

        Args:
            visible_tokens: (B, N_vis, D) encoder output for visible patches
            ids_restore: (B, N) indices to unshuffle tokens back to original order
            H, W: patch grid dimensions

        Returns:
            (B, C, H*P, W*P) reconstructed image
        """
        B, N_vis, _ = visible_tokens.shape
        N = ids_restore.shape[1]
        x = self.decoder_embed(visible_tokens)  # (B, N_vis, D_dec)
        mask_tokens = self.mask_token.expand(B, N - N_vis, -1)  # (B, N_mask, D_dec)
        x_full = torch.cat([x, mask_tokens], dim=1)  # (B, N, D_dec)

        x_full = torch.gather(
            x_full,
            dim=1,
            index=ids_restore.unsqueeze(-1).expand(-1, -1, x_full.shape[2]),
        )  # (B, N, D_dec)

        if N != self.decoder_pos_embed.shape[1]:
            old_size = int(math.sqrt(self.decoder_pos_embed.shape[1]))

            pos = self.decoder_pos_embed.reshape(1, old_size, old_size, -1).permute(
                0, 3, 1, 2
            )

            pos = F.interpolate(pos, size=(H, W), mode="bilinear", align_corners=False)
            pos = pos.permute(0, 2, 3, 1).reshape(1, H * W, -1)

        else:
            pos = self.decoder_pos_embed

        x_full = x_full + pos

        # Decode
        for blk in self.decoder_blocks:
            x_full = blk(x_full)

        x_full = self.decoder_norm(x_full)

        # Predict pixels
        x_full = self.decoder_pred(x_full)  # (B, N, P*P*C)
        P = self.patch_size
        C = self.in_channels
        x_full = x_full.reshape(B, H, W, P, P, C)
        x_full = x_full.permute(0, 5, 1, 3, 2, 4)  # (B, C, H, P, W, P)
        x_full = x_full.reshape(B, C, H * P, W * P)
        
        return x_full
