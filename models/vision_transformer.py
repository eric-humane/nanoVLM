import math
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_patch_size(patch_size):
    if patch_size is None:
        return None
    if torch.is_tensor(patch_size):
        patch_size = patch_size.tolist()
    if isinstance(patch_size, (tuple, list)):
        if len(set(patch_size)) == 1:
            return int(patch_size[0])
        warnings.warn(f"Non-square patch size {patch_size} detected; using max dimension.")
        return int(max(patch_size))
    return int(patch_size)


def _get_image_size(backbone):
    default_cfg = getattr(backbone, "default_cfg", {}) or {}
    input_size = default_cfg.get("input_size") or getattr(backbone, "img_size", None)
    if isinstance(input_size, (tuple, list)) and len(input_size) == 3:
        return tuple(int(x) for x in input_size[1:])
    if isinstance(input_size, (tuple, list)) and len(input_size) == 2:
        return tuple(int(x) for x in input_size)
    if isinstance(input_size, int):
        return (input_size, input_size)
    return None


def _get_patch_size(backbone):
    if hasattr(backbone, "patch_embed") and hasattr(backbone.patch_embed, "patch_size"):
        return backbone.patch_embed.patch_size
    default_cfg = getattr(backbone, "default_cfg", {}) or {}
    ps = default_cfg.get("patch_size") or getattr(backbone, "patch_size", None)
    return ps


def _apply_backbone_metadata(cfg, backbone):
    patch_size = _normalize_patch_size(_get_patch_size(backbone))
    if patch_size is not None:
        cfg.vit_patch_size = patch_size

    img_hw = _get_image_size(backbone)
    if img_hw is not None:
        side = max(img_hw)
        cfg.vit_img_size = side
        if getattr(cfg, "max_img_size", None) in (None, 2048, -1):
            cfg.max_img_size = side

    cfg.vit_hidden_dim = getattr(backbone, "num_features", getattr(backbone, "embed_dim", cfg.vit_hidden_dim))
    cfg.vit_n_blocks = getattr(backbone, "num_layers", getattr(backbone, "depth", cfg.vit_n_blocks))
    cfg.vit_n_heads = getattr(backbone, "num_heads", cfg.vit_n_heads)
    cfg.vit_dropout = getattr(backbone, "drop_rate", cfg.vit_dropout)
    cfg.vit_ln_eps = getattr(getattr(backbone, "norm", None), "eps", cfg.vit_ln_eps)
    mlp_ratio = getattr(backbone, "mlp_ratio", None)
    if mlp_ratio is not None:
        cfg.vit_inter_dim = int(cfg.vit_hidden_dim * mlp_ratio)


def _load_timm_backbone(cfg, pretrained=True):
    import timm

    candidates = [cfg.vit_model_type]
    if "/" in cfg.vit_model_type and not cfg.vit_model_type.startswith("hf-hub:"):
        candidates.append(f"hf-hub:{cfg.vit_model_type}")

    errors = []
    backbone = None
    used_name = None

    for name in candidates:
        try:
            backbone = timm.create_model(
                name,
                pretrained=pretrained,
                num_classes=0,
                global_pool="",
                scriptable=False,
            )
            used_name = name
            break
        except Exception as e:  # pragma: no cover - error path
            errors.append(f"{name}: {e}")

    if backbone is None:
        raise ValueError(
            f"Could not load vision backbone '{cfg.vit_model_type}' via timm. "
            f"Tried {candidates}. Errors: {errors}"
        )

    return backbone, used_name


def hydrate_vision_cfg_from_timm(cfg):
    """
    Populate cfg fields (patch size, image size, hidden dim) from a timm backbone without loading pretrained weights.
    Useful before dataloader/image processor creation to ensure resizing matches the backbone defaults.
    """
    backbone, used_name = _load_timm_backbone(cfg, pretrained=False)
    _apply_backbone_metadata(cfg, backbone)
    return used_name


class TimmVisionAdapter(nn.Module):
    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone

    def forward(self, x):
        if hasattr(self.backbone, "forward_features"):
            feats = self.backbone.forward_features(x)
        else:  # pragma: no cover - most timm models expose forward_features
            feats = self.backbone(x)

        if isinstance(feats, dict):
            if "x" in feats:
                feats = feats["x"]
            elif "last_hidden_state" in feats:
                feats = feats["last_hidden_state"]
            else:
                feats = next(iter(feats.values()))

        if isinstance(feats, (list, tuple)):
            feats = feats[-1]

        if torch.is_tensor(feats) and feats.ndim == 2:
            feats = feats.unsqueeze(1)

        if torch.is_tensor(feats) and feats.ndim == 4:
            feats = feats.flatten(2).transpose(1, 2)

        if feats.ndim != 3:
            raise ValueError(f"Vision backbone returned unexpected shape {feats.shape}; expected (B, seq, dim).")

        seq_len = feats.shape[1]
        seq_root = int(seq_len ** 0.5)
        if seq_len > 1 and seq_root**2 != seq_len and (seq_len - 1) == seq_root**2:
            feats = feats[:, 1:, :]

        return feats

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L245
class ViTPatchEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.img_size = cfg.vit_img_size
        self.patch_size = cfg.vit_patch_size
        self.num_patches = (self.img_size // self.patch_size) ** 2
        self.cls_flag = cfg.vit_cls_flag
        self.embd_dim = cfg.vit_hidden_dim

        # Conv layer to extract the patches
        self.conv = nn.Conv2d(
            in_channels=3,
            out_channels=self.embd_dim,
            kernel_size=self.patch_size,
            stride=self.patch_size,
            padding="valid",
        )

        if self.cls_flag:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embd_dim))
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches + 1, self.embd_dim))
        else:
            self.position_embedding = nn.Parameter(torch.rand(1, self.num_patches, self.embd_dim))


    def forward(self, x):
        x = self.conv(x)  # extract patches
        x = x.flatten(2)  # flatten the patches into a single dimension
        x = x.transpose(1, 2)  # transpose to (batch_size, num_patches, hidden_dim)

        # Add CLS token (according to original ViT Paper) and position embeddings
        if self.cls_flag:
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        x = x + self.position_embedding
        return x

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L381
# https://github.com/karpathy/nanoGPT/blob/master/model.py#L29
class ViTMultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.n_heads = cfg.vit_n_heads
        self.embd_dim = cfg.vit_hidden_dim
        assert self.embd_dim % self.n_heads == 0, "embd_dim must be divisible by num_heads"
        self.head_dim = self.embd_dim // self.n_heads
        self.dropout = cfg.vit_dropout

        # Combined projections for all heads
        self.qkv_proj = nn.Linear(self.embd_dim, 3 * self.embd_dim, bias=True)
        self.out_proj = nn.Linear(self.embd_dim, self.embd_dim, bias=True)

        # Dropout layers
        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        # Use scaled dot product attention if available
        self.sdpa = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.sdpa:
            print("Warning: scaled dot product attention not available. Using standard attention in ViT.")

    def forward(self, x):
        B, T, C = x.size()

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split(C, dim=2)
        # Reshape  [B, T, C] -> [B, T, n_heads, head_dim] and transpose -> [B, n_heads, T, head_dim]
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)  # (B, n_heads, T, head_dim)

        if self.sdpa:
            y = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=False # ViT attention is bidirectional
            )
        else:
            attn = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            attn = F.softmax(attn, dim=-1)
            attn = self.attn_dropout(attn)
            y = attn @ v  # (B, n_heads, T, T) x (B, n_heads, T, head_dim) -> (B, n_heads, T, head_dim)
        
        # Transpose back from [B, n_heads, T, head_dim] to [B, T, n_heads * head_dim] and combine all heads to [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)  
        y = self.out_proj(y)
        y = self.resid_dropout(y)

        return y

# https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py#L453
class ViTMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.activation_fn = nn.GELU(approximate='tanh')
        self.fc1 = nn.Linear(cfg.vit_hidden_dim, cfg.vit_inter_dim)
        self.fc2 = nn.Linear(cfg.vit_inter_dim, cfg.vit_hidden_dim)
        self.dropout = nn.Dropout(cfg.vit_dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation_fn(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

# https://github.com/karpathy/nanoGPT/blob/master/model.py#L94    
class ViTBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.attn = ViTMultiHeadAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)
        self.mlp = ViTMLP(cfg)
    
    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
    

class ViT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cfg.vit_patch_size = int(self.cfg.vit_patch_size)
        self.patch_embedding = ViTPatchEmbeddings(cfg)
        self.cls_flag = cfg.vit_cls_flag
        self.dropout = nn.Dropout(cfg.vit_dropout)
        self.blocks = nn.ModuleList([ViTBlock(cfg) for _ in range(cfg.vit_n_blocks)])
        self.layer_norm = nn.LayerNorm(cfg.vit_hidden_dim, eps=cfg.vit_ln_eps)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, x):
        x = self.patch_embedding(x) 
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)

        if self.cls_flag:
            x = self.layer_norm(x[:, 0])
        else:
            x = self.layer_norm(x)
            #x = x.mean(dim=1)
        
        return x
    
    @classmethod
    def from_pretrained(cls, cfg, *, pretrained=True):
        backbone, used_name = _load_timm_backbone(cfg, pretrained=pretrained)
        _apply_backbone_metadata(cfg, backbone)
        print(f"Loaded vision backbone '{used_name}' via timm (pretrained={pretrained}).")
        return TimmVisionAdapter(backbone)
