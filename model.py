"""
Multimodal Sentiment Analysis Model
Architecture: BERT+BiGRU (text) + ViT (image) + Stable Diffusion cross-modal generation
             + Self-Attention + Co-Attention + Fusion → Sentiment Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, ViTModel
from typing import Optional, Tuple


# ─────────────────────────────────────────────
#  1. TEXT ENCODER  (BERT + Bi-GRU)
# ─────────────────────────────────────────────
class TextEncoder(nn.Module):
    """Encodes raw text with BERT token embeddings fed into a Bidirectional GRU."""

    def __init__(self, hidden_dim: int = 256, gru_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        bert_dim = self.bert.config.hidden_size          # 768

        self.bigru = nn.GRU(
            input_size=bert_dim,
            hidden_size=hidden_dim,
            num_layers=gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim * 2)  # keep dim = 512
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            input_ids:      (B, L)
            attention_mask: (B, L)
        Returns:
            z: (B, L, hidden_dim*2)   sequence of contextual text features
        """
        with torch.no_grad():
            bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        token_embs = bert_out.last_hidden_state          # (B, L, 768)
        token_embs = self.dropout(token_embs)

        gru_out, _ = self.bigru(token_embs)              # (B, L, 512)
        z = self.proj(gru_out)
        return z                                          # Z_T or Z_GT


# ─────────────────────────────────────────────
#  2. IMAGE ENCODER  (ViT)
# ─────────────────────────────────────────────
class ImageEncoder(nn.Module):
    """Encodes an image with a Vision Transformer and projects to a shared dim."""

    def __init__(self, out_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        vit_dim = self.vit.config.hidden_size            # 768

        self.proj = nn.Sequential(
            nn.Linear(vit_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: (B, 3, 224, 224)
        Returns:
            z: (B, N+1, out_dim)   patch sequence incl. [CLS]
        """
        with torch.no_grad():
            vit_out = self.vit(pixel_values=pixel_values)
        patch_embs = vit_out.last_hidden_state           # (B, 197, 768)
        z = self.proj(patch_embs)                        # (B, 197, 512)
        return z                                          # Z_I  (or generated-image branch)


# ─────────────────────────────────────────────
#  3. SELF-ATTENTION MODULE
# ─────────────────────────────────────────────
class SelfAttention(nn.Module):
    """Standard multi-head self-attention over a sequence."""

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, key_padding_mask: Optional[torch.Tensor] = None):
        """
        Args:
            x: (B, S, dim)
        Returns:
            ẑ: (B, S, dim)   self-attended features
        """
        attn_out, _ = self.mha(x, x, x, key_padding_mask=key_padding_mask)
        return self.norm(x + self.dropout(attn_out))


# ─────────────────────────────────────────────
#  4. CO-ATTENTION MODULE
# ─────────────────────────────────────────────
class CoAttention(nn.Module):
    """
    Bidirectional cross-modal attention:
      - Text-to-Image attention  →  Z̃_T
      - Image-to-Text attention  →  Z̃_I
    """

    def __init__(self, dim: int = 512, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        # Text attends to Image
        self.text2img = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        # Image attends to Text
        self.img2text = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )

        self.norm_t = nn.LayerNorm(dim)
        self.norm_i = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        z_t: torch.Tensor,    # (B, Lt, dim)  text sequence
        z_i: torch.Tensor,    # (B, Li, dim)  image patch sequence
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            z_tilde_T: (B, Lt, dim)   image-aware text
            z_tilde_I: (B, Li, dim)   text-aware image
        """
        # Text-to-image: queries=text, keys/values=image
        t2i_out, _ = self.text2img(z_t, z_i, z_i)
        z_tilde_T = self.norm_t(z_t + self.dropout(t2i_out))

        # Image-to-text: queries=image, keys/values=text
        i2t_out, _ = self.img2text(z_i, z_t, z_t)
        z_tilde_I = self.norm_i(z_i + self.dropout(i2t_out))

        return z_tilde_T, z_tilde_I


# ─────────────────────────────────────────────
#  5. FUSION + CLASSIFIER
# ─────────────────────────────────────────────
class FusionClassifier(nn.Module):
    """
    Pools all four representation streams, concatenates, and classifies sentiment.
    F = [ẑ_T ; ẑ_I ; Z̃_T ; Z̃_I]
    """

    def __init__(self, dim: int = 512, num_classes: int = 3, dropout: float = 0.2):
        super().__init__()
        fused_dim = dim * 4   # four streams

        self.classifier = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fused_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 2, fused_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fused_dim // 4, num_classes),
        )

    @staticmethod
    def mean_pool(x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Mean-pool a sequence (B, S, D) → (B, D), respecting optional mask."""
        if mask is not None:
            # mask: (B, S), True = valid
            mask = mask.unsqueeze(-1).float()
            return (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return x.mean(dim=1)

    def forward(
        self,
        z_hat_T:   torch.Tensor,   # (B, Lt, dim)  self-attended text
        z_hat_I:   torch.Tensor,   # (B, Li, dim)  self-attended image
        z_tilde_T: torch.Tensor,   # (B, Lt, dim)  co-attended text
        z_tilde_I: torch.Tensor,   # (B, Li, dim)  co-attended image
        text_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns logits: (B, num_classes)"""
        v_T  = self.mean_pool(z_hat_T,   text_mask)   # (B, dim)
        v_I  = self.mean_pool(z_hat_I)                # (B, dim)
        v_tT = self.mean_pool(z_tilde_T, text_mask)  # (B, dim)
        v_tI = self.mean_pool(z_tilde_I)              # (B, dim)

        F = torch.cat([v_T, v_I, v_tT, v_tI], dim=-1)  # (B, dim*4)
        return self.classifier(F)                         # (B, num_classes)


# ─────────────────────────────────────────────
#  6. FULL MODEL
# ─────────────────────────────────────────────
class MultimodalSentimentModel(nn.Module):
    """
    Full pipeline (without Stable Diffusion at inference time – generated
    embeddings are passed in pre-computed, matching the paper's architecture).

    Inputs at forward():
        text_ids, text_mask         → original post text  T
        image_pixels                → original post image I
        gen_text_ids, gen_text_mask → generated text T_g  (from image via SD captioning)
        gen_image_pixels            → generated image     (from text via SD)

    The model concatenates (C) representations exactly as in the diagram:
        Z_T_concat = concat(Z_T, Z_gen_image)   along sequence dim
        Z_I_concat = concat(Z_I, Z_GT)          along sequence dim
    """

    def __init__(
        self,
        hidden_dim: int = 256,    # per-direction in BiGRU → 512 total
        num_heads:  int = 8,
        num_classes: int = 3,     # negative / neutral / positive
        dropout: float = 0.1,
    ):
        super().__init__()
        dim = hidden_dim * 2      # 512

        # Encoders
        self.text_encoder      = TextEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.image_encoder     = ImageEncoder(out_dim=dim, dropout=dropout)

        # Self-attention (one per modality stream)
        self.self_attn_text    = SelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)
        self.self_attn_image   = SelfAttention(dim=dim, num_heads=num_heads, dropout=dropout)

        # Co-attention
        self.co_attention      = CoAttention(dim=dim, num_heads=num_heads, dropout=dropout)

        # Fusion + classifier
        self.fusion            = FusionClassifier(dim=dim, num_classes=num_classes, dropout=dropout)

    def forward(
        self,
        text_ids:        torch.Tensor,           # (B, Lt)
        text_mask:       torch.Tensor,           # (B, Lt)
        image_pixels:    torch.Tensor,           # (B, 3, 224, 224)
        gen_text_ids:    torch.Tensor,           # (B, Lg)
        gen_text_mask:   torch.Tensor,           # (B, Lg)
        gen_image_pixels: torch.Tensor,          # (B, 3, 224, 224)
    ) -> torch.Tensor:

        # ── Encode all four inputs ──────────────────────────────────────────
        Z_T        = self.text_encoder(text_ids, text_mask)           # (B, Lt, 512)
        Z_I        = self.image_encoder(image_pixels)                 # (B, 197, 512)
        Z_GT       = self.text_encoder(gen_text_ids, gen_text_mask)   # (B, Lg, 512)
        Z_gen_img  = self.image_encoder(gen_image_pixels)             # (B, 197, 512)

        # ── Concatenate cross-modal pairs  (the C boxes in the diagram) ────
        # Upper concat: text branch + generated-image branch  →  text-side stream
        Z_T_full   = torch.cat([Z_T, Z_gen_img], dim=1)   # (B, Lt+197, 512)

        # Lower concat: image branch + generated-text branch →  image-side stream
        Z_I_full   = torch.cat([Z_I, Z_GT],      dim=1)   # (B, 197+Lg, 512)

        # ── Self-attention ──────────────────────────────────────────────────
        z_hat_T    = self.self_attn_text(Z_T_full)   # (B, Lt+197, 512)
        z_hat_I    = self.self_attn_image(Z_I_full)  # (B, 197+Lg, 512)

        # ── Co-attention ────────────────────────────────────────────────────
        z_tilde_T, z_tilde_I = self.co_attention(z_hat_T, z_hat_I)

        # ── Fusion & prediction ─────────────────────────────────────────────
        logits = self.fusion(z_hat_T, z_hat_I, z_tilde_T, z_tilde_I)
        return logits   # (B, num_classes)
