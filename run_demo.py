"""
run_demo.py  –  End-to-end runnable demo of MultimodalSentimentModel.

Uses synthetic (random) tensors so no real dataset or GPU is required.
Run with:
    python run_demo.py
"""

import torch
import torch.nn as nn
from model import MultimodalSentimentModel


# ── Hyper-parameters ─────────────────────────────────────────────────────────
BATCH_SIZE   = 4
SEQ_LEN      = 32     # token length for BERT inputs
NUM_CLASSES  = 3      # negative / neutral / positive
HIDDEN_DIM   = 256    # BiGRU per-direction dim  →  512 total
NUM_HEADS    = 8
VOCAB_SIZE   = 30522  # BERT vocab size
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Build model ──────────────────────────────────────────────────────────────
print("Building MultimodalSentimentModel …")
model = MultimodalSentimentModel(
    hidden_dim=HIDDEN_DIM,
    num_heads=NUM_HEADS,
    num_classes=NUM_CLASSES,
    dropout=0.1,
).to(DEVICE)

total_params = sum(p.numel() for p in model.parameters())
print(f"  Total parameters : {total_params:,}")
print(f"  Running on       : {DEVICE}\n")


# ── Synthetic batch ───────────────────────────────────────────────────────────
def make_batch():
    """Creates a random batch that matches the model's expected input shapes."""
    return {
        "text_ids":          torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE),
        "text_mask":         torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long).to(DEVICE),
        "image_pixels":      torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE),
        "gen_text_ids":      torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN)).to(DEVICE),
        "gen_text_mask":     torch.ones(BATCH_SIZE, SEQ_LEN, dtype=torch.long).to(DEVICE),
        "gen_image_pixels":  torch.randn(BATCH_SIZE, 3, 224, 224).to(DEVICE),
        "labels":            torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)).to(DEVICE),
    }


# ── Single forward pass ───────────────────────────────────────────────────────
print("Running forward pass on synthetic batch …")
batch = make_batch()

model.eval()
with torch.no_grad():
    logits = model(
        batch["text_ids"],
        batch["text_mask"],
        batch["image_pixels"],
        batch["gen_text_ids"],
        batch["gen_text_mask"],
        batch["gen_image_pixels"],
    )

probs = torch.softmax(logits, dim=-1)
preds = logits.argmax(dim=-1)

LABEL_NAMES = ["negative", "neutral", "positive"]
print(f"\n{'Sample':<8} {'Pred':<12} {'Neg':>8} {'Neu':>8} {'Pos':>8}")
print("-" * 48)
for i in range(BATCH_SIZE):
    print(
        f"{i:<8} {LABEL_NAMES[preds[i].item()]:<12} "
        f"{probs[i,0].item():>8.4f} {probs[i,1].item():>8.4f} {probs[i,2].item():>8.4f}"
    )


# ── Quick training step ───────────────────────────────────────────────────────
print("\n\nRunning one training step …")
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss()

optimizer.zero_grad()
logits = model(
    batch["text_ids"],
    batch["text_mask"],
    batch["image_pixels"],
    batch["gen_text_ids"],
    batch["gen_text_mask"],
    batch["gen_image_pixels"],
)
loss = criterion(logits, batch["labels"])
loss.backward()
optimizer.step()
print(f"  Loss after one step : {loss.item():.4f}")


# ── Shape report ─────────────────────────────────────────────────────────────
print("\n── Intermediate tensor shapes (batch_size=4, seq_len=32) ──")
with torch.no_grad():
    model.eval()
    b = make_batch()

    Z_T  = model.text_encoder(b["text_ids"], b["text_mask"])
    Z_I  = model.image_encoder(b["image_pixels"])
    Z_GT = model.text_encoder(b["gen_text_ids"], b["gen_text_mask"])
    Z_GI = model.image_encoder(b["gen_image_pixels"])

    Z_T_full = torch.cat([Z_T,  Z_GI], dim=1)
    Z_I_full = torch.cat([Z_I,  Z_GT], dim=1)

    z_hat_T = model.self_attn_text(Z_T_full)
    z_hat_I = model.self_attn_image(Z_I_full)

    z_tilde_T, z_tilde_I = model.co_attention(z_hat_T, z_hat_I)

    shapes = {
        "Z_T  (text features)":                Z_T.shape,
        "Z_I  (image features)":               Z_I.shape,
        "Z_GT (generated-text features)":      Z_GT.shape,
        "Z_GI (generated-image features)":     Z_GI.shape,
        "Z_T_full  [Z_T ⊕ Z_GI]":             Z_T_full.shape,
        "Z_I_full  [Z_I ⊕ Z_GT]":             Z_I_full.shape,
        "ẑ_T  (self-attended text)":           z_hat_T.shape,
        "ẑ_I  (self-attended image)":          z_hat_I.shape,
        "Z̃_T  (co-attended text→image)":      z_tilde_T.shape,
        "Z̃_I  (co-attended image→text)":      z_tilde_I.shape,
        "logits":                               logits.shape,
    }
    for name, shape in shapes.items():
        print(f"  {name:<42} {str(shape)}")

print("\n✓ Demo completed successfully.")
