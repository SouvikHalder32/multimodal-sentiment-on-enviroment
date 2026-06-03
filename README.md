# Multimodal Sentiment Analysis
## Architecture: BERT+BiGRU · ViT · Stable Diffusion cross-generation · Co-Attention

This repository implements the full pipeline shown in the architecture diagram:

```
Social Media Post (text T + image I)
        │
        ├── Text T  ──────────────────────────────────────────────────────────────┐
        │     └── Stable Diffusion  → Generated Image                             │
        │                               └── ViT ──────────────────────┐           │
        │     └── BERT + Bi-GRU ──────────────────────────────── Z_T ─┤           │
        │                                                               ╞═══(C)    │
        │                                                          Z_gen_img ──────┘
        │
        ├── Image I ───────────────────────────────────────────────────────────────┐
        │     └── ViT ─────────────────────────────────────────────── Z_I ──┐      │
        │     └── Stable Diffusion  → Generated Text T_g                    ╞══(C) │
        │                               └── BERT + Bi-GRU ──────────── Z_GT ┘      │
        │                                                                           │
        └────────────────────────────────────────────────────────────────────────┘
                      ↓                                   ↓
              Self-Attention (text)             Self-Attention (image)
                   ẑ_T                                ẑ_I
                      └───────────── Co-Attention ───────────────┘
                                     Z̃_T   Z̃_I
                                          ↓
                         F = [ẑ_T ; ẑ_I ; Z̃_T ; Z̃_I]
                                          ↓
                              Sentiment Prediction ŷ
```

## File Structure

```
multimodal_sentiment/
├── model.py          # Full model: encoders, self-attn, co-attn, fusion
├── dataset.py        # Dataset + DataLoader factory
├── trainer.py        # train_one_epoch, evaluate, train()
├── run_demo.py       # Runnable end-to-end demo (synthetic data)
└── requirements.txt
```

## Quick Start

```bash
pip install -r requirements.txt
python run_demo.py
```

The demo runs a complete forward pass and one training step using random tensors
— no GPU or real dataset required.

## Using Real Data

Prepare a list of dicts and call `build_dataloaders`:

```python
from dataset import build_dataloaders
from model import MultimodalSentimentModel
from trainer import train

samples = [
    {
        "text":           "Global warming is causing polar ice caps to melt ...",
        "image_path":     "data/polar_bear.jpg",
        "gen_text":       "A polar bear stranded on a melting ice floe ...",
        "gen_image_path": "data/generated_polar.png",   # pre-generated with SD
        "label":          "negative",
    },
    # ...
]

train_samples = samples[:800]
val_samples   = samples[800:900]
test_samples  = samples[900:]

loaders = build_dataloaders(train_samples, val_samples, test_samples)
model   = MultimodalSentimentModel(num_classes=3)
model   = train(model, loaders["train"], loaders["val"], num_epochs=20)
```

## Sentiment Labels

| Index | Label    |
|-------|----------|
| 0     | Negative |
| 1     | Neutral  |
| 2     | Positive |

## Notes on Stable Diffusion

The paper generates cross-modal counterparts with Stable Diffusion:
- **Text → Image**: SD generates an image from the post text
- **Image → Text**: SD (or BLIP/InstructBLIP) generates a caption from the post image

These are pre-generated and stored on disk before training.
To generate them programmatically, use `diffusers`:

```python
from diffusers import StableDiffusionPipeline
pipe = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
image = pipe(prompt=text).images[0]
image.save("generated.png")
```
