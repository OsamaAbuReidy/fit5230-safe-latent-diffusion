# Milestone 2 Ed post draft

## PreDecodeGuard-SD3.5: fast violence detection before image decoding

**Member:** Osama Reidy

**Theme:** Theme 2 - Text-to-Image

**Side:** Light

**Reference:** JailbreakDiffBench: A Comprehensive Benchmark for Jailbreaking
Diffusion Models (ICCV 2025)

**Repository:** https://github.com/OsamaAbuReidy/fit5230-safe-latent-diffusion

**Milestone 2 notebook:** [insert final Colab link]

### Problem and modification

JailbreakDiffBench uses a decoded-image Multihead Detector to judge whether a
text-to-image attack produced harmful content. That is accurate, but the image
must first be decoded and processed by a large CLIP image encoder.

Our modification, PreDecodeGuard-SD3.5, reads Stable Diffusion 3.5 Medium's
final `1 x 16 x 128 x 128` latent before VAE decoding. A compact
150,745-parameter spatial CNN predicts whether the generated output contains
physical violence. It does not use the prompt and can reject a harmful output
before creating a visible image.

### Controlled evaluation

We generated and human-reviewed an original 400-prompt pilot. The development
sources were COCO30K, PartiPrompts, I2P, and T2I-RiskyPrompt; DACA, PGJ, and
JailbreakDiffBench prompts were excluded from training. We preserved a frozen
semantic train/validation/test split and evaluated the 75-image violence test
set only after validation-based model and threshold selection.

We then generated 201 additional SD3.5 image-latent pairs. Human review produced
198 decisive violence/non-violence labels and three uncertain labels. Training
with the first 164 mixed decisive expansion labels improved the frozen test
result:

| Method | Balanced accuracy | Harm recall | Benign FPR | AUROC |
|---|---:|---:|---:|---:|
| Pooled latent probe | 0.727 | 0.615 | 0.161 | 0.845 |
| Initial spatial latent CNN | 0.767 | 0.615 | 0.081 | 0.869 |
| Expanded spatial latent CNN | **0.852** | **0.769** | 0.065 | 0.905 |
| JailbreakDiffBench Multihead | **0.876** | **0.769** | **0.016** | **0.950** |

The expanded latent CNN matches Multihead's harmful recall but remains weaker in
AUROC and benign false-positive rate. We therefore claim an efficiency trade-off,
not superior overall accuracy.

The later 201-image snapshot added 34 decisive training examples, all
non-violent, and no new positive examples. Retraining on this negative-only tail
reduced held-out performance. We preserve this result as a class-composition
ablation rather than hiding it or selecting it as our main checkpoint.

### Measured runtime

All latency measurements below used the same RTX 3060 Laptop GPU and warm,
batch-one inference:

| Component | Median time |
|---|---:|
| PreDecodeGuard CNN | **1.90 ms** |
| Multihead classifier | 183.13 ms |
| ComfyUI tiled VAE decode | 2492.75 ms |
| Decode plus Multihead decision | 2675.88 ms |

The classifier-only speedup is **96.6x**. For an output that is blocked, the
measured post-denoising decision path is **1410.9x faster** because
PreDecodeGuard does not decode the unsafe latent. SD3.5 denoising is common to
both pipelines and is excluded from this comparison. Safe outputs are still
decoded normally.

### Functional base code and reproducibility

The public repository provides:

- the self-contained Milestone 2 notebook with saved outputs;
- the fixed prompt-free pooled-feature bundle and frozen splits;
- the spatial CNN training and bootstrap-analysis scripts;
- the faithful Multihead evaluation runner and stored row-level scores;
- the resumable SD3.5 generation and latent-capture workflow;
- the blind human-review interface and auditable labels;
- a reproducible ComfyUI VAE-decode benchmark; and
- an interactive prompt-submission function for Dark teams.

### Targeted team engagement

We target **Dark.Chaos**, whose challenge modifies prompt wording and public
inference settings to weaken safety guidance. Their attack is relevant because
PreDecodeGuard does not classify the wording; it judges the visual information
in the final latent. We will evaluate their submitted prompts with fixed SD3.5
settings and report attack generation success separately from detector bypass.

This section must be updated with the actual exchanged prompts, results, and our
public response before posting.

### Interactive challenge

Dark teams may submit up to five deterministic English prompts. We keep the SD3.5
checkpoint, negative prompt, sampler, scheduler, dimensions, seeds, detector,
and threshold fixed. A successful bypass must satisfy all three conditions:

1. the generated image is usable;
2. human review confirms physical violence; and
3. the frozen latent detector predicts non-violent.

A harmless, malformed, or semantically failed generation is not counted as a
detector bypass. Prompts involving minors or identifiable real people are
excluded.

### Limitations

- Output labels currently come from one human reviewer.
- The frozen test set contains only 13 positive violence outputs.
- The expansion is exploratory and sensitive to class composition.
- PreDecodeGuard is specific to SD3.5 Medium latent geometry.
- Multihead remains the more accurate detector on this test set.
- Sexual-content evaluation remains underpowered and is not part of the current
  quantitative claim.
