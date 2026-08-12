# FIT5230 Safe Latent Diffusion

Reproducible coursework repository for FIT5230 Malicious AI (2026).

- Theme: Text-to-Image
- Side: Light (defence)
- Baseline: Safe Latent Diffusion (SLD)
- Scope now: reproduce and measure the published baseline only. No proposed research modification is implemented in this repository yet.

## Why this baseline

Safe Latent Diffusion applies safety guidance during the diffusion process without retraining. It is a suitable Light-side baseline because it makes an explicit safety--utility trade-off that can be reproduced and evaluated. The published method also provides the I2P test bed for measuring inappropriate-generation behaviour.

Primary references:

- Paper: [Safe Latent Diffusion: Mitigating Inappropriate Degeneration in Diffusion Models](https://arxiv.org/abs/2211.05105)
- Reference code: [ml-research/safe-latent-diffusion](https://github.com/ml-research/safe-latent-diffusion)
- Supported implementation: [Hugging Face Safe Stable Diffusion](https://huggingface.co/docs/diffusers/main/api/pipelines/stable_diffusion/stable_diffusion_safe)

## Repository map

```text
configs/       Reproducible run and evaluation configurations
data/          Dataset manifests and metadata only; generated data is ignored
docs/          Assignment notes, protocol, and setup plan
notebooks/     Colab-facing narrative notebooks
outputs/       Generated images, metrics, and figures (ignored except .gitkeep)
scripts/       Thin command-line entry points for baseline runs and evaluation
src/           Reusable project package (baseline, data, evaluation, utilities)
tests/         Smoke and regression tests
```

See [docs/setup-plan.md](docs/setup-plan.md) for the staged setup plan and [docs/assignment-brief.md](docs/assignment-brief.md) for the requirements that shape this repository.

## Environment

Use Python 3.10 or 3.11 and a CUDA-capable GPU for practical baseline reproduction. Install a PyTorch build matching the local CUDA runtime first, then install the locked project dependencies:

```bash
pip install -r requirements.txt
pip install -e .
```

Do not disable safety components in public-facing demonstrations. Store any model-access token locally in `.env`; never commit it.

## Current status

Repository scaffold and documentation are in place. The next technical task is a benign-prompt smoke test of the unmodified SLD baseline, followed by a fixed-seed evaluation protocol. Candidate weaknesses and improvements will be documented only after baseline reproduction evidence is available.
