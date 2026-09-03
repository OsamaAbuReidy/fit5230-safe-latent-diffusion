# SD3.5 latent-guard dataset pipeline

## Purpose

This pipeline creates paired observations for training a compact diffusion-native safety student:

```text
prompt -> SD3.5 final latent -> VAE-decoded image -> ShieldGemma 2 teacher scores
```

Prompt intent remains metadata. ShieldGemma scores are noisy teacher labels, and generated-image
content must be human-labelled for validation and testing.

## Fixed generation configuration

- Checkpoint: `sd3.5_medium_incl_clips_t5xxlfp8scaled.safetensors`
- Resolution: 1024 x 1024
- Steps: 30
- CFG: 3.5
- Sampler: Euler
- Scheduler: beta
- Negative prompt: empty
- VAE decode: tiled, 512px tiles with 64px overlap
- One deterministic manifest seed per sample

The custom `FIT5230FinalLatentSaver` captures KSampler's final output before VAE decoding as a
compressed FP16 NPZ. Each NPZ contains the tensor and embedded JSON metadata. The batch runner
saves the decoded image, complete generation record, three ShieldGemma policy scores, timings,
and ComfyUI prompt ID.

## Commands

```powershell
python scripts/run_latent_guard_batch.py --limit 1 --run-id latent_guard_smoke_v1
python scripts/run_latent_guard_batch.py --run-id latent_guard_pilot_v1
python scripts/report_latent_guard_progress.py --run-id latent_guard_pilot_v1
```

The runner is resumable. A sample is skipped only when its immutable completion record exists and
has `status: success`. Partial attempts remain in `attempts.jsonl`; existing completed artifacts
are never overwritten. Use a new run ID for a fresh experiment.

## Validated smoke result

Sample `lgp_benign_001` completed successfully:

- Latent shape: `[1, 16, 128, 128]`
- Stored dtype: FP16
- Compressed latent size: 428,950 bytes
- All latent elements finite: yes
- Decoded image: one 1024px PNG
- ShieldGemma sexual probability: 0.000014
- ShieldGemma violence/gore probability: 0.0
- ShieldGemma dangerous-content probability: 0.0
- Teacher decision: safe
- Three-policy teacher time: 121.0835 seconds on CPU
- Cold end-to-end time: 361.4605 seconds

The repeated cached copy of this smoke sample completed much faster and must not be used as a
latency measurement. Final timing summaries should exclude cached or interrupted attempts.
The smoke used the standard VAE decoder. The collection manifest uses tiled VAE decoding after a
standard 1024px decode saturated the 6GB GPU on the second uncached sample. This changes only the
pixel decoding stage; the captured final latent is unchanged, and the decode configuration is
recorded in every completion record.

## Storage layout

```text
data/raw/latent_guard/
  latents/<run_id>/<sample_id>/seed_<seed>.npz
  <run_id>/attempts.jsonl
  <run_id>/records/<sample_id>.json
  <run_id>/shieldgemma_decisions.jsonl
  <run_id>/shieldgemma_server.log
```

Decoded images are stored in the Assignment ComfyUI output folder under
`output/FIT5230/<run_id>/<split>/<sample_id>/`. All generated data is ignored by Git.
