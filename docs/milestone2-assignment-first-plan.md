# Milestone 2 assignment-first plan

Status date: 3 September 2026

## Deadline and rubric target

Milestone 2 is due 18 September 2026 at 11:55 PM MYT. The submission is an Ed
post plus its link on Moodle. The post must show current results and demos,
provide functional base code for other teams, and identify meaningful engagement
with another team.

The marking allocation is:

- quality of description: 2%
- technical depth: 3%
- documentation and reproducibility: 1%
- engagement with other teams: 2%

## Frozen assignment story

The reference is JailbreakDiffBench and its released Multihead image detector.
Our implemented variation is PreDecodeGuard-SD3.5: capture SD3.5 Medium's final
latent before VAE decoding and classify the generated output from that latent.

The current result is a measured safety-efficiency trade-off. A mixed,
human-labelled expansion improved the spatial latent CNN substantially. It now
matches Multihead's harmful recall on the frozen violence test set while
remaining weaker in AUROC and benign false-positive rate. Its contribution is
the pre-decode decision point and much lower latency, not superior accuracy.

## Work required before research extensions

1. Publish a self-contained Milestone 2 notebook with pre-rendered outputs.
2. Make the notebook train the pooled latent probe from the tracked, prompt-free
   feature bundle and evaluate the untouched test split.
3. Reproduce the Multihead-versus-latent comparison table and at least one
   visualization from row-level data.
4. Include a functional challenge interface and an explicit scoring contract.
5. Remove hardcoded local paths and document every public dependency and artifact.
6. Select one public team, run or analyze its shared challenge, and preserve the
   evidence needed for the 2% engagement criterion.
7. Draft the Milestone 2 Ed post only after the notebook has been executed from
   a clean Colab runtime.

## Completed in the assignment-first sprint

- Built a deterministic, prompt-free 1.43 MiB public feature bundle containing
  all 372 usable samples, 1,088 pooled latent features, frozen splits, human
  output labels, and the seven relevant Multihead scores.
- Added an end-to-end runner that selects regularization on validation data,
  refits on train plus validation, and evaluates the test split once.
- Reproduced the frozen pooled-latent and Multihead results exactly for both
  any-harm and violence/gore tasks.
- Verified byte-for-byte deterministic bundle generation and passed the full
  local test suite.
- Trained a deterministic 150,745-parameter spatial CNN on the full final
  SD3.5 latent for violence-versus-benign classification. On the frozen test
  split it improved AUROC from 0.8449 to 0.8685, balanced accuracy from 0.7270
  to 0.7674, and reduced benign FPR from 16.13% to 8.06% while retaining 61.54%
  harmful recall.
- Added paired stratified bootstrap intervals and row-level error analysis. The
  CNN-minus-pooled AUROC interval crosses zero, so the improvement is reported
  as promising but not statistically conclusive on the 75-image test set.
- Updated the executed Milestone 2 notebook with the spatial result, runtime,
  uncertainty, error analysis, and explicit comparison against Multihead.
- Generated 201 additional SD3.5 image-latent pairs and human-reviewed all of
  them. The labels contain 198 decisive cases and three uncertain cases.
- Training with the first 164 decisive mixed expansion labels improved the
  frozen violence test result to 0.8524 balanced accuracy, 0.7692 harmful
  recall, 0.0645 benign FPR, and 0.9045 AUROC.
- Audited the later 201-image snapshot as a negative-only-tail ablation. The 34
  additional decisive labels were all non-violent and reduced test performance;
  this is reported transparently rather than silently discarded.
- Benchmarked the latent CNN and Multihead on the same RTX 3060 Laptop GPU. Warm
  batch-one median classifier latency was 1.8965 ms versus 183.1275 ms, a 96.6x
  classifier-only speedup.
- Measured the exact ComfyUI tiled VAE decode used by the dataset pipeline at
  2492.7486 ms median. For a blocked output, the post-denoising decision path is
  therefore 1.8965 ms for PreDecodeGuard versus 2675.8761 ms for decode plus
  Multihead, a 1410.9x speedup. Denoising is common to both and excluded.
- Rebuilt the Milestone 2 notebook with these results and executed all nine code
  cells sequentially from the repository environment without error.

## Immediate next action

Run the Dark.Chaos modified-prompt challenge against the frozen detector and
preserve the prompts, fixed generation settings, human output labels, detector
scores, and public response as Milestone 2 peer-engagement evidence. Then
execute the notebook in a clean Colab runtime and publish the Ed post.

## Parked research branch

Decoded-image-to-latent knowledge distillation, proxy latents, and publication
claims remain postponed until the held-out attack evaluation and peer-engagement
evidence are secure. They remain candidate extensions for the Milestone 2-to-3
period.
