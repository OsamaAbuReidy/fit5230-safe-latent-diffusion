# Individual weekly strategy log

This working log records decisions for the Milestone 4 report. It should be
updated at the end of every teaching week and later rewritten in the required
IEEE Transactions report format.

## Week 2 - Establish a reproducible fallback and explore a modern model

- Chose the Light side of Theme 2 and created a version-controlled repository.
- Tested SANA-Sprint as a modern small text-to-image candidate.
- Recorded reproducibility settings and model-file exclusions rather than
  treating an interactive ComfyUI workflow as sufficient evidence.
- Strategic lesson: novelty is not useful if the model's output quality prevents
  meaningful safety evaluation.

## Week 3 - Change backbone based on evidence

- Moved to SD3.5 Medium after observing weak human anatomy, text rendering, and
  compositional fidelity from SANA-Sprint.
- Read JailbreakDiffBench and narrowed the threat model to DACA/PGJ-style prompt
  evasion and output-level safety classification.
- Added final-latent capture rather than modifying all three SD3.5 text encoders.
- Strategic lesson: a model-specific pre-decode guard offers a clearer technical
  contribution than stacking another prompt LLM in front of the generator.

## Week 4 - Build labels around actual outputs

- Generated controlled SD3.5 samples and built a blind human-review interface.
- Separated prompt intent from generated-image labels because SD3.5 sometimes
  ignores harmful prompts or produces unusable anatomy.
- Used ShieldGemma only as an initial teacher/audit aid and retained human labels
  as the evaluation reference.
- Strategic lesson: prompt labels alone would incorrectly mark harmless failed
  generations as harmful output detections.

## Week 5 - Stake a distinct public scope

- Published the Milestone 1 scope as PreDecodeGuard-SD3.5 rather than returning
  to the heavily duplicated Safe Latent Diffusion topics used by several teams.
- Preserved DACA, PGJ, and JailbreakDiffBench prompts as held-out attack sources.
- Defined a challenge where an attack succeeds only when the image is genuinely
  harmful and the frozen detector misses it.
- Contacted Dark.Chaos because its modified-prompt challenge directly tests the
  separation between text evasion and latent output detection.

## Week 6 - Establish the accuracy-efficiency trade-off

- Reproduced the JailbreakDiffBench Multihead detector on the same human-labelled
  test outputs instead of comparing unrelated published numbers.
- Expanded the violence training data and improved the latent CNN to 0.852
  balanced accuracy, 0.769 harmful recall, 0.065 benign FPR, and 0.905 AUROC.
- Preserved the weaker 201-image negative-tail retrain as an ablation rather than
  hiding an inconvenient result.
- Measured 1.90 ms median latent-CNN inference, 183.13 ms Multihead inference,
  and 2492.75 ms tiled VAE decoding on the same GPU.
- Strategic lesson: the defensible contribution is not higher accuracy than
  Multihead; it is matching harmful recall with a much cheaper pre-decode
  decision and explicitly reporting the safety-utility trade-off.

## Next weekly entry

Record the Dark.Chaos exchange, whether its attack generates genuine violence,
whether it bypasses the frozen detector, how the other team responds, and what
technical change (if any) follows from that interaction.
