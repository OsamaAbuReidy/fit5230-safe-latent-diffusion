# Parked research branch: counterfactual early-generation safety detection

## Status

This document preserves a possible publication-oriented extension. It is not the
current assignment implementation plan and it is not yet a supported research
claim. The immediate project priority is the FIT5230 milestone work.

Last literature check: 16 August 2026.

## Motivation from the pilot

The project tested whether early conditional-unconditional denoiser residuals
from SD3.5 Medium could distinguish unsafe generations from benign controls.
The current frozen pilot detector uses the first six sampling steps and an 8 x 8
spatial pooling representation.

The held-out attack pilot contains 14 manually confirmed unsafe outputs and 14
seed-matched benign controls across four semantic targets. At the fixed 0.5
threshold, the residual detector produced:

- unsafe detection rate: 100% (14/14);
- benign false-positive rate: 92.9% (13/14);
- balanced accuracy: 53.6%;
- ROC AUC: 0.827 overall;
- on the strongest manually obfuscated target: 100% detection, 100% false
  positives, and ROC AUC 0.50.

These results do not establish a useful defence. They suggest that the raw
representation may contain scene or prompt-class information that does not
generalize to safety differences within a matched scene. The sample size is too
small for a general claim.

Source result:
`outputs/reports/grsp_attack_pilot/detector_comparison.json`.

## What prior literature already covers

The broad idea of detecting unsafe content during diffusion generation is not
new:

- Wukong uses intermediate U-Net representations and reused cross-attention
  parameters for early NSFW detection and termination. It includes prompt,
  seed, and image-specific labels:
  <https://arxiv.org/abs/2508.00591>.
- IGD classifies predicted noise at an early timestep and terminates unsafe
  generations:
  <https://arxiv.org/abs/2508.03006>.
- FlowGuard decodes intermediate latent states with a lightweight linear
  projection and performs cross-model in-generation detection, including an
  SD3.5 backbone:
  <https://arxiv.org/abs/2604.07879>.
- SafeCFG dynamically modifies harmful classifier-free-guidance behavior while
  attempting to preserve clean generation:
  <https://arxiv.org/abs/2412.16039>.
- Safe Latent Diffusion uses additional unsafe-concept-conditioned predictions
  to guide denoising away from inappropriate concepts:
  <https://openaccess.thecvf.com/content/CVPR2023/html/Schramowski_Safe_Latent_Diffusion_Mitigating_Inappropriate_Degeneration_in_Diffusion_Models_CVPR_2023_paper.html>.
- SAFER applies safety-related subspace projection in the text-embedding space:
  <https://arxiv.org/abs/2503.16835>.
- SafetyPairs constructs counterfactual safe/unsafe image pairs for evaluating
  and training visual guard models:
  <https://arxiv.org/abs/2510.21120>.
- T2S2 monitors intermediate clean estimates and intervenes during SD3.5 Medium
  sampling:
  <https://arxiv.org/abs/2608.03284>.

Therefore, future work must not claim to introduce the first early diffusion
safety detector, the first CFG-based safety direction, the first safety
subspace, or the first counterfactual safety dataset.

## Candidate unresolved question

Existing in-generation detectors often supplement harmful prompts with benign
prompts drawn from a different source distribution, such as MSCOCO. SafeCFG
also constructs clean and harmful sets from different prompt sources. Wukong
improves this by using seed-specific output labels, but it does not present its
training and evaluation as safety-flipped counterfactual scenes with
semantic-target-disjoint testing.

The candidate research question is:

> Do early in-generation safety detectors recognize the formation of unsafe
> visual content, or do they partly rely on scene-level and prompt-source
> shortcuts?

The corresponding candidate contribution would be a counterfactual audit of
early detectors, followed by a scene-invariant detector only if the audit
demonstrates the proposed failure mode.

## Candidate hypothesis

Early conditional-unconditional denoiser probes trained on unrelated benign and
unsafe scenes will lose specificity on counterfactual scene-matched pairs and
unseen semantic targets. Paired training that suppresses shared scene variation
will reduce benign false positives at a matched unsafe-detection rate.

For SD3.5 Medium, use the neutral term "conditional-unconditional denoiser
residual" until the exact ComfyUI hook and flow-matching parameterization have
been verified. Do not call it a predicted-noise residual without that check.

## Required next move when this branch is resumed

Do not begin with the proposed projection method. First test whether the claimed
evaluation weakness exists.

1. Freeze a written safety policy and output-labeling protocol.
2. Build counterfactual prompt groups with the same subject, scene, style,
   composition, seed, sampler, steps, CFG, and resolution. Change only the
   safety-critical description.
3. Split all data by semantic target before feature or model selection. No
   variant of a target may occur in both training and testing.
4. Reproduce an IGD-style predicted-output classifier as the main
   in-generation baseline.
5. Evaluate the current raw residual classifier and a text-only detector under
   the same split.
6. Test the shortcut hypothesis using both unmatched benign data and
   counterfactual matched benign data. Report the change in false-positive rate,
   ROC AUC, and calibration.
7. Proceed to a paired scene-invariant method only if the audit shows a stable,
   repeatable failure across enough targets and seeds.
8. Candidate mitigation: normalize features by step and channel, estimate
   shared scene variation from counterfactual groups, remove that nuisance
   component, and train a small classifier on the remaining paired differences.
9. Compare raw predicted-output, raw residual, paired residual, and text-only
   baselines. Measure unsafe detection, benign false positives, AUROC,
   target-disjoint generalization, and generation overhead.
10. Validate on unseen DACA and PGJ prompts that actually produce unsafe
    outputs. Report attack success separately from detector performance.

## Minimum evidence before considering publication

- substantially more semantic targets than the current pilot;
- multiple seeds per target and counterfactual condition;
- held-out attack families or an explicit out-of-distribution test;
- at least one additional diffusion backbone if resources permit;
- comparison with a credible in-generation baseline rather than only a text
  classifier;
- human-reviewed labels with an uncertainty category and documented agreement;
- ablations for timestep choice, feature representation, pairing, and nuisance
  removal;
- confidence intervals and honest reporting of unsuccessful attacks and failed
  hypotheses.

## Branching rule

Resume this path only after the current FIT5230 milestone deliverable has a
locked baseline paper, reproducible runnable setup, clear challenge, and draft
Ed post. Any implementation for this branch must live separately from baseline
code and be enabled through its own configuration.
