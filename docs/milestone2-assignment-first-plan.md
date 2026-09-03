# Milestone 2 assignment-first plan

Status date: 2 September 2026

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

The current result is deliberately reported as a feasibility result, not a
successful final defence. The pooled latent probe contains measurable signal,
especially for violence, but is weaker than the image-space Multihead baseline.

## Work required before research extensions

1. Publish a self-contained Milestone 2 notebook with pre-rendered outputs.
2. Make the notebook train the pooled latent probe from the tracked, prompt-free
   feature bundle and evaluate the untouched test split.
3. Reproduce the Multihead-versus-latent comparison table and at least one
   visualization from row-level data.
4. Include a functional challenge interface and an explicit scoring contract.
5. Remove hardcoded local paths and document every dependency and artifact.
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

## Immediate next action

Build and execute `notebooks/milestone2_show_of_force.ipynb` around the public
feature bundle. The notebook must contain pre-rendered audit output, model
selection output, the final comparison table, confusion matrices, interpretation,
limitations, and challenge instructions. It should run from a fresh Colab clone
without any local Windows paths.

## Parked research branch

Spatial CNNs, decoded-image-to-latent knowledge distillation, proxy latents, and
publication claims are postponed until the assignment notebook and peer
engagement evidence are secure. They remain the planned improvement for the
Milestone 2-to-3 period.
