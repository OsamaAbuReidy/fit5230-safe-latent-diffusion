# Milestone 1 Ed post draft

## Items to fill before posting

- Team name: `[TEAM NAME]`
- Team member names: `[MEMBER NAMES]`
- Member photos: `[ATTACH PHOTOS]`
- Final Colab URL: `[COLAB URL]`
- Repository URL: <https://github.com/OsamaAbuReidy/fit5230-safe-latent-diffusion>

## Suggested forum heading

`[TEAM NAME] - Defending SD3.5 Medium from prompt jailbreaks`

## Post body

### Project

We are a Light-side Text-to-Image team investigating how to defend Stable
Diffusion 3.5 Medium against natural-language jailbreak attacks while preserving
normal generation for benign users.

Our main reference is **JailbreakDiffBench: A Comprehensive Benchmark for
Jailbreaking Diffusion Models** (ICCV 2025):

- Paper: <https://www.openaccess.thecvf.com/content/ICCV2025/papers/Jin_JailbreakDiffBench_A_Comprehensive_Benchmark_for_Jailbreaking_Diffusion_Models_ICCV_2025_paper.pdf>
- Reference implementation:
  <https://github.com/Jinxiaolong1129/JailbreakDiffusionBench>
- Our repository:
  <https://github.com/OsamaAbuReidy/fit5230-safe-latent-diffusion>
- Our Colab: `[COLAB URL]`

### Why this problem is interesting

Text-only moderation can be bypassed by prompts that express prohibited intent
indirectly. JailbreakDiffBench evaluates attacks including DACA and PGJ and
shows that modern diffusion pipelines, including SD3.5 Medium, are not uniformly
robust. However, blocking every unusual or human-related prompt is not a useful
defence. A practical safeguard must detect genuinely unsafe generations while
maintaining a low false-positive rate for benign requests.

This makes the problem technically challenging because prompt intent, model
behavior, random seed, generated content, and the safety detector can disagree.
A prompt bypass is not necessarily a successful attack unless it also produces
a policy-violating output.

### Problem statement

Our goal is to build and evaluate a lightweight defence for SD3.5 Medium against
DACA/PGJ-style prompt jailbreaks. We will measure:

1. whether the adversarial prompt bypasses the input defence;
2. whether the generated image is actually unsafe;
3. whether the output remains aligned with the attacker's intended concept;
4. how often matched benign prompts are incorrectly blocked; and
5. the generation-time overhead introduced by the defence.

Attack success and detector performance will be reported separately. Seeds,
model settings, prompt identifiers, outputs, and manual labels are recorded for
reproducibility.

### Initial customization and setup

We have created our own reproducible SD3.5 Medium workflow rather than only
cloning the reference repository. The current project includes:

- a working SD3.5 Medium generation workflow in ComfyUI;
- fixed-seed automated prompt execution;
- structured manifests for benign, direct, DACA, and PGJ-style conditions;
- a custom node that records early conditional-unconditional denoiser features;
- scripts for group-aware feature analysis and detector comparison; and
- a small, human-reviewed pilot with matched benign controls.

The pilot is deliberately treated as preliminary. It demonstrated that our raw
early-feature detector over-blocked benign controls, so we are not presenting it
as a successful defence. This failure gives us a concrete baseline and prevents
us from selecting a method using only favorable examples.

### Interactive challenge: break the gate without breaking utility

Can another team expose a weakness in our defence under a fixed SD3.5 Medium
configuration?

There are two challenge tracks:

- **Unsafe bypass:** provide a natural-language adversarial prompt that passes
  the defence and produces a human-verified policy-violating output.
- **Benign false positive:** provide a clearly benign prompt that the defence
  incorrectly blocks.

The released challenge will freeze the safety policy, model configuration,
seeds, query budget, submission format, and scoring code. We will report prompt
bypass, human-verified attack success, semantic alignment, benign false
positives, and runtime overhead separately. Uncertain or failed generations
will not be silently counted as successful attacks.

This challenge is non-trivial because maximizing unsafe bypass alone is not
enough: an aggressive defence can reduce attacks simply by blocking benign
users. Other teams can therefore attack either the security or the utility of
our method.

### Next step

For Milestone 2, we will freeze the baseline configuration, expand the
target-disjoint evaluation set, compare the unmodified pipeline with simple
text and in-generation detector baselines, and select one technically justified
defence modification.
