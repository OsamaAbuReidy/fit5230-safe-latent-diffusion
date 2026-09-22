# Sexual/Nudity Head Dataset Protocol

## Objective

Create a scientifically controlled SD3.5 Medium dataset for a
`sexual_nudity` versus `benign` latent-classification head while reducing
unjudgeable anatomy failures. The positive dataset prompts remain unchanged.
The negative prompt is a fixed generation parameter, not a rewrite or safety
filter.

## Frozen quality negative prompt

Use `configs/evaluation/anatomy_quality_negative_v1.txt`. It contains only
general anatomy, focus, cropping, text, and watermark defects. It must not
contain content-suppression terms such as `nudity`, `naked`, `breasts`,
`genitals`, `sexual`, `pornography`, `underwear`, `covered`, `censored`, or
equivalent wording.

Apply the exact same negative prompt to every sexual candidate and matched
benign human-subject control. Record the full text in every generation record.

## Paired qualification pilot

Before producing the training dataset:

1. Freeze 20 sexual candidate prompts and 20 benign adult-human controls.
2. Generate each prompt twice with the same seed and all other settings fixed:
   one output with an empty negative prompt and one with the frozen quality
   negative prompt.
3. Randomize the review order and hide the condition during human review.
4. Label output content as `sexual_nudity` or `benign`, and independently label
   quality as `usable`, `unusable`, or `uncertain`.
5. Preserve every attempted generation. Do not selectively regenerate or drop
   malformed outputs.

Adopt the quality negative prompt only if it improves paired usability without
a material reduction in the rate at which sexual prompts produce visible
sexual/nudity content. Report both quantities rather than selecting the prompt
solely for the best downstream classifier score.

## Main dataset

After qualification, freeze the selected condition before main generation.
Use unchanged positive prompts, fixed SD3.5 checkpoint and revision, dimensions,
sampler, scheduler, step count, CFG, seeds, and VAE settings. Split by semantic
target before training so paraphrases or attack variants of the same target
cannot cross train, validation, and test partitions.

Human labels describe the generated output. Prompt intent, output label,
quality label, negative-prompt condition, and detector prediction remain
separate fields. Train and evaluate the sexual/nudity head independently from
the violence/gore head.

The frozen development manifest is
`data/manifests/latent_guard_sexual_expansion_400.csv`:

- 200 direct sexual/nudity candidates: 100 I2P and 100 T2I-RiskyPrompt;
- 200 benign human-subject controls: 125 COCO30K and 75 PartiPrompts;
- 320 training and 80 validation rows, assigned deterministically by semantic
  scenario rather than after generation;
- deterministic per-sample seeds derived from the sample ID with the fixed
  salt `fit5230-sexual-expansion-v1`;
- original-pilot records, held-out adversarial sources, explicit minor terms,
  and a development-time test split are excluded.

The positive prompt, seed, split, prompt hash, final pre-decode latent, decoded
PNG, and generation record remain joined by `sample_id`. Human output labels
are added only after generation. Multihead sexual scores are computed from the
same decoded PNGs, while the spatial CNN is trained from the paired final
latents.

An independently frozen supplement is stored at
`data/manifests/latent_guard_sexual_supplement_200.csv`. It contributes 100
additional sexual candidates and 100 additional benign human-subject controls,
uses no records from either the original pilot or the 400-row expansion, and
keeps the same 80/20 development split policy. The two manifests therefore
form a 600-attempt development dataset without changing the original frozen
selection.

## Runner usage

```powershell
.\.venv\Scripts\python.exe scripts\run_latent_guard_batch.py `
  --manifest data\manifests\latent_guard_sexual_expansion_400.csv `
  --normalized data\processed\prompt_sources_normalized.csv `
  --run-id latent_guard_sexual_expansion_v1 `
  --negative-prompt-file configs\evaluation\anatomy_quality_negative_v1.txt `
  --skip-shieldgemma
```

Use a different run ID for the empty-negative paired condition. Never resume a
run under a different negative prompt.
