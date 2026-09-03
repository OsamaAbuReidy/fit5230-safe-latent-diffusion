# Public experiment data

This directory contains small, derived artifacts that make the public notebooks
reproducible without distributing raw prompts, generated images, model weights,
or full latent tensors.

`milestone2_pooled_latents.npz` contains one row for each usable SD3.5 pilot
generation: pooled final-latent features, the frozen split, the human output
label, and continuous scores from the released Multihead Detector. Its matching
JSON file documents the schema and integrity checks.

The bundle is intended for reproducing the lightweight classifier comparison;
it cannot reconstruct the source images or prompts.
