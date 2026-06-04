# ComfyUI KSampler Matrix Lab

ComfyUI KSampler Matrix Lab is a custom node for comparing multiple sampler and scheduler combinations in one labeled image grid.

The node runs each selected sampler/scheduler pair sequentially with the same model, conditioning, latent, VAE, seed, steps, CFG, and denoise settings. It then decodes the results and returns one final `IMAGE` output that can be connected to `Preview Image` or `Save Image`.

## Features

- Compare sampler and scheduler combinations in a single grid.
- Dynamic sampler and scheduler dropdowns from the local ComfyUI installation.
- Up to 9 sampler slots and 9 scheduler slots.
- `None` option for unused sampler or scheduler slots.
- Sequential generation to avoid large all-at-once batches.
- Same-seed comparison mode.
- Increment-per-cell seed mode.
- Per-cell labels with sampler and scheduler names.
- Top run header with model, VAE, CLIP, steps, CFG, and denoise metadata.
- Error placeholder cells when `continue_on_error` is enabled.
- Safety limit for maximum combinations.

## Installation

Clone this repository into your ComfyUI `custom_nodes` directory:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/btitkin/ComfyUI-KSampler-Matrix-Lab.git
```

Restart ComfyUI after installation.

## Node

The node appears as:

```text
KSampler Matrix Lab
```

Category:

```text
ComfyUI-KSampler-Matrix-Lab
```

Output:

```text
IMAGE
```

## Basic Workflow

Use the node with a standard ComfyUI generation workflow:

```text
Load Checkpoint / model loader
CLIP Text Encode positive
CLIP Text Encode negative
Empty Latent Image
KSampler Matrix Lab
Preview Image or Save Image
```

Connect:

- `MODEL` to `model`
- positive conditioning to `positive`
- negative conditioning to `negative`
- latent to `latent_image`
- `VAE` to `vae`

## Sampler and Scheduler Selection

The node provides dropdown slots:

```text
sampler_01 ... sampler_09
scheduler_01 ... scheduler_09
```

Set unused slots to:

```text
None
```

Duplicate sampler or scheduler selections are ignored after the first occurrence.

## Grid Layout

The final image grid uses:

- columns for schedulers,
- rows for samplers,
- generated images as cells,
- optional grid lines,
- repeated per-cell sampler/scheduler labels,
- an optional top metadata header.

The top metadata header includes:

```text
Model
VAE
CLIP
Steps
CFG
Denoise
```

Model, VAE, and CLIP names are inferred from the ComfyUI workflow when possible. Some custom loaders may not expose enough metadata, in which case the header may show `unknown`.

## Seed Modes

```text
same_seed_for_all
```

Uses the same seed for every cell. This is best for direct sampler/scheduler comparison.

```text
increment_per_cell
```

Uses `seed + cell_index` for each cell. This is useful for quick variation previews.

## Error Handling

When `continue_on_error` is enabled, a failed sampler/scheduler combination produces an error placeholder cell and the rest of the matrix continues.

When disabled, the node stops on the first failed combination.

## Notes

- Generation is sequential by design.
- Very large grids can use significant RAM during final image composition.
- If the input latent batch size is greater than 1, the first decoded image is used for each grid cell.
- Full behavior depends on the samplers and schedulers available in the local ComfyUI installation.

## License

MIT
