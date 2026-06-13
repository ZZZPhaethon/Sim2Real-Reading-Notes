# PAct Smoke-Test Reproduction

This reproduction validates the complete inference path on one NVIDIA
A100-SXM4-40GB GPU:

```text
RGBA image + OpenEXR part mask
  -> DINOv2 conditioning
  -> Stage 1 sparse-structure sampling
  -> Stage 2 SLAT and articulation prediction
  -> articulation and exploded-part rendering
```

The test used the upstream Microwave example and deliberately reduced sampling
to two steps per stage. It verifies executability and output generation, not the
paper's final visual quality or quantitative metrics.

## Setup

1. Clone [PAct](https://github.com/PAct-project/PAct).
2. Create the upstream environment.
3. Download [`PAct000/PAct`](https://huggingface.co/PAct000/PAct).
4. Put one `*_processed.png` image and its matching `*_mask.exr` file under a
   subdirectory of `smoke_inputs/`.

## Command

Run from the cloned PAct repository:

```bash
ATTN_BACKEND=xformers SPARSE_ATTN_BACKEND=xformers \
python infer_imgs.py \
  --model_path pretrain/PAct \
  --data_dir smoke_inputs \
  --outdir outputs/smoke \
  --batch_size 1 \
  --ss_steps 2 \
  --slat_steps 2 \
  --arti_mean_num 2 \
  --render_num_frames 4 \
  --no-save_video_grid \
  --no-save_cond_vis_grid
```

## Compatibility Fixes

The tested upstream snapshot required three small engineering fixes:

1. Add `--model_path` and pass it to `PActPipeline.from_pretrained`.
2. Construct `ImageConditioned_dataset` from `args.data_dir` instead of the
   hard-coded example directory.
3. Read OpenEXR masks explicitly with `cv2.imread(..., cv2.IMREAD_UNCHANGED)`
   and save exploded-part frame sequences with `imageio.mimsave`.

These changes do not alter model architecture or sampling equations.

## Results

![Microwave condition and exploded-part result](../../notes/figures/pact_reproduction_microwave.png)

- [Articulation animation](../../notes/figures/pact_reproduction_articulation.mp4)
- [Exploded-part animation](../../notes/figures/pact_reproduction_exploded.mp4)

The detailed interpretation is in
[`notes/03-PAct.md`](../../notes/03-PAct.md#本地复现实验).
