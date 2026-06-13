# Reproduction Environments

This repository does not track virtual environments, CUDA toolkits, pretrained
weights, or cloned upstream repositories.

## PAct

Start from the upstream
[`PAct_env.yml`](https://github.com/PAct-project/PAct/blob/main/PAct_env.yml).
The successful smoke test documented here used:

```text
Python 3.10
torch 2.1.1
xformers 0.0.23+cu118
spconv 2.3.6
opencv-python 4.9.0
imageio 2.34.0
einops 0.8.1
numpy 1.24.3
Pillow 12.2.0
```

See [`../reproductions/pact/README.md`](../reproductions/pact/README.md).

## O-Voxel

Install O-Voxel from the upstream TRELLIS.2 repository. The successful
visualization test used:

```text
Python 3.10
torch 2.6.0+cu124
trimesh 4.12.2
matplotlib 3.11.0
numpy 2.4.6
scipy 1.17.1
```

O-Voxel also declares `cumesh` and `flex_gemm`; follow the upstream build
instructions for full mesh conversion support.

## 3D Representation Tutorial

```bash
python -m venv .venv
.venv/bin/pip install -r environments/tutorial-requirements.txt
.venv/bin/jupyter nbconvert \
  --to notebook --execute --inplace \
  notes/3d_representations_tutorial.ipynb
```
