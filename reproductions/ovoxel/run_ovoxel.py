"""Run real O-Voxel (TRELLIS.2) on the DamagedHelmet and visualize the result.

Geometry  : o_voxel.convert.mesh_to_flexible_dual_grid  (Flexible Dual Grid + QEF)
Material   : o_voxel.convert.textured_mesh_to_volumetric_attr (volumetric PBR)
Visualize  : dual-grid surface vertices colored by base_color, at two resolutions.
"""
import argparse
import os
import time
import numpy as np
import torch
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import o_voxel

def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a textured mesh to O-Voxel and visualize two resolutions."
    )
    parser.add_argument("--input", required=True, help="Input textured mesh or GLB.")
    parser.add_argument(
        "--output",
        default="notes/figures/ovoxel_real_helmet.png",
        help="Output PNG path.",
    )
    return parser.parse_args()


def normalize(asset):
    aabb = asset.bounding_box.bounds
    center = (aabb[0] + aabb[1]) / 2
    scale = 0.99999 / (aabb[1] - aabb[0]).max()
    asset.apply_translation(-center)
    asset.apply_scale(scale)
    return asset


def to_ovoxel(asset, res):
    mesh = asset.to_mesh() if hasattr(asset, "to_mesh") else asset
    V = torch.from_numpy(mesh.vertices).float()
    F = torch.from_numpy(mesh.faces).long()
    t0 = time.time()
    vidx, dual, inter = o_voxel.convert.mesh_to_flexible_dual_grid(
        V, F, grid_size=res, aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
        face_weight=1.0, boundary_weight=0.2, regularization_weight=1e-2)
    t_geo = time.time() - t0
    # material (volumetric PBR)
    t0 = time.time()
    vidx_m, attr = o_voxel.convert.textured_mesh_to_volumetric_attr(
        asset, grid_size=res, aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]])
    t_mat = time.time() - t0
    # align material color to geometry voxels via encoded index
    enc_g = o_voxel.serialize.encode_seq(vidx).cpu().numpy()
    enc_m = o_voxel.serialize.encode_seq(vidx_m).cpu().numpy()
    bc = attr["base_color"].float().cpu().numpy()
    if bc.max() > 1.5: bc = bc / 255.0
    bc = bc[:, :3]
    m2c = {int(e): bc[i] for i, e in enumerate(enc_m)}
    colors = np.array([m2c.get(int(e), (0.6,0.6,0.6)) for e in enc_g])
    colors = np.clip(colors, 0, 1)
    # dual vertices -> world coords (they are in [0,1]^3 grid-normalized)
    P = dual.cpu().numpy()
    print(
        f"  RES={res:4d}: {len(vidx):>7,} occupied voxels | "
        f"geometry {t_geo:.2f}s material {t_mat:.2f}s | "
        f"metallic mean={attr['metallic'].float().mean():.2f} "
        f"roughness mean={attr['roughness'].float().mean():.2f}"
    )
    return P, colors, len(vidx)


def main():
    args = parse_args()
    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    print("Loading mesh:", input_path)
    asset = normalize(trimesh.load(input_path))
    fig = plt.figure(figsize=(13, 6.2))
    for i, res in enumerate([64, 256]):
        P, C, n = to_ovoxel(asset, res)
        ax = fig.add_subplot(1, 2, i + 1, projection="3d")
        if len(P) > 120000:
            sel = np.random.default_rng(0).choice(len(P), 120000, replace=False)
            P, C = P[sel], C[sel]
        ax.scatter(
            P[:, 0], P[:, 2], P[:, 1], c=C,
            s=(6 if res == 64 else 1.2), marker="o", depthshade=True
        )
        ax.set_title(
            f"O-Voxel @ {res}^3: {n:,} occupied voxels\n"
            "(points: Flexible Dual Grid vertices; color: PBR base color)",
            fontsize=10,
        )
        ax.set_xlabel("x")
        ax.set_ylabel("z")
        ax.set_zlabel("y")
        try:
            ax.set_box_aspect((1, 1, 1))
        except Exception:
            pass
        ax.view_init(elev=18, azim=-70)
    plt.suptitle("TRELLIS.2 O-Voxel: textured mesh conversion", y=0.98, fontsize=12)
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    print("Saved:", output_path)


if __name__ == "__main__":
    main()
