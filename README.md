# VGGT-Splat

Casual phone video → [VGGT](https://github.com/facebookresearch/vggt) feed-forward poses, points and depth → 3D Gaussian Splatting, without COLMAP.

**Questions**

1. **Frame budget** — how many frames does a casual capture need? Compare COLMAP+3DGS and VGGT→3DGS at 8 / 16 / 32 / all frames.
2. **VGGT depth prior** — with few frames, how much quality does VGGT depth supervision recover?

Pose refinement during 3DGS training is used as an engineering step.

**Status**: work in progress (target 2026-11-06).

## License

Code in this repository is released under the [MIT License](LICENSE).

Third-party models and libraries keep their own licenses. In particular, VGGT weights are used under their original license (non-commercial use); see the VGGT repository.
