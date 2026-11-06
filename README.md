# VGGT-Splat

Casual phone video → [VGGT](https://github.com/facebookresearch/vggt) feed-forward poses, points and depth → 3D Gaussian Splatting, compared against the usual COLMAP → 3DGS pipeline.

**Project page**: https://chu-jiaming.github.io/vggt-splat/

**Questions**

1. **Frame budget**: how many frames does a casual capture need? Compare COLMAP + 3DGS and VGGT → 3DGS at 8 / 16 / 32 / all training frames.
2. **VGGT depth prior**: with few frames, how much quality does VGGT depth supervision recover?

6 scenes (self-captured desk / lawn / room; Mip-NeRF 360 garden / bicycle / counter) × 4 frame budgets × 3 methods = 72 runs, plus seed repeats.

## Findings

1. **Few frames: VGGT is reliable, COLMAP is not. Many frames: COLMAP is better. The crossover is around N = 16–32.**
   At N = 8, COLMAP fails outright on 4 of 6 scenes and registers only 3/8 and 6/8 training frames on the other two; at N = 16 it fails on 3 scenes and registers 3/16 (bicycle) and 7/16 (desk) on two more. VGGT never fails. With all frames, COLMAP leads by 2.1–4.6 dB on five scenes and 0.7 dB on desk.
2. **With all frames, VGGT loses on precision, not coverage.** Restricted to test pixels that the training views do see, VGGT is still behind (garden 21.7 vs 26.3 dB, bicycle 20.7 vs 22.8, counter 24.9 vs 27.2). Its renders are visibly smeared (higher PSNR than at N = 32 but worse LPIPS on lawn, garden and bicycle), consistent with feed-forward poses that are not mutually consistent enough for 3DGS to resolve fine detail.
3. **The depth prior changes geometry but not novel-view quality.** It raises the local correlation between rendered depth and VGGT depth on training views from 0.4–0.8 to 0.7–0.9, yet across seed repeats only two differences exceed noise: desk N = 32 gets 0.68 dB worse and lawn N = 8 gets 0.17 dB better.
4. **With few frames, the bottleneck is coverage.** Test pixels not seen by any training view are 0.5–9 dB worse than seen ones, and the per-frame unseen fraction is strongly anti-correlated with PSNR. A depth prior on training views cannot constrain those regions, which also explains finding 3.
5. **Noise level.** Re-running the same configuration with different seeds moves PSNR by a standard deviation of 0.06–0.35 dB, so single-run differences below about 0.3 dB are not conclusive.
6. **Speed.** VGGT estimates poses for all frames in 47–73 s; COLMAP (exhaustive matching, GPU) takes 167–790 s.

## Results

Test PSNR (dB) after test-time pose optimization, averaged over the test frames that every usable method registered for that scene and N. `fail` = pose estimation or alignment failed, or the alignment was flagged as inconsistent (center error > 10%). Numbers in brackets are COLMAP's registered / total test frames when it did not register all of them; the other methods are averaged over the same subset.

| Scene | Method | N=8 | N=16 | N=32 | N=all |
|---|---|---:|---:|---:|---:|
| desk | COLMAP | fail | 25.75 (7/16) | 23.47 | **32.46** |
| desk | VGGT | **24.36** | **30.20** | **32.34** | 31.79 |
| desk | VGGT + depth | **24.36** | **30.20** | 31.79 | 30.97 |
| lawn | COLMAP | 13.75 (15/23) | fail | **18.67** (22/23) | **21.44** (22/23) |
| lawn | VGGT | **15.66** | **17.47** | 18.51 | 19.35 |
| lawn | VGGT + depth | 15.61 | 17.41 | 18.64 | 19.39 |
| room | COLMAP | fail | fail | fail | **23.40** (14/17) |
| room | VGGT | **13.77** | **16.62** | 18.24 | 21.20 |
| room | VGGT + depth | **13.77** | 16.52 | **18.54** | 21.64 |
| garden | COLMAP | 14.27 | **19.57** | **23.19** | **26.34** |
| garden | VGGT | 15.64 | 18.13 | 19.93 | 21.79 |
| garden | VGGT + depth | **15.70** | 18.17 | 19.99 | 22.39 |
| bicycle | COLMAP | fail | 12.03 (9/24) | 14.25 | **23.40** |
| bicycle | VGGT | 14.42 | **16.45** | 18.26 | 21.30 |
| bicycle | VGGT + depth | **14.46** | 16.08 | **18.28** | 20.93 |
| counter | COLMAP | fail | fail | 21.04 | **27.25** |
| counter | VGGT | 14.12 | **19.02** | 21.58 | 25.04 |
| counter | VGGT + depth | **14.40** | 18.72 | **22.44** | 24.71 |

COLMAP at N = 32 on desk and bicycle registers only 15/32 and 10/32 training frames, which is why it drops below N = 16 there. SSIM, LPIPS, timings and the failure list are produced by `scripts/summarize.py` (see below).

Pose estimation time (s, single RTX 4090, includes model loading and image I/O):

| Scene | COLMAP all | VGGT all |
|---|---:|---:|
| desk (110 frames) | 167 | 47 |
| lawn (156) | 790 | 60 |
| room (114) | 505 | 49 |
| garden (161) | 770 | 68 |
| bicycle (166) | 380 | 68 |
| counter (180) | 480 | 73 |

## Protocol

- **Data.** Each scene's frames are split into a test set (every 8th frame after removing blurry frames, as in Mip-NeRF 360) and nested training sets of N = 8 / 16 / 32 / all frames sampled uniformly from the rest (`scripts/extract_frames.py`). Self-captured scenes are 1080p phone videos sampled at 3 fps and trained at 960×540; Mip-NeRF 360 scenes use `images_4`.
- **Poses.** COLMAP 4.2.1 with exhaustive matching (`scripts/run_colmap.sh`), or VGGT-1B in one forward pass with a shared camera and the top 50% most confident points as initialization (`scripts/vggt_to_colmap.py`).
- **Training.** gsplat 3DGS with per-camera pose refinement (`--pose_opt`), stopped at step 7,000 of the default 30k schedule. The depth-prior variant adds a Pearson loss between rendered inverse depth and VGGT depth, λ = 0.2 (chosen by a sweep on desk).
- **Evaluation** (`scripts/eval.py`). Test cameras come from a joint run on training + test frames; a Sim(3) fitted on the shared training cameras (rotation from camera orientations, scale and translation from centers) maps them into the trained model's frame; with the Gaussians frozen, each test pose is refined for 500 steps; PSNR / SSIM / LPIPS are measured after refinement. Methods are compared on the test frames that all of them registered.
- **Analysis.** `scripts/coverage_err.py` splits test error by whether a pixel is seen by any training view; `scripts/depth_diag.py` measures agreement between rendered depth and VGGT depth; `scripts/seed_repeat.sh` re-runs selected configurations with seeds 1 and 2.

## Reproducing

### Environment

Tested on Ubuntu 22.04, CUDA 12.8, one RTX 4090 (24 GB), Python 3.11:

| Component | Version |
|---|---|
| PyTorch | 2.9.1 + cu128 |
| gsplat | commit `512d366` (built from source) |
| VGGT | commit `a288dd0`, weights `facebook/VGGT-1B` |
| COLMAP | 4.2.1 built with CUDA |

```bash
uv venv .venv --python 3.11 && source .venv/bin/activate
uv pip install torch==2.9.1 torchvision==0.24.1 --index-url https://download.pytorch.org/whl/cu128
git clone --recursive https://github.com/nerfstudio-project/gsplat.git && git -C gsplat checkout 512d366b67073d77ca099ede742683c165dfc23b && git -C gsplat submodule update --init --recursive
uv pip install setuptools wheel ninja && uv pip install -e gsplat --no-build-isolation
uv pip install -r gsplat/examples/requirements.txt --no-build-isolation
git clone https://github.com/facebookresearch/vggt.git && git -C vggt checkout a288dd0f14786c93483e45524328726ab7b1b4ce
uv pip install -e vggt --no-deps && uv pip install einops safetensors huggingface_hub trimesh hydra-core omegaconf
git clone https://github.com/chu-jiaming/VGGT-Splat.git
```

3DGS training uses this repository's `trainer/`, a modified copy of gsplat `examples/` at the same commit (changes listed in [`trainer/NOTICE.md`](trainer/NOTICE.md)); the gsplat library and `trainer/` must stay on the same commit.

### Data

```
data/<scene>/frames/NNNN.jpg      all frames
data/<scene>/splits/              test.txt, train_{8,16,32,all}.txt (from extract_frames.py)
```

- **Mip-NeRF 360**: download [`360_v2.zip`](https://jonbarron.info/mipnerf360/), point `data/<scene>/frames` at `360_v2/<scene>/images_4` with a symlink, then run `python scripts/extract_frames.py data/<scene>`.
- **Self-captured scenes**: run `python scripts/extract_frames.py data/<scene>` on a folder containing `video.mp4`. The frames and splits used here will be attached to the v1.0.0 release.

### Run

`scripts/run_all.sh` runs pose estimation → joint run → training → evaluation for every scene × method × N and appends one row per run to `results/<scene>.csv`. It resumes from existing outputs. Paths and the Python environment are set at the top of the script (`DATA`, `RUNS`, `RESULTS`, `PATH`); adjust them for your machine.

```bash
SCENES="desk lawn room garden bicycle counter" METHODS="colmap vggt" scripts/run_all.sh
SCENES="desk lawn room garden bicycle counter" METHODS=vggt_depth DEPTH_LAMBDAS=0.2 scripts/run_all.sh
python scripts/summarize.py --results results --runs runs       # tables, curves, failure list → results/summary/
```

Analysis and exports:

```bash
python scripts/coverage_err.py --runs runs --out results/coverage.csv     # --save_maps for GT | render | error | coverage images
python scripts/depth_diag.py --runs runs --out results/depth_diag.csv
scripts/seed_repeat.sh job desk vggt_depth 32 1 0.2 && scripts/seed_repeat.sh summary
python scripts/export_splats.py --runs runs --out results/export --list "garden/vggt_all garden/colmap_all"   # .splat + compressed .ply
RUNS=runs scripts/rerender_traj.sh garden/vggt_all garden/colmap_all   # slower fly-through videos (450 frames) from existing checkpoints
```

One run (pose estimation, joint run, 7k-step training, evaluation) takes about 5–10 minutes on an RTX 4090, and up to about 35 minutes for COLMAP with all frames.

## Limitations

- One capture per scene and, for the main grid, one seed; seed repeats cover only a subset of configurations.
- Training stops at 7k steps for every run. On desk N = 16, the 30k checkpoint of the same run was slightly worse on test frames (extra steps overfit the training views) with the same method ranking; other configurations were not checked.
- The coverage split uses each model's own geometry to decide what is seen, so it is an approximation.
- Test cameras are obtained from a separate joint run and refined with the scene frozen; this is the standard protocol for pose-free methods but still differs from evaluating with ground-truth test poses.

## License

Code in this repository is released under the [MIT License](LICENSE), except `trainer/`, which is adapted from gsplat and released under the Apache License 2.0 (see [`trainer/NOTICE.md`](trainer/NOTICE.md)).

Third-party models and libraries keep their own licenses. In particular, VGGT weights are used under their original license (non-commercial use); see the VGGT repository.
