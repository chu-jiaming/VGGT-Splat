# trainer/ — 来自 gsplat examples

本目录的代码复制自 [nerfstudio-project/gsplat](https://github.com/nerfstudio-project/gsplat) 的 `examples/`，
commit `512d366b67073d77ca099ede742683c165dfc23b`，按 Apache License 2.0 使用（全文见同目录 `LICENSE`）。
本仓库其余代码的 MIT 许可证不适用于本目录。

## 复制了哪些文件

| 本目录 | gsplat 原路径 |
| --- | --- |
| `simple_trainer.py` | `examples/simple_trainer.py` |
| `utils.py` | `examples/utils.py` |
| `gsplat_viewer.py` | `examples/gsplat_viewer.py` |
| `exif.py` | `examples/exif.py` |
| `lib_bilagrid.py` | `examples/lib_bilagrid.py` |
| `datasets/{__init__,colmap,normalize,traj}.py` | `examples/datasets/` 同名文件 |
| `requirements.txt` | `examples/requirements.txt` |
| `LICENSE` | `LICENSE` |

只复制 `simple_trainer.py` 在 COLMAP 数据上运行需要的文件。没复制的 `datasets/ncore.py` 只在 `--data_type ncore` 时才被 import。

## 版本约束

这些脚本调用 gsplat 库（`rasterization`、`DefaultStrategy` 等，含 CUDA 扩展），库必须是同一个 commit `512d366`
（安装见 `docs/setup.md` B4）。不要单独升级 gsplat 库；要升级就重新从新 commit 复制这些文件，再把下面的修改重新做一遍。

## 相对原版的修改

原样复制的版本是本目录的第一个提交，之后每处修改单独提交，`git log -p trainer/` 可查完整 diff。修改过的文件在这里登记：

| 文件 | 修改 |
| --- | --- |
| `datasets/colmap.py` | `test_every <= 0` 时全部帧训练、验证集为空（原逻辑 index 0 总被拿去验证） |
| `simple_trainer.py` | 验证集为空时 `eval()` 改评训练集（输出前缀 `train_`），渲染时训练帧位姿经过 `pose_perturb` / `pose_adjust`，和训练时一致 |
