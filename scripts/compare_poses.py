"""把一组估计位姿和参考位姿（如官方 COLMAP 的 sparse/0）对齐后比较。

两边都用 pycolmap 读（COLMAP 二进制或文本格式均可），按图像文件名配对：
1. 用相机中心做 Umeyama Sim(3) 对齐（估计 → 参考），消掉坐标系和尺度的差异
2. 中心误差：对齐后的距离，除以参考轨迹的尺度（相机中心到其质心的中位距离），得到无量纲误差
3. 旋转误差：对齐旋转后，每帧相对旋转的角度（度）
4. 焦距：按图像宽度换算到同一分辨率后比较

用法：
    python compare_poses.py --ref data/360_v2/garden/sparse/0 --est runs/garden/colmap_16/sparse/0
"""
import argparse

import numpy as np
import pycolmap


def load(path):
    """返回 {name: (R_w2c, C)} 和 {name: (fx, width)}。"""
    rec = pycolmap.Reconstruction(path)
    poses, focals = {}, {}
    for im in rec.images.values():
        if not im.has_pose:
            continue
        R = im.cam_from_world().rotation.matrix()
        poses[im.name] = (R, im.projection_center())
        cam = rec.cameras[im.camera_id]
        focals[im.name] = (cam.params[0], cam.width)  # PINHOLE / SIMPLE_* 的第一个参数都是 fx（或 f）
    return poses, focals


def umeyama(src, dst):
    """求 s, R, t 使 dst ≈ s R src + t（src, dst: (N, 3)）。"""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    xs, xd = src - mu_s, dst - mu_d
    cov = xd.T @ xs / len(src)
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = np.trace(np.diag(D) @ S) / xs.var(0).sum()
    t = mu_d - s * R @ mu_s
    return s, R, t


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ref", required=True, help="参考模型目录")
    p.add_argument("--est", required=True, help="估计模型目录")
    args = p.parse_args()

    ref, ref_f = load(args.ref)
    est, est_f = load(args.est)
    names = sorted(set(ref) & set(est))
    print(f"参考 {len(ref)} 帧，估计 {len(est)} 帧，共同 {len(names)} 帧")
    if len(names) < 3:
        print("共同帧少于 3，无法对齐")
        return

    C_ref = np.stack([ref[n][1] for n in names])
    C_est = np.stack([est[n][1] for n in names])
    s, R_align, t = umeyama(C_est, C_ref)
    C_aligned = s * C_est @ R_align.T + t

    scale = np.median(np.linalg.norm(C_ref - C_ref.mean(0), axis=1))
    err_c = np.linalg.norm(C_aligned - C_ref, axis=1) / scale

    # 估计的世界系经 R_align 转到参考世界系：R_w2c_aligned = R_est @ R_align^T
    err_r = []
    for n in names:
        R_rel = ref[n][0] @ (est[n][0] @ R_align.T).T
        err_r.append(np.degrees(np.arccos(np.clip((np.trace(R_rel) - 1) / 2, -1, 1))))
    err_r = np.array(err_r)

    # 焦距换算到估计模型的分辨率
    f_ratio = np.array([est_f[n][0] / (ref_f[n][0] * est_f[n][1] / ref_f[n][1]) for n in names])

    print(f"Sim(3) 尺度 {s:.4f}；参考轨迹尺度（中心到质心的中位距离）{scale:.4f}")
    print(f"中心误差 / 轨迹尺度：中位 {np.median(err_c):.4f}，最大 {err_c.max():.4f}")
    print(f"旋转误差（度）：中位 {np.median(err_r):.3f}，最大 {err_r.max():.3f}")
    print(f"fx 估计 / 参考（同分辨率）：中位 {np.median(f_ratio):.4f}，范围 {f_ratio.min():.4f}–{f_ratio.max():.4f}")


if __name__ == "__main__":
    main()
