"""COLMAP 文本格式（cameras.txt / images.txt / points3D.txt）的写出与四元数工具。

只负责写；读用 pycolmap（`pycolmap.Reconstruction(dir)`，gsplat 也是这么读的）。
文本格式跨 COLMAP / pycolmap 版本稳定，避免 pycolmap 构造 API 在 3.x → 4.x 间的不兼容。

约定（与 COLMAP 一致）：
- 外参是 world-to-camera（OpenCV 坐标系），q = (qw, qx, qy, qz)，相机中心 C = -R^T t
- 像素坐标以图像左上角为原点，第一个像素的中心在 (0.5, 0.5)
"""
import os
from dataclasses import dataclass, field

import numpy as np


@dataclass
class Camera:
    id: int
    model: str  # "PINHOLE": fx fy cx cy；"SIMPLE_PINHOLE": f cx cy
    width: int
    height: int
    params: np.ndarray


@dataclass
class Image:
    id: int
    qvec: np.ndarray  # (4,) w2c 旋转，qw qx qy qz
    tvec: np.ndarray  # (3,) w2c 平移
    camera_id: int
    name: str
    xys: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # (M, 2) 观测像素坐标
    point3D_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))  # (M,)，-1 表示无对应点


def rotmat_to_qvec(R):
    """3x3 旋转矩阵 → (qw, qx, qy, qz)，qw >= 0。"""
    R = np.asarray(R, dtype=np.float64)
    # 对称矩阵的最大特征向量法，数值稳定（同 COLMAP 的 read_write_model.py）
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.flat
    K = np.array([
        [Rxx - Ryy - Rzz, 0, 0, 0],
        [Ryx + Rxy, Ryy - Rxx - Rzz, 0, 0],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, 0],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    q = eigvecs[[3, 0, 1, 2], np.argmax(eigvals)]
    return -q if q[0] < 0 else q


def qvec_to_rotmat(q):
    w, x, y, z = q
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * z * x + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * z * x - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def write_model_text(out_dir, cameras, images, xyz, rgb, error=None):
    """写出 COLMAP 文本模型。

    xyz: (P, 3)，rgb: (P, 3) uint8，error: (P,)，默认 0。
    3D 点的 id 为 1..P（第 i 行对应 id i+1），track 由 images 里的 point3D_ids 反推。
    """
    os.makedirs(out_dir, exist_ok=True)
    P = len(xyz)
    error = np.zeros(P) if error is None else error

    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        f.write(f"# Camera list with one line of data per camera:\n#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n# Number of cameras: {len(cameras)}\n")
        for c in cameras:
            f.write(f"{c.id} {c.model} {c.width} {c.height} " + " ".join(f"{p:.10g}" for p in c.params) + "\n")

    tracks = [[] for _ in range(P)]
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        f.write(f"# Image list with two lines of data per image:\n#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n#   POINTS2D[] as (X, Y, POINT3D_ID)\n# Number of images: {len(images)}\n")
        for im in images:
            f.write(f"{im.id} " + " ".join(f"{v:.12g}" for v in (*im.qvec, *im.tvec)) + f" {im.camera_id} {im.name}\n")
            f.write(" ".join(f"{x:.8g} {y:.8g} {pid}" for (x, y), pid in zip(im.xys, im.point3D_ids)) + "\n")
            for idx, pid in enumerate(im.point3D_ids):
                if pid > 0:
                    tracks[pid - 1].append((im.id, idx))

    with open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        f.write(f"# 3D point list with one line of data per point:\n#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n# Number of points: {P}\n")
        for i in range(P):
            x, y, z = xyz[i]
            r, g, b = rgb[i]
            track = " ".join(f"{iid} {idx}" for iid, idx in tracks[i])
            f.write(f"{i + 1} {x:.8g} {y:.8g} {z:.8g} {r} {g} {b} {error[i]:.6g} {track}\n")
