"""测试集评测（M1 ④）：Sim(3) 对齐测试帧初始位姿 → 冻结高斯做测试时位姿优化 → PSNR/SSIM/LPIPS。

协议沿用 BARF / NoPe-NeRF：
1. 测试帧初始位姿来自“训练帧 + 测试帧”的联合运行（VGGT 或 COLMAP，--joint_dir）；训练只用训练帧的结果，
   测试帧不泄露进高斯
2. 对齐目标是训练坐标系里 pose opt 之后的训练帧位姿：Parser(normalize) 的 c2w 再乘 ckpt["pose_adjust"]
   （simple_trainer 的 --ckpt 评测分支不加载 pose_adjust，所以这里不复用 runner.eval）
3. 按帧名配对共同训练帧，用相机中心做 Umeyama Sim(3)（联合坐标系 → 训练坐标系），一个变换作用于全部测试帧；
   内参（--intrinsics）：默认 train，测试帧用训练运行内参的逐元素中位数（所有帧来自同一台相机；
   VGGT 每帧焦距 ±2% 是估计噪声，联合运行会重估焦距，6 自由度位姿优化补不回缩放）。
   joint_ratio：联合运行内参 × 共同训练帧“训练焦距 / 联合焦距”的中位数；joint：联合运行内参原样
4. 每个测试帧单独做测试时位姿优化：高斯冻结，新建 CameraOptModule(1)，loss 与训练一致
   （lerp(L1, 1-SSIM, ssim_lambda)），步数 / 学习率由参数固定，所有方法同一预算
5. 优化前、优化后的指标都记：前者反映位姿 + 对齐质量，后者反映高斯本身的质量
   指标实现与 trainer 一致（torchmetrics；LPIPS 网络取训练 cfg 的 lpips_net）

联合运行在 COLMAP 里没注册上的测试帧记 registered=false、不计入平均，单独报注册率。

输入：
    --result_dir   一次训练的 result_dir（含 cfg.yml、ckpts/），data_dir / data_factor 等从 cfg.yml 读
    --joint_dir    联合运行的输出目录（sparse/0 + images/）；factor > 1 时缺 images_<f> 会自动建软链接
    --test_list    测试帧列表（data/<scene>/splits/test.txt）
输出（--out，默认 <result_dir>/eval.json）：超参、对齐诊断、逐帧指标与位姿、已注册帧的平均

自检：--test_list 传训练帧列表，优化前 PSNR 应接近训练集指标，优化后基本不变。

用法：
    python scripts/eval.py --result_dir runs/desk/vggt_16/gs --joint_dir runs/desk/vggt_16_joint \\
        --test_list data/desk/splits/test.txt
"""
import argparse
import glob
import json
import os
import re
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
import yaml

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "trainer"))
sys.path.insert(0, os.path.join(REPO, "scripts"))

import imageio.v2 as imageio  # noqa: E402
from compare_poses import umeyama  # noqa: E402
from datasets.colmap import Parser  # noqa: E402
from gsplat.losses import ssim_loss  # noqa: E402
from gsplat.rendering import rasterization  # noqa: E402
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure  # noqa: E402
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity  # noqa: E402
from utils import CameraOptModule  # noqa: E402


# ---------- 训练配置与 ckpt ----------

class _CfgLoader(yaml.SafeLoader):
    """cfg.yml 是 yaml.dump(vars(cfg))，含 !!python/object（strategy 等）；只取普通字段，其余按普通容器读。"""


def _ignore_python_tag(loader, suffix, node):
    if isinstance(node, yaml.MappingNode):
        return loader.construct_mapping(node, deep=True)
    if isinstance(node, yaml.SequenceNode):
        return loader.construct_sequence(node, deep=True)
    return loader.construct_scalar(node)


_CfgLoader.add_multi_constructor("tag:yaml.org,2002:python/", _ignore_python_tag)


def load_cfg(result_dir):
    with open(os.path.join(result_dir, "cfg.yml")) as f:
        cfg = yaml.load(f, Loader=_CfgLoader)
    if cfg.get("app_opt"):
        raise ValueError("训练开了 app_opt：测试帧没有每帧外观嵌入，这个脚本不支持")
    if cfg.get("post_processing"):
        raise ValueError(f"训练开了 post_processing={cfg['post_processing']}：这个脚本不支持")
    if cfg.get("camera_model", "pinhole") != "pinhole":
        raise ValueError(f"只支持 pinhole，cfg 里是 {cfg['camera_model']}")
    return cfg


def latest_ckpt(result_dir):
    paths = glob.glob(os.path.join(result_dir, "ckpts", "ckpt_*_rank0.pt"))
    if not paths:
        raise FileNotFoundError(f"{result_dir}/ckpts 下没有 ckpt_*_rank0.pt")
    return max(paths, key=lambda p: int(re.search(r"ckpt_(\d+)_rank0", p).group(1)))


def load_gaussians(ckpt, device):
    """激活后的高斯参数，全部冻结（不求梯度）。"""
    s = {k: v.to(device).detach() for k, v in ckpt["splats"].items()}
    return {
        "means": s["means"],
        "quats": s["quats"],  # rasterization 内部归一化
        "scales": torch.exp(s["scales"]),
        "opacities": torch.sigmoid(s["opacities"]),
        "colors": torch.cat([s["sh0"], s["shN"]], 1),
    }


# ---------- 位姿 ----------

def ensure_factor_dir(data_dir, factor):
    """gsplat Parser 在 factor > 1 时要求 images_<f>/ 存在（是 jpg 就从 images/ 自动缩放成 images_<f>_png/）。"""
    if factor <= 1:
        return
    d = os.path.join(data_dir, f"images_{factor}")
    if not os.path.exists(d):
        os.symlink("images", d)
        print(f"建软链接 {d} -> images")


def load_target(cfg, ckpt):
    """训练坐标系里 pose opt 之后的训练帧位姿 {name: c2w (4, 4) float64}、训练轨迹尺度、
    训练时的内参 {name: K (3, 3)}（已按 data_factor 缩放）。"""
    p = Parser(cfg["data_dir"], factor=cfg["data_factor"],
               normalize=cfg.get("normalize_world_space", True), test_every=0)
    c2w = torch.from_numpy(p.camtoworlds).float()
    if cfg.get("pose_opt"):
        if "pose_adjust" not in ckpt:
            raise KeyError("cfg 里 pose_opt=True，但 ckpt 没有 pose_adjust")
        adj = CameraOptModule(len(c2w))
        adj.load_state_dict(ckpt["pose_adjust"])  # 帧数对不上会在这里报错
        with torch.no_grad():
            # 训练集 image_id 就是 Parser 里按文件名排序后的下标（test_every=0）
            c2w = adj(c2w, torch.arange(len(c2w)))
    c2w = c2w.double().numpy()
    centers = c2w[:, :3, 3]
    scale = float(np.median(np.linalg.norm(centers - centers.mean(0), axis=1)))
    Ks = {n: p.Ks_dict[p.camera_ids[i]].astype(np.float64) for i, n in enumerate(p.image_names)}
    return dict(zip(p.image_names, c2w)), scale, Ks


def focal_ratio(joint, target_Ks):
    """训练内参 / 联合运行内参的焦距比，取共同训练帧的中位数。

    联合运行会重新估计焦距（COLMAP 共享相机 desk 上差 2.6%，VGGT 每帧 ±2%）。高斯是按训练内参拟合的，
    焦距不一致相当于缩放画面，6 自由度的位姿优化补不回来（11-05 自检：COLMAP 优化后 PSNR 卡在 26）。
    和 Sim(3) 一样只用训练帧求，作用于全部测试帧。
    """
    common = sorted(set(joint) & set(target_Ks))
    r = np.array([target_Ks[n][0, 0] / joint[n][1][0, 0] for n in common])
    return float(np.median(r)), {"focal_ratio_median": float(np.median(r)),
                                 "focal_ratio_min": float(r.min()), "focal_ratio_max": float(r.max())}


def load_joint(joint_dir, factor):
    """联合运行（未归一化）的 {name: (c2w, K, (W, H), image_path)}。"""
    ensure_factor_dir(joint_dir, factor)
    p = Parser(joint_dir, factor=factor, normalize=False, test_every=0)
    out = {}
    for i, name in enumerate(p.image_names):
        cid = p.camera_ids[i]
        out[name] = (p.camtoworlds[i].astype(np.float64), p.Ks_dict[cid].astype(np.float64),
                     p.imsize_dict[cid], p.image_paths[i])
    return out


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def align_sim3(joint, target, target_scale):
    """联合坐标系 → 训练坐标系的 Sim(3)，用共同训练帧的相机中心拟合。返回 (4x4 矩阵 S, s, 诊断)。"""
    common = sorted(set(joint) & set(target))
    if len(common) < 3:
        raise ValueError(f"共同训练帧只有 {len(common)} 帧，无法做 Sim(3) 对齐")
    C_src = np.stack([joint[n][0][:3, 3] for n in common])
    C_dst = np.stack([target[n][:3, 3] for n in common])
    s, R, t = umeyama(C_src, C_dst)
    S = np.eye(4)
    S[:3, :3] = s * R
    S[:3, 3] = t

    err_c, err_r = [], []
    for n in common:
        c2w = apply_sim3(S, s, joint[n][0])
        err_c.append(np.linalg.norm(c2w[:3, 3] - target[n][:3, 3]) / target_scale)
        err_r.append(rot_angle_deg(c2w[:3, :3].T @ target[n][:3, :3]))
    info = {
        "n_common": len(common),
        "common": common,
        "scale": float(s),
        "target_traj_scale": target_scale,
        "center_err_rel_median": float(np.median(err_c)),
        "center_err_rel_max": float(np.max(err_c)),
        "rot_err_deg_median": float(np.median(err_r)),
        "rot_err_deg_max": float(np.max(err_r)),
    }
    return S, s, info


def apply_sim3(S, s, c2w):
    """c2w' = S @ c2w，再把旋转部分的尺度除掉（相机中心 s R C + t，朝向 R R_c）。"""
    out = S @ c2w
    out[:3, :3] /= s
    return out


# ---------- 渲染、位姿优化、指标 ----------

def render(gs, c2w, K, W, H, cfg):
    """c2w: (1, 4, 4)，K: (3, 3) → (1, H, W, 3)，未截断。"""
    colors, _, _ = rasterization(
        means=gs["means"], quats=gs["quats"], scales=gs["scales"],
        opacities=gs["opacities"], colors=gs["colors"],
        viewmats=torch.linalg.inv(c2w), Ks=K[None], width=W, height=H,
        sh_degree=cfg.get("sh_degree", 3),
        near_plane=cfg.get("near_plane", 0.01), far_plane=cfg.get("far_plane", 1e10),
        packed=cfg.get("packed", False),
        rasterize_mode="antialiased" if cfg.get("antialiased") else "classic",
        camera_model="pinhole",
    )
    return colors[..., :3]


def test_time_opt(gs, c2w0, K, W, H, gt, cfg, steps, lr, lr_final_ratio):
    """高斯冻结，只优化一个 CameraOptModule(1)（与训练时 pose opt 同一参数化，右乘修正）。"""
    device = c2w0.device
    mod = CameraOptModule(1).to(device)
    mod.zero_init()
    opt = torch.optim.Adam(mod.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=lr_final_ratio ** (1.0 / max(steps, 1)))
    ids = torch.zeros(1, dtype=torch.long, device=device)
    lam = cfg.get("ssim_lambda", 0.2)
    gt_p = gt.permute(0, 3, 1, 2)
    losses = []
    for _ in range(steps):
        c2w = mod(c2w0[None], ids)
        img = render(gs, c2w, K, W, H, cfg)
        l1 = F.l1_loss(img, gt)
        loss = torch.lerp(l1, ssim_loss(img.permute(0, 3, 1, 2), gt_p), lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())
    with torch.no_grad():
        c2w = mod(c2w0[None], ids)[0]
    return c2w, losses


class Metrics:
    def __init__(self, lpips_net, device):
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)
        if lpips_net == "alex":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex", normalize=True).to(device)
        elif lpips_net == "vgg":
            self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=False).to(device)
        else:
            raise ValueError(lpips_net)

    @torch.no_grad()
    def __call__(self, img, gt):
        a = img.clamp(0, 1).permute(0, 3, 1, 2)
        b = gt.permute(0, 3, 1, 2)
        return {"psnr": self.psnr(a, b).item(), "ssim": self.ssim(a, b).item(),
                "lpips": self.lpips(a, b).item()}


def to_uint8(x):
    return (x.clamp(0, 1)[0].cpu().numpy() * 255).round().astype(np.uint8)


# ---------- 主流程 ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--result_dir", required=True, help="训练的 result_dir（含 cfg.yml、ckpts/）")
    ap.add_argument("--ckpt", help="默认取 result_dir/ckpts 里步数最大的 rank0")
    ap.add_argument("--joint_dir", required=True, help="训练帧 + 测试帧联合运行的 COLMAP 目录")
    ap.add_argument("--test_list", required=True, help="测试帧列表，每行一个文件名")
    # 11-05 desk N=16 扫 steps {200,500,1000,2000} × lr {3e-4,1e-3,3e-3}：200 步 lr 1e-3 起平均指标不再变化
    # （VGGT 各帧 PSNR 到小数点后一位相同，COLMAP 平均差 ≤ 0.03 dB）；取 500 步留余量
    ap.add_argument("--tto_steps", type=int, default=500, help="每帧测试时位姿优化步数（所有方法统一）")
    ap.add_argument("--tto_lr", type=float, default=1e-3)
    ap.add_argument("--tto_lr_final_ratio", type=float, default=0.01, help="学习率指数衰减到初值的比例")
    ap.add_argument("--intrinsics", choices=["train", "joint_ratio", "joint"], default="train",
                    help="测试帧内参来源：训练运行中位数 / 联合运行 × 焦距比 / 联合运行原样")
    ap.add_argument("--save_renders", action="store_true", help="保存 GT | 优化前 | 优化后 拼图到 <out 同名目录>/")
    ap.add_argument("--out", help="默认 <result_dir>/eval.json")
    args = ap.parse_args()

    device = "cuda"
    torch.manual_seed(0)
    cfg = load_cfg(args.result_dir)
    ckpt_path = args.ckpt or latest_ckpt(args.result_dir)
    out_path = args.out or os.path.join(args.result_dir, "eval.json")
    render_dir = os.path.splitext(out_path)[0] + "_renders"
    if args.save_renders:
        os.makedirs(render_dir, exist_ok=True)

    with open(args.test_list) as f:
        test_names = [ln.strip() for ln in f if ln.strip()]
    print(f"ckpt {ckpt_path}\ndata_dir {cfg['data_dir']}  factor {cfg['data_factor']}  测试帧 {len(test_names)}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    target, target_scale, target_Ks = load_target(cfg, ckpt)
    leaked = sorted(set(test_names) & set(target))
    if leaked:
        print(f"注意：{len(leaked)} 个测试帧也在训练集里（自检模式才应如此），它们也参与了 Sim(3) 对齐")
    joint = load_joint(args.joint_dir, cfg["data_factor"])
    S, s, align_info = align_sim3(joint, target, target_scale)
    print(f"Sim(3)：共同帧 {align_info['n_common']}，尺度 {s:.4f}；"
          f"中心误差/轨迹尺度 中位 {align_info['center_err_rel_median']:.4f} 最大 {align_info['center_err_rel_max']:.4f}；"
          f"旋转误差 中位 {align_info['rot_err_deg_median']:.3f}° 最大 {align_info['rot_err_deg_max']:.3f}°")
    f_ratio, f_info = focal_ratio(joint, target_Ks)
    align_info.update(f_info)
    K_train = np.median(np.stack(list(target_Ks.values())), axis=0)
    align_info["intrinsics"] = args.intrinsics
    align_info["K_train_median"] = K_train.tolist()
    print(f"焦距比 训练/联合：中位 {f_info['focal_ratio_median']:.4f}，范围 "
          f"{f_info['focal_ratio_min']:.4f}–{f_info['focal_ratio_max']:.4f}；测试帧内参：{args.intrinsics}"
          + (f"（fx {K_train[0, 0]:.1f} fy {K_train[1, 1]:.1f}）" if args.intrinsics == "train" else ""))

    gs = load_gaussians(ckpt, device)
    metrics = Metrics(cfg.get("lpips_net", "alex"), device)
    frames = []
    t0 = time.time()
    for name in test_names:
        if name not in joint:
            print(f"{name}: 联合运行未注册")
            frames.append({"name": name, "registered": False})
            continue
        c2w_joint, K_np, (W, H), img_path = joint[name]
        c2w0 = torch.from_numpy(apply_sim3(S, s, c2w_joint)).float().to(device)
        if args.intrinsics == "train":
            K_np = K_train.copy()
        elif args.intrinsics == "joint_ratio":
            K_np = K_np.copy()
            K_np[0, 0] *= f_ratio
            K_np[1, 1] *= f_ratio
        K = torch.from_numpy(K_np).float().to(device)
        gt = torch.from_numpy(imageio.imread(img_path)[..., :3]).float().to(device)[None] / 255.0
        if gt.shape[1:3] != (H, W):
            raise ValueError(f"{name}: 图像 {tuple(gt.shape[1:3])} 与内参分辨率 {(H, W)} 不符")

        with torch.no_grad():
            img0 = render(gs, c2w0[None], K, W, H, cfg)
        m0 = metrics(img0, gt)
        c2w1, losses = test_time_opt(gs, c2w0, K, W, H, gt, cfg,
                                     args.tto_steps, args.tto_lr, args.tto_lr_final_ratio)
        with torch.no_grad():
            img1 = render(gs, c2w1[None], K, W, H, cfg)
        m1 = metrics(img1, gt)

        a, b = c2w0.double().cpu().numpy(), c2w1.double().cpu().numpy()
        frames.append({
            "name": name, "registered": True, "before": m0, "after": m1,
            "tto_loss_first": losses[0] if losses else None, "tto_loss_last": losses[-1] if losses else None,
            "tto_dt_rel": float(np.linalg.norm(b[:3, 3] - a[:3, 3]) / target_scale),
            "tto_drot_deg": rot_angle_deg(a[:3, :3].T @ b[:3, :3]),
            "c2w_before": a.tolist(), "c2w_after": b.tolist(),
        })
        print(f"{name}: PSNR {m0['psnr']:.2f} → {m1['psnr']:.2f}  SSIM {m0['ssim']:.3f} → {m1['ssim']:.3f}  "
              f"LPIPS {m0['lpips']:.3f} → {m1['lpips']:.3f}  "
              f"Δt/尺度 {frames[-1]['tto_dt_rel']:.4f} Δrot {frames[-1]['tto_drot_deg']:.2f}°")
        if args.save_renders:
            canvas = np.concatenate([to_uint8(gt), to_uint8(img0), to_uint8(img1)], axis=1)
            imageio.imwrite(os.path.join(render_dir, os.path.splitext(name)[0] + ".png"), canvas)

    reg = [f for f in frames if f["registered"]]
    mean = {stage: {k: float(np.mean([f[stage][k] for f in reg])) if reg else None
                    for k in ("psnr", "ssim", "lpips")}
            for stage in ("before", "after")}
    result = {
        "ckpt": os.path.abspath(ckpt_path),
        "data_dir": cfg["data_dir"],
        "joint_dir": os.path.abspath(args.joint_dir),
        "test_list": os.path.abspath(args.test_list),
        "data_factor": cfg["data_factor"],
        "lpips_net": cfg.get("lpips_net", "alex"),
        "tto": {"steps": args.tto_steps, "lr": args.tto_lr, "lr_final_ratio": args.tto_lr_final_ratio,
                "ssim_lambda": cfg.get("ssim_lambda", 0.2)},
        "align": align_info,
        "n_test": len(test_names),
        "n_registered": len(reg),
        "mean": mean,
        "time_s": time.time() - t0,
        "frames": frames,
    }
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"已注册 {len(reg)}/{len(test_names)}；平均 "
          + "  ".join(f"{st}: PSNR {mean[st]['psnr']:.2f} SSIM {mean[st]['ssim']:.3f} LPIPS {mean[st]['lpips']:.3f}"
                      for st in ("before", "after") if reg)
          + f"\n写入 {out_path}")


if __name__ == "__main__":
    main()
