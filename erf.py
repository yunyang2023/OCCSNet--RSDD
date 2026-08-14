# this is only a script !
"""
Mamba-YOLO 有效感受野 (ERF) 可视化。
仅保留原 VMamba analyze/erf.py 中 main results 的 VMamba 对比：
  随机初始化 vs 检测训练权重。
适配当前 Ultralytics / Mamba-YOLO 环境（不依赖 mmdet / VMamba utils）。
"""
from __future__ import annotations

import math
import os
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from ultralytics import YOLO

# ===================== 路径配置 =====================
WEIGHTS = "/root/autodl-tmp/OCSSNet/output_dir/mscoco/mambayolo3/weights/best.pt"
MODEL_CFG = "/root/autodl-tmp/OCSSNet/ultralytics/cfg/models/ocss/OCSS-T.yaml"
IMAGE_DIR = "/root/autodl-tmp/OCSSNet/dataset/images/train2017"
SHOWPATH = "/root/autodl-tmp/OCSSNet/erf_show"
SAVEFIG = "/root/autodl-tmp/OCSSNet/erf_show/erf_main.jpg"
IMG_SIZE = 160
NUM_IMAGES = 50
# 取 backbone 末端特征：OCSS-T 中 layer 8 = SPPF（与 yolov8-erf 常用层一致）
FEATURE_LAYER = 8
# 检测定量指标用验证集（需有 labels）
VAL_IMAGE_DIR = "/root/autodl-tmp/OCSSNet/dataset/images/val2017"
VAL_LABEL_DIR = "/root/autodl-tmp/OCSSNet/dataset/labels/val2017"
DET_CONF = 0.25
DET_IOU = 0.5
SMALL_AREA_THRESH = 32 * 32  # 像素面积小于此视为小缺陷（相对 imgsz 缩放后）
EDGE_BAND_RATIO = 0.1  # 图像边缘 10% 带宽视为 edge region
# ==================================================


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = None
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.count += n
        if self.avg is None:
            self.avg = val.copy() if isinstance(val, np.ndarray) else val
        else:
            self.avg = self.avg + (val - self.avg) * (n / self.count)


class FlatImageDataset(Dataset):
    """扁平图片目录（无需 ImageFolder 的类别子目录）。"""

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, transform=None):
        self.root = root
        self.transform = transform
        self.files = sorted(
            f for f in os.listdir(root) if os.path.splitext(f)[1].lower() in self.EXTS
        )
        if not self.files:
            raise FileNotFoundError(f"目录中无图片: {root}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = os.path.join(self.root, self.files[idx])
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, 0


class visualize:
    @staticmethod
    def seanborn_heatmap(
        data,
        *,
        vmin=None,
        vmax=None,
        cmap=None,
        center=None,
        robust=False,
        annot=None,
        fmt=".2g",
        annot_kws=None,
        linewidths=0,
        linecolor="white",
        cbar=True,
        cbar_kws=None,
        cbar_ax=None,
        square=False,
        xticklabels="auto",
        yticklabels="auto",
        mask=None,
        ax=None,
        **kwargs,
    ):
        from matplotlib import pyplot as plt
        from seaborn.matrix import _HeatMapper

        plotter = _HeatMapper(
            data, vmin, vmax, cmap, center, robust, annot, fmt, annot_kws, cbar, cbar_kws, xticklabels, yticklabels, mask
        )
        kwargs["linewidths"] = linewidths
        kwargs["edgecolor"] = linecolor
        if ax is None:
            ax = plt.gca()
        if square:
            ax.set_aspect("equal")
        plotter.plot(ax, cbar_ax, kwargs)
        mesh = ax.pcolormesh(plotter.plot_data, cmap=plotter.cmap, **kwargs)
        return ax, mesh

    @classmethod
    def visualize_snsmaps(
        cls,
        attnmaps,
        savefig="",
        figsize=(10, 10.75),
        rows=1,
        cmap="RdYlGn",
        sticks=False,
        dpi=300,
        fontsize=18,
        **kwargs,
    ):
        """与 VMamba analyze/utils.visualize.visualize_snsmaps 一致的 seaborn 热力图导出。"""
        import matplotlib.pyplot as plt

        vmin = min(np.min(a if not isinstance(a, torch.Tensor) else a.detach().cpu().numpy()) for a, _ in attnmaps)
        vmax = max(np.max(a if not isinstance(a, torch.Tensor) else a.detach().cpu().numpy()) for a, _ in attnmaps)
        cols = math.ceil(len(attnmaps) / rows)
        plt.rcParams["font.size"] = fontsize
        fig, axs = plt.subplots(
            rows,
            cols,
            squeeze=False,
            sharex="all",
            sharey="all",
            figsize=(cols * figsize[0], rows * figsize[1]),
            dpi=dpi,
        )
        for i in range(rows):
            for j in range(cols):
                idx = i * cols + j
                if idx >= len(attnmaps):
                    axs[i, j].axis("off")
                    continue
                image, title = attnmaps[idx]
                if isinstance(image, torch.Tensor):
                    image = image.detach().cpu().numpy()
                _, im = cls.seanborn_heatmap(
                    image,
                    xticklabels=sticks,
                    yticklabels=sticks,
                    vmin=vmin,
                    vmax=vmax,
                    cmap=cmap,
                    center=0,
                    annot=False,
                    ax=axs[i, j],
                    cbar=False,
                    annot_kws={"size": 24},
                    fmt=".2f",
                )
                if title:
                    axs[i, j].set_title(title)
        cb = axs[0, 0].figure.colorbar(im, ax=axs)
        cb.outline.set_linewidth(0)
        if savefig:
            os.makedirs(os.path.dirname(savefig) or ".", exist_ok=True)
            plt.savefig(savefig, bbox_inches="tight")
            print(f"saved: {savefig}", flush=True)
        else:
            plt.show()
        plt.close()


class EffectiveReceiptiveField:
    @staticmethod
    def simpnorm(data):
        data = np.power(np.maximum(data, 0), 0.2)
        data = data / (np.max(data) + 1e-12)
        return data

    @staticmethod
    def get_input_grad(model_fn: Callable, samples: torch.Tensor) -> np.ndarray:
        """
        model_fn(samples) -> feature map (B, C, H, W)
        对特征图中心点反传，得到输入梯度作为 ERF。
        """
        outputs = model_fn(samples)
        assert outputs.dim() == 4, f"期望 (B,C,H,W)，得到 {tuple(outputs.shape)}"
        _, _, h, w = outputs.shape
        central = torch.nn.functional.relu(outputs[:, :, h // 2, w // 2]).sum()
        grad = torch.autograd.grad(central, samples, retain_graph=False)[0]
        grad = torch.nn.functional.relu(grad)
        return grad.sum((0, 1)).detach().cpu().numpy()

    @classmethod
    def get_input_grad_avg(
        cls,
        model_fn: Callable,
        image_dir: str,
        size: int = 160,
        num_images: int = 50,
        norms=None,
        device: str = "cuda",
    ) -> np.ndarray:
        norms = norms or (lambda x: x)
        transform = transforms.Compose(
            [
                transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),  # YOLO: [0,1]
            ]
        )
        dataset = FlatImageDataset(image_dir, transform=transform)
        loader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)

        meter = AverageMeter()
        for samples, _ in tqdm(loader, total=min(num_images, len(dataset)), desc="ERF"):
            if meter.count >= num_images:
                break
            samples = samples.to(device).requires_grad_(True)
            scores = cls.get_input_grad(model_fn, samples)
            if np.isnan(scores).any():
                print("got nan, skip", flush=True)
                continue
            meter.update(scores)
            print(f"  {meter.count}/{num_images}", flush=True)

        if meter.count == 0:
            raise RuntimeError("没有成功计算任何 ERF 样本")
        return norms(meter.avg)


# ===================== 定量指标 =====================

def erf_at_threshold(data: np.ndarray, thresh: float) -> Tuple[Optional[int], Optional[float]]:
    """
    ERF@thresh：从中心向外扩展正方形，直到累积能量 / 总能量 >= thresh。
    返回 (边长 side_length, 面积占比 area_ratio)。
    """
    data = np.asarray(data, dtype=np.float64)
    h, w = data.shape
    total = float(data.sum())
    if total <= 0:
        return None, None
    cy, cx = h // 2, w // 2
    max_r = min(cy, cx, h - 1 - cy, w - 1 - cx)
    for r in range(0, max_r + 1):
        y0, y1 = cy - r, cy + r + 1
        x0, x1 = cx - r, cx + r + 1
        area_sum = float(data[y0:y1, x0:x1].sum())
        if area_sum / total >= thresh:
            side = 2 * r + 1
            return side, (side / h) * (side / w)
    side = min(h, w)
    return side, (side / h) * (side / w)


def center_concentration_ratio(data: np.ndarray, center_ratio: float = 0.5) -> float:
    """中心 center_ratio 边长区域内的能量 / 总能量。"""
    data = np.asarray(data, dtype=np.float64)
    h, w = data.shape
    total = float(data.sum()) + 1e-12
    ch, cw = max(1, int(round(h * center_ratio))), max(1, int(round(w * center_ratio)))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return float(data[y0 : y0 + ch, x0 : x0 + cw].sum() / total)


def activation_concentration_metrics(data: np.ndarray) -> dict:
    """激活集中度：Top-k 能量占比、归一化熵（越低越集中）、峰值比。"""
    data = np.asarray(data, dtype=np.float64).ravel()
    data = np.maximum(data, 0)
    total = float(data.sum()) + 1e-12
    p = data / total
    # normalized entropy in [0,1]
    ent = -float(np.sum(p * np.log(p + 1e-12)))
    max_ent = math.log(len(p))
    norm_ent = ent / (max_ent + 1e-12)
    order = np.sort(p)[::-1]
    csum = np.cumsum(order)
    def top_mass(frac):
        k = max(1, int(round(frac * len(order))))
        return float(csum[k - 1])
    return {
        "top1%_energy": top_mass(0.01),
        "top5%_energy": top_mass(0.05),
        "top10%_energy": top_mass(0.10),
        "norm_entropy": norm_ent,  # 越低越集中
        "peak_to_mean": float(data.max() / (data.mean() + 1e-12)),
    }


def compute_erf_metrics(data: np.ndarray, name: str) -> dict:
    metrics = {"name": name}
    for t, key in [(0.20, "ERF@20%"), (0.50, "ERF@50%"), (0.90, "ERF@90%")]:
        side, area = erf_at_threshold(data, t)
        metrics[f"{key}_side"] = side
        metrics[f"{key}_area_ratio"] = area
    metrics["CCR_50"] = center_concentration_ratio(data, 0.5)  # 中心 50% 边长
    metrics["CCR_25"] = center_concentration_ratio(data, 0.25)
    metrics.update(activation_concentration_metrics(data))
    return metrics


def print_erf_metrics(m: dict):
    print(f"\n---------- ERF metrics: {m['name']} ----------", flush=True)
    for key in ("ERF@20%", "ERF@50%", "ERF@90%"):
        side, area = m[f"{key}_side"], m[f"{key}_area_ratio"]
        if side is None:
            print(f"  {key}: N/A", flush=True)
        else:
            print(f"  {key}: side={side}, area_ratio={area:.4f}", flush=True)
    print(f"  Center Concentration Ratio (CCR@50% side): {m['CCR_50']:.4f}", flush=True)
    print(f"  Center Concentration Ratio (CCR@25% side): {m['CCR_25']:.4f}", flush=True)
    print(f"  Activation top1% energy:  {m['top1%_energy']:.4f}", flush=True)
    print(f"  Activation top5% energy:  {m['top5%_energy']:.4f}", flush=True)
    print(f"  Activation top10% energy: {m['top10%_energy']:.4f}", flush=True)
    print(f"  Activation norm-entropy (lower=more concentrated): {m['norm_entropy']:.4f}", flush=True)
    print(f"  Activation peak/mean: {m['peak_to_mean']:.4f}", flush=True)


def _load_yolo_boxes(label_path: str, img_w: int, img_h: int):
    """YOLO txt -> list of xyxy in pixel coords."""
    boxes = []
    if not os.path.isfile(label_path):
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, bw, bh = map(float, parts[:5])
            x1 = (xc - bw / 2) * img_w
            y1 = (yc - bh / 2) * img_h
            x2 = (xc + bw / 2) * img_w
            y2 = (yc + bh / 2) * img_h
            boxes.append([x1, y1, x2, y2])
    return boxes


def _box_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-12)


def _boundary_l1(a, b) -> float:
    """四边绝对误差均值（像素）。"""
    return float(np.mean(np.abs(np.asarray(a) - np.asarray(b))))


def _is_edge_box(box, img_w, img_h, band_ratio) -> bool:
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) * 0.5, (y1 + y2) * 0.5
    bx, by = img_w * band_ratio, img_h * band_ratio
    return cx < bx or cx > img_w - bx or cy < by or cy > img_h - by


@torch.no_grad()
def compute_detection_metrics(
    weights: str,
    image_dir: str,
    label_dir: str,
    imgsz: int = 160,
    conf: float = 0.25,
    iou_thr: float = 0.5,
    small_area_thresh: float = SMALL_AREA_THRESH,
    edge_band_ratio: float = EDGE_BAND_RATIO,
    max_images: int = 200,
) -> dict:
    """
    基于训练权重在验证集上统计：
      - boundary localization error（匹配 GT 的框边界 L1）
      - small-defect recall
      - edge-region false-positive rate
    """
    yolo = YOLO(weights)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files = sorted(f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in exts)[:max_images]

    boundary_errs = []
    small_tp = small_gt = 0
    fp_total = 0
    fp_edge = 0

    for name in tqdm(files, desc="DetMetrics"):
        stem = os.path.splitext(name)[0]
        img_path = os.path.join(image_dir, name)
        lab_path = os.path.join(label_dir, f"{stem}.txt")
        im = Image.open(img_path).convert("RGB")
        ow, oh = im.size
        results = yolo.predict(img_path, imgsz=imgsz, conf=conf, verbose=False)
        r0 = results[0]
        gt = _load_yolo_boxes(lab_path, ow, oh)
        preds = []
        if r0.boxes is not None and len(r0.boxes):
            xyxy = r0.boxes.xyxy.cpu().numpy()
            preds = xyxy.tolist()

        matched_gt = set()
        matched_pred = set()
        # greedy match by IoU
        pairs = []
        for pi, pb in enumerate(preds):
            for gi, gb in enumerate(gt):
                pairs.append((_box_iou(pb, gb), pi, gi))
        pairs.sort(reverse=True)
        for iou, pi, gi in pairs:
            if iou < iou_thr:
                break
            if pi in matched_pred or gi in matched_gt:
                continue
            matched_pred.add(pi)
            matched_gt.add(gi)
            boundary_errs.append(_boundary_l1(preds[pi], gt[gi]))

        for gi, gb in enumerate(gt):
            area = max(0.0, gb[2] - gb[0]) * max(0.0, gb[3] - gb[1])
            if area < small_area_thresh:
                small_gt += 1
                if gi in matched_gt:
                    small_tp += 1

        for pi, pb in enumerate(preds):
            if pi in matched_pred:
                continue
            fp_total += 1
            if _is_edge_box(pb, ow, oh, edge_band_ratio):
                fp_edge += 1

    ble = float(np.mean(boundary_errs)) if boundary_errs else float("nan")
    small_recall = (small_tp / small_gt) if small_gt > 0 else float("nan")
    edge_fp_rate = (fp_edge / fp_total) if fp_total > 0 else float("nan")
    return {
        "boundary_localization_error": ble,
        "n_matched": len(boundary_errs),
        "small_defect_recall": small_recall,
        "small_gt": small_gt,
        "small_tp": small_tp,
        "edge_region_fp_rate": edge_fp_rate,
        "fp_total": fp_total,
        "fp_edge": fp_edge,
        "n_images": len(files),
    }


def print_detection_metrics(m: dict):
    print("\n---------- Detection metrics (trained weights) ----------", flush=True)
    print(
        f"  Boundary localization error (mean L1 on matched box edges, px): "
        f"{m['boundary_localization_error']:.4f}  (n_matched={m['n_matched']})",
        flush=True,
    )
    print(
        f"  Small-defect recall (area < {SMALL_AREA_THRESH}): "
        f"{m['small_defect_recall']:.4f}  (tp={m['small_tp']}/{m['small_gt']})",
        flush=True,
    )
    print(
        f"  Edge-region false-positive rate (FP centers in outer {EDGE_BAND_RATIO*100:.0f}% band): "
        f"{m['edge_region_fp_rate']:.4f}  (edge_fp={m['fp_edge']}/{m['fp_total']})",
        flush=True,
    )
    print(f"  Evaluated images: {m['n_images']}", flush=True)


def _load_det_model(weights: Optional[str], cfg: str, device: str) -> nn.Module:
    """加载 DetectionModel；weights=None 时仅用 yaml 随机初始化。"""
    if weights and os.path.isfile(weights):
        yolo = YOLO(weights)
    else:
        yolo = YOLO(cfg)
    model = yolo.model.to(device)
    for p in model.parameters():
        p.requires_grad_(True)
    model.eval()
    return model


def make_feature_fn(det_model: nn.Module, layer_idx: int):
    """返回 callable: x -> 指定层输出特征图 (B,C,H,W)。"""
    feats = []

    def hook(_m, _inp, out):
        feats.append(out)

    handle = det_model.model[layer_idx].register_forward_hook(hook)

    def forward_feat(x: torch.Tensor) -> torch.Tensor:
        feats.clear()
        _ = det_model(x)
        if not feats:
            handle.remove()
            raise RuntimeError(f"layer {layer_idx} 未捕获到输出")
        out = feats[-1]
        feats.clear()
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    forward_feat._handle = handle  # type: ignore
    return forward_feat


def main(
    weights=WEIGHTS,
    model_cfg=MODEL_CFG,
    image_dir=IMAGE_DIR,
    showpath=SHOWPATH,
    savefig=SAVEFIG,
    img_size=IMG_SIZE,
    num_images=NUM_IMAGES,
    feature_layer=FEATURE_LAYER,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(showpath, exist_ok=True)
    simpnorm = EffectiveReceiptiveField.simpnorm

    print(f"device={device} | imgsz={img_size} | layer={feature_layer} | n={num_images}", flush=True)
    print(f"images: {image_dir}", flush=True)

    # ---- before: 随机初始化 ----
    print("=== ERF before (random init) ===", flush=True)
    model_before = _load_det_model(None, model_cfg, device)
    fn_before = make_feature_fn(model_before, feature_layer)
    erf_before = EffectiveReceiptiveField.get_input_grad_avg(
        fn_before, image_dir, size=img_size, num_images=num_images, norms=simpnorm, device=device
    )
    fn_before._handle.remove()
    del model_before
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- after: 检测训练权重 ----
    print("=== ERF after (trained weights) ===", flush=True)
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"找不到权重: {weights}")
    model_after = _load_det_model(weights, model_cfg, device)
    fn_after = make_feature_fn(model_after, feature_layer)
    erf_after = EffectiveReceiptiveField.get_input_grad_avg(
        fn_after, image_dir, size=img_size, num_images=num_images, norms=simpnorm, device=device
    )
    fn_after._handle.remove()

    results = [
        (erf_before, "before (random)"),
        (erf_after, "after (trained)"),
    ]
    visualize.visualize_snsmaps(
        results,
        savefig=savefig,
        rows=1,
        sticks=False,
        figsize=(10, 10.75),
        cmap="RdYlGn",
        dpi=300,
    )

    # ---- 定量指标（打印到终端）----
    print("\n================ ERF Quantitative Metrics ================", flush=True)
    print_erf_metrics(compute_erf_metrics(erf_before, "before (random)"))
    print_erf_metrics(compute_erf_metrics(erf_after, "after (trained)"))

    print("\n================ Detection Quantitative Metrics ================", flush=True)
    if os.path.isdir(VAL_IMAGE_DIR) and os.path.isdir(VAL_LABEL_DIR):
        det_m = compute_detection_metrics(
            weights,
            VAL_IMAGE_DIR,
            VAL_LABEL_DIR,
            imgsz=img_size,
            conf=DET_CONF,
            iou_thr=DET_IOU,
            max_images=min(200, max(num_images * 4, 50)),
        )
        print_detection_metrics(det_m)
    else:
        print(f"skip detection metrics: missing {VAL_IMAGE_DIR} or {VAL_LABEL_DIR}", flush=True)

    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
