# this is only a script !
"""
多缺陷处理规则：
- 一张图有 N 个缺陷 → 拆成 N 次可视化
- 每次只在该缺陷中心画 1 颗红星
- 各自输出 Activation Map + 四向扫描路径 (a0/a1/a2/a3)

适配当前 Mamba-YOLO / Ultralytics 环境（不再依赖 VMamba/mmdet 的 utils）。
"""
from __future__ import annotations

import gc
import json
import math
import os
from collections import defaultdict
from typing import List, Optional, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw
from torchvision import transforms

from ultralytics import YOLO
from ultralytics.nn.modules.common_utils_mbyolo import _continuous_scan_indices
from ultralytics.nn.modules.mamba_yolo import SS2D, VSSBlock

# ===================== 路径配置（按本机数据修改） =====================
WEIGHTS = "/root/autodl-tmp/OCSSNet/output_dir/mscoco/mambayolo3/weights/best.pt"
IMAGE_DIR = "/root/autodl-tmp/OCSSNet/dataset/images/val2017"
# COCO json（可选）；None 或文件不存在时改用 YOLO txt labels
COCO_ANN = None
LABEL_DIR = "/root/autodl-tmp/OCSSNet/dataset/labels/val2017"
SHOWPATH = "/root/autodl-tmp/OCSSNet/attnmap_show"
# None = 目录下全部有标注的图；单张时改为如 "rail_71.jpg"
# 注：当前 val2017 无 rail_43.jpg，暂用最接近的 rail_42.jpg；有 rail_43 后改回即可
TARGET_IMAGE = "rail_42.jpg"
# CPU 低内存配置（当前实例无 GPU / 内存约 2GB）：
# stage2 stride=16 → 160/16=10；有 GPU 时可改回 FEAT_HW=32, IMG_SIZE=512
STAGE = 2
BLOCK_ID = 1
FEAT_HW = 10
IMG_SIZE = FEAT_HW * 16  # 160
# ====================================================================


class visualize:
    """Minimal visualization helpers (from VMamba analyze/utils)."""

    @staticmethod
    def visualize_attnmap(
        attnmap,
        savefig="",
        figsize=(8, 7),
        cmap=None,
        sticks=True,
        dpi=200,
        fontsize=18,
        colorbar=True,
        **kwargs,
    ):
        if isinstance(attnmap, torch.Tensor):
            attnmap = attnmap.detach().cpu().numpy()
        plt.rcParams["font.size"] = fontsize
        plt.figure(figsize=figsize, dpi=dpi, **kwargs)
        ax = plt.gca()
        im = ax.imshow(attnmap, cmap=cmap)
        if not sticks:
            ax.set_axis_off()
        if colorbar:
            ax.figure.colorbar(im, ax=ax)
        if savefig:
            plt.savefig(savefig, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    @staticmethod
    def draw_image_star(image: Image.Image, centers, radius=10, color=(255, 0, 0)):
        """Draw filled 5-point stars at centers on a PIL image."""
        img = image.copy()
        draw = ImageDraw.Draw(img)
        for cx, cy in centers:
            pts = []
            for i in range(10):
                ang = -math.pi / 2 + i * math.pi / 5
                r = radius if i % 2 == 0 else radius * 0.4
                pts.append((cx + r * math.cos(ang), cy + r * math.sin(ang)))
            draw.polygon(pts, fill=color)
        return img


class AttnMamba:
    """
    Port of VMamba analyze.utils.AttnMamba, adapted for:
    - continuous (snake) CrossScan used by this repo
    - non-square H×W feature maps
    - optional query_indices to avoid materializing full (L,L) maps (needed for large featHW)
    """

    @staticmethod
    def checkpostfix(tag, value):
        ret = value[-len(tag) :] == tag
        if ret:
            value = value[: -len(tag)]
        return ret, value

    @staticmethod
    def _build_qk(regs, mode, device=None):
        """Build per-direction Q/K in scan order. Returns list length G of (Q,K), each (Dq,N,L)."""
        As = -torch.exp(regs["A_logs"].to(torch.float32))
        Bs, Cs = regs["Bs"], regs["Cs"]
        dts, delta_bias = regs["dts"], regs["delta_bias"]
        H, W = int(regs["H"]), int(regs["W"])
        L = H * W

        if device is not None:
            As, Bs, Cs = As.to(device), Bs.to(device), Cs.to(device)
            dts, delta_bias = dts.to(device), delta_bias.to(device)

        B, G, N, _ = Bs.shape
        GD, _ = As.shape
        D = GD // G
        assert B == 1, "attnmap 仅支持 batch=1"

        dts = torch.nn.functional.softplus(dts + delta_bias[:, None]).view(B, G, D, L)
        need_w = mode in ("CwBw", "CwBdtw", "ww")
        ws = None
        if need_w:
            dw_logs = As.view(G, D, N)[None, :, :, :, None] * dts[:, :, :, None, :]
            ws = torch.cumsum(dw_logs, dim=-1).exp()

        out = []
        for g in range(G):
            Cg, Bg = Cs[0, g], Bs[0, g]  # (N, L)
            if mode == "CB":
                Q, K = Cg[None], Bg[None]  # (1, N, L)
            elif mode == "CBdt":
                Q = Cg[None].expand(D, -1, -1)
                K = Bg[None] * dts[0, g][:, None, :]
            elif mode == "CwBw":
                Q = Cg[None] * ws[0, g]
                K = Bg[None] / ws[0, g].clamp(min=1e-20)
            elif mode == "CwBdtw":
                Q = Cg[None] * ws[0, g]
                K = Bg[None] * dts[0, g][:, None, :] / ws[0, g].clamp(min=1e-20)
            elif mode == "ww":
                Q = ws[0, g]
                K = 1.0 / ws[0, g].clamp(min=1e-20)
            else:
                raise NotImplementedError(mode)
            out.append((Q.float(), K.float()))
        return out, H, W, L

    @classmethod
    def _dir_rows(cls, Q, K, inv, q_spatial, reverse=False):
        """
        Q,K: (Dq, N, L) in this direction's scan order.
        Returns (n_q, L) attention rows in spatial flat order.
        """
        L = Q.shape[-1]
        device = Q.device
        q_spatial = q_spatial.to(device)

        if not reverse:
            q_scan = inv[q_spatial]
            acc = None
            for d in range(Q.shape[0]):
                rows = Q[d].transpose(0, 1)[q_scan] @ K[d]  # (n_q, L)
                acc = rows if acc is None else acc + rows
            acc = acc / Q.shape[0]
            key_idx = torch.arange(L, device=device)[None, :]
            acc = acc.masked_fill(key_idx > q_scan[:, None], 0.0)
            return acc[:, inv]

        # reverse direction: attn_fwd[i,j] = attn_rev[L-1-i, L-1-j]
        q_rev = (L - 1) - inv[q_spatial]
        acc = None
        for d in range(Q.shape[0]):
            rows = Q[d].transpose(0, 1)[q_rev] @ K[d]
            acc = rows if acc is None else acc + rows
        acc = acc / Q.shape[0]
        key_idx = torch.arange(L, device=device)[None, :]
        acc = acc.masked_fill(key_idx > q_rev[:, None], 0.0)
        # to forward-scan order then spatial
        return acc.flip(-1)[:, inv]

    @classmethod
    @torch.no_grad()
    def attnmap_mamba(
        cls,
        regs,
        mode="CB",
        ret="all",
        absnorm=0,
        scale=1,
        verbose=False,
        device=None,
        query_indices=None,
    ):
        print(f"attn for mode={mode}, ret={ret}, absnorm={absnorm}, scale={scale}", flush=True)
        if absnorm == 1:
            _norm = lambda x: ((x - x.min()) / (x.max() - x.min() + 1e-12))
        elif absnorm == 2:
            _norm = lambda x: (x.abs() / (x.abs().max() + 1e-12))
        else:
            _norm = lambda x: x

        qk_list, H, W, L = cls._build_qk(regs, mode, device=device)
        idx_h, idx_v = _continuous_scan_indices(H, W, qk_list[0][0].device)
        inv_h = torch.argsort(idx_h)
        inv_v = torch.argsort(idx_v)

        specs = {
            "a0": [(0, inv_h, False)],
            "a1": [(1, inv_v, False)],
            "a2": [(2, inv_h, True)],
            "a3": [(3, inv_v, True)],
            "all": [(0, inv_h, False), (1, inv_v, False), (2, inv_h, True), (3, inv_v, True)],
            "nall": [(0, inv_h, False), (1, inv_v, False), (2, inv_h, True), (3, inv_v, True)],
        }
        if ret not in specs:
            raise NotImplementedError(f"{ret} is not allowed (ao* 已省略，请用 a0-a3)")

        if query_indices is None:
            # full map — only for small L
            parts = []
            for g, inv, rev in specs[ret]:
                Q, K = qk_list[g]
                acc = None
                for d in range(Q.shape[0]):
                    a = Q[d].transpose(0, 1) @ K[d]
                    acc = a if acc is None else acc + a
                acc = (acc / Q.shape[0]) * torch.tril(acc.new_ones(L, L))
                if rev:
                    acc = acc.flip(0).flip(1)
                parts.append(acc[inv][:, inv])
            if ret == "nall":
                attn = sum(_norm(p) for p in parts) / len(parts)
            elif ret == "all":
                attn = _norm(sum(parts))
            else:
                attn = _norm(parts[0])
            attn = (scale * attn).clamp(max=attn.max())
            return attn, H, W

        q_spatial = torch.as_tensor(query_indices, dtype=torch.long)
        parts = []
        for g, inv, rev in specs[ret]:
            Q, K = qk_list[g]
            parts.append(cls._dir_rows(Q, K, inv, q_spatial, reverse=rev))
        if ret == "nall":
            rows = sum(_norm(p) for p in parts) / len(parts)
        elif ret == "all":
            rows = _norm(sum(parts))
        else:
            rows = _norm(parts[0])
        rows = (scale * rows).clamp(max=rows.max())
        return rows.view(-1, H, W), H, W

    @classmethod
    @torch.no_grad()
    def get_attnmap_mamba(cls, ss2d, mode="", verbose=False, scale=1, device=None, query_indices=None):
        mode1 = mode.split("_")[-1]
        mode0 = mode[: -(len(mode1) + 1)]

        absnorm = 0
        tag, mode0 = cls.checkpostfix("_absnorm", mode0)
        absnorm = 2 if tag else absnorm
        tag, mode0 = cls.checkpostfix("_norm", mode0)
        absnorm = 1 if tag else absnorm

        regs = getattr(ss2d, "__data__")
        return cls.attnmap_mamba(
            regs,
            mode=mode1,
            ret=mode0,
            absnorm=absnorm,
            verbose=verbose,
            scale=scale,
            device=device,
            query_indices=query_indices,
        )[0]


def build_image_transform(img_size):
    """YOLO-style: resize + /255 (no ImageNet normalize)."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),  # [0,1]
        ]
    )


def load_image_tensor(img_path, transform):
    img = Image.open(img_path).convert("RGB")
    return transform(img)


def _yolo_label_path(label_dir, name):
    stem = os.path.splitext(name)[0]
    return os.path.join(label_dir, f"{stem}.txt")


def _bboxes_from_yolo_txt(txt_path, img_size):
    """Return list of (bx,by,bw,bh) in resized pixel coords."""
    if not os.path.isfile(txt_path):
        return []
    boxes = []
    with open(txt_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            _, xc, yc, w, h = map(float, parts[:5])
            bw, bh = w * img_size, h * img_size
            bx, by = xc * img_size - bw / 2.0, yc * img_size - bh / 2.0
            boxes.append((bx, by, bw, bh))
    return boxes


def build_queries(image_dir, img_size, coco_ann=None, label_dir=None, target_image=None):
    """Build per-defect queries from COCO json or YOLO labels."""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    files_on_disk = {
        f for f in os.listdir(image_dir) if os.path.splitext(f)[1].lower() in exts
    }

    use_coco = coco_ann is not None and os.path.isfile(coco_ann)
    queries = []

    if use_coco:
        data = json.load(open(coco_ann, "r"))
        id2anns = defaultdict(list)
        for a in data["annotations"]:
            id2anns[a["image_id"]].append(a)
        name2im = {os.path.basename(im["file_name"]): im for im in data["images"]}
        files = sorted(files_on_disk & set(name2im.keys()))
        if target_image is not None:
            target = os.path.basename(target_image)
            if target not in files_on_disk:
                raise FileNotFoundError(f"在 {image_dir} 中找不到图片: {target}")
            files = [target]
        for name in files:
            im = name2im[name]
            anns = id2anns[im["id"]]
            if not anns:
                continue
            ow, oh = float(im["width"]), float(im["height"])
            sx, sy = img_size / ow, img_size / oh
            img_path = os.path.join(image_dir, name)
            for defect_k, a in enumerate(anns, start=1):
                x, y, w, h = a["bbox"]
                bx, by, bw, bh = x * sx, y * sy, w * sx, h * sy
                posx = min(max((bx + bw / 2.0) / img_size, 0.0), 1.0 - 1e-6)
                posy = min(max((by + bh / 2.0) / img_size, 0.0), 1.0 - 1e-6)
                queries.append(
                    dict(
                        img_path=img_path,
                        defect_k=defect_k,
                        n_defects=len(anns),
                        posx=posx,
                        posy=posy,
                        bbox=(bx, by, bw, bh),
                        name=name,
                        ann_id=a["id"],
                    )
                )
        print(f"loaded queries from COCO: {coco_ann}", flush=True)
        return queries

    # YOLO labels fallback
    if label_dir is None or not os.path.isdir(label_dir):
        raise FileNotFoundError(
            f"找不到 COCO 标注 ({coco_ann})，且 YOLO label 目录无效: {label_dir}"
        )
    files = sorted(files_on_disk)
    if target_image is not None:
        target = os.path.basename(target_image)
        if target not in files_on_disk:
            raise FileNotFoundError(f"在 {image_dir} 中找不到图片: {target}")
        files = [target]

    for name in files:
        boxes = _bboxes_from_yolo_txt(_yolo_label_path(label_dir, name), img_size)
        if not boxes:
            continue
        img_path = os.path.join(image_dir, name)
        for defect_k, (bx, by, bw, bh) in enumerate(boxes, start=1):
            posx = min(max((bx + bw / 2.0) / img_size, 0.0), 1.0 - 1e-6)
            posy = min(max((by + bh / 2.0) / img_size, 0.0), 1.0 - 1e-6)
            queries.append(
                dict(
                    img_path=img_path,
                    defect_k=defect_k,
                    n_defects=len(boxes),
                    posx=posx,
                    posy=posy,
                    bbox=(bx, by, bw, bh),
                    name=name,
                    ann_id=defect_k,
                )
            )
    print(f"loaded queries from YOLO labels: {label_dir}", flush=True)
    return queries


def save_overlay(deimg_u8, act_map, center_xy, radius, save_path):
    h, w = deimg_u8.shape[:2]
    if isinstance(act_map, torch.Tensor):
        act_map = act_map.detach().cpu().numpy()
    act = act_map.astype(np.float32)
    act = (act - act.min()) / (act.max() - act.min() + 1e-8)
    heat = cv2.resize(act, (w, h), interpolation=cv2.INTER_LINEAR)
    heat_color = cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET)
    heat_color = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)
    blend = 0.55 * deimg_u8.astype(np.float32) + 0.45 * heat_color.astype(np.float32)
    blend = np.clip(blend, 0, 255).astype(np.uint8)
    visualize.draw_image_star(Image.fromarray(blend), centers=[center_xy], radius=radius).save(save_path)


def _iter_vss_blocks(module: nn.Module):
    """Yield VSSBlock modules in backbone order (depth-first)."""
    if isinstance(module, VSSBlock):
        yield module
        return
    if isinstance(module, nn.Sequential):
        for child in module:
            yield from _iter_vss_blocks(child)


def get_backbone_ss2d(det_model: nn.Module, stage: int, block_id: int) -> SS2D:
    """
    OCSS-T backbone stages (after width/depth scale):
      stage0: model[1]  stride /4
      stage1: model[3]  stride /8
      stage2: model[5]  stride /16  (may be Sequential of repeated VSSBlocks)
      stage3: model[7]  stride /32
    """
    stage_to_layer = {0: 1, 1: 3, 2: 5, 3: 7}
    if stage not in stage_to_layer:
        raise ValueError(f"stage must be in {list(stage_to_layer)}, got {stage}")
    layer = det_model.model[stage_to_layer[stage]]
    blocks = list(_iter_vss_blocks(layer))
    if not blocks:
        raise RuntimeError(f"layer {stage_to_layer[stage]} 中找不到 VSSBlock: {type(layer)}")
    if block_id >= len(blocks):
        raise IndexError(f"block_id={block_id} 超出 stage{stage} 的 VSSBlock 数量 {len(blocks)}")
    ss2d = blocks[block_id].op
    if not isinstance(ss2d, SS2D):
        raise RuntimeError(f"期望 SS2D，得到 {type(ss2d)}")
    return ss2d


def stride_of_stage(stage: int) -> int:
    return {0: 4, 1: 8, 2: 16, 3: 32}[stage]


@torch.no_grad()
def main(
    weights=WEIGHTS,
    image_dir=IMAGE_DIR,
    coco_ann=COCO_ANN,
    label_dir=LABEL_DIR,
    showpath=SHOWPATH,
    target_image=TARGET_IMAGE,
    img_size=IMG_SIZE,
    stage=STAGE,
    block_id=BLOCK_ID,
    featHW=FEAT_HW,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    stride = stride_of_stage(stage)
    expect = featHW * stride
    if img_size != expect:
        print(f"warn: IMG_SIZE={img_size} 与 FEAT_HW×stride={expect} 不一致，改用 {expect}", flush=True)
        img_size = expect
    scan_modes = ["a0", "a1", "a2", "a3"]
    act_modes = ["all", "nall"]
    attn_modes_m1 = ["CB", "CwBw", "ww"]
    all_m0 = scan_modes + act_modes

    transform = build_image_transform(img_size)
    queries = build_queries(
        image_dir, img_size, coco_ann=coco_ann, label_dir=label_dir, target_image=target_image
    )
    if not queries:
        raise RuntimeError(f"没有可可视化的缺陷：dir={image_dir}, target={target_image}")
    print(f"{len(queries)} defect(s) | imgsz={img_size} stage={stage} featHW={featHW}", flush=True)

    if not os.path.isfile(weights):
        raise FileNotFoundError(f"找不到权重: {weights}")
    yolo = YOLO(weights)
    det: nn.Module = yolo.model.to(device).eval()

    ss2d = get_backbone_ss2d(det, stage, block_id)
    setattr(ss2d, "__DEBUG__", True)
    print(f"hook SS2D: stage={stage} block_id={block_id} -> {ss2d.__class__.__name__}", flush=True)

    os.makedirs(showpath, exist_ok=True)

    # group defects by image
    from itertools import groupby

    queries_sorted = sorted(queries, key=lambda x: x["img_path"])
    for img_path, group in groupby(queries_sorted, key=lambda x: x["img_path"]):
        group = list(group)
        img = load_image_tensor(img_path, transform)
        _ = det(img[None].to(device))
        deimg_u8 = (img.permute(1, 2, 0).cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)

        if not hasattr(ss2d, "__data__"):
            raise RuntimeError("SS2D 未写入 __data__，请确认已开启 __DEBUG__ 且前向经过该层")

        # move debug tensors to CPU for attention algebra
        regs = ss2d.__data__
        for k, v in list(regs.items()):
            if torch.is_tensor(v):
                regs[k] = v.detach().cpu()
        H, W = int(regs["H"]), int(regs["W"])
        if H != featHW or W != featHW:
            print(f"warn: 实际特征图 {H}x{W} 与 FEAT_HW={featHW} 不一致，改用实际值", flush=True)
            featHW = H

        q_tokens = []
        for q in group:
            qy = min(max(int(q["posy"] * featHW), 0), featHW - 1)
            qx = min(max(int(q["posx"] * featHW), 0), featHW - 1)
            q_tokens.append(qy * featHW + qx)

        # prepare out dirs + base images
        out_dirs = []
        for q in group:
            stem = os.path.splitext(q["name"])[0]
            out_dir = (
                f"{showpath}/{stem}/"
                f"defect_{q['defect_k']:02d}_of_{q['n_defects']:02d}_ann{q['ann_id']}"
            )
            os.makedirs(out_dir, exist_ok=True)
            out_dirs.append(out_dir)
            bx, by, bw, bh = q["bbox"]
            cx, cy = bx + bw / 2.0, by + bh / 2.0
            radius = max(6, min(bw, bh) * 0.35)
            q["_cx"], q["_cy"], q["_radius"] = cx, cy, radius
            Image.fromarray(deimg_u8).save(f"{out_dir}/imori.jpg")
            visualize.draw_image_star(
                Image.fromarray(deimg_u8.copy()), centers=[(cx, cy)], radius=radius
            ).save(f"{out_dir}/imori_star.jpg")
            print(
                f"[{q['name']}] defect {q['defect_k']}/{q['n_defects']} "
                f"ann={q['ann_id']} pos=({q['posx']:.3f},{q['posy']:.3f}) -> {out_dir}",
                flush=True,
            )

        # CPU 低内存：降低绘图 DPI，并在每组模式后主动回收
        plot_dpi = 150 if device == "cpu" else 600
        for m0 in all_m0:
            for m1 in attn_modes_m1:
                key = f"{m0}_norm_{m1}"
                acts = AttnMamba.get_attnmap_mamba(
                    ss2d, key, device="cpu", query_indices=q_tokens
                )  # (n_q, H, W)
                for qi, q in enumerate(group):
                    act = acts[qi]
                    out_dir = out_dirs[qi]
                    cx, cy, radius = q["_cx"], q["_cy"], q["_radius"]
                    if m0 in act_modes:
                        visualize.visualize_attnmap(
                            act,
                            f"{out_dir}/activation_{m0}_norm_{m1}.jpg",
                            colorbar=False,
                            sticks=False,
                            dpi=plot_dpi,
                            figsize=(5, 4),
                        )
                        save_overlay(
                            deimg_u8, act, (cx, cy), radius,
                            f"{out_dir}/overlay_activation_{m0}_norm_{m1}.jpg",
                        )
                    if m0 in scan_modes:
                        visualize.visualize_attnmap(
                            act,
                            f"{out_dir}/scan_{m0}_norm_{m1}.jpg",
                            colorbar=False,
                            sticks=False,
                            dpi=plot_dpi,
                            figsize=(5, 4),
                        )
                        save_overlay(
                            deimg_u8, act, (cx, cy), radius,
                            f"{out_dir}/overlay_scan_{m0}_norm_{m1}.jpg",
                        )
                del acts
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    print(f"done. results -> {showpath}", flush=True)


if __name__ == "__main__":
    main()
