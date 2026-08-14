"""
自定义预测可视化：
1. bbox 颜色为黄色
2. 只显示置信度分数，不显示类别
3. 置信度保留两位小数（小数形式，如 0.87）
4. 置信度文本：Times New Roman（新罗马）黄字 + 紧凑黑底，上下左右居中
5. 文本放在 bbox 外侧（优先框上方），避免挡住缺陷
6. 高分辨率绘制并以 600 DPI 导出，提升清晰度
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO

# BGR / RGB
BBOX_COLOR_BGR = (0, 255, 255)
TEXT_COLOR_RGB = (255, 255, 0)  # yellow
TEXT_BG_RGB = (0, 0, 0)

# Times New Roman 及兼容衬线字体候选路径
_FONT_CANDIDATES = [
    # Windows / 常见 Times New Roman
    "C:/Windows/Fonts/times.ttf",
    "C:/Windows/Fonts/timesnr.ttf",
    "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSerif-Regular.ttf",
    # 本机可用的 Times 风格衬线（metric-compatible / 近似）
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    str(Path(__file__).resolve().parent / "fonts" / "LiberationSerif-Regular.ttf"),
    str(Path(__file__).resolve().parent / "fonts" / "TimesNewRoman.ttf"),
]


def get_times_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """加载 Times New Roman；若系统无则回退到 Liberation/DejaVu Serif。"""
    for path in _FONT_CANDIDATES:
        p = Path(path)
        if p.is_file() and p.stat().st_size > 1000:
            try:
                return ImageFont.truetype(str(p), size=size)
            except Exception:
                continue
    return ImageFont.load_default()


def get_label_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """标签字体（与 get_times_font 相同入口，兼容旧调用）。"""
    return get_times_font(size)


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    """返回文本宽高（不含多余空白）。"""
    if hasattr(draw, "textbbox"):
        l, t, r, b = draw.textbbox((0, 0), text, font=font)
        return int(r - l), int(b - t)
    if hasattr(font, "getbbox"):
        l, t, r, b = font.getbbox(text)
        return int(r - l), int(b - t)
    return font.getsize(text)


def _place_label_outside(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    box_w: int,
    box_h: int,
    img_w: int,
    img_h: int,
    gap: int,
) -> tuple[int, int, int, int]:
    """将紧凑标签框放在检测框外侧，优先上方。返回 (bx1, by1, bx2, by2)。"""
    cx = (x1 + x2) / 2.0
    candidates = [
        # top
        (int(round(cx - box_w / 2)), y1 - gap - box_h, int(round(cx - box_w / 2)) + box_w, y1 - gap),
        # bottom
        (int(round(cx - box_w / 2)), y2 + gap, int(round(cx - box_w / 2)) + box_w, y2 + gap + box_h),
        # left
        (x1 - gap - box_w, y1, x1 - gap, y1 + box_h),
        # right
        (x2 + gap, y1, x2 + gap + box_w, y1 + box_h),
    ]

    def clamp(bx1, by1, bx2, by2):
        if bx1 < 0:
            bx2 -= bx1
            bx1 = 0
        if bx2 > img_w:
            shift = bx2 - img_w
            bx1 -= shift
            bx2 -= shift
        bx1 = int(np.clip(bx1, 0, max(0, img_w - 1)))
        bx2 = int(np.clip(bx2, bx1 + 1, img_w))
        return bx1, by1, bx2, by2

    for cand in candidates:
        bx1, by1, bx2, by2 = clamp(*cand)
        if by1 >= 0 and by2 <= img_h and (bx2 - bx1) >= box_w * 0.8 and (by2 - by1) >= box_h * 0.8:
            return bx1, by1, bx2, by2

    # 兜底：贴顶
    bx1, by1, bx2, by2 = clamp(*candidates[0])
    by1 = max(0, min(by1, img_h - box_h))
    by2 = by1 + box_h
    return bx1, by1, bx2, by2


def draw_custom_detections(
    image: np.ndarray,
    boxes_xyxy: np.ndarray,
    scores: np.ndarray,
    line_width: int = 2,
) -> np.ndarray:
    """
    在图像上绘制自定义检测结果。
    置信度：Times 新罗马字体、黄字、紧凑黑底、框内居中；放在检测框外侧。
    """
    im = image.copy()
    h, w = im.shape[:2]
    lw = max(int(line_width), 1)

    if boxes_xyxy is None or len(boxes_xyxy) == 0:
        return im

    boxes_xyxy = np.asarray(boxes_xyxy, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)

    # 先画黄色 bbox（OpenCV）
    for box in boxes_xyxy:
        x1, y1, x2, y2 = map(int, np.round(box))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)
        cv2.rectangle(im, (x1, y1), (x2, y2), BBOX_COLOR_BGR, thickness=lw, lineType=cv2.LINE_AA)

    # 再用 PIL 画 Times 字体分数（紧凑黑底 + 居中）
    pil_im = Image.fromarray(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_im)

    # 字号接近第一版 OpenCV 无衬线视觉大小
    font_size = max(28, int(round(lw * 8.5)))
    font = get_label_font(font_size)
    # 上下左右各留很少 padding，使黑框紧贴数字
    pad_x = max(3, int(round(font_size * 0.10)))
    pad_y = max(2, int(round(font_size * 0.06)))  # 只比数字高度高一点
    gap = max(2, lw)

    for box, score in zip(boxes_xyxy, scores):
        x1, y1, x2, y2 = map(int, np.round(box))
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        label_text = f"{float(score):.2f}"
        tw, th = _text_size(draw, label_text, font)
        box_w = tw + 2 * pad_x
        box_h = th + 2 * pad_y

        bx1, by1, bx2, by2 = _place_label_outside(x1, y1, x2, y2, box_w, box_h, w, h, gap)

        # 黑底
        draw.rectangle([bx1, by1, bx2, by2], fill=TEXT_BG_RGB)

        # 分数在黑框内上下左右居中
        # textbbox 可能有负的 top offset，用 anchor='mm' 更稳
        cx = (bx1 + bx2) / 2.0
        cy = (by1 + by2) / 2.0
        try:
            draw.text((cx, cy), label_text, font=font, fill=TEXT_COLOR_RGB, anchor="mm")
        except TypeError:
            # 旧版 PIL 无 anchor：手动居中
            if hasattr(draw, "textbbox"):
                l, t, r, b = draw.textbbox((0, 0), label_text, font=font)
                tw2, th2 = r - l, b - t
                tx = cx - tw2 / 2 - l
                ty = cy - th2 / 2 - t
            else:
                tx = cx - tw / 2
                ty = cy - th / 2
            draw.text((tx, ty), label_text, font=font, fill=TEXT_COLOR_RGB)

    return cv2.cvtColor(np.asarray(pil_im), cv2.COLOR_RGB2BGR)


def save_image_dpi(image_bgr: np.ndarray, out_path: Path, dpi: int = 600) -> Path:
    """以指定 DPI 保存图像（PNG，保证清晰度）。"""
    out_path = Path(out_path)
    if out_path.suffix.lower() not in {".png", ".tif", ".tiff"}:
        out_path = out_path.with_suffix(".png")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    Image.fromarray(rgb).save(out_path, dpi=(dpi, dpi))
    return out_path


def visualize_and_save(
    model: YOLO,
    image_path: str | Path,
    save_dir: str | Path,
    imgsz: int = 160,
    conf: float = 0.50,
    device: str = "0",
    line_width: int = 2,
    save_txt: bool = True,
    dpi: int = 600,
    render_scale: float | None = None,
) -> Path:
    """对单张图预测，并用自定义样式保存可视化结果。"""
    image_path = Path(image_path)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = model.predict(
        source=str(image_path),
        imgsz=imgsz,
        conf=conf,
        device=device,
        save=False,
        save_txt=False,
        verbose=False,
    )
    result = results[0]
    im0 = result.orig_img.copy()
    h0, w0 = im0.shape[:2]

    if render_scale is None:
        render_scale = max(dpi / 96.0, 1.0)
    render_scale = float(render_scale)

    new_w = max(1, int(round(w0 * render_scale)))
    new_h = max(1, int(round(h0 * render_scale)))
    im_hi = cv2.resize(im0, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    lw_hi = max(1, int(round(line_width * render_scale)))

    if result.boxes is not None and len(result.boxes):
        boxes = result.boxes.xyxy.cpu().numpy() * render_scale
        scores = result.boxes.conf.cpu().numpy()
        vis = draw_custom_detections(im_hi, boxes, scores, line_width=lw_hi)
        n_det = len(scores)
    else:
        scores = np.zeros((0,), dtype=np.float32)
        vis = im_hi
        n_det = 0

    out_img = save_dir / f"{image_path.stem}.png"
    out_img = save_image_dpi(vis, out_img, dpi=dpi)

    if save_txt:
        labels_dir = save_dir / "labels"
        labels_dir.mkdir(parents=True, exist_ok=True)
        out_txt = labels_dir / f"{image_path.stem}.txt"
        lines = []
        if result.boxes is not None and len(result.boxes):
            xywhn = result.boxes.xywhn.cpu().numpy()
            cls = result.boxes.cls.cpu().numpy().astype(int)
            confs = result.boxes.conf.cpu().numpy()
            for c, box, s in zip(cls, xywhn, confs):
                lines.append(
                    f"{c} {box[0]:.6f} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {s:.6f}"
                )
        out_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    font_used = next((p for p in _FONT_CANDIDATES if Path(p).is_file() and Path(p).stat().st_size > 1000), "default")
    print(
        f"Saved: {out_img}  (detections={n_det}, "
        f"size={vis.shape[1]}x{vis.shape[0]}, dpi={dpi}, scale={render_scale:.2f}, font={font_used})"
    )
    return out_img


if __name__ == "__main__":
    model = YOLO("/root/autodl-tmp/OCSSNet/output_dir/mscoco/best.pt")

    img_dir = Path("/root/autodl-tmp/OCSSNet/dataset/images/test2017")
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    image_paths = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in exts)
    if not image_paths:
        raise FileNotFoundError(f"目录中无图片: {img_dir}")
    print(f"Predict {len(image_paths)} images from {img_dir}")

    save_root = Path("/root/autodl-tmp/OCSSNet/Defect_Sample/Type2/[predict_results_custom]")

    for i, image_path in enumerate(image_paths, 1):
        print(f"[{i}/{len(image_paths)}] {image_path.name}")
        visualize_and_save(
            model=model,
            image_path=image_path,
            save_dir=save_root,
            imgsz=160,
            conf=0.50,
            device="0",
            line_width=2.5,
            save_txt=True,
            dpi=2400,
        )
