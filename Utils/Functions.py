import sys
if 'Utils.Imports' in sys.modules:
    del sys.modules['Utils.Imports']
from Utils.Imports import *



import os
import json
from PIL import Image, ImageDraw, ImageFont


def draw_bones_from_json_withimage_V2(
    json_path: str,
    image_path: Optional[str] = None,
    out_path: Optional[str] = None,
    show: bool = True,
    draw_names: bool = True,
    only_in_front: bool = False,
    line_width: int = 3,
    head_radius: int = 4,
    highlight_bones: tuple = ("spine 3", "spine 4"),
    highlight_color=(255, 0, 0, 220),
    normal_color=(0, 0, 0, 180),
    font_size: int = 16,
):
    """
    JSON'daki bones sözlüğünden (head_px, tail_px) ile iskeleti çizer.
    - image_path verilirse görsel üzerine overlay yapar
    - image_path yoksa resolution boyutunda beyaz tuval oluşturur
    - draw_names=True ise tüm kemik isimlerini yazar
    - only_in_front=True ise sadece in_front_head & in_front_tail True olanları çizer
    - highlight_bones: (spine 3, spine 4) gibi vurgulanacak kemikler
    """

    json_path = os.path.abspath(json_path)
    base_dir = os.path.dirname(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bones = data.get("bones", {})
    if not isinstance(bones, dict) or not bones:
        raise ValueError("JSON içinde 'bones' boş veya yok.")

    # --- resolution (W,H)
    res = data.get("resolution", None)
    if not res or len(res) != 2:
        raise ValueError("JSON içinde 'resolution' (W,H) yok veya hatalı.")
    W, H = int(res[0]), int(res[1])

    # --- Görsel (varsa) yükle; yoksa beyaz tuval
    if image_path is None:
        # beyaz zemin
        img = Image.new("RGB", (W, H), (255, 255, 255)).convert("RGBA")
    else:
        image_path = os.path.abspath(image_path)
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")
        img = Image.open(image_path).convert("RGBA")

        # görsel boyutu resolution ile farklıysa ölçek uygula
        W_img, H_img = img.size
        sx = W_img / float(W)
        sy = H_img / float(H)
        W, H = W_img, H_img
    # scale (image yoksa 1)
    if image_path is None:
        sx = sy = 1.0

    # --- out_path default
    if out_path is None:
        name = os.path.splitext(os.path.basename(json_path))[0]
        out_path = os.path.join(base_dir, f"{name}__bones.png")
    out_path = os.path.abspath(out_path)

    overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # --- Font
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    def _scale_pt(pt):
        return (float(pt[0]) * sx, float(pt[1]) * sy)

    def _draw_label(x, y, text, pad=3):
        if not text:
            return
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle(
            (bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad),
            fill=(255, 255, 255, 200),
        )
        draw.text((x, y), text, fill=(0, 0, 0, 255), font=font)

    # --- Çizim
    for bone_name, b in bones.items():
        head_px = b.get("head_px")
        tail_px = b.get("tail_px")
        if not head_px or not tail_px:
            continue

        if only_in_front:
            if not (b.get("in_front_head", False) and b.get("in_front_tail", False)):
                continue

        x1, y1 = _scale_pt(head_px)
        x2, y2 = _scale_pt(tail_px)

        is_hl = bone_name in highlight_bones
        color = highlight_color if is_hl else normal_color
        w = int(line_width * (1.8 if is_hl else 1.0))

        draw.line([(x1, y1), (x2, y2)], fill=color, width=w)

        r = int(head_radius * (1.4 if is_hl else 1.0))
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r),
                     fill=(255, 255, 255, 255),
                     outline=color)

        if draw_names:
            _draw_label(x1 + 6, y1 + 6, bone_name)

    out_img = Image.alpha_composite(img, overlay).convert("RGB")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_img.save(out_path)

    if show:
        out_img.show()

    return out_path



# ---------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------
def load_json_folder_as_dict(
    folder_path: str,
    key_mode: str = "filename",   # "filename" | "stem"
    recursive: bool = False,
    encoding: str = "utf-8",
) -> Tuple[Dict[str, dict], Dict[str, str]]:
    """
    folder_path altındaki *.json dosyalarını okur ve sözlük döndürür.

    Returns:
      (data_dict, errors)
        data_dict: { key: json_dict }
        errors:    { filepath: "error message" }
    """
    folder = Path(folder_path)
    if not folder.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {folder}")

    pattern = "**/*.json" if recursive else "*.json"
    files = sorted(folder.glob(pattern))

    data_dict: Dict[str, dict] = {}
    errors: Dict[str, str] = {}

    for fp in files:
        key = fp.name if key_mode == "filename" else fp.stem
        try:
            with fp.open("r", encoding=encoding) as f:
                data_dict[key] = json.load(f)
        except Exception as e:
            errors[str(fp)] = str(e)

    return data_dict, errors


def load_json_folder_to_dict(folder_path: str, key_mode: str = "filename") -> Dict[str, dict]:
    """
    Backward-compatible loader: glob ile okur.

    key_mode:
      - "filename": fullbody_001_180.json
      - "stem":     fullbody_001_180
    """
    folder_path = os.path.abspath(folder_path)
    paths = sorted(glob(os.path.join(folder_path, "*.json")))
    out: Dict[str, dict] = {}

    for p in paths:
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)

        base = os.path.basename(p)
        stem = os.path.splitext(base)[0]
        key = base if key_mode == "filename" else stem
        out[key] = data

    return out




def compute_angles_for_folder(folder_path: str):
    import pandas as pd
    d = load_json_folder_to_dict(folder_path, key_mode="stem")
    rows = []
    for k, js in d.items():
        ang = compute_reba_angles_from_json(js)
        ang["file"] = k
        try:
            last = k.split("_")[-1]
            ang["angle_deg_in_filename"] = int(last) if last.isdigit() else None
        except Exception:
            ang["angle_deg_in_filename"] = None
        rows.append(ang)
    df = pd.DataFrame(rows).sort_values(["angle_deg_in_filename", "file"])
    return df


def print_reba_angles_pretty(angles: dict, title: Optional[str] = None):
    """
    Açıları okunur biçimde yazdırır.
    """
    def fmt(v):
        return "—" if v is None else f"{float(v):.1f}°"

    if title:
        print("\n" + "=" * len(title))
        print(title)
        print("=" * len(title))

    print(f"  • Trunk (signed)             : {fmt(angles.get('trunk_flex_ext_deg'))}  | abs: {fmt(angles.get('trunk_abs_deg'))}")
    print(f"  • Hip hinge (signed)         : {fmt(angles.get('hip_hinge_deg'))}  | abs: {fmt(angles.get('hip_hinge_abs_deg'))}")
    print(f"  • Neck (signed)              : {fmt(angles.get('neck_flex_ext_deg'))}  | abs: {fmt(angles.get('neck_abs_deg'))}")

    print(f"  • Upperarm L vs spine mean   : {fmt(angles.get('upperarmL_vs_spine_deg'))}")
    print(f"  • Upperarm R vs spine mean   : {fmt(angles.get('upperarmR_vs_spine_deg'))}")
    print(f"  • Thigh L vs spine mean      : {fmt(angles.get('thighL_vs_spine_deg'))}")
    print(f"  • Thigh R vs spine mean      : {fmt(angles.get('thighR_vs_spine_deg'))}")

    print(f"  • Hand L vs Forearm L        : {fmt(angles.get('handL_vs_forearmL_deg'))}")
    print(f"  • Hand R vs Forearm R        : {fmt(angles.get('handR_vs_forearmR_deg'))}")

    print(f"  • Spine.004 vs Spine.003     : {fmt(angles.get('spine004_vs_spine003_deg'))}")
    print(f"  • Spine.004 vs Spine.005     : {fmt(angles.get('spine004_vs_spine005_deg'))}")

    print(f"  • Elbow L/R                  : {fmt(angles.get('elbow_l_deg'))} / {fmt(angles.get('elbow_r_deg'))}")
    print(f"  • Knee  L/R                  : {fmt(angles.get('knee_l_deg'))} / {fmt(angles.get('knee_r_deg'))}")
    print(f"  • Ankle L/R                  : {fmt(angles.get('ankle_l_deg'))} / {fmt(angles.get('ankle_r_deg'))}")

    print("\n" + "-" * 55)


# ---------------------------------------------------------------------
# Drawing bones on image (+ angle overlay)
# ---------------------------------------------------------------------


# =========================================================
# draw skeleton + overlay angles (aligned with your compute_reba_angles_from_json)
# =========================================================
def draw_bones_from_json_withimage(
    json_path: str,
    image_path: str,          # required
    out_path: str,            # required
    show: bool = True,
    draw_names: bool = True,
    only_in_front: bool = False,
    line_width: int = 3,
    head_radius: int = 4,
    alpha: int = 200,

    # angles overlay
    draw_angles: bool = True,
    angles: Optional[Dict] = None,
    draw_angle_box: bool = True,
    angle_decimals: int = 1,
    angle_font_size: int = 18,

    # angle labels toggles
    draw_joint_labels: bool = True,
):
    """
    bones_*.json içeriğinden (head_px, tail_px) ile görsele iskelet çizer.
    + compute_reba_angles_from_json çıktısını (SENİN SON KODUNDAKİ KEY'LER ile) overlay eder.

    Beklenen angle key'leri (compute_reba_angles_from_json):
      - thighL_vs_shinL_deg
      - thighR_vs_shinR_deg
      - upperarmL_vs_spine4_deg
      - upperarmR_vs_spine4_deg
      - upperarmL_vs_forearmL_deg
      - upperarmR_vs_forearmR_deg
      - handL_vs_forearmL_deg
      - handR_vs_forearmR_deg
      - head_vs_neck_deg
      - thighL_vs_spine2_deg
      - thighR_vs_spine2_deg

    image_path ve out_path zorunlu.
    """

    if not image_path:
        raise ValueError("image_path zorunlu.")
    if not out_path:
        raise ValueError("out_path zorunlu.")

    json_path  = os.path.abspath(json_path)
    image_path = os.path.abspath(image_path)
    out_path   = os.path.abspath(out_path)

    if not os.path.exists(json_path):
        raise FileNotFoundError(f"JSON bulunamadı: {json_path}")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    bones = _bones(data)
    if not bones:
        raise ValueError("JSON içinde bones boş.")

    # --- resolution scaling (json resolution -> actual image size)
    res = data.get("resolution", None)
    res_w = float(res[0]) if isinstance(res, (list, tuple)) and len(res) == 2 else None
    res_h = float(res[1]) if isinstance(res, (list, tuple)) and len(res) == 2 else None

    img = Image.open(image_path).convert("RGBA")
    W, H = img.size

    sx = (W / res_w) if (res_w and res_w > 0) else 1.0
    sy = (H / res_h) if (res_h and res_h > 0) else 1.0

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # --- font
    try:
        font = ImageFont.truetype("arial.ttf", angle_font_size)
    except Exception:
        font = ImageFont.load_default()

    def _scale_pt(pt):
        return (float(pt[0]) * sx, float(pt[1]) * sy)

    # =========================================================
    # 1) draw bones
    # =========================================================
    for bone_name, b in bones.items():
        head_px = b.get("head_px")
        tail_px = b.get("tail_px")
        if not head_px or not tail_px:
            continue

        if only_in_front:
            if not (b.get("in_front_head", False) and b.get("in_front_tail", False)):
                continue

        x1, y1 = _scale_pt(head_px)
        x2, y2 = _scale_pt(tail_px)

        in_front = bool(b.get("in_front_head", False) or b.get("in_front_tail", False))
        line_color = (0, 255, 0, alpha) if in_front else (255, 0, 0, alpha)

        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=int(line_width))
        r = int(head_radius)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r),
                     fill=(255, 255, 255, alpha),
                     outline=line_color)

        if draw_names:
            draw.text((x1 + 6, y1 + 6), str(bone_name), fill=(255, 255, 255, alpha), font=font)

    # =========================================================
    # 2) overlay angles
    # =========================================================
    def _fmt(v):
        if v is None:
            return None
        try:
            return f"{float(v):.{angle_decimals}f}°"
        except Exception:
            return None

    def _draw_label_xy(x, y, text, pad=4):
        if not text:
            return
        bbox = draw.textbbox((x, y), text, font=font)
        draw.rectangle((bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad), fill=(0, 0, 0, 140))
        draw.text((x, y), text, fill=(255, 255, 0, 230), font=font)

    def _get_px(bone_candidates, which="head_px"):
        b, _ = _get_bone(data, *bone_candidates)
        if b is None:
            return None
        pt = b.get(which, None)
        if pt is None:
            return None
        return _scale_pt(pt)

    if draw_angles:
        if angles is None:
            # compute from JSON using your function
            angles = compute_reba_angles_from_json(data)

        # -----------------------------------------------------
        # Anchor positions for each angle label (screen coords)
        # -----------------------------------------------------
        anchors = {
            # Thigh vs Shin -> knee position (shin head_px)
            "thighL_vs_shinL_deg": _get_px(["shin.l", "shin_l", "l_shin", "lowerleg.l"], which="head_px"),
            "thighR_vs_shinR_deg": _get_px(["shin.r", "shin_r", "r_shin", "lowerleg.r"], which="head_px"),

            # Upperarm vs Spine4 -> elbow position (upperarm tail_px)
            "upperarmL_vs_spine4_deg": _get_px(["upperarm.l", "upper_arm.l", "uparm.l"], which="tail_px"),
            "upperarmR_vs_spine4_deg": _get_px(["upperarm.r", "upper_arm.r", "uparm.r"], which="tail_px"),

            # Upperarm vs Forearm -> elbow position as well
            "upperarmL_vs_forearmL_deg": _get_px(["upperarm.l", "upper_arm.l", "uparm.l"], which="tail_px"),
            "upperarmR_vs_forearmR_deg": _get_px(["upperarm.r", "upper_arm.r", "uparm.r"], which="tail_px"),

            # Hand vs Forearm -> wrist position (forearm tail_px)
            "handL_vs_forearmL_deg": _get_px(["forearm.l", "lowerarm.l"], which="tail_px"),
            "handR_vs_forearmR_deg": _get_px(["forearm.r", "lowerarm.r"], which="tail_px"),

            # Head vs Neck -> neck/head region
            "head_vs_neck_deg": _get_px(["neck", "nack"], which="head_px") or _get_px(["head"], which="head_px"),

            # Thigh vs Spine2 -> hip region (upperleg head_px) or spine.002
            "thighL_vs_spine2_deg": _get_px(["upperleg.l", "upper_leg.l", "thigh.l", "upleg.l"], which="head_px")
                                 or _get_px(["spine.002", "spine002", "spine 2"], which="head_px"),
            "thighR_vs_spine2_deg": _get_px(["upperleg.r", "upper_leg.r", "thigh.r", "upleg.r"], which="head_px")
                                 or _get_px(["spine.002", "spine002", "spine 2"], which="head_px"),
        }

        # -----------------------------------------------------
        # Short label names (for per-joint labels + box)
        # -----------------------------------------------------
        label_names = {
            "thighL_vs_shinL_deg": "THIGH L vs SHIN L",
            "thighR_vs_shinR_deg": "THIGH R vs SHIN R",
            "upperarmL_vs_spine4_deg": "UPPERARM L vs SPINE4",
            "upperarmR_vs_spine4_deg": "UPPERARM R vs SPINE4",
            "upperarmL_vs_forearmL_deg": "UPPERARM L vs FOREARM L",
            "upperarmR_vs_forearmR_deg": "UPPERARM R vs FOREARM R",
            "handL_vs_forearmL_deg": "HAND L vs FOREARM L",
            "handR_vs_forearmR_deg": "HAND R vs FOREARM R",
            "head_vs_neck_deg": "HEAD vs NECK",
            "thighL_vs_spine2_deg": "THIGH L vs SPINE2",
            "thighR_vs_spine2_deg": "THIGH R vs SPINE2",
        }

        # -----------------------------------------------------
        # 2a) per-joint labels near anchors
        # -----------------------------------------------------
        if draw_joint_labels:
            for k, label in label_names.items():
                v = angles.get(k)
                t = _fmt(v)
                p = anchors.get(k)
                if t and p:
                    _draw_label_xy(p[0] + 10, p[1] + 10, f"{label}: {t}")

        # -----------------------------------------------------
        # 2b) summary angle box (top-left)
        # -----------------------------------------------------
        if draw_angle_box:
            box_keys = [
                "thighL_vs_shinL_deg",
                "thighR_vs_shinR_deg",
                "upperarmL_vs_spine4_deg",
                "upperarmR_vs_spine4_deg",
                "upperarmL_vs_forearmL_deg",
                "upperarmR_vs_forearmR_deg",
                "handL_vs_forearmL_deg",
                "handR_vs_forearmR_deg",
                "head_vs_neck_deg",
                "thighL_vs_spine2_deg",
                "thighR_vs_spine2_deg",
            ]

            lines = []
            for k in box_keys:
                t = _fmt(angles.get(k))
                if t:
                    lines.append(f"{label_names.get(k, k)}: {t}")

            if lines:
                x0, y0 = 15, 15
                text = "\n".join(lines)
                bbox = draw.multiline_textbbox((x0, y0), text, font=font, spacing=4)
                pad = 8
                draw.rectangle((bbox[0]-pad, bbox[1]-pad, bbox[2]+pad, bbox[3]+pad),
                               fill=(0, 0, 0, 140))
                draw.multiline_text((x0, y0), text, fill=(255, 255, 0, 230), font=font, spacing=4)

    # =========================================================
    # save / show
    # =========================================================
    out_img = Image.alpha_composite(img, overlay).convert("RGB")
    out_img.save(out_path)

    if show:
        out_img.show()

    return out_path

####################################################################################################
####################################################################################################
####################################################################################################
#####################################    ANGLES   ##################################################
####################################################################################################
####################################################################################################
####################################################################################################
import json
import numpy as np
from typing import Dict, Iterable, Optional, Tuple

# =========================================================
# Geometry helpers
# =========================================================
def _as_np(p) -> Optional[np.ndarray]:
    """[x,y,z] -> np.array(3,)"""
    if p is None:
        return None
    try:
        a = np.asarray(p, dtype=float)
        if a.ndim != 1 or a.shape[0] != 3:
            return None
        return a
    except Exception:
        return None


def _norm(v: np.ndarray) -> Optional[np.ndarray]:
    """Normalize vector, return None if too small."""
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return None
    return v / n


def angle_0_180(v1: np.ndarray, v2: np.ndarray) -> Optional[float]:
    """Klasik iki 3B vektör arası açı (0..180)."""
    u1 = _norm(v1)
    u2 = _norm(v2)
    if u1 is None or u2 is None:
        return None
    c = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def _angle_2d(a2: np.ndarray, b2: np.ndarray) -> Optional[float]:
    """Klasik iki 2B vektör arası açı (0..180)."""
    u1 = _norm(a2)
    u2 = _norm(b2)
    if u1 is None or u2 is None:
        return None
    c = float(np.clip(np.dot(u1, u2), -1.0, 1.0))
    return float(np.degrees(np.arccos(c)))


def angle_on_plane(v1: np.ndarray, v2: np.ndarray, plane: str) -> Optional[float]:
    """
    v1 ve v2'nin belirtilen düzleme projeksiyonları arasındaki açı (0..180).
    plane: "XY" | "XZ" | "YZ"
    """
    plane = plane.upper().strip()
    if plane == "XY":
        a2 = np.array([v1[0], v1[1]], dtype=float)
        b2 = np.array([v2[0], v2[1]], dtype=float)
    elif plane == "XZ":
        a2 = np.array([v1[0], v1[2]], dtype=float)
        b2 = np.array([v2[0], v2[2]], dtype=float)
    elif plane == "YZ":
        a2 = np.array([v1[1], v1[2]], dtype=float)
        b2 = np.array([v2[1], v2[2]], dtype=float)
    else:
        raise ValueError("plane must be one of {'XY','XZ','YZ'}")

    return _angle_2d(a2, b2)


def joint_angle(prox: Optional[np.ndarray],
                joint: Optional[np.ndarray],
                dist: Optional[np.ndarray]) -> Optional[float]:
    """
    Eklem iç açısı (0..180):
      v1 = prox - joint
      v2 = dist - joint
    """
    if prox is None or joint is None or dist is None:
        return None
    return angle_0_180(prox - joint, dist - joint)


def joint_angle_on_plane(prox: Optional[np.ndarray],
                         joint: Optional[np.ndarray],
                         dist: Optional[np.ndarray],
                         plane: str) -> Optional[float]:
    """Eklem iç açısını düzleme projekte edip ölçer (0..180)."""
    if prox is None or joint is None or dist is None:
        return None
    v1 = prox - joint
    v2 = dist - joint
    return angle_on_plane(v1, v2, plane)


def score_180_same_0_opposite(a_dir: Optional[np.ndarray],
                             b_dir: Optional[np.ndarray]) -> Optional[float]:
    """
    - aynı doğrultu => 180
    - dik          => 90
    - zıt          => 0
    score = 180 - angle_0_180(a, b)
    """
    if a_dir is None or b_dir is None:
        return None
    base = angle_0_180(a_dir, b_dir)
    if base is None:
        return None
    return float(np.clip(180.0 - base, 0.0, 180.0))


def score_0_same_180_opposite(a_dir: Optional[np.ndarray],
                             b_dir: Optional[np.ndarray]) -> Optional[float]:
    """
    - aynı doğrultu => 0
    - dik          => 90
    - zıt          => 180
    score = angle_0_180(a, b)
    """
    if a_dir is None or b_dir is None:
        return None
    base = angle_0_180(a_dir, b_dir)
    if base is None:
        return None
    return float(np.clip(base, 0.0, 180.0))


def _plane_pack(prefix: str,
                v1: Optional[np.ndarray],
                v2: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """
    v1-v2 için:
      - full 3D angle
      - XY/XZ/YZ plane angles
    """
    out: Dict[str, Optional[float]] = {
        f"{prefix}_3d_deg": None,
        f"{prefix}_xy_deg": None,
        f"{prefix}_xz_deg": None,
        f"{prefix}_yz_deg": None,
    }
    if v1 is None or v2 is None:
        return out

    out[f"{prefix}_3d_deg"] = angle_0_180(v1, v2)
    out[f"{prefix}_xy_deg"] = angle_on_plane(v1, v2, "XY")
    out[f"{prefix}_xz_deg"] = angle_on_plane(v1, v2, "XZ")
    out[f"{prefix}_yz_deg"] = angle_on_plane(v1, v2, "YZ")
    return out


def _plane_pack_joint(prefix: str,
                      prox: Optional[np.ndarray],
                      joint: Optional[np.ndarray],
                      dist: Optional[np.ndarray]) -> Dict[str, Optional[float]]:
    """
    prox-joint-dist eklem açısı için:
      - full 3D joint angle
      - XY/XZ/YZ plane joint angles
    """
    out: Dict[str, Optional[float]] = {
        f"{prefix}_3d_deg": None,
        f"{prefix}_xy_deg": None,
        f"{prefix}_xz_deg": None,
        f"{prefix}_yz_deg": None,
    }
    if prox is None or joint is None or dist is None:
        return out

    out[f"{prefix}_3d_deg"] = joint_angle(prox, joint, dist)
    out[f"{prefix}_xy_deg"] = joint_angle_on_plane(prox, joint, dist, "XY")
    out[f"{prefix}_xz_deg"] = joint_angle_on_plane(prox, joint, dist, "XZ")
    out[f"{prefix}_yz_deg"] = joint_angle_on_plane(prox, joint, dist, "YZ")
    return out


# =========================================================
# Bone access helpers
# =========================================================
def _bones(json_data: dict) -> Dict[str, dict]:
    b = (json_data or {}).get("bones", {}) or {}
    return b if isinstance(b, dict) else {}


def _get_bone(json_data: dict, *candidates: str) -> Tuple[Optional[dict], Optional[str]]:
    bones = _bones(json_data)

    def _canon(s: str) -> str:
        s = str(s).strip().lower().replace(" ", ".")
        if s.startswith("spine.") and s[6:].isdigit():
            s = "spine." + s[6:].zfill(3)
        return s

    # exact
    for c in candidates:
        if c in bones:
            return bones[c], c

    # case-insensitive
    low = {k.lower(): k for k in bones.keys()}
    for c in candidates:
        ck = str(c).lower()
        if ck in low:
            real = low[ck]
            return bones[real], real

    # canonical
    canon_map = {_canon(k): k for k in bones.keys()}
    for c in candidates:
        cc = _canon(c)
        if cc in canon_map:
            real = canon_map[cc]
            return bones[real], real

    return None, None


def _get_point(json_data: dict, candidates: Iterable[str], which: str) -> Optional[np.ndarray]:
    b, _ = _get_bone(json_data, *list(candidates))
    if b is None:
        return None
    return _as_np(b.get(which, None))


def _dir_from_points(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """Normalize(b-a)"""
    if a is None or b is None:
        return None
    return _norm(b - a)


def _get_dir_from_bone(json_data: dict,
                       candidates: Iterable[str],
                       head_key="head_world",
                       tail_key="tail_world") -> Optional[np.ndarray]:
    h = _get_point(json_data, candidates, head_key)
    t = _get_point(json_data, candidates, tail_key)
    return _dir_from_points(h, t)


# =========================================================
# Main
# =========================================================
from typing import Dict, List, Optional
import numpy as np


def compute_reba_angles_from_json(json_data: dict) -> Dict[str, Optional[float]]:
    """
    Mevcut açı seti + her açı için XY/XZ/YZ düzlem projeksiyon açıları.

    Eklenenler:
      - shoulder.* ile upperarm* arasındaki açı -> 3D + XY/XZ/YZ
      - seçili kemikler için rotation_local_quaternion bilgileri

    Çıktı anahtar formatı:
      - ..._3d_deg
      - ..._xy_deg
      - ..._xz_deg
      - ..._yz_deg
      - ..._rotation_local_quaternion
    """
    out: Dict[str, Optional[float]] = {}

    # -----------------------------------------------------
    # Yardımcı: quaternion çek
    # -----------------------------------------------------
    def _get_quat(json_data: dict, bone_names) -> Optional[List[float]]:
        bones = json_data.get("bones", {})
        bone = None

        for bn in bone_names:
            if bn in bones:
                bone = bones[bn]
                break

        if bone is None:
            return None

        q = bone.get("rotation_local_quaternion", None)
        if q is None:
            return None

        try:
            return [float(v) for v in q]
        except Exception:
            return None

    # -----------------------------------------------------
    # Spine directions
    # -----------------------------------------------------
    sp1_head = _get_point(json_data, ["spine 1", "spine1", "spine.001", "spine001"], "head_world")
    sp4_tail = _get_point(json_data, ["spine 4", "spine4", "spine.004", "spine004"], "tail_world")

    spine_up = _dir_from_points(sp1_head, sp4_tail)
    if spine_up is None:
        spine_up = _get_dir_from_bone(json_data, ["spine 4", "spine4", "spine.004", "spine004"])

    spine1_dir = _get_dir_from_bone(json_data, ["spine 1", "spine1", "spine.001", "spine001"])
    spine2_dir = _get_dir_from_bone(json_data, ["spine 2", "spine2", "spine.002", "spine002"])
    spine3_dir = _get_dir_from_bone(json_data, ["spine 3", "spine3", "spine.003", "spine003"])
    spine4_dir = _get_dir_from_bone(json_data, ["spine 4", "spine4", "spine.004", "spine004"])

    out.update(_plane_pack("spine1_vs_spine2", spine1_dir, spine2_dir))
    out.update(_plane_pack("spine2_vs_spine3", spine2_dir, spine3_dir))
    out.update(_plane_pack("spine3_vs_spine4", spine3_dir, spine4_dir))

    # -----------------------------------------------------
    # Arms points
    # -----------------------------------------------------
    sh_l_pt = _get_point(json_data, ["sholder.l", "shoulder.l", "shoulder_l"], "head_world")
    sh_r_pt = _get_point(json_data, ["sholder.r", "shoulder.r", "shoulder_r"], "head_world")

    el_l = _get_point(json_data, ["upperarm.l", "upper_arm.l", "uparm.l"], "tail_world")
    el_r = _get_point(json_data, ["upperarm.r", "upper_arm.r", "uparm.r"], "tail_world")

    wr_l = _get_point(json_data, ["forearm.l", "lowerarm.l"], "tail_world")
    wr_r = _get_point(json_data, ["forearm.r", "lowerarm.r"], "tail_world")

    hand_lt = _get_point(json_data, ["hand.l", "hand_l"], "tail_world")
    hand_rt = _get_point(json_data, ["hand.r", "hand_r"], "tail_world")

    upperarm_l_dir = _dir_from_points(sh_l_pt, el_l)
    upperarm_r_dir = _dir_from_points(sh_r_pt, el_r)

    # -----------------------------------------------------
    # upperarm vs spine_up
    # -----------------------------------------------------
    if upperarm_l_dir is not None and spine_up is not None:
        out["upperarmL_vs_spine_up_3d_deg"] = score_180_same_0_opposite(upperarm_l_dir, spine_up)
        for pl in ("XY", "XZ", "YZ"):
            ang = angle_on_plane(upperarm_l_dir, spine_up, pl)
            out[f"upperarmL_vs_spine_up_{pl.lower()}_deg"] = None if ang is None else float(np.clip(180.0 - ang, 0.0, 180.0))
    else:
        out["upperarmL_vs_spine_up_3d_deg"] = None
        out["upperarmL_vs_spine_up_xy_deg"] = None
        out["upperarmL_vs_spine_up_xz_deg"] = None
        out["upperarmL_vs_spine_up_yz_deg"] = None

    if upperarm_r_dir is not None and spine_up is not None:
        out["upperarmR_vs_spine_up_3d_deg"] = score_180_same_0_opposite(upperarm_r_dir, spine_up)
        for pl in ("XY", "XZ", "YZ"):
            ang = angle_on_plane(upperarm_r_dir, spine_up, pl)
            out[f"upperarmR_vs_spine_up_{pl.lower()}_deg"] = None if ang is None else float(np.clip(180.0 - ang, 0.0, 180.0))
    else:
        out["upperarmR_vs_spine_up_3d_deg"] = None
        out["upperarmR_vs_spine_up_xy_deg"] = None
        out["upperarmR_vs_spine_up_xz_deg"] = None
        out["upperarmR_vs_spine_up_yz_deg"] = None

    # -----------------------------------------------------
    # Elbow / Wrist
    # -----------------------------------------------------
    out.update(_plane_pack_joint("elbowL", sh_l_pt, el_l, wr_l))
    out.update(_plane_pack_joint("elbowR", sh_r_pt, el_r, wr_r))

    out.update(_plane_pack_joint("wristL", el_l, wr_l, hand_lt))
    out.update(_plane_pack_joint("wristR", el_r, wr_r, hand_rt))

    # -----------------------------------------------------
    # Shoulder bone direction vs upperarm bone direction
    # -----------------------------------------------------
    shoulder_l_dir = _get_dir_from_bone(json_data, ["sholder.l", "shoulder.l", "shoulder_l"])
    shoulder_r_dir = _get_dir_from_bone(json_data, ["sholder.r", "shoulder.r", "shoulder_r"])

    upperarm_l_bone_dir = _get_dir_from_bone(json_data, ["upperarm.l", "upper_arm.l", "uparm.l"])
    upperarm_r_bone_dir = _get_dir_from_bone(json_data, ["upperarm.r", "upper_arm.r", "uparm.r"])

    out.update(_plane_pack("shoulderL_vs_upperarmL", shoulder_l_dir, upperarm_l_bone_dir))
    out.update(_plane_pack("shoulderR_vs_upperarmR", shoulder_r_dir, upperarm_r_bone_dir))

    # -----------------------------------------------------
    # Legs points
    # -----------------------------------------------------
    hip_l = _get_point(json_data, ["upperleg.l", "upper_leg.l", "thigh.l", "upleg.l"], "head_world")
    hip_r = _get_point(json_data, ["upperleg.r", "upper_leg.r", "thigh.r", "upleg.r"], "head_world")
    knee_l = _get_point(json_data, ["shin.l", "shin_l", "l_shin", "lowerleg.l"], "head_world")
    knee_r = _get_point(json_data, ["shin.r", "shin_r", "r_shin", "lowerleg.r"], "head_world")
    ankle_l = _get_point(json_data, ["foot.l", "foot_l", "l_foot"], "head_world")
    ankle_r = _get_point(json_data, ["foot.r", "foot_r", "r_foot"], "head_world")

    out.update(_plane_pack_joint("kneeL", hip_l, knee_l, ankle_l))
    out.update(_plane_pack_joint("kneeR", hip_r, knee_r, ankle_r))

    thigh_l_dir = _dir_from_points(hip_l, knee_l)
    thigh_r_dir = _dir_from_points(hip_r, knee_r)

    out["thighL_vs_spine1_3d_deg"] = (
        score_0_same_180_opposite(thigh_l_dir, spine1_dir)
        if (thigh_l_dir is not None and spine1_dir is not None) else None
    )
    out["thighR_vs_spine1_3d_deg"] = (
        score_0_same_180_opposite(thigh_r_dir, spine1_dir)
        if (thigh_r_dir is not None and spine1_dir is not None) else None
    )

    for side, tdir in (("L", thigh_l_dir), ("R", thigh_r_dir)):
        if tdir is not None and spine1_dir is not None:
            out[f"thigh{side}_vs_spine1_xy_deg"] = angle_on_plane(tdir, spine1_dir, "XY")
            out[f"thigh{side}_vs_spine1_xz_deg"] = angle_on_plane(tdir, spine1_dir, "XZ")
            out[f"thigh{side}_vs_spine1_yz_deg"] = angle_on_plane(tdir, spine1_dir, "YZ")
        else:
            out[f"thigh{side}_vs_spine1_xy_deg"] = None
            out[f"thigh{side}_vs_spine1_xz_deg"] = None
            out[f"thigh{side}_vs_spine1_yz_deg"] = None

    # -----------------------------------------------------
    # Head vs neck
    # -----------------------------------------------------
    d_neck = _get_dir_from_bone(json_data, ["nack", "neck"])
    d_head = _get_dir_from_bone(json_data, ["head"])
    out.update(_plane_pack("head_vs_neck", d_head, d_neck))

    # -----------------------------------------------------
    # Local quaternion bilgileri
    # -----------------------------------------------------
    out["head_rotation_local_quaternion"] = _get_quat(json_data, ["head"])
    out["neck_rotation_local_quaternion"] = _get_quat(json_data, ["nack", "neck"])

    out["shoulderL_rotation_local_quaternion"] = _get_quat(json_data, ["sholder.l", "shoulder.l", "shoulder_l"])
    out["shoulderR_rotation_local_quaternion"] = _get_quat(json_data, ["sholder.r", "shoulder.r", "shoulder_r"])

    out["upperarmL_rotation_local_quaternion"] = _get_quat(json_data, ["upperarm.l", "upper_arm.l", "uparm.l"])
    out["upperarmR_rotation_local_quaternion"] = _get_quat(json_data, ["upperarm.r", "upper_arm.r", "uparm.r"])

    out["forearmL_rotation_local_quaternion"] = _get_quat(json_data, ["forearm.l", "lowerarm.l"])
    out["forearmR_rotation_local_quaternion"] = _get_quat(json_data, ["forearm.r", "lowerarm.r"])

    out["handL_rotation_local_quaternion"] = _get_quat(json_data, ["hand.l", "hand_l"])
    out["handR_rotation_local_quaternion"] = _get_quat(json_data, ["hand.r", "hand_r"])

    out["thighL_rotation_local_quaternion"] = _get_quat(json_data, ["upperleg.l", "upper_leg.l", "thigh.l", "upleg.l"])
    out["thighR_rotation_local_quaternion"] = _get_quat(json_data, ["upperleg.r", "upper_leg.r", "thigh.r", "upleg.r"])

    out["shinL_rotation_local_quaternion"] = _get_quat(json_data, ["shin.l", "shin_l", "l_shin", "lowerleg.l"])
    out["shinR_rotation_local_quaternion"] = _get_quat(json_data, ["shin.r", "shin_r", "r_shin", "lowerleg.r"])

    out["footL_rotation_local_quaternion"] = _get_quat(json_data, ["foot.l", "foot_l", "l_foot"])
    out["footR_rotation_local_quaternion"] = _get_quat(json_data, ["foot.r", "foot_r", "r_foot"])

    out["spine1_rotation_local_quaternion"] = _get_quat(json_data, ["spine 1", "spine1", "spine.001", "spine001"])
    out["spine2_rotation_local_quaternion"] = _get_quat(json_data, ["spine 2", "spine2", "spine.002", "spine002"])
    out["spine3_rotation_local_quaternion"] = _get_quat(json_data, ["spine 3", "spine3", "spine.003", "spine003"])
    out["spine4_rotation_local_quaternion"] = _get_quat(json_data, ["spine 4", "spine4", "spine.004", "spine004"])

    return out
def compute_reba_angles(json_path: str) -> Dict[str, Optional[float]]:
    """Dosyadan oku ve compute_reba_angles_from_json çağır."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return compute_reba_angles_from_json(data)


# ---------------------------------------------------------
# Örnek kullanım:
# res = compute_reba_angles(r"C:\Python_Output\Makehuman\bones_angle0.json")
# for k, v in res.items():
#     print(k, "=", v)
# ---------------------------------------------------------
# --------------------------------------------------
# Yardımcı fonksiyonlar
# --------------------------------------------------
def _get_point(row, kp_name):
    """
    Örn: kp_name='left_shoulder' için
    [left_shoulder_x, left_shoulder_y, left_shoulder_z] döndürür.
    """
    cols = [f"{kp_name}_x", f"{kp_name}_y", f"{kp_name}_z"]
    if not all(c in row.index for c in cols):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return np.array([row[cols[0]], row[cols[1]], row[cols[2]]], dtype=float)


def _safe_norm(v):
    if np.any(np.isnan(v)):
        return np.nan
    return np.linalg.norm(v)


def _angle_between_vectors(v1, v2):
    """
    İki vektör arasındaki 3D açıyı derece cinsinden döndürür.
    """
    n1 = _safe_norm(v1)
    n2 = _safe_norm(v2)

    if np.isnan(n1) or np.isnan(n2) or n1 == 0 or n2 == 0:
        return np.nan

    cosang = np.dot(v1, v2) / (n1 * n2)
    cosang = np.clip(cosang, -1.0, 1.0)
    return np.degrees(np.arccos(cosang))


def _project_vector(v, plane="xy"):
    """
    Vektörü ilgili düzleme projeler.
    xy -> z=0
    xz -> y=0
    yz -> x=0
    """
    if np.any(np.isnan(v)):
        return np.array([np.nan, np.nan, np.nan], dtype=float)

    v = v.copy().astype(float)

    if plane == "xy":
        v[2] = 0.0
    elif plane == "xz":
        v[1] = 0.0
    elif plane == "yz":
        v[0] = 0.0
    else:
        raise ValueError("plane yalnızca 'xy', 'xz' veya 'yz' olabilir.")

    return v


def _joint_angle(row, p1, p2, p3, plane=None):
    """
    p1 - p2 - p3 şeklindeki eklem açısını hesaplar.
    Açı p2 noktasında oluşur.

    Örnek:
    omuz açısı = elbow - shoulder - hip
    dirsek açısı = shoulder - elbow - wrist
    """
    P1 = _get_point(row, p1)
    P2 = _get_point(row, p2)
    P3 = _get_point(row, p3)

    v1 = P1 - P2
    v2 = P3 - P2

    if plane is not None:
        v1 = _project_vector(v1, plane=plane)
        v2 = _project_vector(v2, plane=plane)

    return _angle_between_vectors(v1, v2)


def _segment_angle(row, a1, a2, b1, b2, plane=None):
    """
    a1->a2 segmenti ile b1->b2 segmenti arasındaki açıyı hesaplar.
    """
    A1 = _get_point(row, a1)
    A2 = _get_point(row, a2)
    B1 = _get_point(row, b1)
    B2 = _get_point(row, b2)

    v1 = A2 - A1
    v2 = B2 - B1

    if plane is not None:
        v1 = _project_vector(v1, plane=plane)
        v2 = _project_vector(v2, plane=plane)

    return _angle_between_vectors(v1, v2)


def _midpoint(row, p1, p2):
    P1 = _get_point(row, p1)
    P2 = _get_point(row, p2)
    if np.any(np.isnan(P1)) or np.any(np.isnan(P2)):
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return (P1 + P2) / 2.0


# --------------------------------------------------
# Ana fonksiyon
# --------------------------------------------------
def add_pose_angles(df_input, add_planar_angles=True):
    """
    df_input içindeki 17 keypoint koordinatlarından açı kolonları üretir.
    
    Parametreler
    ------------
    df_input : pd.DataFrame
        Kolon yapısı:
        nose_x, nose_y, nose_z, left_shoulder_x, ...
    add_planar_angles : bool
        True ise ayrıca xy/xz/yz düzlemlerindeki açıları da ekler.

    Döndürür
    --------
    df_out : pd.DataFrame
        Yeni açı kolonları eklenmiş kopya dataframe
    """
    df_out = df_input.copy()

    # -----------------------------
    # 1) Temel 3D eklem açıları
    # -----------------------------
    df_out["left_elbow_angle_3d"] = df_out.apply(
        lambda r: _joint_angle(r, "left_shoulder", "left_elbow", "left_wrist"),
        axis=1
    )
    df_out["right_elbow_angle_3d"] = df_out.apply(
        lambda r: _joint_angle(r, "right_shoulder", "right_elbow", "right_wrist"),
        axis=1
    )

    df_out["left_shoulder_angle_3d"] = df_out.apply(
        lambda r: _joint_angle(r, "left_elbow", "left_shoulder", "left_hip"),
        axis=1
    )
    df_out["right_shoulder_angle_3d"] = df_out.apply(
        lambda r: _joint_angle(r, "right_elbow", "right_shoulder", "right_hip"),
        axis=1
    )

    df_out["left_hip_angle_3d"] = df_out.apply(
        lambda r: _joint_angle(r, "left_shoulder", "left_hip", "left_knee"),
        axis=1
    )
    df_out["right_hip_angle_3d"] = df_out.apply(
        lambda r: _joint_angle(r, "right_shoulder", "right_hip", "right_knee"),
        axis=1
    )

    df_out["left_knee_angle_3d"] = df_out.apply(
        lambda r: _joint_angle(r, "left_hip", "left_knee", "left_ankle"),
        axis=1
    )
    df_out["right_knee_angle_3d"] = df_out.apply(
        lambda r: _joint_angle(r, "right_hip", "right_knee", "right_ankle"),
        axis=1
    )

    # -----------------------------
    # 2) Gövde / boyun / pelvis ilişkileri
    # -----------------------------
    def calc_extra_angles(row):
        out = {}

        shoulder_mid = _midpoint(row, "left_shoulder", "right_shoulder")
        hip_mid = _midpoint(row, "left_hip", "right_hip")
        nose = _get_point(row, "nose")
        left_hip = _get_point(row, "left_hip")
        right_hip = _get_point(row, "right_hip")
        left_knee = _get_point(row, "left_knee")
        right_knee = _get_point(row, "right_knee")
        left_shoulder = _get_point(row, "left_shoulder")
        right_shoulder = _get_point(row, "right_shoulder")

        # trunk: hip_mid -> shoulder_mid
        trunk_vec = shoulder_mid - hip_mid
        neck_vec = nose - shoulder_mid   # yaklaşık boyun/baş yönü

        left_thigh_vec = left_knee - left_hip
        right_thigh_vec = right_knee - right_hip

        shoulder_line = right_shoulder - left_shoulder
        hip_line = right_hip - left_hip

        out["trunk_vs_left_thigh_angle_3d"] = _angle_between_vectors(trunk_vec, left_thigh_vec)
        out["trunk_vs_right_thigh_angle_3d"] = _angle_between_vectors(trunk_vec, right_thigh_vec)
        out["neck_vs_trunk_angle_3d"] = _angle_between_vectors(neck_vec, trunk_vec)
        out["shoulder_line_vs_hip_line_angle_3d"] = _angle_between_vectors(shoulder_line, hip_line)

        return pd.Series(out)

    df_out = pd.concat([df_out, df_out.apply(calc_extra_angles, axis=1)], axis=1)

    # -----------------------------
    # 3) İsteğe bağlı düzlemsel açılar
    # -----------------------------
    if add_planar_angles:
        planes = ["xy", "xz", "yz"]

        joint_defs = {
            "left_elbow_angle": ("left_shoulder", "left_elbow", "left_wrist"),
            "right_elbow_angle": ("right_shoulder", "right_elbow", "right_wrist"),
            "left_shoulder_angle": ("left_elbow", "left_shoulder", "left_hip"),
            "right_shoulder_angle": ("right_elbow", "right_shoulder", "right_hip"),
            "left_hip_angle": ("left_shoulder", "left_hip", "left_knee"),
            "right_hip_angle": ("right_shoulder", "right_hip", "right_knee"),
            "left_knee_angle": ("left_hip", "left_knee", "left_ankle"),
            "right_knee_angle": ("right_hip", "right_knee", "right_ankle"),
        }

        for plane in planes:
            for col_base, (p1, p2, p3) in joint_defs.items():
                df_out[f"{col_base}_{plane}"] = df_out.apply(
                    lambda r, p1=p1, p2=p2, p3=p3, plane=plane: _joint_angle(r, p1, p2, p3, plane=plane),
                    axis=1
                )

            # omuz hattı ile kalça hattı düzlemsel açı
            df_out[f"shoulder_line_vs_hip_line_angle_{plane}"] = df_out.apply(
                lambda r, plane=plane: _segment_angle(r, "left_shoulder", "right_shoulder",
                                                      "left_hip", "right_hip", plane=plane),
                axis=1
            )

    return df_out