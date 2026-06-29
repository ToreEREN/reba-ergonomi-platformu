import sys
if 'Utils.Imports' in sys.modules:
    del sys.modules['Utils.Imports']
from Utils.Imports import *


import numpy as np
import bpy, math, json, pickle, os
from math import radians
from mathutils import Vector, Euler, Quaternion

# =======================
# YARDIMCI FONKSİYONLAR
# =======================
def ensure_dir(p: str):
    if not os.path.exists(p):
        os.makedirs(p, exist_ok=True)

def scene_bbox(objects):
    """Sahnedeki görünür MESH/ARMATURE birleşik bbox (mins, maxs, center, size) döndürür."""
    mins = Vector(( 1e9,  1e9,  1e9))
    maxs = Vector((-1e9, -1e9, -1e9))
    any_found = False
    for o in objects:
        if not o.visible_get(): 
            continue
        if o.type not in {"MESH", "ARMATURE"}:
            continue
        any_found = True
        for v in o.bound_box:
            wv = o.matrix_world @ Vector(v)
            mins.x = min(mins.x, wv.x); mins.y = min(mins.y, wv.y); mins.z = min(mins.z, wv.z)
            maxs.x = max(maxs.x, wv.x); maxs.y = max(maxs.y, wv.y); maxs.z = max(maxs.z, wv.z)
    if not any_found:
        raise RuntimeError("Sahnede görünür MESH/ARMATURE bulunamadı.")
    center = (mins + maxs) * 0.5
    size   = maxs - mins
    return mins, maxs, center, size

def get_or_create_camera(name="AutoCamera", loc=(0, -5, 1.6), rot_euler=(radians(90), 0, 0)):
    """UI context gerekmeden kamera oluşturur ve sahneye linkler."""
    scene = bpy.context.scene
    for o in bpy.data.objects:
        if o.type == 'CAMERA':
            cam_obj = o
            break
    else:
        cam_data = bpy.data.cameras.new(name)
        cam_obj  = bpy.data.objects.new(name, cam_data)
        scene.collection.objects.link(cam_obj)
    cam_obj.location = loc
    cam_obj.rotation_mode = 'XYZ'
    cam_obj.rotation_euler = rot_euler
    scene.camera = cam_obj
    try:
        bpy.context.view_layer.objects.active = cam_obj
        cam_obj.select_set(True)
    except Exception:
        pass
    return cam_obj

def look_at_with_trackto(cam_obj, target_vec):
    """Kamerayı hedefe baktırmak için Track-To kurar (Empty ile)."""
    old = bpy.data.objects.get("Aim_Target")
    if old:
        try: bpy.data.objects.remove(old, do_unlink=True)
        except Exception: pass
    empty = bpy.data.objects.new("Aim_Target", None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = target_vec
    for c in list(cam_obj.constraints):
        if c.type == 'TRACK_TO':
            cam_obj.constraints.remove(c)
    c = cam_obj.constraints.new(type='TRACK_TO')
    c.target = empty
    c.track_axis = 'TRACK_NEGATIVE_Z'
    c.up_axis    = 'UP_Y'
    return empty

def camera_space_extents(cam_obj, objects):
    """
    Objelerin eval edilmiş vertekslerini kamera uzayına dönüştürüp
    X (genişlik) ve Y (yükseklik) yayılımlarını döndürür.
    """
    deps = bpy.context.evaluated_depsgraph_get()
    M_cam = cam_obj.matrix_world.inverted()
    min_x, max_x =  1e9, -1e9
    min_y, max_y =  1e9, -1e9
    any_mesh = False
    for o in objects:
        if not o.visible_get() or o.type != 'MESH':
            continue
        any_mesh = True
        o_eval = o.evaluated_get(deps)
        me = o_eval.to_mesh()
        MW = o.matrix_world
        for v in me.vertices:
            p_cam = M_cam @ (MW @ v.co)   # cam local: X sag, Y yukarı, -Z ileri
            x, y = p_cam.x, p_cam.y
            if x < min_x: min_x = x
            if x > max_x: max_x = x
            if y < min_y: min_y = y
            if y > max_y: max_y = y
        o_eval.to_mesh_clear()
    if not any_mesh:
        raise RuntimeError("Sahnede MESH yok (camera_space_extents).")
    return (max_x - min_x), (max_y - min_y)
from bpy_extras.object_utils import world_to_camera_view

def fit_ortho_scale_to_screen(cam_obj, objects, aspect, margin=0.03, max_iters=25):
    """
    Ortho kameranın ortho_scale'ini büyüterek, objelerin tamamını ekran uzayında
    [margin, 1-margin] içine sığdırır. (Kesilmeyi kesin çözer)
    """
    scene = bpy.context.scene

    def screen_bounds():
        umin, umax =  1e9, -1e9
        vmin, vmax =  1e9, -1e9
        deps = bpy.context.evaluated_depsgraph_get()
        for o in objects:
            if not o.visible_get() or o.type != 'MESH':
                continue
            o_eval = o.evaluated_get(deps)
            me = o_eval.to_mesh()
            MW = o.matrix_world
            for v in me.vertices:
                co_world = MW @ v.co
                uvw = world_to_camera_view(scene, cam_obj, co_world)  # (u,v,w)  u,v: 0..1 ekran
                u, v_ = uvw.x, uvw.y
                if u < umin: umin = u
                if u > umax: umax = u
                if v_ < vmin: vmin = v_
                if v_ > vmax: vmax = v_
            o_eval.to_mesh_clear()
        return umin, umax, vmin, vmax

    # Her iterasyonda ekrandan taşma varsa scale'i arttır
    for _ in range(max_iters):
        umin, umax, vmin, vmax = screen_bounds()

        # Taşma miktarlarını hesapla (sol/sağ/alt/üst)
        over_left   = max(0.0, (margin - umin))
        over_right  = max(0.0, (umax   - (1.0 - margin)))
        over_bottom = max(0.0, (margin - vmin))
        over_top    = max(0.0, (vmax   - (1.0 - margin)))

        # Hiç taşma yoksa kır
        if over_left == over_right == over_bottom == over_top == 0.0:
            break

        # Ekran genişliği/h yüksekliği taşmalarına göre gereken ölçek katsayısı
        # Ortho'da ölçek lineer: ufak artışlarla yaklaş.
        overflow_u = max(over_left, over_right)
        overflow_v = max(over_bottom, over_top)

        # Taşma oranını kaba şekilde ölçeğe çevir: (1 + 2*overflow)
        # (margin'li iç kareye sığana kadar büyüt)
        grow_u = 1.0 + 2.0 * overflow_u
        grow_v = 1.0 + 2.0 * overflow_v

        grow = max(grow_u, grow_v)
        # Çok küçük artışlarda takılmaması için min büyütme
        if grow < 1.02:
            grow = 1.02

        cam_obj.data.ortho_scale *= grow
import json
import os
from PIL import Image, ImageDraw, ImageFont
import os, json
import bpy
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

def _find_armature(arm_name=None):
    if arm_name and arm_name in bpy.data.objects:
        obj = bpy.data.objects[arm_name]
        if obj.type == "ARMATURE":
            return obj
    # otomatik: sahnedeki ilk armature
    for o in bpy.context.scene.objects:
        if o.type == "ARMATURE" and o.visible_get():
            return o
    return None

def _world_and_screen_of_bones(scene, cam_obj, arm_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_eval = arm_obj.evaluated_get(depsgraph)  # poz uygulanmış değerlendirilmiş hali

    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    bones_out = {}
    for pb in arm_eval.pose.bones:
        # pose bone head/tail armature-local uzayda gelir; world'e çevir
        head_w = arm_eval.matrix_world @ pb.head
        tail_w = arm_eval.matrix_world @ pb.tail

        # 2D ekran (0..1) + piksel
        head_ndc = world_to_camera_view(scene, cam_obj, head_w)  # x,y,z in 0..1
        tail_ndc = world_to_camera_view(scene, cam_obj, tail_w)

        head_px = [head_ndc.x * res_x, (1.0 - head_ndc.y) * res_y]  # y ekseni ters
        tail_px = [tail_ndc.x * res_x, (1.0 - tail_ndc.y) * res_y]

        bones_out[pb.name] = {
            "head_world": [head_w.x, head_w.y, head_w.z],
            "tail_world": [tail_w.x, tail_w.y, tail_w.z],
            "head_ndc":   [head_ndc.x, head_ndc.y, head_ndc.z],
            "tail_ndc":   [tail_ndc.x, tail_ndc.y, tail_ndc.z],
            "head_px":    head_px,
            "tail_px":    tail_px,
            "in_front_head": (0.0 <= head_ndc.z <= 1.0),
            "in_front_tail": (0.0 <= tail_ndc.z <= 1.0),
        }
    return bones_out


def draw_bones_from_json_withimage_blend(
    json_path: str,
    image_path: str | None = None,
    out_path: str | None = None,
    show: bool = True,
    draw_names: bool = True,
    only_in_front: bool = False,
    line_width: int = 3,
    head_radius: int = 4,
):
    """
    bones_*.json içeriğinden (head_px, tail_px) kullanarak görsele bone çizimi yapar.
    - image_path verilmezse json içindeki "image_file" alanını json klasörüne göre arar.
    - Eğer görsel boyutu json'daki resolution ile farklıysa px koordinatlarını ölçekler.
    - only_in_front=True ise sadece in_front_head & in_front_tail True olanları çizer.
    """

    json_path = os.path.abspath(json_path)
    base_dir = os.path.dirname(json_path)

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Görsel yolu
    if image_path is None:
        img_name = data.get("image_file", None)
        if not img_name:
            raise ValueError("image_path verilmedi ve JSON içinde 'image_file' yok.")
        image_path = os.path.join(base_dir, img_name)
    image_path = os.path.abspath(image_path)

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Görsel bulunamadı: {image_path}")

    # Çıktı yolu
    if out_path is None:
        name, ext = os.path.splitext(os.path.basename(image_path))
        out_path = os.path.join(base_dir, f"{name}__bones{ext}")
    out_path = os.path.abspath(out_path)

    # JSON resolution (width,height) bekliyoruz
    res = data.get("resolution", None)
    if not res or len(res) != 2:
        # fallback: ölçekleme yapmayız
        res_w = res_h = None
    else:
        res_w, res_h = float(res[0]), float(res[1])

    img = Image.open(image_path).convert("RGBA")
    W, H = img.size

    # Ölçek katsayıları
    if res_w and res_h and res_w > 0 and res_h > 0:
        sx = W / res_w
        sy = H / res_h
    else:
        sx = sy = 1.0

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    # Basit font (bulamazsa default)
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    bones = data.get("bones", {})
    if not isinstance(bones, dict) or not bones:
        raise ValueError("JSON içinde 'bones' sözlüğü boş ya da yok.")

    def _scale_pt(pt):
        x, y = float(pt[0]), float(pt[1])
        return (x * sx, y * sy)

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

        # Renk: önden/arkadan bilgisi varsa ona göre ayır
        # (Senin örnekte hepsi False olduğu için genelde "arkada" rengi olur.)
        in_front = bool(b.get("in_front_head", False) or b.get("in_front_tail", False))
        line_color = (0, 255, 0, 200) if in_front else (255, 0, 0, 200)  # yeşil/ kırmızı

        # çizgi
        draw.line([(x1, y1), (x2, y2)], fill=line_color, width=int(line_width))

        # head noktası
        r = int(head_radius)
        draw.ellipse((x1 - r, y1 - r, x1 + r, y1 + r), fill=(255, 255, 255, 220), outline=line_color)

        # isim
        if draw_names:
            draw.text((x1 + 6, y1 + 6), bone_name, fill=(255, 255, 255, 220), font=font)

    out_img = Image.alpha_composite(img, overlay).convert("RGB")
    out_img.save(out_path)

    if show:
        out_img.show()

    return out_path

import math
import json
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view


def _safe_angle_deg(v1, v2):
    """İki vektör arasındaki 3D açıyı derece cinsinden döndürür."""
    if v1.length < 1e-8 or v2.length < 1e-8:
        return None
    dot = max(-1.0, min(1.0, v1.normalized().dot(v2.normalized())))
    return math.degrees(math.acos(dot))


def _safe_angle_deg_2d(a1, a2, b1, b2):
    """
    İki 2D vektör arasındaki açıyı derece cinsinden döndürür.
    v1=(a1,a2), v2=(b1,b2)
    """
    n1 = math.sqrt(a1*a1 + a2*a2)
    n2 = math.sqrt(b1*b1 + b2*b2)
    if n1 < 1e-8 or n2 < 1e-8:
        return None

    dot = (a1*b1 + a2*b2) / (n1*n2)
    dot = max(-1.0, min(1.0, dot))
    return math.degrees(math.acos(dot))


def _bone_world_vector(head_w, tail_w):
    return tail_w - head_w


def _angles_between_bones(vec_a, vec_b):
    """
    İki 3D vektör arasında 3d, xy, xz, yz açılarını hesaplar.
    """
    return {
        "angle_3d_deg": _safe_angle_deg(vec_a, vec_b),
        "angle_xy_deg": _safe_angle_deg_2d(vec_a.x, vec_a.y, vec_b.x, vec_b.y),
        "angle_xz_deg": _safe_angle_deg_2d(vec_a.x, vec_a.z, vec_b.x, vec_b.z),
        "angle_yz_deg": _safe_angle_deg_2d(vec_a.y, vec_a.z, vec_b.y, vec_b.z),
    }

from pathlib import Path

def list_blend_files_as_array(folder: str, recursive: bool = True):
    """
    Verilen klasördeki tüm .blend dosyalarını bulur ve tam path listesi döndürür.
    recursive=True ise alt klasörleri de tarar.
    """
    base = Path(folder)

    if not base.exists():
        raise FileNotFoundError(f"Klasör bulunamadı: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Verilen yol klasör değil: {base}")

    pattern = "**/*.blend" if recursive else "*.blend"
    blend_paths = sorted(p.resolve() for p in base.glob(pattern))

    # str array (istenen “aray” genelde bu oluyor)
    blend_array = [str(p) for p in blend_paths]

    # Ek: sadece dosya isimleri
    blend_names = [p.name for p in blend_paths]

    return blend_array, blend_names
def _world_screen_and_angles_of_bones(scene, cam_obj, arm_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_eval = arm_obj.evaluated_get(depsgraph)

    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    bones_out = {}

    # Önce tüm kemiklerin dünya koordinatlarını toplayalım
    temp_world = {}
    for pb in arm_eval.pose.bones:
        head_w = arm_eval.matrix_world @ pb.head
        tail_w = arm_eval.matrix_world @ pb.tail
        temp_world[pb.name] = {
            "pose_bone": pb,
            "head_w": head_w,
            "tail_w": tail_w,
            "vec_w": tail_w - head_w
        }

    # Sonra ekran projeksiyonu + parent açılarını hesaplayalım
    for bone_name, info in temp_world.items():
        pb = info["pose_bone"]
        head_w = info["head_w"]
        tail_w = info["tail_w"]
        bone_vec = info["vec_w"]

        head_ndc = world_to_camera_view(scene, cam_obj, head_w)
        tail_ndc = world_to_camera_view(scene, cam_obj, tail_w)

        head_px = [head_ndc.x * res_x, (1.0 - head_ndc.y) * res_y]
        tail_px = [tail_ndc.x * res_x, (1.0 - tail_ndc.y) * res_y]

        parent_name = pb.parent.name if pb.parent else None
        parent_angles = None

        if pb.parent and pb.parent.name in temp_world:
            parent_vec = temp_world[pb.parent.name]["vec_w"]
            parent_angles = _angles_between_bones(parent_vec, bone_vec)

        bones_out[bone_name] = {
            "parent_name": parent_name,
            "head_world": [head_w.x, head_w.y, head_w.z],
            "tail_world": [tail_w.x, tail_w.y, tail_w.z],
            "bone_vector_world": [bone_vec.x, bone_vec.y, bone_vec.z],

            "head_ndc": [head_ndc.x, head_ndc.y, head_ndc.z],
            "tail_ndc": [tail_ndc.x, tail_ndc.y, tail_ndc.z],
            "head_px": head_px,
            "tail_px": tail_px,

            "in_front_head": (0.0 <= head_ndc.z <= 1.0),
            "in_front_tail": (0.0 <= tail_ndc.z <= 1.0),

            # parent ile açı bilgileri
            "angle_vs_parent_3d_deg": None if parent_angles is None else parent_angles["angle_3d_deg"],
            "angle_vs_parent_xy_deg": None if parent_angles is None else parent_angles["angle_xy_deg"],
            "angle_vs_parent_xz_deg": None if parent_angles is None else parent_angles["angle_xz_deg"],
            "angle_vs_parent_yz_deg": None if parent_angles is None else parent_angles["angle_yz_deg"],
        }

    return bones_out

def _world_screen_and_angles_of_bones(scene, cam_obj, arm_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_eval = arm_obj.evaluated_get(depsgraph)

    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    bones_out = {}

    # Önce tüm kemiklerin dünya koordinatlarını toplayalım
    temp_world = {}
    for pb in arm_eval.pose.bones:
        head_w = arm_eval.matrix_world @ pb.head
        tail_w = arm_eval.matrix_world @ pb.tail
        temp_world[pb.name] = {
            "pose_bone": pb,
            "head_w": head_w,
            "tail_w": tail_w,
            "vec_w": tail_w - head_w
        }

    # Sonra ekran projeksiyonu + parent açılarını hesaplayalım
    for bone_name, info in temp_world.items():
        pb = info["pose_bone"]
        head_w = info["head_w"]
        tail_w = info["tail_w"]
        bone_vec = info["vec_w"]

        head_ndc = world_to_camera_view(scene, cam_obj, head_w)
        tail_ndc = world_to_camera_view(scene, cam_obj, tail_w)

        head_px = [head_ndc.x * res_x, (1.0 - head_ndc.y) * res_y]
        tail_px = [tail_ndc.x * res_x, (1.0 - tail_ndc.y) * res_y]

        parent_name = pb.parent.name if pb.parent else None
        parent_angles = None

        if pb.parent and pb.parent.name in temp_world:
            parent_vec = temp_world[pb.parent.name]["vec_w"]
            parent_angles = _angles_between_bones(parent_vec, bone_vec)

        bones_out[bone_name] = {
            "parent_name": parent_name,
            "head_world": [head_w.x, head_w.y, head_w.z],
            "tail_world": [tail_w.x, tail_w.y, tail_w.z],
            "bone_vector_world": [bone_vec.x, bone_vec.y, bone_vec.z],

            "head_ndc": [head_ndc.x, head_ndc.y, head_ndc.z],
            "tail_ndc": [tail_ndc.x, tail_ndc.y, tail_ndc.z],
            "head_px": head_px,
            "tail_px": tail_px,

            "in_front_head": (0.0 <= head_ndc.z <= 1.0),
            "in_front_tail": (0.0 <= tail_ndc.z <= 1.0),

            # parent ile açı bilgileri
            "angle_vs_parent_3d_deg": None if parent_angles is None else parent_angles["angle_3d_deg"],
            "angle_vs_parent_xy_deg": None if parent_angles is None else parent_angles["angle_xy_deg"],
            "angle_vs_parent_xz_deg": None if parent_angles is None else parent_angles["angle_xz_deg"],
            "angle_vs_parent_yz_deg": None if parent_angles is None else parent_angles["angle_yz_deg"],
        }

    return bones_out
import math
import json
import os
import bpy
from mathutils import Vector

# =========================================================
# ROTATION YARDIMCI
# =========================================================
def _rotation_payload_from_pose_bone(arm_eval, pb):
    """
    Pose bone için hem local hem world rotation bilgisini üretir.
    JSON'a uygun saf float listeleri döndürür.
    """

    # -----------------------------------------------------
    # 1) Local rotation (pose space / bone'un kendi rotasyonu)
    # -----------------------------------------------------
    rot_mode = pb.rotation_mode

    local_rot_euler_deg = None
    local_rot_quaternion = None
    local_rot_axis_angle = None

    try:
        if rot_mode == 'QUATERNION':
            q_local = pb.rotation_quaternion.copy()
            e_local = q_local.to_euler('XYZ')

            local_rot_quaternion = [
                float(q_local.w), float(q_local.x), float(q_local.y), float(q_local.z)
            ]
            local_rot_euler_deg = [
                float(math.degrees(e_local.x)),
                float(math.degrees(e_local.y)),
                float(math.degrees(e_local.z)),
            ]

        elif rot_mode == 'AXIS_ANGLE':
            aa = pb.rotation_axis_angle[:]   # [angle, ax, ay, az]
            local_rot_axis_angle = [float(v) for v in aa]

            angle_rad = aa[0]
            axis = Vector((aa[1], aa[2], aa[3]))
            if axis.length > 1e-12:
                axis.normalize()
                q_local = Vector(axis).to_track_quat('Z', 'Y')
                # axis-angle'i quaternion'a manuel çevirmek daha doğru:
                from mathutils import Quaternion
                q_local = Quaternion(axis, angle_rad)
                e_local = q_local.to_euler('XYZ')

                local_rot_quaternion = [
                    float(q_local.w), float(q_local.x), float(q_local.y), float(q_local.z)
                ]
                local_rot_euler_deg = [
                    float(math.degrees(e_local.x)),
                    float(math.degrees(e_local.y)),
                    float(math.degrees(e_local.z)),
                ]
        else:
            # XYZ / XZY / YXZ / YZX / ZXY / ZYX
            e_local = pb.rotation_euler.copy()
            local_rot_euler_deg = [
                float(math.degrees(e_local.x)),
                float(math.degrees(e_local.y)),
                float(math.degrees(e_local.z)),
            ]

            q_local = e_local.to_quaternion()
            local_rot_quaternion = [
                float(q_local.w), float(q_local.x), float(q_local.y), float(q_local.z)
            ]
    except Exception:
        pass

    # -----------------------------------------------------
    # 2) Pose matrix rotation (armature space)
    # -----------------------------------------------------
    pose_quat = None
    pose_euler_deg = None
    pose_matrix_4x4 = None

    try:
        q_pose = pb.matrix.to_quaternion()
        e_pose = q_pose.to_euler('XYZ')

        pose_quat = [
            float(q_pose.w), float(q_pose.x), float(q_pose.y), float(q_pose.z)
        ]
        pose_euler_deg = [
            float(math.degrees(e_pose.x)),
            float(math.degrees(e_pose.y)),
            float(math.degrees(e_pose.z)),
        ]
        pose_matrix_4x4 = [[float(v) for v in row] for row in pb.matrix]
    except Exception:
        pass

    # -----------------------------------------------------
    # 3) World rotation
    # -----------------------------------------------------
    world_quat = None
    world_euler_deg = None
    world_matrix_4x4 = None

    try:
        world_mat = arm_eval.matrix_world @ pb.matrix
        q_world = world_mat.to_quaternion()
        e_world = q_world.to_euler('XYZ')

        world_quat = [
            float(q_world.w), float(q_world.x), float(q_world.y), float(q_world.z)
        ]
        world_euler_deg = [
            float(math.degrees(e_world.x)),
            float(math.degrees(e_world.y)),
            float(math.degrees(e_world.z)),
        ]
        world_matrix_4x4 = [[float(v) for v in row] for row in world_mat]
    except Exception:
        pass

    return {
        "rotation_mode": rot_mode,

        "rotation_local_euler_deg": local_rot_euler_deg,
        "rotation_local_quaternion": local_rot_quaternion,
        "rotation_local_axis_angle": local_rot_axis_angle,

        "rotation_pose_euler_deg": pose_euler_deg,
        "rotation_pose_quaternion": pose_quat,
        "pose_matrix_4x4": pose_matrix_4x4,

        "rotation_world_euler_deg": world_euler_deg,
        "rotation_world_quaternion": world_quat,
        "world_matrix_4x4": world_matrix_4x4,
    }

def _world_screen_and_angles_of_bones(scene, cam_obj, arm_obj):
    depsgraph = bpy.context.evaluated_depsgraph_get()
    arm_eval = arm_obj.evaluated_get(depsgraph)

    res_x = scene.render.resolution_x
    res_y = scene.render.resolution_y

    bones_out = {}

    # Önce tüm kemiklerin dünya koordinatlarını toplayalım
    temp_world = {}
    for pb in arm_eval.pose.bones:
        head_w = arm_eval.matrix_world @ pb.head
        tail_w = arm_eval.matrix_world @ pb.tail

        temp_world[pb.name] = {
            "pose_bone": pb,
            "head_w": head_w,
            "tail_w": tail_w,
            "vec_w": tail_w - head_w
        }

    # Sonra ekran projeksiyonu + parent açıları + rotation
    for bone_name, info in temp_world.items():
        pb = info["pose_bone"]
        head_w = info["head_w"]
        tail_w = info["tail_w"]
        bone_vec = info["vec_w"]

        head_ndc = world_to_camera_view(scene, cam_obj, head_w)
        tail_ndc = world_to_camera_view(scene, cam_obj, tail_w)

        head_px = [head_ndc.x * res_x, (1.0 - head_ndc.y) * res_y]
        tail_px = [tail_ndc.x * res_x, (1.0 - tail_ndc.y) * res_y]

        parent_name = pb.parent.name if pb.parent else None
        parent_angles = None

        if pb.parent and pb.parent.name in temp_world:
            parent_vec = temp_world[pb.parent.name]["vec_w"]
            parent_angles = _angles_between_bones(parent_vec, bone_vec)

        rot_payload = _rotation_payload_from_pose_bone(arm_eval, pb)

        bones_out[bone_name] = {
            "parent_name": parent_name,

            "head_world": [float(head_w.x), float(head_w.y), float(head_w.z)],
            "tail_world": [float(tail_w.x), float(tail_w.y), float(tail_w.z)],
            "bone_vector_world": [float(bone_vec.x), float(bone_vec.y), float(bone_vec.z)],

            "head_ndc": [float(head_ndc.x), float(head_ndc.y), float(head_ndc.z)],
            "tail_ndc": [float(tail_ndc.x), float(tail_ndc.y), float(tail_ndc.z)],
            "head_px": [float(head_px[0]), float(head_px[1])],
            "tail_px": [float(tail_px[0]), float(tail_px[1])],

            "in_front_head": bool(0.0 <= head_ndc.z <= 1.0),
            "in_front_tail": bool(0.0 <= tail_ndc.z <= 1.0),

            "angle_vs_parent_3d_deg": None if parent_angles is None else parent_angles["angle_3d_deg"],
            "angle_vs_parent_xy_deg": None if parent_angles is None else parent_angles["angle_xy_deg"],
            "angle_vs_parent_xz_deg": None if parent_angles is None else parent_angles["angle_xz_deg"],
            "angle_vs_parent_yz_deg": None if parent_angles is None else parent_angles["angle_yz_deg"],

            # yeni eklenen rotation alanları
            **rot_payload
        }

    return bones_out
def save_bones_json_for_angle(json_path, angle_deg, image_name, cam_obj, arm_name=None):
    scene = bpy.context.scene
    arm_obj = _find_armature(arm_name=arm_name)
    if arm_obj is None:
        raise RuntimeError("ARMATURE bulunamadı. arm_name ver veya sahnede armature olduğundan emin ol.")

    bones = _world_screen_and_angles_of_bones(scene, cam_obj, arm_obj)

    payload = {
        "angle_deg": angle_deg,
        "image_file": image_name,
        "camera_name": cam_obj.name,
        "armature_name": arm_obj.name,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y],
        "bones": bones
    }

    out_dir = os.path.dirname(json_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return json_path