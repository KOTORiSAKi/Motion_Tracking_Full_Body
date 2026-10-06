import os
import time
import math
import threading
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.fusion import fuse_conv_bn_eval
from ultralytics import YOLO
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from scipy.spatial.transform import Rotation as R
from pythonosc import udp_client
from pythonosc.osc_bundle_builder import OscBundleBuilder, IMMEDIATELY
from pythonosc.osc_message_builder import OscMessageBuilder

# ==========================================
# 0. Settings
# ==========================================
OSC_HOST, OSC_PORT = "127.0.0.1", 39539
USE_OSC_BUNDLES = True
OSC_BUNDLE_CHUNK = 20

CAMERA_INDEX = 0
FRAME_W, FRAME_H = 640, 480
CAMERA_FPS = 30
TARGET_FPS = 30

BODY_IMGSZ = 320                
BODY_EVERY_N = 2                
HAND_EVERY_N = 2                # skip hand detection on alternate frames
BODY_LOST_FRAMES = 5 # 🌟 ถ้าไม่เจอคน 5 เฟรม ให้เข้าสู่โหมด T-Pose อัตโนมัติ

SHOW_PREVIEW = True             
PREVIEW_EVERY_N = 2             

HAND_CROP_SCALE = 1.50
X_SIGN = -1.0
Z_SIGN = 1.0
USE_DEPTH_HINT = True
CONF_MIN = 0.4
FINGER_GAIN = 1.15
FINGER_DEADZONE_DEG = 8.0
TWIST_SIGN = 1.0
THUMB_SIGN = 1.0
SWAP_MEDIAPIPE_LABELS = False

BODY_KP_FILTER = dict(min_cutoff=0.5, beta=0.01)
FACE_LM_FILTER = dict(min_cutoff=1.2, beta=0.02)
HAND_LM_FILTER = dict(min_cutoff=1.5, beta=0.03)
BONE_FILTER = dict(min_cutoff=1.0, beta=2.0)
HEAD_FILTER = dict(min_cutoff=1.5, beta=3.0)
FINGER_FILTER = dict(min_cutoff=2.0, beta=0.02)
MOUTH_FILTER = dict(min_cutoff=1.0, beta=2.0)
BLINK_FILTER = dict(min_cutoff=2.0, beta=6.0)

MAX_FLEX = (85.0, 100.0, 70.0)
MAX_FLEX_THUMB = (35.0, 50.0, 60.0)
FINGERS = {"Index": (5, 6, 7, 8), "Middle": (9, 10, 11, 12), "Ring": (13, 14, 15, 16), "Little": (17, 18, 19, 20)}
JOINTS = ("Proximal", "Intermediate", "Distal")
HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
                    (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
BODY_SKELETON = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
                 (5, 11), (6, 12), (11, 12),
                 (11, 13), (13, 15), (12, 14), (14, 16)]

FACE_3D_MODEL_POINTS = np.array([
    (0.0, 0.0, 0.0),             # 54: Nose
    (0.0, 330.0, 65.0),          # 16: Chin
    (-225.0, -170.0, 135.0),     # 60: Left Eye
    (225.0, -170.0, 135.0),      # 72: Right Eye
    (-150.0, 150.0, 125.0),      # 76: Left Mouth
    (150.0, 150.0, 125.0)        # 82: Right Mouth
], dtype=np.float64)
HEAD_PNP_IDX = [54, 16, 60, 72, 76, 82]

_tri, _names, _max = [], [], []
for _n, (_m, _p, _d, _t) in FINGERS.items():
    _chain = [0, _m, _p, _d, _t]
    for _j, _joint in enumerate(JOINTS):
        _tri.append((_chain[_j], _chain[_j + 1], _chain[_j + 2]))
        _names.append(f"{_n}{_joint}")
        _max.append(MAX_FLEX[_j])
_N_FINGER_JOINTS = len(_tri)
for _j, _joint in enumerate(JOINTS):
    _tri.append(([0, 1, 2, 3, 4][_j], [0, 1, 2, 3, 4][_j + 1], [0, 1, 2, 3, 4][_j + 2]))
    _names.append(f"Thumb{_joint}")
    _max.append(MAX_FLEX_THUMB[_j])
_TRI = np.array(_tri)
TRI_A, TRI_B, TRI_C = _TRI[:, 0], _TRI[:, 1], _TRI[:, 2]
FINGER_NAMES = _names
FINGER_MAX = np.array(_max, dtype=np.float64)

# ==========================================
# 1. Filters
# ==========================================
def _alpha(cutoff, dt):
    tau = 1.0 / (2.0 * np.pi * cutoff)
    return 1.0 / (1.0 + tau / dt)

class OneEuro:
    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0, max_gap=0.3):
        self.min_cutoff, self.beta, self.d_cutoff, self.max_gap = min_cutoff, beta, d_cutoff, max_gap
        self.x = self.dx = self.t = None

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=np.float64)
        if self.x is None or (t - self.t) > self.max_gap or x.shape != self.x.shape:
            self.x, self.dx, self.t = x.copy(), np.zeros_like(x), t
            return x.copy()
        dt = max(t - self.t, 1e-3)
        dx = (x - self.x) / dt
        a_d = _alpha(self.d_cutoff, dt)
        self.dx = a_d * dx + (1.0 - a_d) * self.dx
        a = _alpha(self.min_cutoff + self.beta * np.abs(self.dx), dt)
        self.x = a * x + (1.0 - a) * self.x
        self.t = t
        return self.x.copy()

class QuatOneEuro:
    def __init__(self, min_cutoff=1.0, beta=1.0, d_cutoff=1.0, max_gap=0.3):
        self.min_cutoff, self.beta, self.d_cutoff, self.max_gap = min_cutoff, beta, d_cutoff, max_gap
        self.q = self.t = None
        self.speed = 0.0

    def __call__(self, q, t):
        q = np.asarray(q, dtype=np.float64)
        n = np.linalg.norm(q)
        if n < 1e-8: return self.q.copy() if self.q is not None else np.array([0.0, 0.0, 0.0, 1.0])
        q = q / n
        if self.q is None or (t - self.t) > self.max_gap:
            self.q, self.t, self.speed = q.copy(), t, 0.0
            return q
        dot = float(np.dot(self.q, q))
        if dot < 0.0: q, dot = -q, -dot
        dt = max(t - self.t, 1e-3)
        angle = 2.0 * math.acos(min(1.0, dot))
        a_d = _alpha(self.d_cutoff, dt)
        self.speed = a_d * (angle / dt) + (1.0 - a_d) * self.speed
        a = _alpha(self.min_cutoff + self.beta * self.speed, dt)
        out = a * q + (1.0 - a) * self.q
        out /= np.linalg.norm(out)
        self.q, self.t = out, t
        return out.copy()

# ==========================================
# 2. OSC / VMC output 
# ==========================================
def _msg(address, args):
    b = OscMessageBuilder(address=address)
    for a in args: b.add_arg(a)
    return b.build()

class VMCSender:
    def __init__(self, host, port, bundle=True, chunk=12):
        self.client = udp_client.SimpleUDPClient(host, port)
        self.bundle, self.chunk = bundle, chunk
        self.msgs = []
        self.has_blend = False

    def bone(self, name, q):
        self.msgs.append(_msg("/VMC/Ext/Bone/Pos", [name, 0.0, 0.0, 0.0, float(q[0]), float(q[1]), float(q[2]), float(q[3])]))
        
    def root_pos(self, name, x, y, z):
        self.msgs.append(_msg("/VMC/Ext/Root/Pos", [name, float(x), float(y), float(z), 0.0, 0.0, 0.0, 1.0]))

    def blend(self, name, value):
        self.msgs.append(_msg("/VMC/Ext/Blend/Val", [name, float(value)]))
        self.has_blend = True

    def flush(self):
        if self.has_blend: self.msgs.append(_msg("/VMC/Ext/Blend/Apply", []))
        msgs, self.msgs, self.has_blend = self.msgs, [], False
        if not msgs: return
        if not self.bundle:
            for m in msgs: self.client.send(m)
            return
        for i in range(0, len(msgs), self.chunk):
            b = OscBundleBuilder(IMMEDIATELY)
            for m in msgs[i:i + self.chunk]: b.add_content(m)
            self.client.send(b.build())

# ==========================================
# 3. Camera Thread
# ==========================================
class CameraStream:
    def __init__(self, index, width, height, fps):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.frame, self.t = None, 0.0
        self.failed = not self.cap.isOpened()
        self.running = True
        self.lock = threading.Lock()
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.failed = True
                self.event.set()
                break
            with self.lock:
                self.frame, self.t = frame, time.perf_counter()
            self.event.set()

    def read(self, timeout=1.0):
        if not self.event.wait(timeout): return None
        self.event.clear()
        with self.lock: return (self.frame, self.t) if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

# ==========================================
# 4. Utilities
# ==========================================
def get_square_crop(frame, cx, cy, size):
    h, w = frame.shape[:2]
    half = int(size // 2)
    x1, y1, x2, y2 = cx - half, cy - half, cx + half, cy + half
    crop = frame[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    pt, pb, pl, pr = max(0, -y1), max(0, y2 - h), max(0, -x1), max(0, x2 - w)
    if crop.size and (pt or pb or pl or pr):
        crop = cv2.copyMakeBorder(crop, pt, pb, pl, pr, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return crop, x1, y1

def to_input(crops_u8, device, dtype):
    arr = np.stack(crops_u8)
    t = torch.from_numpy(arr).to(device)
    t = t.permute(0, 3, 1, 2).float().mul_(1.0 / 127.5).sub_(1.0)
    return t.to(dtype)

# ==========================================
# 5. PyTorch Models
# ==========================================
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = round(inp * expand_ratio)
        layers = []
        if expand_ratio != 1: layers.extend([nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True)])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x): return x + self.conv(x) if self.use_res_connect else self.conv(x)

class LandmarkModel(nn.Module):
    def __init__(self, c, num_points):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(c, c, 3, 1, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True))
        self.block3 = nn.Sequential(InvertedResidual(c, c, 2, 2), *[InvertedResidual(c, c, 1, 2) for _ in range(4)])
        self.block4 = InvertedResidual(c, c * 2, 2, 2)
        self.block5 = nn.Sequential(*[InvertedResidual(c * 2, c * 2, 1, 4) for _ in range(6)])
        self.out1_branch = InvertedResidual(c * 2, c // 4, 1, 2)
        self.out2_branch = nn.Sequential(nn.Conv2d(c // 4, c // 2, 3, 2, 1, bias=False), nn.BatchNorm2d(c // 2), nn.ReLU(True))
        self.out3_branch = nn.Sequential(nn.Conv2d(c // 2, c * 2, 7, 1, 0, bias=False), nn.BatchNorm2d(c * 2), nn.ReLU(True))
        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(c // 4 + c // 2 + c * 2, num_points * 2)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.block5(self.block4(self.block3(x)))
        out1 = self.out1_branch(x)
        out2 = self.out2_branch(out1)
        out3 = self.out3_branch(out2)
        n = x.size(0)
        feat = torch.cat([self.avg_pool1(out1).reshape(n, -1), self.avg_pool2(out2).reshape(n, -1), out3.reshape(n, -1)], 1)
        return torch.sigmoid(self.fc(feat))

def FaceLandmarkModel(): return LandmarkModel(64, 98)
def HandLandmarkModel(): return LandmarkModel(128, 21)

def fuse_conv_bn(module):
    for child in module.children(): fuse_conv_bn(child)
    if isinstance(module, nn.Sequential):
        keys = list(module._modules.keys())
        i = 0
        while i < len(keys) - 1:
            a, b = module._modules[keys[i]], module._modules[keys[i + 1]]
            if isinstance(a, nn.Conv2d) and isinstance(b, nn.BatchNorm2d):
                module._modules[keys[i]] = fuse_conv_bn_eval(a, b)
                module._modules[keys[i + 1]] = nn.Identity()
                i += 2
            else: i += 1

def load_model(model, path, device, fp16):
    try: sd = torch.load(path, map_location="cpu", weights_only=True)
    except Exception: sd = torch.load(path, map_location="cpu")
    for key in ("state_dict", "model_state_dict"):
        if isinstance(sd, dict) and key in sd:
            sd = sd[key]; break
    clean = {k.replace("module.", "").replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(clean, strict=False)
    model.eval()
    fuse_conv_bn(model)
    model = model.to(device).to(memory_format=torch.channels_last)
    return model.half() if fp16 else model

def warmup(model, device, dtype, batches=(1, 2)):
    for n in batches:
        x = torch.zeros(n, 3, 112, 112, device=device, dtype=dtype).contiguous(memory_format=torch.channels_last)
        model(x)
    if device.type == "cuda": torch.cuda.synchronize()

# ==========================================
# 6. Body Remapper (T-Pose & Fallback Logic)
# ==========================================
def arc(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9: return R.identity()
    a, b = a / na, b / nb
    d = float(np.dot(a, b))
    if d > 0.999999: return R.identity()
    if d < -0.999999:
        axis = np.cross(a, [0.0, 0.0, 1.0])
        if np.linalg.norm(axis) < 1e-6: axis = np.cross(a, [0.0, 1.0, 0.0])
        return R.from_rotvec((axis / np.linalg.norm(axis)) * math.pi)
    axis = np.cross(a, b)
    return R.from_rotvec((axis / np.linalg.norm(axis)) * math.acos(max(-1.0, min(1.0, d))))

def flex_all(pts):
    v1, v2 = pts[TRI_B] - pts[TRI_A], pts[TRI_C] - pts[TRI_B]
    n = np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1)
    cos = np.einsum("ij,ij->i", v1, v2) / np.maximum(n, 1e-9)
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    return np.where(n < 1e-9, 0.0, ang)

class Remapper:
    def __init__(self, sender):
        self.sender = sender
        self.now = 0.0
        self.bone_filters = {}
        self.finger_filters = {"Left": OneEuro(**FINGER_FILTER), "Right": OneEuro(**FINGER_FILTER)}
        self.kp_filter = OneEuro(**BODY_KP_FILTER)
        self.was_valid = None
        self.max_len = {}
        self.lower_world = {"Left": None, "Right": None}
        self.kp = self.conf = None
        
        self.rest_dirs = {
            "Spine": np.array([0.0, 1.0, 0.0]),
            "LeftU": np.array([X_SIGN, 0.0, 0.0]), "LeftL": np.array([X_SIGN, 0.0, 0.0]),
            "RightU": np.array([-X_SIGN, 0.0, 0.0]), "RightL": np.array([-X_SIGN, 0.0, 0.0]),
            "LeftUL": np.array([0.0, -1.0, 0.0]), "LeftLL": np.array([0.0, -1.0, 0.0]),
            "RightUL": np.array([0.0, -1.0, 0.0]), "RightLL": np.array([0.0, -1.0, 0.0])
        }
        self.is_calibrating = False
        self.calib_frames = 0
        self.calib_accum = {k: np.zeros(3) for k in self.rest_dirs}

    def start_calibration(self):
        self.is_calibrating = True
        self.calib_frames = 0
        self.calib_accum = {k: np.zeros(3) for k in self.rest_dirs}
        print("🔄 [CALIBRATION] เริ่มเก็บค่า T-Pose กรุณายืนตัวตรง กางแขนขนานพื้น...")

    def on_body_lost(self):
        self.kp = self.conf = None
        self.lower_world = {"Left": None, "Right": None}
        self.kp_filter.reset()
        self.was_valid = None

    def _send(self, name, rot):
        f = self.bone_filters.get(name)
        if f is None: f = self.bone_filters[name] = QuatOneEuro(**BONE_FILTER)
        self.sender.bone(name, f(rot.as_quat(), self.now))

    def _dir(self, key, p1, p2):
        dx, dy = p2[0] - p1[0], p2[1] - p1[1]
        length = math.hypot(dx, dy)
        z = 0.0
        if USE_DEPTH_HINT and length > 1e-6:
            m = max(self.max_len.get(key, 0.0) * 0.9995, length)
            self.max_len[key] = m
            z = math.sqrt(max(m * m - length * length, 0.0)) * Z_SIGN
        v = np.array([X_SIGN * dx, -dy, z])
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else np.zeros(3)

    def _filter_keypoints(self, raw_kp, conf):
        valid = conf >= CONF_MIN
        flt = self.kp_filter
        if flt.x is not None and self.was_valid is not None and flt.x.shape == raw_kp.shape:
            snap = valid & ~self.was_valid
            if snap.any():
                flt.x[snap], flt.dx[snap] = raw_kp[snap], 0.0
            raw_kp = np.where(valid[:, None], raw_kp, flt.x)
        self.was_valid = valid
        return flt(raw_kp, self.now)

    def update_body(self, raw_kp, conf):
        kp = self._filter_keypoints(np.asarray(raw_kp, dtype=np.float64), conf)
        self.kp, self.conf = kp, conf
        self.lower_world = {"Left": None, "Right": None}
        arms = {"Left": (5, 7, 9), "Right": (6, 8, 10)}

        # 🌟 1. Fallback ลำตัว: ถ้าไม่เห็นสะโพก/ไหล่ ให้ตัวตรง
        valid_spine = conf[5] >= CONF_MIN and conf[6] >= CONF_MIN and conf[11] >= CONF_MIN and conf[12] >= CONF_MIN
        if valid_spine:
            hips = (kp[11] + kp[12]) / 2.0
            neck = (kp[5] + kp[6]) / 2.0
            spine_dir = self._dir("Spine", hips, neck)
            root_x = (hips[0] - (FRAME_W / 2)) / (FRAME_W * 0.5)
            root_y = (FRAME_H - hips[1]) / (FRAME_H * 0.5) 
            self.sender.root_pos("Root", root_x, root_y, 0.0)
        else:
            spine_dir = self.rest_dirs["Spine"]

        if self.is_calibrating:
            all_ok = True
            if valid_spine: self.calib_accum["Spine"] += spine_dir
            for side, (s, e, w) in arms.items():
                if all(conf[i] >= CONF_MIN for i in (s, e, w)):
                    self.calib_accum[f"{side}U"] += self._dir(f"{side}U", kp[s], kp[e])
                    self.calib_accum[f"{side}L"] += self._dir(f"{side}L", kp[e], kp[w])
                else: all_ok = False
            for side, (h, k, a) in (("Left", (11, 13, 15)), ("Right", (12, 14, 16))):
                if all(conf[i] >= CONF_MIN for i in (h, k, a)):
                    self.calib_accum[f"{side}UL"] += self._dir(f"{side}UL", kp[h], kp[k])
                    self.calib_accum[f"{side}LL"] += self._dir(f"{side}LL", kp[k], kp[a])
                else: all_ok = False
            
            if all_ok: self.calib_frames += 1
            if self.calib_frames >= 20:
                for k in self.rest_dirs:
                    n = np.linalg.norm(self.calib_accum[k])
                    if n > 0: self.rest_dirs[k] = self.calib_accum[k] / n
                self.is_calibrating = False
                print("✅ [CALIBRATION] ตั้งค่า T-Pose สมบูรณ์ ขยับตัวได้เลย!")
            return

        spine_rot = arc(self.rest_dirs["Spine"], spine_dir)
        self._send("Spine", spine_rot)

        # 🌟 2. Fallback แขน: ถ้าหลุด ให้กางแขนเป็น T-Pose สมูทๆ
        for side, (s, e, w) in arms.items():
            valid_U = conf[s] >= CONF_MIN and conf[e] >= CONF_MIN
            dir_U = self._dir(f"{side}U", kp[s], kp[e]) if valid_U else self.rest_dirs[f"{side}U"]
            up = arc(self.rest_dirs[f"{side}U"], dir_U)
            self._send(f"{side}UpperArm", up)
            
            valid_L = valid_U and conf[w] >= CONF_MIN
            dir_L = self._dir(f"{side}L", kp[e], kp[w]) if valid_L else self.rest_dirs[f"{side}L"]
            low = arc(self.rest_dirs[f"{side}L"], dir_L)
            self._send(f"{side}LowerArm", up.inv() * low)
            self.lower_world[side] = low

        # 🌟 Legs: ถ้าหลุด ให้ขาตรง
        for side, (h, k, a) in (("Left", (11, 13, 15)), ("Right", (12, 14, 16))):
            valid_U = conf[h] >= CONF_MIN and conf[k] >= CONF_MIN
            dir_U = self._dir(f"{side}UL", kp[h], kp[k]) if valid_U else self.rest_dirs[f"{side}UL"]
            up = arc(self.rest_dirs[f"{side}UL"], dir_U)
            self._send(f"{side}UpperLeg", up)

            valid_L = valid_U and conf[a] >= CONF_MIN
            dir_L = self._dir(f"{side}LL", kp[k], kp[a]) if valid_L else self.rest_dirs[f"{side}LL"]
            low = arc(self.rest_dirs[f"{side}LL"], dir_L)
            self._send(f"{side}LowerLeg", up.inv() * low)

    def hand_side(self, mp_label, wrist_xy):
        if self.kp is not None and self.conf[9] >= CONF_MIN and self.conf[10] >= CONF_MIN:
            dl = math.hypot(wrist_xy[0] - self.kp[9][0], wrist_xy[1] - self.kp[9][1])
            dr = math.hypot(wrist_xy[0] - self.kp[10][0], wrist_xy[1] - self.kp[10][1])
            return "Left" if dl < dr else "Right"
        # Fallback if body wrists aren't visible
        if SWAP_MEDIAPIPE_LABELS:
            return "Right" if mp_label == "Left" else "Left"
        return mp_label

    def send_hand(self, side, pts):
        pts = np.asarray(pts, dtype=float)
        is_left = (side == "Left")
        rest = np.array([X_SIGN if is_left else -X_SIGN, 0.0, 0.0])
        wrist = pts[0]
        low = self.lower_world.get(side)
        if low is not None:
            # --- Bone Direction (Local X) ---
            # d is the world direction of the fingers
            centroid = pts[[5, 9, 13, 17]].mean(axis=0)
            d = np.array([X_SIGN * (centroid[0] - wrist[0]), -(centroid[1] - wrist[1]), -(centroid[2] - wrist[2])])
            dn = np.linalg.norm(d)
            if dn > 1e-6: d /= dn
            # VRM Left Hand bone points to -X, Right Hand bone points to +X
            x_axis = -d if is_left else d
            
            # --- Palm Normal (Local Y) ---
            # Calculate cross product in world space to find the palm's normal vector
            w_idx = np.array([X_SIGN * (pts[5][0] - wrist[0]), -(pts[5][1] - wrist[1]), -(pts[5][2] - wrist[2])])
            w_pin = np.array([X_SIGN * (pts[17][0] - wrist[0]), -(pts[17][1] - wrist[1]), -(pts[17][2] - wrist[2])])
            y_axis = np.cross(w_idx, w_pin)
            # For Left Hand, cross(idx, pin) points to Palm (-Y). For Right Hand, it points to Back (+Y).
            # We want y_axis to be the Back of the hand (Local +Y).
            if is_left: y_axis = -y_axis
            yn = np.linalg.norm(y_axis)
            if yn > 1e-6: y_axis /= yn
            y_axis *= TWIST_SIGN
            
            # --- Thumb Direction (Local Z) ---
            z_axis = np.cross(x_axis, y_axis)
            y_axis = np.cross(z_axis, x_axis) # re-orthogonalize just in case
            
            # Build the World Rotation matrix and apply it relative to the lower arm
            M = np.column_stack((x_axis, y_axis, z_axis))
            try:
                swing = R.from_matrix(M)
                self._send(f"{side}Hand", low.inv() * swing)
            except ValueError:
                pass # fallback to identity if landmarks collapse

        ang = self.finger_filters[side](flex_all(pts), self.now)
        bend = np.clip((ang - FINGER_DEADZONE_DEG) * FINGER_GAIN, 0.0, FINGER_MAX)
        rx = float(rest[0])
        half = np.radians(bend) * 0.5
        sin_h, cos_h = np.sin(half), np.cos(half)
        for i, name in enumerate(FINGER_NAMES):
            if i < _N_FINGER_JOINTS: q = (0.0, 0.0, -rx * sin_h[i], cos_h[i])
            else: q = (0.0, THUMB_SIGN * rx * sin_h[i], 0.0, cos_h[i])
            self.sender.bone(f"{side}{name}", q)

    # 🌟 3. Fallback นิ้วมือ: ถ้านิ้วหลุด ให้เหยียดตรง ข้อมือตรง
    def send_neutral_hand(self, side, t):
        self._send(f"{side}Hand", R.identity())
        
        zero_flex = np.zeros(15, dtype=np.float64)
        ang = self.finger_filters[side](zero_flex, t)
        bend = np.clip((ang - FINGER_DEADZONE_DEG) * FINGER_GAIN, 0.0, FINGER_MAX)
        
        rest = np.array([X_SIGN if side == "Left" else -X_SIGN, 0.0, 0.0])
        rx = float(rest[0])
        half = np.radians(bend) * 0.5
        sin_h, cos_h = np.sin(half), np.cos(half)
        for i, name in enumerate(FINGER_NAMES):
            if i < _N_FINGER_JOINTS: q = (0.0, 0.0, -rx * sin_h[i], cos_h[i])
            else: q = (0.0, THUMB_SIGN * rx * sin_h[i], 0.0, cos_h[i])
            self.sender.bone(f"{side}{name}", q)

# ==========================================
# 7. Face Tracker
# ==========================================
class FaceTracker:
    def __init__(self, model, device, dtype, sender, img_w, img_h):
        self.model, self.device, self.dtype, self.sender = model, device, dtype, sender
        self.lm_filter = OneEuro(**FACE_LM_FILTER)
        self.head_filter = QuatOneEuro(**HEAD_FILTER)
        self.f_mouth = OneEuro(**MOUTH_FILTER)
        self.f_blink_l = OneEuro(**BLINK_FILTER)
        self.f_blink_r = OneEuro(**BLINK_FILTER)
        focal = float(max(img_w, img_h))
        self.cam = np.array([[focal, 0, img_w / 2.0], [0, focal, img_h / 2.0], [0, 0, 1.0]], dtype=np.float64)
        self.dist = np.zeros((4, 1), dtype=np.float64)
        self.rvec = self.tvec = None
        self.have_pose = False
        self.prev_face_size = 0
        self.pending = None
        self.last_seen = 0.0 # 🌟 เวลาล่าสุดที่เจอหน้า

    def launch(self, frame, kp, conf):
        self.pending = None
        v = conf >= CONF_MIN
        if not v[0]: return
        if v[3] and v[4]: face_width = np.linalg.norm(kp[3] - kp[4])
        elif v[1] and v[2]: face_width = np.linalg.norm(kp[1] - kp[2]) * 2.5
        else: face_width = self.prev_face_size / 1.5 if self.prev_face_size > 0 else 150
        
        face_size = int(face_width * 1.5)
        if self.prev_face_size > 0: face_size = int(0.7 * face_size + 0.3 * self.prev_face_size)
        self.prev_face_size = face_size
        if face_size <= 50: return
        
        crop, fx, fy = get_square_crop(frame, int(kp[0][0]), int(kp[0][1]), face_size)
        if crop.size == 0: return
        x = to_input([cv2.resize(crop, (112, 112))], self.device, self.dtype)
        self.pending = (self.model(x), fx, fy, face_size)

    def collect(self, t, preview=None):
        if self.pending is None: 
            # 🌟 4. Fallback หน้า: ถ้าไม่เจอหน้าเกิน 0.5 วิ ให้หันหน้าตรง ตาปกติ ปากหุบ
            if t - self.last_seen > 0.5:
                self.sender.bone("Head", self.head_filter([0.0, 0.0, 0.0, 1.0], t))
                self.sender.blend("A", float(self.f_mouth(0.0, t)))
                self.sender.blend("Blink_R", float(self.f_blink_l(0.0, t)))
                self.sender.blend("Blink_L", float(self.f_blink_r(0.0, t)))
            return
            
        self.last_seen = t
        pred, fx, fy, size = self.pending
        self.pending = None
        lm = pred.float().cpu().numpy()[0].reshape(98, 2)
        lm = np.clip(lm, 0.0, 1.0) * size + np.array([fx, fy], dtype=np.float64)
        lm = self.lm_filter(lm, t)
        
        if preview is not None:
            for pt in lm:
                cv2.circle(preview, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
                
        self._head_pose(lm, t)
        self._blendshapes(lm, t)

    def _head_pose(self, lm, t):
        from_guess = self.have_pose
        ok, rvec, tvec = cv2.solvePnP(
            FACE_3D_MODEL_POINTS, lm[HEAD_PNP_IDX], self.cam, self.dist,
            rvec=self.rvec.copy() if from_guess else None,
            tvec=self.tvec.copy() if from_guess else None,
            useExtrinsicGuess=from_guess, flags=cv2.SOLVEPNP_ITERATIVE)
        if not ok:
            self.have_pose = False
            return
        self.rvec, self.tvec, self.have_pose = rvec, tvec, True
        pitch, yaw, roll = R.from_rotvec(rvec.reshape(3,)).as_euler("xyz", degrees=True)
        q = R.from_euler("xyz", [-pitch, yaw, roll], degrees=True).as_quat()
        self.sender.bone("Head", self.head_filter(q, t))

    def _blendshapes(self, lm, t):
        def d(i, j): return float(np.linalg.norm(lm[i] - lm[j]))
        mouth_w, mouth_h = d(76, 82), d(90, 94)
        left_w = d(60, 64)
        left_h = (d(61, 67) + d(62, 66) + d(63, 65)) / 3.0
        right_w = d(68, 72)
        right_h = (d(69, 75) + d(70, 74) + d(71, 73)) / 3.0
        ear_l, ear_r = left_h / (left_w + 1e-6), right_h / (right_w + 1e-6)

        raw_mouth = min(1.0, max(0.0, ((mouth_h / (mouth_w + 1e-6)) - 0.10) / 0.40))
        blink_a = min(1.0, max(0.0, (0.325 - ear_l) / 0.15))
        blink_b = min(1.0, max(0.0, (0.325 - ear_r) / 0.15))
        self.sender.blend("A", float(self.f_mouth(raw_mouth, t)))
        self.sender.blend("Blink_R", float(self.f_blink_l(blink_a, t)))
        self.sender.blend("Blink_L", float(self.f_blink_r(blink_b, t)))

# ==========================================
# 8. Hand Tracker
# ==========================================
def draw_body(frame, kp, conf):
    for a, b in BODY_SKELETON:
        if conf[a] >= CONF_MIN and conf[b] >= CONF_MIN:
            cv2.line(frame, (int(kp[a][0]), int(kp[a][1])), (int(kp[b][0]), int(kp[b][1])), (0, 255, 0), 2)
    for i in range(len(kp)):
        if conf[i] >= CONF_MIN:
            cv2.circle(frame, (int(kp[i][0]), int(kp[i][1])), 4, (0, 0, 255), -1)

def draw_hand(frame, pts, color=(0, 255, 0)):
    p = [tuple(xy[:2]) for xy in pts.astype(int).tolist()]
    for a, b in HAND_CONNECTIONS: cv2.line(frame, p[a], p[b], (255, 0, 0), 2)
    for xy in p: cv2.circle(frame, xy, 4, color, -1)

class HandTracker:
    def __init__(self, detector, mapper):
        self.detector = detector
        self.mapper = mapper
        self.pt_filters = {"Left": OneEuro(**HAND_LM_FILTER), "Right": OneEuro(**HAND_LM_FILTER)}
        self.last_ts = -1
        self.last_seen = {"Left": 0.0, "Right": 0.0}  # 🌟 เวลาล่าสุดที่เจอมือแต่ละข้าง

    def run(self, img_rgb, t, preview=None):
        ts = int(t * 1000)
        if ts <= self.last_ts: ts = self.last_ts + 1
        self.last_ts = ts
        res = self.detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb), ts)
        
        sent_sides = set()
        if res.hand_landmarks:
            h, w = img_rgb.shape[:2]
            for i, lms in enumerate(res.hand_landmarks):
                pts = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in lms], dtype=np.float64)
                try: label = res.handedness[i][0].category_name
                except (IndexError, AttributeError):
                    label = "Unknown"
                
                side = self.mapper.hand_side(label, pts[0])
                if side in sent_sides: continue
                
                sent_sides.add(side)
                self.last_seen[side] = t
                pts = self.pt_filters[side](pts, t)
                self.mapper.send_hand(side, pts)
                if preview is not None: draw_hand(preview, pts)

        if "Left" not in sent_sides and t - self.last_seen["Left"] > 0.5:
            self.mapper.send_neutral_hand("Left", t)
        if "Right" not in sent_sides and t - self.last_seen["Right"] > 0.5:
            self.mapper.send_neutral_hand("Right", t)


@torch.inference_mode()
def main():
    print("="*50)
    # --- เช็คและประกาศการใช้งาน CUDA ---
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_name = torch.cuda.get_device_name(0)
        is_fp16 = True
        print(f"✅ [CUDA ENABLED] ตรวจพบการรองรับ CUDA!")
        print(f"🚀 กำลังรันระบบด้วย GPU: {device_name}")
    else:
        device = torch.device("cpu")
        is_fp16 = False
        print(f"⚠️ [CUDA DISABLED] ไม่พบ CUDA หรือ GPU ไม่รองรับ")
        print(f"🐢 กำลังรันระบบด้วย CPU (อาจทำงานช้ากว่าปกติ)")

    dtype = torch.float16 if is_fp16 else torch.float32
    print(f"⚙️ โหมดประมวลผล: {device} | ใช้ FP16: {is_fp16}")
    print("="*50)
    
    if is_fp16: torch.backends.cudnn.benchmark = True
    else: torch.set_num_threads(max(1, min(4, (os.cpu_count() or 2) // 2)))
    cv2.setNumThreads(2)

    base = os.path.dirname(os.path.abspath(__file__))
    face_model = load_model(FaceLandmarkModel(), os.path.join(base, "best_face_model.pth"), device, is_fp16)
    warmup(face_model, device, dtype, batches=(1,))
    
    body_model = YOLO(os.path.join(base, "best.pt"))
    body_model.to(device) # บังคับให้ YOLO โหลดลง Device (CUDA/CPU) ที่ตรวจพบอย่างชัดเจน

    hand_task = os.path.join(base, "hand_landmarker.task")
    if not os.path.exists(hand_task):
        urllib.request.urlretrieve("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task", hand_task)
    hand_detector = mp_vision.HandLandmarker.create_from_options(mp_vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=hand_task), num_hands=2,
        min_hand_detection_confidence=0.5, min_hand_presence_confidence=0.5, min_tracking_confidence=0.5,
        running_mode=mp_vision.RunningMode.VIDEO))

    sender = VMCSender(OSC_HOST, OSC_PORT, USE_OSC_BUNDLES, OSC_BUNDLE_CHUNK)
    mapper = Remapper(sender)
    face = FaceTracker(face_model, device, dtype, sender, FRAME_W, FRAME_H)
    hands = HandTracker(hand_detector, mapper)

    cam = CameraStream(CAMERA_INDEX, FRAME_W, FRAME_H, CAMERA_FPS)
    if cam.failed:
        print("❌ เปิดกล้องไม่ได้")
        return

    print("✅ System Ready! กด C เพื่อตั้ง T-Pose | กด V เพื่อซ่อน/โชว์ภาพ")
    show_preview = SHOW_PREVIEW
    min_dt = 1.0 / TARGET_FPS
    last_t, frame_idx, body_miss = 0.0, 0, 0
    fps_t0, fps_count, shown_fps = time.perf_counter(), 0, 0.0
    status_img = np.zeros((60, 320, 3), dtype=np.uint8)
    _zero_kp = np.zeros((17, 2), dtype=np.float64)
    _zero_conf = np.zeros(17, dtype=np.float64)

    try:
        while True:
            item = cam.read()
            if item is None:
                if cam.failed: break
                continue
            frame, t = item
            if t - last_t < min_dt - 0.002: continue
            last_t = t
            mapper.now = t
            frame = cv2.flip(frame, 1)
            frame_idx += 1

            # --- Body (YOLO) ---
            if (frame_idx - 1) % BODY_EVERY_N == 0:
                r = body_model.track(frame, imgsz=BODY_IMGSZ, device=device.type, verbose=False, max_det=1, persist=True)[0]
                data = r.keypoints.data if r.keypoints is not None else None
                if data is not None and len(data) > 0:
                    d = data[0].cpu().numpy()
                    conf = d[:, 2] if d.shape[1] > 2 else np.ones(len(d))
                    mapper.update_body(d[:, :2], conf)
                    body_miss = 0
                else:
                    body_miss += 1
                    if body_miss > BODY_LOST_FRAMES:
                        # 🌟 5. Fallback ลำตัวทั้งหมด: ส่งข้อมูลความมั่นใจ = 0 เข้าไป เพื่อให้ร่างกายสมูทกลับไปเป็น T-Pose
                        mapper.update_body(_zero_kp, _zero_conf)

            # --- Face ---
            if mapper.kp is not None: face.launch(frame, mapper.kp, mapper.conf)
            else: face.pending = None

            draw_now = show_preview and frame_idx % PREVIEW_EVERY_N == 0

            # --- Draw body skeleton on preview ---
            if draw_now and mapper.kp is not None:
                draw_body(frame, mapper.kp, mapper.conf)

            # --- Hands (skip on alternate frames for perf) ---
            if (frame_idx - 1) % HAND_EVERY_N == 0:
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                hands.run(img_rgb, t, frame if draw_now else None)

            face.collect(t, frame if draw_now else None)
            sender.flush()

            # --- FPS / UI ---
            fps_count += 1
            now = time.perf_counter()
            if now - fps_t0 >= 0.5:
                shown_fps = fps_count / (now - fps_t0)
                fps_count, fps_t0 = 0, now

            if show_preview:
                if draw_now:
                    cv2.putText(frame, f"FPS: {shown_fps:.1f}", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    if mapper.is_calibrating:
                        cv2.putText(frame, "CALIBRATING T-POSE...", (FRAME_W // 2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 3)
                    cv2.imshow("Warudo Tracker", frame)
            elif frame_idx % 15 == 0:
                status_img[:] = 0
                cv2.putText(status_img, f"FPS {shown_fps:.1f}  (v = preview)", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.imshow("Warudo Tracker", status_img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"): break
            elif key == ord("c"): mapper.start_calibration()
            elif key == ord("v"):
                show_preview = not show_preview
                cv2.destroyAllWindows()
    finally:
        cam.stop()
        hand_detector.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()