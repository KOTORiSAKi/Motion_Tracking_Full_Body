import cv2
import numpy as np
import time
import math
import torch
import torch.nn as nn
from ultralytics import YOLO
import mediapipe as mp

# ==========================================
# ⚙️ 1. ตั้งค่า Path สำหรับไฟล์ .pth และ .pt
# ==========================================
BODY_MODEL_PATH = "best.pt" 
FACE_MODEL_PATH = "best_face_model.pth"
HAND_MODEL_PATH = "best_hand_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 กำลังรันโมเดลบนอุปกรณ์: {device}")

# ==========================================
# 🏗️ 2. วางโครงสร้าง Neural Network ให้ PyTorch รู้จัก
# ==========================================
# (หากโหลดไฟล์ .pth จำเป็นต้องมี Class โครงสร้างเหล่านี้ในไฟล์ใช้งาน)

class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super().__init__()
        self.use_res_connect = stride == 1 and inp == oup
        hidden_dim = round(inp * expand_ratio)
        layers = []
        if expand_ratio != 1:
            layers.extend([nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False), nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True)])
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim), nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False), nn.BatchNorm2d(oup)
        ])
        self.conv = nn.Sequential(*layers)
    def forward(self, x): return x + self.conv(x) if self.use_res_connect else self.conv(x)

class FaceLandmarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.block3 = nn.Sequential(InvertedResidual(64, 64, 2, 2), InvertedResidual(64, 64, 1, 2), InvertedResidual(64, 64, 1, 2), InvertedResidual(64, 64, 1, 2), InvertedResidual(64, 64, 1, 2))
        self.block4 = InvertedResidual(64, 128, 2, 2)
        self.block5 = nn.Sequential(InvertedResidual(128, 128, 1, 4), InvertedResidual(128, 128, 1, 4), InvertedResidual(128, 128, 1, 4), InvertedResidual(128, 128, 1, 4), InvertedResidual(128, 128, 1, 4), InvertedResidual(128, 128, 1, 4))
        self.out1_branch = InvertedResidual(128, 16, 1, 2)
        self.out2_branch = nn.Sequential(nn.Conv2d(16, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(True))
        self.out3_branch = nn.Sequential(nn.Conv2d(32, 128, 7, 1, 0, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(16 + 32 + 128, 98 * 2)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.block5(self.block4(self.block3(x)))
        out1 = self.out1_branch(x)
        out2 = self.out2_branch(out1)
        out3 = self.out3_branch(out2)
        out1 = self.avg_pool1(out1).view(out1.size(0), -1)
        out2 = self.avg_pool2(out2).view(out2.size(0), -1)
        out3 = out3.view(out3.size(0), -1)
        multi_scale = torch.cat([out1, out2, out3], 1)
        return torch.sigmoid(self.fc(multi_scale))

class HandLandmarkModel(nn.Module):
    def __init__(self):
        super().__init__()
        # 🌟 อัปเกรด: ขยายช่องสัญญาณจาก 64 -> 128
        self.conv1 = nn.Sequential(nn.Conv2d(3, 128, 3, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True))
        self.block3 = nn.Sequential(InvertedResidual(128, 128, 2, 2), InvertedResidual(128, 128, 1, 2), InvertedResidual(128, 128, 1, 2), InvertedResidual(128, 128, 1, 2), InvertedResidual(128, 128, 1, 2))

        # 🌟 อัปเกรด: ขยายช่องสัญญาณจาก 128 -> 256
        self.block4 = InvertedResidual(128, 256, 2, 2)
        self.block5 = nn.Sequential(InvertedResidual(256, 256, 1, 4), InvertedResidual(256, 256, 1, 4), InvertedResidual(256, 256, 1, 4), InvertedResidual(256, 256, 1, 4), InvertedResidual(256, 256, 1, 4), InvertedResidual(256, 256, 1, 4))

        self.out1_branch = InvertedResidual(256, 32, 1, 2)
        self.out2_branch = nn.Sequential(nn.Conv2d(32, 64, 3, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True))
        self.out3_branch = nn.Sequential(nn.Conv2d(64, 256, 7, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True))

        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)

        self.fc = nn.Linear(32 + 64 + 256, 21 * 2)

    def forward(self, x):
        x = self.conv2(self.conv1(x))
        x = self.block5(self.block4(self.block3(x)))
        out1 = self.out1_branch(x)
        out2 = self.out2_branch(out1)
        out3 = self.out3_branch(out2)
        multi_scale = torch.cat([self.avg_pool1(out1).view(x.size(0), -1), self.avg_pool2(out2).view(x.size(0), -1), out3.view(x.size(0), -1)], 1)
        return torch.sigmoid(self.fc(multi_scale))

# ==========================================
# 🚀 3. โหลดน้ำหนัก .pth เข้าโมเดล
# ==========================================
body_model = YOLO(BODY_MODEL_PATH, task='pose') # ของ Body รองรับ .pt ปกติ

face_model = FaceLandmarkModel().to(device)
face_model.load_state_dict(torch.load(FACE_MODEL_PATH, map_location=device, weights_only=True))
face_model.eval()

hand_model = HandLandmarkModel().to(device)
hand_model.load_state_dict(torch.load(HAND_MODEL_PATH, map_location=device, weights_only=True))
hand_model.eval()

# ==========================================
# 🛠️ 4. ฟังก์ชันจัดการภาพก่อนเข้า PFLD
# ==========================================
def preprocess_crop_torch(crop_img, size=(112, 112)):
    img_resized = cv2.resize(crop_img, size)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = (img_rgb.astype(np.float32) / 255.0 - 0.5) / 0.5
    img_chw = np.transpose(img_norm, (2, 0, 1))
    tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    return tensor

def get_square_crop(frame, center_x, center_y, size):
    h, w = frame.shape[:2]
    half_size = int(size // 2)
    x1, y1 = center_x - half_size, center_y - half_size
    x2, y2 = center_x + half_size, center_y + half_size
    
    x1_real, y1_real = max(0, x1), max(0, y1)
    x2_real, y2_real = min(w, x2), min(h, y2)
    crop = frame[y1_real:y2_real, x1_real:x2_real]
    
    pad_top = max(0, -y1)
    pad_bottom = max(0, y2 - h)
    pad_left = max(0, -x1)
    pad_right = max(0, x2 - w)
    
    if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
        crop = cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        
    return crop, x1, y1

# ==========================================
# ✋ 4.5. เตรียม MediaPipe สำหรับหาตำแหน่งมือ (Bounding Box)
# ==========================================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ==========================================
# 🎥 5. วงจรการทำงานกล้อง (Camera Loop)
# ==========================================
cap = cv2.VideoCapture(0)
pTime = 0
prev_face_size = 0
print("🎥 กล้องพร้อมใช้งาน! กด 'q' เพื่อออกจากโปรแกรม")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break
    frame = cv2.flip(frame, 1)
    
    # ------------------ STEP 1: Body Model ------------------
    results = body_model(frame, device=device.type, verbose=False)
    
    if len(results[0].keypoints.xy) > 0:
        body_kpts = results[0].keypoints.xy.cpu().numpy()[0].copy()
        
        # 🛡️ ดึงค่าความมั่นใจ (Confidence) ของแต่ละจุด
        if results[0].keypoints.conf is not None:
            body_confs = results[0].keypoints.conf.cpu().numpy()[0]
            for i in range(len(body_confs)):
                if body_confs[i] < 0.5: # ถ้าความมั่นใจต่ำกว่า 50% ให้มองว่าไม่เจอจุดนั้น
                    body_kpts[i] = [0, 0]
        
        # วาดจุดของ Body Model (17 จุดของ YOLO) ด้วยสีแดง
        for pt in body_kpts:
            x, y = int(pt[0]), int(pt[1])
            if x != 0 and y != 0:
                cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)
        
        nose = body_kpts[0]
        l_eye, r_eye = body_kpts[1], body_kpts[2]
        l_ear, r_ear = body_kpts[3], body_kpts[4]
        l_elbow, r_elbow = body_kpts[7], body_kpts[8]
        l_wrist, r_wrist = body_kpts[9], body_kpts[10]
        
        # ------------------ STEP 2: Face Model ------------------
        if nose[0] != 0 and nose[1] != 0:
            # คำนวณความกว้างใบหน้าจากระยะหู หรือระยะตา
            if l_ear[0] != 0 and r_ear[0] != 0:
                face_width = np.linalg.norm(l_ear - r_ear)
            elif l_eye[0] != 0 and r_eye[0] != 0:
                face_width = np.linalg.norm(l_eye - r_eye) * 2.5
            else:
                face_width = prev_face_size / 1.5 if prev_face_size > 0 else 150
            
            face_size = int(face_width * 1.5)
            
            if prev_face_size > 0:
                face_size = int(0.7 * face_size + 0.3 * prev_face_size)
            prev_face_size = face_size 
            
            if face_size > 50:
                face_crop, fx, fy = get_square_crop(frame, int(nose[0]), int(nose[1]), face_size)
                cv2.rectangle(frame, (fx, fy), (fx + face_size, fy + face_size), (0, 255, 0), 2)
                
                face_input = preprocess_crop_torch(face_crop)
                with torch.no_grad():
                    face_preds = face_model(face_input)[0].cpu().numpy()
                
                for i in range(98):
                    px = int(face_preds[i*2] * face_size) + fx
                    py = int(face_preds[i*2 + 1] * face_size) + fy
                    cv2.circle(frame, (px, py), 2, (0, 255, 255), -1)

        # ------------------ STEP 3: Hand Model (using MediaPipe for precise BBox) ------------------
        # ใช้ MediaPipe หากล่องครอบมือที่เป๊ะ 100% แทนการเดาจากข้อศอก
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        hands_results = hands_detector.process(rgb_frame)
        if hands_results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(hands_results.multi_hand_landmarks, hands_results.multi_handedness):
                # label จะบอกว่าเป็น "Left" หรือ "Right" ในรูปที่ถูก Mirror แล้ว
                label = handedness.classification[0].label 
                
                frame_h, frame_w = frame.shape[:2]
                
                # หาขอบเขต Bounding Box ของมือจาก 21 จุดของ MediaPipe
                x_min = min([lm.x for lm in hand_landmarks.landmark])
                x_max = max([lm.x for lm in hand_landmarks.landmark])
                y_min = min([lm.y for lm in hand_landmarks.landmark])
                y_max = max([lm.y for lm in hand_landmarks.landmark])
                
                x_min, x_max = int(x_min * frame_w), int(x_max * frame_w)
                y_min, y_max = int(y_min * frame_h), int(y_max * frame_h)
                
                center_x = (x_min + x_max) / 2
                center_y = (y_min + y_max) / 2
                
                box_w = x_max - x_min
                box_h = y_max - y_min
                
                # 🌟 สร้างกล่องให้เหมือน FreiHAND Dataset 
                # (FreiHAND ใช้ BBox ของเนื้อรูปมือจริงๆ ซึ่งใหญ่กว่า BBox ของโครงกระดูก MediaPipe เล็กน้อย)
                # จึงต้องขยายขนาดกล่องจาก 1.5 เป็น 1.9 เพื่อชดเชยให้สัดส่วนมือเท่ากับตอนเทรน
                hand_size = int(max(box_w, box_h) * 1.5)
                
                if hand_size > 40:
                    hand_crop, hx, hy = get_square_crop(frame, int(center_x), int(center_y), hand_size)
                    
                    # สี: ซ้ายในจอ(ขวาจริง)=Cyan, ขวาในจอ(ซ้ายจริง)=Magenta
                    color = (255, 255, 0) if label == "Left" else (255, 0, 255)
                    cv2.rectangle(frame, (hx, hy), (hx + hand_size, hy + hand_size), color, 2)
                    
                    # 🌟 TRICK: ภาพถูกกลับซ้ายขวา (Mirror)
                    # ถ้า MediaPipe บอกว่านี่คือ "Left" (นิ้วโป้งอยู่ขวา) -> ต้อง Flip ให้กลายเป็นมือขวาก่อนเข้าโมเดล
                    needs_flip = (label == "Left")
                    if needs_flip:
                        hand_crop_model = cv2.flip(hand_crop, 1)
                    else:
                        hand_crop_model = hand_crop
                    
                    hand_input = preprocess_crop_torch(hand_crop_model)
                    with torch.no_grad():
                        hand_preds = hand_model(hand_input)[0].cpu().numpy()
                    
                    for i in range(21):
                        px_norm = hand_preds[i*2]
                        py_norm = hand_preds[i*2 + 1]
                        
                        # ถ้าพลิกภาพไป ต้องพลิกพิกัด X กลับคืนให้ตรงกับภาพบนจอ
                        if needs_flip:
                            px_norm = 1.0 - px_norm
                            
                        px = int(px_norm * hand_size) + hx
                        py = int(py_norm * hand_size) + hy
                        cv2.circle(frame, (px, py), 3, color, -1)
        
    # ------------------ FPS Calculation ------------------
    cTime = time.time()
    fps = 1 / (cTime - pTime) if pTime != 0 else 0
    pTime = cTime
    cv2.putText(frame, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    cv2.imshow("VTuber (PyTorch .pth Version)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



