import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import os
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import threading
import time

# ==========================================
# KONFIGURASI & INISIALISASI
# ==========================================
WIDTH, HEIGHT = 1000, 700
NUM_PARTICLES = 1500

lock = threading.Lock()
shared_data = {
    "mode": 1,
    "target_x": 0.0,
    "target_y": 0.0,
    "target_z": -12.0,
    "frame": None,
    "running": True
}

# ==========================================
# GENERASI DATA BENTUK (PARTIKEL 3D)
# ==========================================
pos_space = np.random.uniform(-4.0, 4.0, (NUM_PARTICLES, 3))

# Mode 2: Saturnus (Bola Inti + Cincin)
pos_planet = np.zeros((NUM_PARTICLES, 3))
NUM_SPHERE = 700
for i in range(NUM_SPHERE):
    phi = np.random.uniform(0, 2 * np.pi)
    costheta = np.random.uniform(-1, 1)
    theta = np.arccos(costheta)
    r = 1.3
    pos_planet[i, 0] = r * np.sin(theta) * np.cos(phi)
    pos_planet[i, 1] = r * np.sin(theta) * np.sin(phi)
    pos_planet[i, 2] = r * np.cos(theta)

for i in range(NUM_SPHERE, NUM_PARTICLES):
    theta = np.random.uniform(0, 2 * np.pi)
    r = np.random.uniform(1.8, 3.8)
    pos_planet[i, 0] = r * np.cos(theta)
    pos_planet[i, 1] = np.random.uniform(-0.05, 0.05)
    pos_planet[i, 2] = r * np.sin(theta)

# Mode 3: Tulisan "I LOVE YOU"
text_img = np.zeros((200, 800), dtype=np.uint8)
cv2.putText(text_img, "I LOVE YOU", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 3.5, 255, 12, cv2.LINE_AA)
y_indices, x_indices = np.where(text_img > 0)
x_text = (x_indices - 400) / 70.0
y_text = -(y_indices - 100) / 70.0
z_text = np.random.uniform(-0.1, 0.1, len(x_text))
pos_text = np.zeros((NUM_PARTICLES, 3))
text_points = np.stack((x_text, y_text, z_text), axis=-1)
chosen_indices = np.random.choice(len(text_points), NUM_PARTICLES)
pos_text = text_points[chosen_indices]

# Mode 4: Teks "SAYA AKAN LAWAN"
text_img_lawan = np.zeros((200, 800), dtype=np.uint8)
cv2.putText(text_img_lawan, "HIDUP JOKOWI", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.5, 255, 10, cv2.LINE_AA)
y_idx_lawan, x_idx_lawan = np.where(text_img_lawan > 0)
x_lawan = (x_idx_lawan - 400) / 70.0
y_lawan = -(y_idx_lawan - 100) / 70.0
z_lawan = np.random.uniform(-0.1, 0.1, len(x_lawan))
lawan_points = np.stack((x_lawan, y_lawan, z_lawan), axis=-1)
chosen_idx_lawan = np.random.choice(len(lawan_points), NUM_PARTICLES)
pos_lawan = lawan_points[chosen_idx_lawan]

current_pos = np.copy(pos_space)
target_pos = np.copy(pos_space)

# ==========================================
# FUNGSI LOGIKA DETEKSI GESTUR
# ==========================================
def hitung_mode_gestur(landmarks):
    """landmarks is a list of 21 NormalizedLandmark objects (new Tasks API format)."""
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]
    jari_berdiri = [landmarks[t].y < landmarks[p].y for t, p in zip(tips, pips)]

    if sum(jari_berdiri) == 0:
        return 4  # Genggam -> SAYA AKAN LAWAN
    if jari_berdiri[0] and not any(jari_berdiri[1:]):
        return 2  # Telunjuk -> Saturnus
    if jari_berdiri[0] and jari_berdiri[1] and not jari_berdiri[2] and not jari_berdiri[3]:
        return 3  # Peace -> Teks I LOVE YOU
    return 1  # Default -> Kosmos


def draw_hand_landmarks(frame, landmarks, w, h):
    """Draw hand landmarks and connections manually on the frame."""
    HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (5,9),(9,10),(10,11),(11,12),
        (9,13),(13,14),(14,15),(15,16),
        (13,17),(17,18),(18,19),(19,20),(0,17)
    ]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for pt in pts:
        cv2.circle(frame, pt, 4, (0, 0, 255), -1)

# ==========================================
# THREAD BACKGROUND: PROSES KAMERA & AI
# ==========================================
def camera_thread_func():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[WARN] Camera index 0 failed, trying index 1...")
        cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera. Check if the camera is connected and not used by another app.")
        shared_data["running"] = False
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 480)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

    # Setup new mediapipe Tasks API for hand landmark detection
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_options,
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    landmarker = mp_vision.HandLandmarker.create_from_options(options)

    frame_timestamp_ms = 0

    while shared_data["running"]:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.01)
            continue

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms += 33  # ~30fps
        results = landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        local_mode = 1
        local_x, local_y, local_z = 0.0, 0.0, -12.0
        h, w = frame.shape[:2]

        if results.hand_landmarks:
            for landmarks in results.hand_landmarks:
                draw_hand_landmarks(frame, landmarks, w, h)
                local_mode = hitung_mode_gestur(landmarks)

                wrist = landmarks[0]
                local_x = (wrist.x - 0.5) * 10.0
                local_y = -(wrist.y - 0.5) * 7.0

                pinky_mcp = landmarks[17]
                distance = math.sqrt((wrist.x - pinky_mcp.x)**2 + (wrist.y - pinky_mcp.y)**2)
                local_z = -10.0 - (1.0 / (distance + 0.01)) * 0.2

        with lock:
            shared_data["mode"] = local_mode
            shared_data["target_x"] = local_x
            shared_data["target_y"] = local_y
            shared_data["target_z"] = local_z
            shared_data["frame"] = frame

    landmarker.close()
    cap.release()

# Jalankan Thread Kamera terlebih dahulu
camera_thread = threading.Thread(target=camera_thread_func, daemon=False)
camera_thread.start()

# Berikan jeda 1 detik agar kamera macOS siap sebelum Pygame mengambil alih GUI utama
time.sleep(1.0)

# ==========================================
# MAIN THREAD: RENDERING GRAFIS 3D (Pygame diinisialisasi di sini)
# ==========================================
pygame.init()
pygame.mixer.init()
sounds = {
    1: pygame.mixer.Sound("beep_mode1.wav") if os.path.exists("beep_mode1.wav") else None,
    2: pygame.mixer.Sound("beep_mode2.wav") if os.path.exists("beep_mode2.wav") else None,
    3: pygame.mixer.Sound("beep_mode3.wav") if os.path.exists("beep_mode3.wav") else None,
    4: pygame.mixer.Sound("hidup_jokowi.wav") if os.path.exists("hidup_jokowi.wav") else None,
}
last_mode = shared_data["mode"]
pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("Space Gesture Controller - Mac Optimized")

glMatrixMode(GL_PROJECTION)
glLoadIdentity()
gluPerspective(45, (WIDTH / HEIGHT), 0.1, 50.0)
glMatrixMode(GL_MODELVIEW)
glEnable(GL_DEPTH_TEST)

clock = pygame.time.Clock()
rotation_angle = 0.0
hand_x, hand_y, hand_z = 0.0, 0.0, -12.0

while shared_data["running"]:
    pygame.event.pump()
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == KEYDOWN and event.key == K_ESCAPE):
            shared_data["running"] = False

    with lock:
        current_mode = shared_data["mode"]
        target_hand_x = shared_data["target_x"]
        target_hand_y = shared_data["target_y"]
        target_hand_z = shared_data["target_z"]
        frame = shared_data["frame"]

    if current_mode != last_mode:
        if sounds.get(current_mode):
            sounds[current_mode].play()
        last_mode = current_mode

    if frame is not None:
        mode_labels = {
            1: "ROTASI ANGCOSMOS (Terbuka)",
            2: "SATURNUS 3D (Satu Jari)",
            3: "TEKS I LOVE YOU (Peace)",
            4: "HIDUP JOKOWI (Kepal)"
        }
        cv2.putText(frame, f"MODE: {mode_labels[current_mode]}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.imshow("Hand Sensor Monitor", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            shared_data["running"] = False

    glClearColor(0.0, 0.0, 0.0, 1.0)
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()

    hand_x += (target_hand_x - hand_x) * 0.25
    hand_y += (target_hand_y - hand_y) * 0.25
    hand_z += (target_hand_z - hand_z) * 0.25

    if current_mode == 1:
        target_pos = pos_space
        rotation_angle += 0.5
    elif current_mode == 2:
        target_pos = pos_planet
        rotation_angle += 2.0
    elif current_mode == 3:
        target_pos = pos_text
        rotation_angle = 0.0
    elif current_mode == 4:
        target_pos = pos_lawan
        rotation_angle = 0.0

    current_pos += (target_pos - current_pos) * 0.15

    if current_mode in [2, 3, 4]:
        glTranslatef(hand_x, hand_y, hand_z)
        if current_mode == 2:
            glRotatef(25, 1.0, 0.0, 0.5)
    else:
        glTranslatef(0.0, 0.0, -12.0)

    glRotatef(rotation_angle, 0.0, 1.0, 0.0)

    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glPointSize(4.5)

    glBegin(GL_POINTS)
    for i in range(NUM_PARTICLES):
        if current_mode == 3:
            glColor4f(0.0, 0.8, 1.0, 0.9)
        elif current_mode == 4:
            glColor4f(1.0, 0.2, 0.2, 0.95)
        elif current_mode == 2 and i >= NUM_SPHERE:
            glColor4f(1.0, 0.7, 0.3, 0.6)
        elif current_mode == 2 and i < NUM_SPHERE:
            glColor4f(1.0, 0.5, 0.0, 0.85)
        else:
            glColor4f(0.1, 0.5, 1.0, 0.8)

        glVertex3f(current_pos[i, 0], current_pos[i, 1], current_pos[i, 2])
    glEnd()

    pygame.display.flip()
    clock.tick(60)

# Wait for camera thread to finish cleanup (landmarker.close())
camera_thread.join(timeout=3.0)
cv2.destroyAllWindows()
pygame.quit()
