import cv2
import mediapipe as mp
import time
import math
import subprocess

# ---------------- MediaPipe ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# ---------------- STATES ----------------
volume_last = 0
nav_last = 0
last_pause = 0
last_mute = 0
last_fullscreen = 0
last_subtitle = 0

# ---------------- COOLDOWNS ----------------
PAUSE_CD = 2.0
MUTE_CD = 2.0
FULL_CD = 1.2
SUB_CD = 1.2
VOL_INTERVAL = 0.2
NAV_INTERVAL = 0.3

PINCH_CLOSE = 0.05

# ---------------- FPS ----------------
prev_time = time.time()
fps = 0

# ---------------- UTILS ----------------
def dist(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)

def fingers_up(hand):
    wrist = hand.landmark[0]
    tips = [4, 8, 12, 16, 20]
    joints = [2, 6, 10, 14, 18]

    states = []
    for t, j in zip(tips, joints):
        states.append(
            dist(hand.landmark[t], wrist) >
            dist(hand.landmark[j], wrist) * 1.1
        )
    return states

def palm_open_wide(hand):
    f = fingers_up(hand)
    if sum(f) != 5:
        return False

    index = hand.landmark[8]
    middle = hand.landmark[12]
    ring = hand.landmark[16]
    pinky = hand.landmark[20]

    return (dist(index, middle) > 0.05 and
            dist(middle, ring) > 0.05 and
            dist(ring, pinky) > 0.05)

def press(key):
    subprocess.Popen(["xdotool", "key", key])

# --- ADDED: BACK-SIDE PREVENTION LOGIC ---
def is_palm_facing_camera(hand, label):
    index_x = hand.landmark[5].x  # Index finger knuckle
    pinky_x = hand.landmark[17].x # Pinky finger knuckle

    if label == "Right":
        return index_x < pinky_x
    else:
        return index_x > pinky_x
# -----------------------------------------

# ---------------- MAIN LOOP ----------------
while True:

    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Smaller processing size for speed
    small = cv2.resize(frame, (224, 168))
    rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

    results = hands.process(rgb)

    gesture = ""
    action = ""

    if results.multi_hand_landmarks:
        hand = results.multi_hand_landmarks[0]
        
        # --- ADDED: Get Left/Right Hand Label ---
        label = results.multi_handedness[0].classification[0].label

        f = fingers_up(hand)

        wrist = hand.landmark[0]
        thumb = hand.landmark[4]
        index = hand.landmark[8]
        middle = hand.landmark[12]
        ring = hand.landmark[16]
        pinky = hand.landmark[20]

        pinch1 = dist(thumb, index)
        pinch2 = dist(thumb, pinky)
        pinch3 = dist(thumb, ring)

        now = time.time()

        # --- ADDED: Check if Back of Hand is showing ---
        if not is_palm_facing_camera(hand, label):
            gesture = "BACK OF HAND"
            action = "IGNORED"
        else:
            # -------- SEEK (Index + Pinky zones) --------
            if f[1] and f[4] and not f[2] and not f[3]:

                wrist_x = wrist.x

                if wrist_x < 0.35:
                    if now - nav_last > NAV_INTERVAL:
                        press("Left")
                        nav_last = now
                    gesture = "LEFT"
                    action = "BACK 10s"

                elif wrist_x > 0.65:
                    if now - nav_last > NAV_INTERVAL:
                        press("Right")
                        nav_last = now
                    gesture = "RIGHT"
                    action = "NEXT 10s"

                else:
                    gesture = "CENTER"
                    action = "READY"

            # -------- VOLUME UP --------
            elif f[0] and f[1] and not f[2] and not f[3] and not f[4]:
                if now - volume_last > VOL_INTERVAL:
                    press("0")
                    volume_last = now
                gesture = "THUMB+INDEX"
                action = "VOL UP"

            # -------- VOLUME DOWN --------
            elif pinch1 < PINCH_CLOSE and f[2] and f[3] and f[4]:
                if now - volume_last > VOL_INTERVAL:
                    press("9")
                    volume_last = now
                gesture = "OK SIGN"
                action = "VOL DOWN"

            # -------- PAUSE --------
            elif palm_open_wide(hand):
                if now - last_pause > PAUSE_CD:
                    press("space")
                    last_pause = now
                gesture = "OPEN PALM"
                action = "PAUSE / PLAY"

            # -------- MUTE --------
            elif f[1] and f[2] and pinch3 < PINCH_CLOSE:
                if now - last_mute > MUTE_CD:
                    press("m")
                    last_mute = now
                gesture = "INDEX+MID"
                action = "MUTE"

            # -------- FULLSCREEN --------
            elif f[1] and f[2] and f[3] and pinch2 < PINCH_CLOSE:
                if now - last_fullscreen > FULL_CD:
                    press("f")
                    last_fullscreen = now
                gesture = "3 FINGERS"
                action = "FULLSCREEN"

            # -------- SUBTITLE --------
            elif f[0] and f[4] and not f[1] and not f[2] and not f[3]:
                if now - last_subtitle > SUB_CD:
                    press("v")
                    last_subtitle = now
                gesture = "THUMB+PINKY"
                action = "SUBTITLE"

    # -------- FPS Calculation --------
    current_time = time.time()
    instant_fps = 1 / max((current_time - prev_time), 0.001)
    fps = 0.9 * fps + 0.1 * instant_fps
    prev_time = current_time

    # -------- DISPLAY --------
    cv2.putText(frame, "FPS: " + str(int(fps)), (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.putText(frame, "Gesture: " + gesture, (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    cv2.putText(frame, "Action: " + action, (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

    cv2.imshow("Jetson MPV Gesture Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
