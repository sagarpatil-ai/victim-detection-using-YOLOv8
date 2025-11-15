from flask import Flask, render_template, request, redirect, url_for, session, flash, send_from_directory
from ultralytics import YOLO
import os, cv2, torch, numpy as np, sys, sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from flask_bcrypt import Bcrypt
from PIL import Image

# ---- Depth-Anything path (adjust if different) ----
sys.path.append(os.path.join(os.getcwd(), "Depth-Anything-V2", "Depth-Anything-V2-main"))
from depth_anything_v2.dpt import DepthAnythingV2

app = Flask(__name__)

# ---------------------------- AUTH & DB ----------------------------
# Change this to a secure random secret before production
app.secret_key = "replace_this_with_a_secure_random_key"

bcrypt = Bcrypt(app)
DB_FILE = "users.db"

def get_db_connection():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """
    Ensures DB and tables exist:
      - users (original)
      - posts (community)
      - comments (community comments)
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # users table (your original)
    cur.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE,
                    email TEXT UNIQUE,
                    password TEXT
                )''')

    # posts table for community posts (title, body, optional image, optional coords)
    cur.execute('''CREATE TABLE IF NOT EXISTS posts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT,
                    body TEXT NOT NULL,
                    image_path TEXT,
                    latitude REAL,
                    longitude REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )''')

    # comments table
    cur.execute('''CREATE TABLE IF NOT EXISTS comments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    post_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    body TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(post_id) REFERENCES posts(id),
                    FOREIGN KEY(user_id) REFERENCES users(id)
                )''')

    conn.commit()
    conn.close()

def get_user_by_username(username):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = cur.fetchone()
    conn.close()
    return user

# Ensure DB and tables exist
init_db()

# ---------------------------- Folders ----------------------------
UPLOAD_FOLDER = "static/uploads"
PREDICT_FOLDER = "static/results/predict"
COMMUNITY_UPLOAD_FOLDER = os.path.join("static", "community_uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PREDICT_FOLDER, exist_ok=True)
os.makedirs(COMMUNITY_UPLOAD_FOLDER, exist_ok=True)

# ---------------------------- Config ----------------------------
VFOV_DEG_DEFAULT = 55.0
PERSON_HEIGHT_M_DEFAULT = 1.70  # assume adult

# Depth fusion thresholds (tuned for rubble/landslide scenes)
TORSO_MARGIN = 0.05
COVERAGE_THRESH = 0.05

FUSION_COVERAGE_NEEDHELP = 0.25
FUSION_COVERAGE_EMERGENCY = 0.45

TORSO_BG_DELTA_NEEDHELP = 0.06
TORSO_BG_DELTA_EMERGENCY = 0.10

# Feet occlusion (YOLOv8-pose keypoint conf & depth delta)
ANKLE_CONF_LOW = 0.30
FEET_DEPTH_OCCLUSED_DELTA = 0.05

# Risk score → label
RISK_EMERGENCY = 0.70
RISK_NEEDHELP = 0.40

# ---------------------------- Models ----------------------------
print("🔍 Loading YOLOv8 models...")
yolo_model = YOLO("yolov8s.pt")          # boxes
pose_model = YOLO("yolov8s-pose.pt")     # pose

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Device: {device}")

depth_model = DepthAnythingV2(
    encoder="vits", features=64, out_channels=[48, 96, 192, 384]
).to(device).eval()

depth_ckpt = "Depth-Anything-V2/checkpoints/depth_anything_v2_vits.pth"
if os.path.exists(depth_ckpt):
    depth_model.load_state_dict(torch.load(depth_ckpt, map_location=device))
    print("✅ Depth model loaded")
else:
    print("⚠️ Depth checkpoint missing! Depth maps will not be meaningful without weights.")

# ---------------------------- Helpers ----------------------------
def infer_depth(img: np.ndarray) -> np.ndarray:
    """Return normalized depth (0..1) same size as img."""
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (518, 518))
    t = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = t.to(device)
    with torch.no_grad():
        d = depth_model(t).squeeze().cpu().numpy()
    d = (d - d.min()) / (d.max() - d.min() + 1e-6)
    return cv2.resize(d, (img.shape[1], img.shape[0]))

def get_gps(path):
    """Extract (lat, lon) from EXIF; return (None, None) if absent."""
    try:
        ex = Image.open(path)._getexif()
        if not ex: return None, None
        gps = ex.get(34853)
        if not gps: return None, None
        def conv(v): return float(v[0]) / v[1]
        lat = conv(gps[2][0]) + conv(gps[2][1]) / 60 + conv(gps[2][2]) / 3600
        if gps[1] != "N": lat = -lat
        lon = conv(gps[4][0]) + conv(gps[4][1]) / 60 + conv(gps[4][2]) / 3600
        if gps[3] != "E": lon = -lon
        return lat, lon
    except:
        return None, None

# COCO skeleton pairs
SKELETON = [(15,13),(13,11),(16,14),(14,12),(11,12),(5,11),(6,12),(5,6),
            (5,7),(7,9),(6,8),(8,10),(1,2),(0,1),(0,2),(1,3),(2,4),(3,5),(4,6)]

def detect_priority_pose(k, box, conf):
    """
    Pose-only decision (conservative). Returns (priority, reason, raw_angle_deg).
    """
    if k is None or len(k) < 17:
        return ("Need Help" if conf > 0.5 else "Low Priority", "few/occluded keypoints", 0.0)

    def y(i): return k[i][1]
    def x(i): return k[i][0]

    x1, y1, x2, y2 = box
    bh = max(1, y2 - y1)

    nose = y(0); ls = y(5); rs = y(6); lh = y(11); rh = y(12)
    lw = y(9); rw = y(10); la = y(15); ra = y(16)

    mid_sh = ((x(5)+x(6))/2, (ls+rs)/2)
    mid_hp = ((x(11)+x(12))/2, (lh+rh)/2)
    dx = mid_hp[0] - mid_sh[0]
    dy = mid_hp[1] - mid_sh[1]
    # angle from vertical (0 upright → 90 lying)
    angle = float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))

    lying = angle > 65
    collapse = (abs(nose - la) < 0.25*bh) or (abs(nose - ra) < 0.25*bh)
    hands_up = (lw < nose - 0.12*bh) or (rw < nose - 0.12*bh)

    if lying or collapse:
        return ("Emergency" if conf >= 0.60 else "Need Help", f"lying({angle:.1f}°) or collapse", angle)
    if hands_up:
        return ("Need Help", "hands above head", angle)
    return ("Low Priority", "normal posture", angle)

def estimate_distance_m_from_fov(img_h, bbox_h, vfov_deg=VFOV_DEG_DEFAULT, person_h_m=PERSON_HEIGHT_M_DEFAULT):
    bbox_h = max(1.0, float(bbox_h))
    vfov_rad = np.deg2rad(vfov_deg)
    focal_px = (img_h / 2.0) / np.tan(vfov_rad / 2.0)
    return float((focal_px * person_h_m) / bbox_h)

def torso_box_from_kpts(k, x1, y1, x2, y2, margin=TORSO_MARGIN):
    if k is None or len(k) < 17: return x1, y1, x2, y2
    xs = [k[5][0], k[6][0], k[11][0], k[12][0]]
    ys = [k[5][1], k[6][1], k[11][1], k[12][1]]
    if any(v <= 0 for v in xs + ys): return x1, y1, x2, y2
    tx1, tx2 = int(min(xs)), int(max(xs))
    ty1, ty2 = int(min(ys)), int(max(ys))
    bw, bh = x2 - x1, y2 - y1
    pad_x, pad_y = int(bw * margin), int(bh * margin)
    tx1, tx2 = max(x1, tx1 - pad_x), min(x2, tx2 + pad_x)
    ty1, ty2 = max(y1, ty1 - pad_y), min(y2, ty2 + pad_y)
    if tx2 <= tx1 or ty2 <= ty1: return x1, y1, x2, y2
    return tx1, ty1, tx2, ty2

def local_bg_stats(depth, x1, y1, x2, y2, pad_frac=0.06):
    H, W = depth.shape
    pad = max(4, int(pad_frac * max(W, H)))
    bx1, by1 = max(0, x1 - pad), max(0, y1 - pad)
    bx2, by2 = min(W-1, x2 + pad), min(H-1, y2 + pad)
    bg = depth[by1:by2, bx1:bx2].copy()
    # cut out bbox region
    bg[y1-by1:y2-by1, x1-bx1:x2-bx1] = np.nan
    return float(np.nanmedian(bg))

def depth_fusion_analysis(depth, x1, y1, x2, y2, k=None):
    """Return (coverage, delta_torso_bg, bg_median, torso_mean, reason)."""
    H, W = depth.shape
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0, 0.0, 0.0, 0.0, "invalid bbox"
    bbox_depth = depth[y1:y2, x1:x2]
    bg_median = local_bg_stats(depth, x1, y1, x2, y2)

    tx1, ty1, tx2, ty2 = torso_box_from_kpts(k, x1, y1, x2, y2)
    torso = depth[ty1:ty2, tx1:tx2]
    torso_mean = float(np.nanmean(torso)) if torso.size > 0 else float(np.nanmean(bbox_depth))

    delta = float(torso_mean - bg_median)
    coverage = float(np.mean(bbox_depth > (bg_median + COVERAGE_THRESH))) if bbox_depth.size > 0 else 0.0

    if delta > TORSO_BG_DELTA_EMERGENCY and coverage >= FUSION_COVERAGE_EMERGENCY:
        reason = f"torso deeper(+{delta:.2f}) & coverage {coverage:.2f}"
    elif delta > TORSO_BG_DELTA_EMERGENCY:
        reason = f"torso deeper(+{delta:.2f})"
    elif coverage >= FUSION_COVERAGE_EMERGENCY:
        reason = f"coverage {coverage:.2f} ≥ {FUSION_COVERAGE_EMERGENCY}"
    else:
        reason = "no strong burial signal"
    return coverage, delta, bg_median, torso_mean, reason

def split_cov(depth, x1, y1, x2, y2, bg_med, frac=0.5, thresh=COVERAGE_THRESH):
    """Return (upper_cov, lower_cov) deeper than bg_med + thresh."""
    H, W = depth.shape
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
    if x2 <= x1 or y2 <= y1: return 0.0, 0.0
    mid = int(y1 + frac * (y2 - y1))
    upper = depth[y1:mid, x1:x2]
    lower = depth[mid:y2, x1:x2]
    def cov(arr): return float(np.mean(arr > (bg_med + thresh))) if arr.size else 0.0
    return cov(upper), cov(lower)

def head_airway_risk(depth, k, bg_med, r=6, z_delta=0.06):
    """Check if nose/eyes are deeper than bg (possible coverage)."""
    if k is None or len(k) < 17: return 0.0, "no kpts"
    H, W = depth.shape
    idxs = [0,1,2]  # nose, left eye, right eye
    deep_hits, total = 0, 0
    for i in idxs:
        x, y, s = k[i]
        if x <= 0 or y <= 0: continue
        x = int(np.clip(x, 0, W-1)); y = int(np.clip(y, 0, H-1))
        x0, x1 = max(0, x-r), min(W, x+r+1)
        y0, y1 = max(0, y-r), min(H, y+r+1)
        patch = depth[y0:y1, x0:x1]
        if patch.size == 0: continue
        total += 1
        if float(np.nanmean(patch) - bg_med) > z_delta:
            deep_hits += 1
    score = deep_hits / total if total else 0.0
    reason = "airway covered" if score >= 0.67 else ("partially covered" if score > 0 else "airway ok")
    return float(score), reason

def edge_contact_pressure(depth, x1, y1, x2, y2, torso_mean, band=6):
    """Large depth step at edges vs torso suggests pressure/pinning."""
    H, W = depth.shape
    x1 = max(0, x1); y1 = max(0, y1); x2 = min(W-1, x2); y2 = min(H-1, y2)
    if x2 <= x1 or y2 <= y1: return 0.0
    L = depth[y1:y2, max(0, x1-band):x1]
    R = depth[y1:y2, x2:min(W, x2+band)]
    T = depth[max(0, y1-band):y1, x1:x2]
    B = depth[y2:min(H, y2+band), x1:x2]
    edges = [arr for arr in [L, R, T, B] if arr.size]
    if not edges: return 0.0
    deltas = [float(np.nanmean(e) - torso_mean) for e in edges]
    mx = max(deltas)  # positive -> edge deeper (object in front)
    return float(np.clip((mx - 0.04) / 0.10, 0.0, 1.0))

def scene_slope_from_depth(depth, sample=8):
    """Rough vertical slope using column-wise medians; degrees."""
    H, W = depth.shape
    xs = np.linspace(0, W-1, sample).astype(int)
    meds = [np.nanmedian(depth[:, x]) for x in xs]
    m = np.polyfit(xs, meds, 1)[0] if len(xs) >= 2 else 0.0
    angle = np.degrees(np.arctan(m * (W / H)))
    return float(angle)

def slope_corrected_lying_angle(raw_angle, slope_deg):
    """Reduce lying threshold if scene is tilted; returns adjusted angle."""
    return float(raw_angle + 0.4 * abs(slope_deg))

def clamp01(x): return float(np.clip(x, 0.0, 1.0))

def feet_occluded(depth, k, x1, y1, x2, y2, bg_med):
    """Feet occluded if ankle kp low or ankle depth >> bg."""
    if k is None or len(k) < 17: return False, "no kpts"
    def kpt(i): return k[i] if i < len(k) else (0, 0, 0)
    l_ax, l_ay, l_as = kpt(15)  # left ankle
    r_ax, r_ay, r_as = kpt(16)  # right ankle
    low_score = (l_as < ANKLE_CONF_LOW) or (r_as < ANKLE_CONF_LOW)

    H, W = depth.shape
    def patch_mean(cx, cy, r=6):
        if cx <= 0 or cy <= 0: return None
        cx = int(np.clip(cx, 0, W-1)); cy = int(np.clip(cy, 0, H-1))
        x0, x1p = max(0, cx-r), min(W, cx+r+1)
        y0, y1p = max(0, cy-r), min(H, cy+r+1)
        p = depth[y0:y1p, x0:x1p]
        return float(np.nanmean(p)) if p.size else None

    ankle_deep = False
    reason = []
    lm = patch_mean(l_ax, l_ay)
    rm = patch_mean(r_ax, r_ay)
    if lm is not None and lm - bg_med > FEET_DEPTH_OCCLUSED_DELTA:
        ankle_deep = True; reason.append("left feet deeper")
    if rm is not None and rm - bg_med > FEET_DEPTH_OCCLUSED_DELTA:
        ankle_deep = True; reason.append("right feet deeper")
    if low_score: reason.append("ankle kpt low")
    return (low_score or ankle_deep), (", ".join(reason) if reason else "feet ok")

# ---------------------------- Routes ----------------------------
@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        email = request.form.get("email", "").strip()
        password = request.form.get("password", "")
        if not username or not email or not password:
            flash("Please fill all fields.", "danger")
            return render_template("register.html")
        hashed_pw = bcrypt.generate_password_hash(password).decode("utf-8")

        conn = get_db_connection()
        cur = conn.cursor()
        try:
            cur.execute("INSERT INTO users (username, email, password) VALUES (?, ?, ?)",
                        (username, email, hashed_pw))
            conn.commit()
            flash("Account created successfully. Please log in.", "success")
            return redirect(url_for("login"))
        except sqlite3.IntegrityError:
            flash("Username or email already exists.", "danger")
        finally:
            conn.close()
    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        if not username or not password:
            flash("Please enter username and password.", "danger")
            return render_template("login.html")
        user = get_user_by_username(username)
        if user and bcrypt.check_password_hash(user["password"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome, {user['username']}!", "success")
            return redirect(url_for("index"))
        else:
            flash("Invalid username or password.", "danger")
    return render_template("login.html")

@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))

@app.route("/", methods=["GET", "POST"])
def index():
    # protect route: require login
    if "user_id" not in session:
        flash("Please log in to use DisasterSight.", "warning")
        return redirect(url_for("login"))

    result_img = pose_img = depth_img = None
    victims = []
    locs = []
    total = emer = need_help = low = 0

    if request.method == "POST":
        f = request.files.get("file")
        if not f:
            return redirect("/")
        name = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(f.filename)
        up = os.path.join(UPLOAD_FOLDER, name)
        f.save(up)

        lat, lon = get_gps(up)
        if not lat:
            lat, lon = 28.61, 77.20  # fallback

        img = cv2.imread(up)
        H, W = img.shape[:2]

        # --- Boxes view
        det = yolo_model(up)[0]
        det_out = det.plot()
        cv2.imwrite(os.path.join(PREDICT_FOLDER, f"det_{name}"), det_out)
        result_img = f"results/predict/det_{name}"

        # --- Depth
        depth = infer_depth(img)
        depth_vis = (depth * 255).astype(np.uint8)
        depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_MAGMA)
        cv2.imwrite(os.path.join(PREDICT_FOLDER, f"depth_{name}"), depth_vis)
        depth_img = f"results/predict/depth_{name}"

        # --- Pose
        pose = pose_model(up)[0]
        kp = pose.keypoints.data.cpu().numpy() if pose.keypoints is not None else None  # [N,17,3]
        boxes = pose.boxes
        pose_canvas = img.copy()

        # Precompute scene slope once
        scene_slope = scene_slope_from_depth(depth)

        if boxes is not None:
            # (Optional) neighbor risk scaffold
            all_boxes = [b.xyxy[0].tolist() for b in boxes]
            # Loop
            for i, b in enumerate(boxes):
                xy = b.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, xy)
                conf = float(b.conf)

                kpts = kp[i] if kp is not None and i < len(kp) else None

                # Pose-only
                pose_priority, pose_reason, raw_angle = detect_priority_pose(kpts, (x1, y1, x2, y2), conf)

                # Depth fusion metrics
                coverage, delta, bg_med, torso_mean, depth_reason = depth_fusion_analysis(depth, x1, y1, x2, y2, kpts)
                upper_cov, lower_cov = split_cov(depth, x1, y1, x2, y2, bg_med)
                airway_score, airway_note = head_airway_risk(depth, kpts, bg_med)
                feet_blocked, feet_note = feet_occluded(depth, kpts, x1, y1, x2, y2, bg_med)
                contact_pin_score = edge_contact_pressure(depth, x1, y1, x2, y2, torso_mean)

                # Slope-aware lying contribution
                adj_angle = slope_corrected_lying_angle(raw_angle, scene_slope)
                lying01 = clamp01((adj_angle - 55.0) / 35.0)  # contributes >55°

                # Distance (FOV-based)
                # Optional: if bbox is small (child), you can reduce assumed height
                person_h_m = PERSON_HEIGHT_M_DEFAULT
                if (y2 - y1) < 0.18 * H:
                    person_h_m = 1.35
                distance_m = estimate_distance_m_from_fov(H, (y2 - y1), VFOV_DEG_DEFAULT, person_h_m)

                # Neighbor risk (simple placeholder = 0 for single image)
                neighbor_risk = 0.0

                # ------------- Unified Risk Score -------------
                risk = (
                    0.25 * clamp01(lower_cov / 0.55) +
                    0.20 * clamp01(upper_cov / 0.45) +
                    0.20 * clamp01((delta - 0.06) / 0.10) +
                    0.10 * (1.0 if feet_blocked else 0.0) +
                    0.10 * clamp01(contact_pin_score) +
                    0.05 * lying01 +
                    0.05 * clamp01(neighbor_risk)
                )
                # Pose biasing
                if "hands above head" in pose_reason:
                    risk = max(risk, 0.45)
                if "lying" in pose_reason and conf >= 0.60:
                    risk = max(risk, 0.70)

                # Risk → label
                if risk >= RISK_EMERGENCY:
                    fusion = "Emergency"
                elif risk >= RISK_NEEDHELP:
                    fusion = "Need Help"
                else:
                    fusion = "Low Priority"

                # --- Draw pose canvas box/label ---
                if fusion == "Emergency": color = (0, 0, 255)
                elif fusion == "Need Help": color = (0, 165, 255)
                else: color = (0, 255, 0)

                label_txt = f"{fusion} • {round(conf*100,1)}% • {distance_m:.1f}m • risk {risk:.2f}"
                (tw, th), base = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                cv2.rectangle(pose_canvas, (x1, y1), (x2, y2), color, 4)
                cv2.rectangle(pose_canvas, (x1, max(0, y1 - th - base - 4)), (x1 + tw + 10, y1), color, -1)
                cv2.putText(pose_canvas, label_txt, (x1 + 5, y1 - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                if kpts is not None:
                    for (xk, yk, _z) in kpts:
                        if xk > 0 and yk > 0:
                            cv2.circle(pose_canvas, (int(xk), int(yk)), 5, color, -1)
                    for j, k2 in SKELETON:
                        if kpts[j][0] > 0 and kpts[j][1] > 0 and kpts[k2][0] > 0 and kpts[k2][1] > 0:
                            cv2.line(pose_canvas,
                                     (int(kpts[j][0]), int(kpts[j][1])),
                                     (int(kpts[k2][0]), int(kpts[k2][1])),
                                     color, 3)

                # Counts
                if fusion == "Emergency": emer += 1
                elif fusion == "Need Help": need_help += 1
                else: low += 1

                # Collect for UI
                victims.append({
                    "id": len(victims) + 1,
                    "confidence": round(conf * 100, 1),
                    "priority": fusion,
                    "distance_m": round(distance_m, 2),
                    "coverage": round(coverage, 2),
                    "upper_cov": round(upper_cov, 2),
                    "lower_cov": round(lower_cov, 2),
                    "airway": round(airway_score, 2),
                    "contact_pin": round(contact_pin_score, 2),
                    "depth_delta": round(delta, 2),
                    "risk": round(risk, 2),
                    "pose_reason": pose_reason,
                    "depth_reason": depth_reason
                })

                # Approx geo offset for map
                xc, yc = (x1 + x2) / 2, (y1 + y2) / 2
                locs.append({
                    "id": len(victims),
                    "lat": lat + (yc - H / 2) * 1e-5,
                    "lng": lon + (xc - W / 2) * 1e-5,
                    "confidence": round(conf * 100, 1),
                    "priority": fusion
                })

        total = len(victims)
        cv2.imwrite(os.path.join(PREDICT_FOLDER, f"pose_{name}"), pose_canvas)
        pose_img = f"results/predict/pose_{name}"

    return render_template(
        "index.html",
        result_img=result_img,
        pose_img=pose_img,
        depth_img=depth_img,
        victim_info=victims,
        victim_locations=locs,
        total_count=total,
        emergency_count=emer,
        need_help_count=need_help,
        low_priority_count=low
    )

# ---------------------------- COMMUNITY ROUTES ----------------------------
@app.route("/community", methods=["GET", "POST"])
def community():
    if "user_id" not in session:
        flash("Please log in to access the rescue community.", "warning")
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()

    # create post with optional image + optional coords
    if request.method == "POST":
        title = request.form.get("title", "").strip()
        body = request.form.get("body", "").strip()
        lat = request.form.get("latitude")
        lng = request.form.get("longitude")
        image = request.files.get("image")
        image_path = None
        if image and image.filename:
            fname = datetime.now().strftime("%Y%m%d_%H%M%S_") + secure_filename(image.filename)
            save_path = os.path.join(COMMUNITY_UPLOAD_FOLDER, fname)
            image.save(save_path)
            image_path = save_path.replace("\\", "/")

        if not body and not image_path:
            flash("Please provide a message or image for the post.", "danger")
            return redirect(url_for("community"))

        cur.execute("""INSERT INTO posts (user_id, title, body, image_path, latitude, longitude)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (session["user_id"], title, body, image_path, lat, lng))
        conn.commit()
        flash("Post created.", "success")
        return redirect(url_for("community"))

    # fetch posts with author username
    cur.execute("""
        SELECT posts.*, users.username
        FROM posts
        JOIN users ON posts.user_id = users.id
        ORDER BY posts.created_at DESC
    """)
    posts = cur.fetchall()
    conn.close()
    return render_template("community.html", posts=posts)

@app.route("/post/<int:post_id>", methods=["GET", "POST"])
def post_detail(post_id):
    if "user_id" not in session:
        return redirect(url_for("login"))

    conn = get_db_connection()
    cur = conn.cursor()

    # add comment
    if request.method == "POST":
        comment_body = request.form.get("comment", "").strip()
        if comment_body:
            cur.execute("INSERT INTO comments (post_id, user_id, body) VALUES (?, ?, ?)",
                        (post_id, session["user_id"], comment_body))
            conn.commit()
            flash("Comment added.", "success")
        else:
            flash("Comment cannot be empty.", "danger")
        return redirect(url_for("post_detail", post_id=post_id))

    # fetch post
    cur.execute("""
        SELECT posts.*, users.username
        FROM posts
        JOIN users ON posts.user_id = users.id
        WHERE posts.id = ?
    """, (post_id,))
    post = cur.fetchone()

    # fetch comments
    cur.execute("""
        SELECT comments.*, users.username
        FROM comments
        JOIN users ON comments.user_id = users.id
        WHERE comments.post_id = ?
        ORDER BY comments.created_at DESC
    """, (post_id,))
    comments = cur.fetchall()
    conn.close()
    return render_template("post_detail.html", post=post, comments=comments)

# serve community uploads (for convenience)
@app.route('/community_uploads/<path:filename>')
def community_uploads(filename):
    return send_from_directory(COMMUNITY_UPLOAD_FOLDER, filename)

if __name__ == "__main__":
    print("🚀 Starting DisasterSight at http://127.0.0.1:5000")
    app.run(debug=True)
