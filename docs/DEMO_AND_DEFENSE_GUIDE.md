# AI Gym Trainer — Demo & Defense Guide

A one-stop sheet for the viva: the exact commands to run, a plain-English
explanation of the whole system, why each technology was chosen (with
alternatives), honest limitations, and likely examiner questions with answers.

> Keep **Section 1 (Runbook)** open on a second screen during the demo so you
> never have to remember a command.

---

## 0. 60-second pitch (say this first)

> "It's a **real-time AI gym coach** that uses just a **laptop webcam** — no
> special hardware, no GPU, and **no cloud**. It watches your body, **counts your
> reps**, **scores your form** out of 100 with live feedback, and **detects the
> gym equipment** in view. Everything runs **on the device**, so it's private and
> works offline. It supports three exercises — **bicep curl, bent-over row, and
> push-up** — tuned for three age groups (children, adults, seniors)."

---

## 1. DEMO RUNBOOK (the commands)

The app has **three programs** that run at the same time, each in **its own
terminal**. Think of them as: **the brain** (AI), **the notebook** (accounts &
history), and **the screen** (the website you click).

| # | Terminal | What it is | Port |
|---|----------|------------|------|
| 1 | **ML brain** (Python) | Pose, rep counting, form scoring, equipment | **8000** |
| 2 | **App-data server** (Node) | Login, sign-up, workout history (SQLite) | **5001** |
| 3 | **Web UI** (Vite/React) | The website you present | **5173** |

### 1.1 One-time setup (do this BEFORE viva day, needs internet)

```powershell
# From the project root: e:\FYP 1\AI_Gym_FYP
pip install -r requirements.txt           # Python libraries
python -m scripts.fetch_yolo_weights      # download base YOLO weights

cd frontend\web_app\FYP1
npm install                               # web UI libraries
cd server
npm install                               # app-data server libraries
cd ..\..\..\..                            # back to project root
```

### 1.2 Start the three services (3 terminals, in this order)

**Terminal 1 — ML brain (start this first):**
```powershell
python -m uvicorn api_backend:app --host 127.0.0.1 --port 8000
```
✅ Wait until you see: `Model availability: 9/9 models loaded` and
`Application startup complete.`

**Terminal 2 — App-data server (must be run from the `server` folder):**
```powershell
cd frontend\web_app\FYP1\server
npm start
```
✅ Wait until you see: `🚀 Server running on port 5001`

**Terminal 3 — Web UI:**
```powershell
cd frontend\web_app\FYP1
npm run dev
```
✅ It prints a link like `http://localhost:5173/` — open that in the browser.

### 1.3 The demo flow (what to click)

1. **Sign up** (or log in) — name, email, password, age, etc.
2. Pick **today's exercise** (e.g. a Biceps / Back / Chest exercise).
3. Choose **Camera Mode** → **Allow camera**.
4. Press **Start Workout** → do a few reps.
5. Show the panel: **rep count**, **form quality %**, the **skeleton overlay**
   on your body, **live feedback**, and the **Equipment Detected** card (hold a
   dumbbell/bench in view).

### 1.4 Prove it works — run the tests

```powershell
python -m pytest tests\ -q
```
✅ Expect: **258 passed**. (Good line to show an examiner: "every part is unit-
and integration-tested, on Windows and Linux CI.")

### 1.5 Desktop fallback demo (no servers needed — just the camera)

If a server misbehaves on demo day, you can still show the core AI:
```powershell
python main.py
```
This opens an OpenCV window with the live skeleton + rep counting (no website).

### 1.6 Optional / advanced commands (only if asked)

```powershell
# Retrain all 9 form models
python training\train_universal.py

# Re-train the equipment detector (CPU-friendly preset)
python -m training.finetune_equipment_yolo --fast `
  --data "E:\FYP 1\data set\gym-equipment.v2i.yolov8\data_local.yaml" `
  --output models\equipment_yolov8n.pt

# Record a per-frame trace of a live session (for analysis / tuning)
$env:AI_GYM_LIVE_TRACE = "logs\live_trace.jsonl"
python -m uvicorn api_backend:app --host 127.0.0.1 --port 8000
```

### 1.7 Troubleshooting (read this the morning of the viva)

| Symptom | Cause | Fix |
|---------|-------|-----|
| **"Failed to fetch"** on login/signup | App-data server (5001) isn't running | Start **Terminal 2**. *(Not a database/XAMPP problem — the app uses local SQLite. Do **not** start XAMPP/MySQL.)* |
| App-data server says port 5000 | You ran `npm start` from the wrong folder | Run it from inside `frontend\web_app\FYP1\server` |
| **"ML Offline"** badge in browser | Python ML server (8000) isn't running | Start **Terminal 1** |
| Equipment card stays empty | Backend started before the gym model existed | Restart Terminal 1 so it loads `models/equipment_yolov8n.pt` |
| Camera is black | Another app is using the webcam | Close Zoom/Teams/Camera app; re-allow permission |
| "Address already in use" | An old server is still running | Close the old terminal / kill the process on that port |

---

## 2. How it works (plain English)

### 2.1 The big picture

```
   Your webcam
       │  (video frames)
       ▼
  [ Web UI :5173 ] ──login/history──► [ App-data server :5001 ] ──► SQLite file
       │
       │  each frame (as an image)
       ▼
  [ ML brain :8000 ]  ── MediaPipe finds your body → angles → rep count + form score
                       └─ YOLO (once a second) → equipment in view
```

- **Web UI** = the website (React). It shows the camera, the skeleton, the
  numbers, and the feedback.
- **App-data server** = a small Node/Express service with a **SQLite** file. It
  only stores *accounts and workout history* — **no AI runs here**.
- **ML brain** = the Python (FastAPI) service. **All AI is here**, on your
  device.

### 2.2 From video to a rep (the core loop)

1. **Pose estimation** — **MediaPipe Pose** finds **33 body points** (shoulder,
   elbow, wrist, hip, knee, …) in each frame.
2. **Angles** — we turn those points into **joint angles**. The main one (the
   "primary angle") is the elbow bend for arms, etc.

   | Exercise | Primary angle |
   |----------|---------------|
   | Bicep curl | shoulder → elbow → wrist (elbow bend) |
   | Bent-over row | shoulder → elbow → wrist |
   | Push-up | shoulder → elbow → wrist (with body-line checks) |

3. **Rep counting** — as you move, the primary angle goes up and down like a
   wave. We **smooth** the wave (to ignore camera jitter) and count **one rep
   per full dip-and-rise** that is big enough and far enough apart from the last
   one. (Technique: *prominence-based peak detection*.)
4. **Form scoring** — when a rep finishes, we score it **0–100** by combining:
   - **Rules** (e.g. "did you go through the full range of motion?"),
   - a **Random Forest** model (good vs bad form from the angles),
   - (optionally a CNN+LSTM model — *off by default*, see limitations).

### 2.3 The 3-pillar form score (one number you can defend)

The final score is a **weighted blend**, so no single component can be wrong on
its own:
- With rules + Random Forest: **rules 0.6 / RF 0.4**.
- If the experimental CNN+LSTM is enabled: **rules 0.4 / RF 0.3 / NN 0.3**.

### 2.4 Equipment detection (YOLO)

A **YOLOv8-nano** model runs **once per second** in the background and reports
what gym equipment is in view (**barbell, bench, dumbbell, parallel bars,
pull-up bar, resistance band**). It's separate from the pose work so it never
slows down the rep counting.

### 2.5 Why three age groups?

The same movement looks different for a child, an adult, and a senior (range of
motion, speed). We keep **separate models per age group** (children / adult /
senior) so feedback is fair. The app picks the group from the user's age.

---

## 3. Why these technologies (defending your choices)

> Core theme to repeat: **"This runs on a normal laptop CPU, in real time, with
> no GPU and no internet. Every choice optimises for *light, fast, on-device*."**

### 3.1 Pose estimation — **MediaPipe Pose** ✅

**Why it fits:** runs in **real time on CPU**, gives **33 3D landmarks** for a
single person, is **pre-trained** (no training needed), and is **lightweight**
and easy to use from Python.

| Alternative | Why we didn't use it |
|-------------|----------------------|
| **OpenPose** | Very accurate and multi-person, but **needs a GPU** for real-time and is heavy to install — overkill for one user on a laptop. |
| **PoseNet / MoveNet** | Fast and browser-friendly, but fewer/weaker 3D landmarks; MediaPipe gives cleaner 3D points that pass straight into our Python math. |
| **YOLOv8-Pose** | Capable, but MediaPipe is more mature for **single-person fitness pose** and lighter on CPU. |
| **Wearable sensors / smartwatches** | Accurate, but need **extra hardware**; our goal is **webcam-only**. |

**One-liner for the viva:** *"MediaPipe gives me real-time 3D body points on a
CPU with zero hardware — the others are either GPU-hungry or need wearables."*

### 3.2 Form classification — **Random Forest (scikit-learn)** ✅

**Why it fits:** our input is just **3 numbers (joint angles)** per frame. Random
Forest is **fast, tiny (<10 MB), works on a CPU instantly, handles non-linear
thresholds, and is interpretable** (you can show which angle mattered).

| Alternative | Why we didn't use it (as the main scorer) |
|-------------|-------------------------------------------|
| **Deep neural network (CNN/LSTM)** | Needs **lots of data + a GPU**; overkill for 3 features. *(We do have a CNN+LSTM, but it's **off by default** — see §4.)* |
| **SVM / Logistic Regression** | Workable, but Random Forest handled the angle bands better and gives feature importance for free. |
| **Rules only** | Too rigid; Random Forest + rules together are more robust. |

**One-liner:** *"For three angle features, a Random Forest is the right-sized
tool — instant on CPU, small, and explainable. A deep net would be a hammer for
a thumbtack."*

### 3.3 Equipment detection — **YOLOv8-nano** ✅

**Why it fits:** **nano** is the **smallest, fastest** YOLOv8 (~**5.5 MB**,
~**70 ms** per frame on CPU). It hits our 1-per-second budget, and **fine-tuning
from COCO** lets us teach it gym gear with a small dataset.

| Alternative | Why we didn't use it |
|-------------|----------------------|
| **YOLOv8 s / m / l / x** | More accurate but **bigger and slower** — would blow the CPU budget. |
| **Faster R-CNN** | Two-stage and heavy; YOLO (single-stage) is much faster. |
| **SSD / MobileNet-SSD** | Light, but YOLOv8-nano is more accurate and far easier to fine-tune today. |

**One-liner:** *"Nano is the sweet spot of size, speed, and accuracy for
real-time detection on a CPU."*

### 3.4 The rest of the stack

- **FastAPI** (Python API): async + automatic data validation (Pydantic) + auto
  docs; faster and more modern than Flask/Django for this.
- **SQLite** (via Node/Express): a **file-based database** — no database server
  to install. It's the local stand-in for MySQL from the original proposal.
- **React + Vite**: fast, modern web UI with instant hot-reload during dev.
- **OpenCV**: reads/encodes camera frames.

---

## 4. Honest limitations (say these *before* the examiner finds them)

Examiners respect honesty + a mitigation plan more than a perfect-sounding
claim. Lead with these:

1. **The form models were trained on *synthetic* (computer-generated) data.**
   - *Mitigation / honesty:* we are upfront about it in the report; the rule
     engine (based on real exercise science, e.g. range-of-motion) anchors the
     score, and the synthetic models add a second opinion. Collecting a large
     real labelled dataset was out of scope for the FYP timeline.

2. **Rep counting is accurate to about ±1 rep.** On real bicep clips it matched
   the true count within ±1 every time; a live test counted 10 reps as 11 (one
   extra at the start). *Mitigation:* this is within the documented tolerance;
   the counter is smoothed and debounced.

3. **The experimental CNN+LSTM form scorer is turned OFF by default.** It was
   trained on synthetic sequences and has a **measured "sim-to-real" gap** (it
   recognises only some real good-form reps), so including it would *hurt* real
   scores. It's behind a flag for demos only. *(This is a strength: we measured
   it and made the honest call.)*

4. **Equipment detector is trained on a small dataset (≈387 images).** It detects
   **bench, dumbbell, and barbell well**; the rarer classes (pull-up bar,
   parallel bars, resistance band) need more training images to be reliable.

5. **Only three exercises** are supported by the AI (bicep curl, bent-over row,
   push-up). Other muscle groups show a clear "not supported yet" message rather
   than guessing.

---

## 5. Likely examiner questions + crisp answers

- **"Why no cloud / GPU?"** → Privacy (video never leaves the device), zero
  running cost, works offline, and it matches real edge-AI constraints. The whole
  design targets a normal laptop CPU.
- **"Why MediaPipe and not OpenPose?"** → Real-time on CPU with no GPU; OpenPose
  needs a GPU and is multi-person overkill for one user.
- **"How do you count a rep?"** → I smooth the elbow-angle signal and detect each
  significant dip-and-rise (a peak with enough "prominence"), with a minimum time
  gap so jitter doesn't double-count.
- **"How is the form score made?"** → A weighted blend of exercise rules and a
  Random Forest model (and an optional neural net), so it's robust.
- **"What's the latency?"** → Budget is ≤100 ms per frame; the live path measures
  ~25 ms per frame on this laptop, so it feels real-time.
- **"Is your data real?"** → The form models use synthetic data — we're explicit
  about that; the rule engine uses real exercise-science thresholds. *(Don't
  hide it.)*
- **"What happens if it can't see you?"** → It degrades gracefully: no pose →
  neutral "no pose detected", count held, never a fake score.
- **"How do you know it works?"** → 258 automated tests on Windows + Linux CI;
  plus the live demo.
- **"How would you improve it?"** → Collect a real labelled dataset, add more
  exercises, expand the equipment dataset, and (with a GPU) revisit the CNN+LSTM.

---

## 6. Key numbers to remember (cheat line)

- **3 exercises × 3 age groups = 9 form models**; each `.pkl` ≤ 10 MB.
- **33** MediaPipe body landmarks; **3** angles per exercise.
- Latency budget **≤100 ms/frame**; measured **~25 ms**. Memory **≤512 MB**.
- Equipment: **YOLOv8-nano**, ~**5.5 MB**, ~**70 ms/frame**, runs at **1 Hz**,
  **6** gym classes, overall **mAP50 ≈ 0.60**.
- Ports: **5173** (UI), **5001** (accounts), **8000** (AI).
- **258** automated tests passing.

---

## 7. Team & acknowledgements

- **Developer / Author:** Muhammad Huzaifa
- **External Supervisor & Contributor:** _<add name here>_ — provided guidance
  and review as the external supervisor on this project.

> To formally credit the supervisor in the codebase/GitHub, we can add a
> `CONTRIBUTORS.md` and/or co-author trailers on commits once their name and
> email are provided.

---

*This guide reflects the project as built (see `CLAUDE.md` for the engineering
charter and `docs/architecture_3pillar.md` for the detailed pipeline).*
