import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { ArrowLeft, Camera, Play, Square, AlertCircle, CheckCircle, XCircle, Wifi, WifiOff } from 'lucide-react';

const ML_API = 'http://localhost:8000';

// Live overlay cadence. The backend answers in ~25ms (measured from the live
// trace), so the old 150ms (~6.7fps) was the bottleneck, not latency; 80ms
// (~12fps) roughly halves the visible skeleton lag and stays within the
// ≤100ms/frame budget. The isProcessingRef guard drops a tick if a frame ever
// overruns, so this never pipelines overlapping requests.
const FRAME_INTERVAL_MS = 80;

// MediaPipe Pose connections — upper + lower body skeleton
const POSE_CONNECTIONS: [number, number][] = [
  [11, 12], // shoulders
  [11, 13], [13, 15], // left arm
  [12, 14], [14, 16], // right arm
  [11, 23], [12, 24], // torso sides
  [23, 24], // hips
  [23, 25], [25, 27], [27, 29], [27, 31], // left leg
  [24, 26], [26, 28], [28, 30], [28, 32], // right leg
];

type MLExerciseType = 'bicep_curl' | 'bent_over_row' | 'push_up';

/**
 * Resolve the long-form ML exercise key the FastAPI service expects.
 *
 * The exercise's `muscleGroup` is AUTHORITATIVE (project charter §11's
 * body-part → exercise map: Biceps → bicep_curl, Back → bent_over_row,
 * Chest → push_up). Muscle groups without an ML model (Triceps, Shoulders,
 * Legs, Abdominal) return null, which the caller treats as a terminal
 * "not supported yet" state.
 *
 * Name matching is only a fallback for callers that don't pass a muscleGroup;
 * it is deliberately NOT used when one is given, because exercise names leak
 * across muscle groups — e.g. a triceps "Pushdown" contains "push", and
 * "Bench Dips" / "Close-Grip Bench Press" contain "bench", so name matching
 * silently misroutes them to push_up.
 */
function toMLExerciseType(exercise: string, muscleGroup?: string): MLExerciseType | null {
  if (muscleGroup) {
    switch (muscleGroup.toLowerCase()) {
      case 'biceps': return 'bicep_curl';
      case 'back':   return 'bent_over_row';
      case 'chest':  return 'push_up';
      default:       return null; // triceps, shoulders, legs, abdominal — no model
    }
  }
  const name = exercise.toLowerCase();
  if (name.includes('bicep') || name.includes('curl')) return 'bicep_curl';
  if (name.includes('back') || name.includes('row') || name.includes('pull')) return 'bent_over_row';
  if (name.includes('chest') || name.includes('push') || name.includes('bench')) return 'push_up';
  return null;
}

// Derive feedback severity from rep_quality score and message text
function classifyFeedback(repQuality: number, message: string): FeedbackType {
  const lower = message.toLowerCase();
  if (lower.includes('good') || lower.includes('perfect') || lower.includes('excellent') || lower.includes('great')) return 'good';
  if (lower.includes('wrong') || lower.includes('incorrect') || lower.includes('bad')) return 'error';
  if (repQuality >= 75) return 'good';
  if (repQuality >= 50) return 'warning';
  return 'error';
}

type FeedbackType = 'good' | 'warning' | 'error';

interface AIFeedback {
  type: FeedbackType;
  message: string;
  timestamp: number;
}

interface PoseLandmark {
  x: number;
  y: number;
  z?: number;
  visibility: number;
}

interface MLResponse {
  success: boolean;
  counter: number;
  feedback: string;
  stage?: string;
  primary_angle: number;
  secondary_angle?: number;
  rep_quality: number | null;   // live path: null until the first rep closes
  confidence?: number;
  detected_exercise?: string;
  is_valid_rep?: boolean;
  model_missing?: boolean;
  degraded?: boolean;
  session_id: string;
  latency_ms: number;
  landmarks?: PoseLandmark[];
}

interface CameraWorkoutProps {
  exercise: string;
  muscleGroup?: string;   // authoritative source for ML routing (see toMLExerciseType)
  userAge?: number;
  onComplete: (session: { exerciseName: string; reps: number; accuracy: number; mistakes: string[] }) => void;
  onBack: () => void;
}

export function CameraWorkout({ exercise, muscleGroup, userAge = 25, onComplete, onBack }: CameraWorkoutProps) {
  const [isWorkoutActive, setIsWorkoutActive] = useState(false);
  const [countdown, setCountdown] = useState(3);
  const [showCountdown, setShowCountdown] = useState(false);
  const [repsCount, setRepsCount] = useState(0);
  const [currentFeedback, setCurrentFeedback] = useState<AIFeedback[]>([]);
  const [mistakes, setMistakes] = useState<string[]>([]);
  const [cameraPermission, setCameraPermission] = useState<'granted' | 'denied' | 'prompt'>('prompt');
  const [formQuality, setFormQuality] = useState<number | null>(null);
  const [primaryAngle, setPrimaryAngle] = useState<number | null>(null);
  const [mlOnline, setMlOnline] = useState<boolean | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);

  const mlExerciseType = useMemo<MLExerciseType | null>(
    () => toMLExerciseType(exercise, muscleGroup),
    [exercise, muscleGroup],
  );

  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);       // pose overlay
  const captureCanvasRef = useRef<HTMLCanvasElement>(null); // hidden, for frame grabbing
  const streamRef = useRef<MediaStream | null>(null);
  const frameLoopRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const qualityHistoryRef = useRef<number[]>([]);
  const mistakesSetRef = useRef<Set<string>>(new Set());
  const lastFeedbackRef = useRef<string | null>(null); // dedupe held/neutral feedback
  const isProcessingRef = useRef(false); // prevent overlapping API calls

  // Check ML backend health on mount
  useEffect(() => {
    fetch(`${ML_API}/health`)
      .then(r => setMlOnline(r.ok))
      .catch(() => setMlOnline(false));
  }, []);

  // ── Camera helpers ──────────────────────────────────────────────────────────

  const requestCameraAccess = async () => {
    try {
      if (!navigator.mediaDevices?.getUserMedia) { setCameraPermission('denied'); return; }
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: 640, height: 480 } });
      streamRef.current = stream;
      if (videoRef.current) videoRef.current.srcObject = stream;
      setCameraPermission('granted');
    } catch {
      setCameraPermission('denied');
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach(t => t.stop());
    streamRef.current = null;
  };

  useEffect(() => {
    requestCameraAccess();
    return () => stopCamera();
  }, []);

  // ── Frame capture ───────────────────────────────────────────────────────────

  const captureFrame = (): string | null => {
    const video = videoRef.current;
    const canvas = captureCanvasRef.current;
    if (!video || !canvas || video.readyState < 2) return null;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    return canvas.toDataURL('image/jpeg', 0.7).split(',')[1]; // base64 only
  };

  // ── Pose landmark drawing ───────────────────────────────────────────────────

  const drawLandmarks = (landmarks: PoseLandmark[]) => {
    const canvas = canvasRef.current;
    const video = videoRef.current;
    if (!canvas || !video) return;
    canvas.width = video.clientWidth;
    canvas.height = video.clientHeight;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Skeleton lines
    ctx.lineWidth = 2;
    ctx.strokeStyle = '#a855f7';
    POSE_CONNECTIONS.forEach(([i, j]) => {
      const a = landmarks[i];
      const b = landmarks[j];
      if (!a || !b || a.visibility < 0.3 || b.visibility < 0.3) return;
      ctx.beginPath();
      ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
      ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
      ctx.stroke();
    });

    // Joint dots — cyan for key joints, purple for the rest
    const KEY_JOINTS = new Set([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]);
    landmarks.forEach((lm, i) => {
      if (lm.visibility < 0.3) return;
      ctx.fillStyle = KEY_JOINTS.has(i) ? '#06b6d4' : '#a855f7';
      ctx.beginPath();
      ctx.arc(lm.x * canvas.width, lm.y * canvas.height, KEY_JOINTS.has(i) ? 6 : 4, 0, 2 * Math.PI);
      ctx.fill();
    });
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    if (canvas) canvas.getContext('2d')?.clearRect(0, 0, canvas.width, canvas.height);
  };

  // ── ML session management ───────────────────────────────────────────────────

  const startMLSession = async (): Promise<string | null> => {
    if (mlExerciseType === null) return null;
    try {
      const form = new FormData();
      form.append('user_id', 'local_user');
      form.append('exercise_type', mlExerciseType);
      form.append('age', String(userAge));
      const res = await fetch(`${ML_API}/api/start-session`, { method: 'POST', body: form });
      const data = await res.json();
      return data.session_id ?? null;
    } catch { return null; }
  };

  const endMLSession = async (sid: string) => {
    try {
      const form = new FormData();
      form.append('session_id', sid);
      await fetch(`${ML_API}/api/end-session`, { method: 'POST', body: form });
    } catch { /* ignore */ }
  };

  // ── Per-frame processing ────────────────────────────────────────────────────

  const processFrame = useCallback(async (sid: string | null) => {
    if (mlExerciseType === null) return;
    if (isProcessingRef.current) return; // skip if previous call still in flight
    isProcessingRef.current = true;

    try {
      const frameBase64 = captureFrame();
      if (!frameBase64) return;

      const body: Record<string, unknown> = {
        frame_data: frameBase64,
        exercise_type: mlExerciseType,
        age: userAge,
        auto_detect: false,
      };
      if (sid) body.session_id = sid;

      const res = await fetch(`${ML_API}/api/live-frame`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });

      if (!res.ok) { setMlOnline(false); return; }

      const data: MLResponse = await res.json();
      if (!data.success) return;

      setMlOnline(true);

      // Rep count (backend is authoritative)
      if (typeof data.counter === 'number') setRepsCount(data.counter);

      // Form metrics
      if (typeof data.rep_quality === 'number') {
        setFormQuality(Math.round(data.rep_quality));
        qualityHistoryRef.current.push(data.rep_quality);
      }
      if (typeof data.primary_angle === 'number') setPrimaryAngle(Math.round(data.primary_angle));

      // Feedback updates on rep close and holds between reps. The live backend
      // returns the held (or neutral pre-first-rep) message every frame, so
      // dedupe identical consecutive messages to keep the panel from filling
      // with repeats — this realises the per-rep cadence without spam.
      if (data.feedback && data.feedback !== lastFeedbackRef.current) {
        lastFeedbackRef.current = data.feedback;
        const quality = data.rep_quality ?? 75;
        const type = classifyFeedback(quality, data.feedback);
        setCurrentFeedback(prev => [...prev, { type, message: data.feedback, timestamp: Date.now() }].slice(-3));

        // Track unique error messages as mistakes
        if (type === 'error' && !mistakesSetRef.current.has(data.feedback)) {
          mistakesSetRef.current.add(data.feedback);
          setMistakes(Array.from(mistakesSetRef.current));
        }
      }

      // Draw real landmarks if the backend returns them
      if (Array.isArray(data.landmarks) && data.landmarks.length > 0) {
        drawLandmarks(data.landmarks);
      }
    } catch {
      setMlOnline(false);
    } finally {
      isProcessingRef.current = false;
    }
  }, [mlExerciseType, userAge]);

  // Start / stop frame loop when workout state changes
  useEffect(() => {
    if (!isWorkoutActive) {
      if (frameLoopRef.current) { clearInterval(frameLoopRef.current); frameLoopRef.current = null; }
      clearCanvas();
      return;
    }

    const sid = sessionId;
    frameLoopRef.current = setInterval(() => processFrame(sid), FRAME_INTERVAL_MS);

    return () => {
      if (frameLoopRef.current) { clearInterval(frameLoopRef.current); frameLoopRef.current = null; }
    };
  }, [isWorkoutActive, sessionId, processFrame, mlExerciseType]);

  // ── Workout controls ────────────────────────────────────────────────────────

  const handleStartWorkout = async () => {
    if (mlExerciseType === null) {
      // Selected exercise isn't in the ML service's supported set;
      // the camera-area UX surfaces this — don't try to start.
      return;
    }
    setShowCountdown(true);
    setCountdown(3);
    setRepsCount(0);
    setCurrentFeedback([]);
    setMistakes([]);
    setFormQuality(null);
    setPrimaryAngle(null);
    qualityHistoryRef.current = [];
    mistakesSetRef.current = new Set();
    lastFeedbackRef.current = null;

    const sid = await startMLSession();
    setSessionId(sid);

    const countdownInterval = setInterval(() => {
      setCountdown(prev => {
        if (prev <= 1) {
          clearInterval(countdownInterval);
          setShowCountdown(false);
          setIsWorkoutActive(true);
          return 0;
        }
        return prev - 1;
      });
    }, 1000);
  };

  const handleStopWorkout = async () => {
    setIsWorkoutActive(false);
    if (sessionId) await endMLSession(sessionId);

    const qualities = qualityHistoryRef.current;
    const avgQuality = qualities.length > 0
      ? Math.round(qualities.reduce((a, b) => a + b, 0) / qualities.length)
      : 0;

    onComplete({
      exerciseName: exercise,
      reps: repsCount,
      accuracy: avgQuality,
      mistakes: Array.from(mistakesSetRef.current),
    });
  };

  // ── UI helpers ──────────────────────────────────────────────────────────────

  const getFeedbackIcon = (type: FeedbackType) => {
    switch (type) {
      case 'good':    return <CheckCircle className="w-5 h-5 text-green-400" />;
      case 'warning': return <AlertCircle className="w-5 h-5 text-yellow-400" />;
      case 'error':   return <XCircle className="w-5 h-5 text-red-400" />;
    }
  };

  // ── Render ──────────────────────────────────────────────────────────────────

  return (
    <div className="min-h-screen p-4 md:p-8">
      <div className="max-w-7xl mx-auto">

        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-4">
            <button
              onClick={onBack}
              className="p-3 bg-white/10 backdrop-blur-lg rounded-xl border border-white/20 text-white hover:bg-white/20 transition-all"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>
            <div>
              <h1 className="text-white text-2xl">{exercise}</h1>
              <p className="text-gray-400 flex items-center gap-1 text-sm">
                {mlOnline === true  && <><Wifi    className="w-3 h-3 text-green-400" /><span className="text-green-400">ML Connected</span></>}
                {mlOnline === false && <><WifiOff className="w-3 h-3 text-red-400"   /><span className="text-red-400">ML Offline</span></>}
                {mlOnline === null  && <span>AI will guide your form</span>}
              </p>
            </div>
          </div>

          <div className="flex gap-3">
            {formQuality !== null && (
              <div className="bg-white/10 backdrop-blur-lg rounded-xl px-4 py-3 border border-white/20 text-center">
                <p className="text-gray-400 text-sm mb-1">Form Quality</p>
                <p className={`text-2xl font-bold ${formQuality >= 75 ? 'text-green-400' : formQuality >= 50 ? 'text-yellow-400' : 'text-red-400'}`}>
                  {formQuality}%
                </p>
              </div>
            )}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl px-6 py-3 border border-white/20">
              <p className="text-gray-400 mb-1">Reps Count</p>
              <p className="text-white text-3xl">{repsCount}</p>
            </div>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

          {/* Camera Feed */}
          <div className="lg:col-span-2">
            <div className="bg-black rounded-2xl overflow-hidden relative aspect-video border-2 border-white/20">

              {mlExerciseType === null ? (
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center max-w-md px-4">
                    <AlertCircle className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-400 mb-2">Not supported yet</p>
                    <p className="text-gray-500">
                      The ML coach currently supports bicep curls, bent-over rows, and
                      push-ups. Other exercises (legs, shoulders, triceps, abdominal)
                      are coming soon — pick one of the supported variants to start a
                      guided workout.
                    </p>
                  </div>
                </div>
              ) : cameraPermission === 'granted' ? (
                <>
                  <video
                    ref={videoRef}
                    autoPlay
                    playsInline
                    muted
                    className="w-full h-full object-cover"
                  />

                  {/* Hidden capture canvas */}
                  <canvas ref={captureCanvasRef} className="hidden" />

                  {/* Real pose landmark overlay */}
                  <canvas
                    ref={canvasRef}
                    className="absolute inset-0 w-full h-full pointer-events-none"
                  />

                  {/* Countdown overlay */}
                  {showCountdown && (
                    <div className="absolute inset-0 bg-black/70 flex items-center justify-center">
                      <div className="text-white text-9xl animate-pulse">{countdown}</div>
                    </div>
                  )}

                  {/* Live angle badge */}
                  {isWorkoutActive && primaryAngle !== null && (
                    <div className="absolute bottom-4 left-4 bg-black/60 backdrop-blur rounded-lg px-3 py-1">
                      <p className="text-white text-sm">
                        Angle: <span className="text-cyan-400 font-bold">{primaryAngle}°</span>
                      </p>
                    </div>
                  )}
                </>
              ) : cameraPermission === 'denied' ? (
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center">
                    <Camera className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                    <p className="text-gray-400 mb-2">Camera access denied</p>
                    <p className="text-gray-500">Please enable camera to use AI trainer</p>
                  </div>
                </div>
              ) : (
                <div className="w-full h-full flex items-center justify-center">
                  <div className="text-center">
                    <Camera className="w-16 h-16 text-gray-400 mx-auto mb-4 animate-pulse" />
                    <p className="text-gray-400">Requesting camera access...</p>
                  </div>
                </div>
              )}
            </div>

            {/* Control Buttons */}
            <div className="mt-4 flex gap-4">
              {!isWorkoutActive ? (
                <button
                  onClick={handleStartWorkout}
                  disabled={cameraPermission !== 'granted' || showCountdown || mlExerciseType === null}
                  className="flex-1 bg-green-500 text-white py-4 rounded-xl hover:bg-green-600 transition-all disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                >
                  <Play className="w-6 h-6" />
                  {showCountdown ? 'Starting...' : 'Start Workout'}
                </button>
              ) : (
                <button
                  onClick={handleStopWorkout}
                  className="flex-1 bg-red-500 text-white py-4 rounded-xl hover:bg-red-600 transition-all flex items-center justify-center gap-2"
                >
                  <Square className="w-6 h-6" />
                  Stop Workout
                </button>
              )}
            </div>
          </div>

          {/* AI Feedback Panel */}
          <div className="space-y-4">

            {/* Reference Video placeholder */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
              <h3 className="text-white mb-3 flex items-center gap-2">
                <Camera className="w-5 h-5 text-blue-400" />
                Reference Video
              </h3>
              <div className="bg-black rounded-lg aspect-video flex items-center justify-center">
                <div className="text-center">
                  <Play className="w-12 h-12 text-gray-500 mx-auto mb-2" />
                  <p className="text-gray-400">{exercise} Demo</p>
                </div>
              </div>
            </div>

            {/* Real-time AI Feedback */}
            <div className="bg-white/10 backdrop-blur-lg rounded-xl p-4 border border-white/20">
              <h3 className="text-white mb-3">AI Feedback</h3>
              <div className="space-y-2 max-h-48 overflow-y-auto">
                {currentFeedback.length === 0 ? (
                  <p className="text-gray-400">Start workout to receive AI feedback</p>
                ) : (
                  currentFeedback.map(feedback => (
                    <div
                      key={feedback.timestamp}
                      className={`p-3 rounded-lg flex items-start gap-3 ${
                        feedback.type === 'good'
                          ? 'bg-green-500/20 border border-green-500/30'
                          : feedback.type === 'warning'
                          ? 'bg-yellow-500/20 border border-yellow-500/30'
                          : 'bg-red-500/20 border border-red-500/30'
                      }`}
                    >
                      {getFeedbackIcon(feedback.type)}
                      <p className="text-white flex-1">{feedback.message}</p>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Mistakes Detected */}
            {mistakes.length > 0 && (
              <div className="bg-red-500/10 backdrop-blur-lg rounded-xl p-4 border border-red-500/30">
                <h3 className="text-red-400 mb-2">Mistakes to Correct</h3>
                <ul className="space-y-1">
                  {mistakes.map((mistake, i) => (
                    <li key={i} className="text-gray-300 flex items-center gap-2">
                      <span className="w-1.5 h-1.5 bg-red-400 rounded-full" />
                      {mistake}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
