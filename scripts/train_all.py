"""
Train all missing exercise models (bicep/back/chest × children/adult/senior)
- Skips models that already exist in data/models/
- Optionally runs video processing first (default: true)

Usage:
    python scripts/train_all.py               # process videos + train missing models
    python scripts/train_all.py --skip-video  # just train missing models
    python scripts/train_all.py --only back   # limit to one exercise
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "data" / "models"

EXERCISES = ["bicep", "back", "chest"]
AGES = ["children", "adult", "senior"]

MODEL_PATH = lambda ex, age: MODELS_DIR / f"{ex}_{age}.pkl"


def run(cmd: list[str]) -> int:
    print(f"\n[RUN] {' '.join(cmd)}")
    return subprocess.call(cmd, cwd=ROOT)


def main():
    parser = argparse.ArgumentParser(description="Train all missing models")
    parser.add_argument("--skip-video", action="store_true", help="Skip video processing")
    parser.add_argument("--only", choices=EXERCISES, help="Train only a specific exercise")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1) Optionally process videos to CSVs
    if not args.skip_video:
        print("[INFO] Processing videos to generate training CSVs...")
        ret = run([sys.executable, "video_pipeline.py"])  # scans videos/ and writes data/training/*.csv
        if ret != 0:
            print("[WARN] video_pipeline.py exited with errors. You can retry or run with --skip-video if CSVs exist.")

    # 2) Train missing models
    exercises = [args.only] if args.only else EXERCISES

    missing = []
    for ex in exercises:
        for age in AGES:
            target = MODEL_PATH(ex, age)
            if target.exists():
                print(f"[SKIP] {target.name} already exists")
                continue
            missing.append((ex, age, target))

    if not missing:
        print("[DONE] No missing models. All target .pkl files exist.")
        return

    print(f"[INFO] Training {len(missing)} missing models...")
    for ex, age, target in missing:
        print(f"[TRAIN] {ex}/{age} → {target.name}")
        ret = run([sys.executable, "train_universal.py", "--exercise", ex, "--age", age])
        if ret != 0:
            print(f"[ERROR] Training failed for {ex}/{age}. See logs above.")
        else:
            exists = target.exists()
            print(f"[OK] {'Created' if exists else 'Completed'}: {target.name}")

    print("\n[SUMMARY]")
    for ex in exercises:
        for age in AGES:
            target = MODEL_PATH(ex, age)
            status = "✅" if target.exists() else "❌"
            print(f"  {status} {target.name}")


if __name__ == "__main__":
    main()
