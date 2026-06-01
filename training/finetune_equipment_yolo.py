"""
training.finetune_equipment_yolo — fine-tune YOLOv8-nano on a gym-equipment
dataset so the equipment detector (``core/object_detector.py``) can actually
see dumbbells / kettlebells / barbells / benches / mats instead of the stock
COCO classes (person / bottle / chair / ...).

This is the VIVA-BLOCKING step recorded in ``TODO.md`` ("Equipment Detector —
Fine-tune YOLOv8-nano on Gym Dataset"). The runtime code is already complete and
swappable: once you produce a fine-tuned ``.pt`` and point the detector at it
(see the integration notes printed at the end of a run), equipment detection and
``GET /api/equipment`` start returning real gym classes — no code change needed.

REQUIRES A GPU. YOLO training on CPU is not viable — it can take days. A
multi-hour / 72-hour ETA almost always means you are on CPU: a Colab runtime left
on "CPU", or this 8 GB laptop (no dGPU). On a free Colab/Kaggle T4 GPU,
fine-tuning YOLOv8-nano is minutes-to-an-hour. The charter forbids cloud
*inference*, not cloud *training*, so a cloud GPU is allowed. This script prints a
loud warning when it cannot see a GPU.

Speed knobs (use ``--fast`` for a quick / CPU-tolerable run):
    --fast        preset: imgsz=320, epochs=40, freeze=10 (train the detection
                  head only), early-stop patience=10, RAM image cache. Much
                  faster, slightly lower ceiling — ideal for a first working
                  model. Individual flags below still override it.
    --freeze N    freeze the first N layers (10 ≈ the COCO backbone). Fine-tuning
                  only the head is far cheaper and works well for small datasets.
    --patience N  stop early after N epochs with no improvement.
    --imgsz / --epochs / --batch  override individually.

--------------------------------------------------------------------------------
Colab / Kaggle quickstart
--------------------------------------------------------------------------------
    # 1. ENABLE THE GPU FIRST (this is what fixes the multi-day training):
    #    Colab : Runtime -> Change runtime type -> Hardware accelerator -> T4 GPU
    #    Kaggle: Settings -> Accelerator -> GPU
    !pip install ultralytics==8.3.0 roboflow
    # 2a. Option A — let the script pull a Roboflow Universe dataset:
    !python -m training.finetune_equipment_yolo --fast \
        --roboflow-api-key YOUR_KEY --roboflow-workspace WORKSPACE \
        --roboflow-project PROJECT --roboflow-version 1
    # 2b. Option B — you already have an Ultralytics-format dataset (a data.yaml):
    !python -m training.finetune_equipment_yolo --fast --data /path/to/data.yaml

Finding a dataset: search Roboflow Universe (https://universe.roboflow.com) for
"gym equipment" / "workout equipment" / "dumbbell" — several public projects
export directly to the YOLOv8 format. Pick one whose classes match the equipment
you want to coach on; its "YOLOv8" export gives you the workspace/project/version
used above (or a downloadable folder with a ``data.yaml`` for Option B).

After training, download the produced ``equipment_yolov8n.pt`` and follow the
integration notes the script prints (drop it in ``models/`` and set the
detector's ``allowed_classes``).
"""
from __future__ import annotations

import argparse
import logging
import shutil
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
#: Where the runtime detector loads weights from (see core/object_detector.py
#: and scripts/fetch_yolo_weights.py). Printed in the integration notes.
_RUNTIME_WEIGHTS = _REPO_ROOT / "models" / "yolov8n.pt"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments (see the module docstring for usage)."""
    parser = argparse.ArgumentParser(
        description="Fine-tune YOLOv8-nano on a gym-equipment dataset.",
    )
    # --- dataset source (exactly one of: --data, or the Roboflow group) -------
    parser.add_argument(
        "--data",
        type=Path,
        default=None,
        help="Path to an Ultralytics dataset YAML (data.yaml). Use this if you "
             "already downloaded/exported a dataset.",
    )
    parser.add_argument("--roboflow-api-key", default=None)
    parser.add_argument("--roboflow-workspace", default=None)
    parser.add_argument("--roboflow-project", default=None)
    parser.add_argument("--roboflow-version", type=int, default=None)

    # --- training hyper-parameters -------------------------------------------
    parser.add_argument(
        "--base-weights",
        default="yolov8n.pt",
        help="Pretrained checkpoint to fine-tune from. Default 'yolov8n.pt' "
             "(ultralytics fetches the COCO-pretrained nano automatically).",
    )
    parser.add_argument(
        "--fast", action="store_true",
        help="Speed preset for a quick / CPU-tolerable run: imgsz=320, "
             "epochs=40, freeze=10 (head only), patience=10, RAM cache. "
             "Individual flags below still override.",
    )
    parser.add_argument("--epochs", type=int, default=None,
                        help="Default 60 (40 with --fast).")
    parser.add_argument("--imgsz", type=int, default=None,
                        help="Default 640 (320 with --fast).")
    parser.add_argument(
        "--freeze", type=int, default=None,
        help="Freeze the first N layers (transfer-learn the head only). "
             "Default 0 (10 with --fast); 10 ≈ the COCO backbone.",
    )
    parser.add_argument(
        "--patience", type=int, default=None,
        help="Early-stop after N epochs with no improvement. "
             "Default 25 (10 with --fast).",
    )
    parser.add_argument(
        "--cache", choices=("ram", "disk"), default=None,
        help="Cache images for faster epochs. --fast uses 'ram' unless set; "
             "skip on very low-RAM machines.",
    )
    parser.add_argument(
        "--batch", type=int, default=16,
        help="Batch size. Pass -1 for ultralytics auto-batch.",
    )
    parser.add_argument(
        "--device", default=None,
        help="Ultralytics device string: e.g. '0' (first GPU) or 'cpu'. "
             "Default: auto-select.",
    )
    parser.add_argument("--project", default="runs/equipment_finetune")
    parser.add_argument("--name", default="yolov8n_gym")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("equipment_yolov8n.pt"),
        help="Where to copy the best checkpoint after training. Default "
             "'equipment_yolov8n.pt' in the working directory.",
    )
    return parser.parse_args(argv)


def _apply_speed_preset(args: argparse.Namespace) -> None:
    """Fill unset speed knobs, applying the ``--fast`` preset.

    Explicit flags always win — only ``None`` (unset) values are filled. With
    ``--fast`` the cheaper option is chosen for each; otherwise the GPU-sized
    default is used.
    """
    fast = args.fast
    if args.epochs is None:
        args.epochs = 40 if fast else 60
    if args.imgsz is None:
        args.imgsz = 320 if fast else 640
    if args.freeze is None:
        args.freeze = 10 if fast else 0
    if args.patience is None:
        args.patience = 10 if fast else 25
    if args.cache is None and fast:
        args.cache = "ram"


def _warn_if_no_gpu(device: str | None) -> None:
    """Print a loud warning when training would fall back to CPU.

    A CPU run can take hours-to-days — the symptom that motivates ``--fast``.
    ``torch`` is pulled in transitively by ultralytics, so this check only runs
    inside a real training environment.
    """
    try:
        import torch  # type: ignore[import-not-found]
        cuda = bool(torch.cuda.is_available())
    except ImportError:
        cuda = False
    explicit_cpu = device is not None and str(device).lower() == "cpu"
    if cuda and not explicit_cpu:
        return
    logger.warning(
        "\n%s\nNO GPU IN USE — training on CPU can take many hours to days.\n"
        "Enable a GPU runtime and re-run (Colab: Runtime -> Change runtime type "
        "-> T4 GPU; Kaggle: Settings -> Accelerator -> GPU).\n"
        "If you must stay on CPU, --fast makes it as cheap as possible.\n%s",
        "!" * 70, "!" * 70,
    )


def resolve_dataset(args: argparse.Namespace) -> Path:
    """Return the path to a ``data.yaml``, downloading from Roboflow if asked.

    Two mutually exclusive sources:

    * ``--data`` — a local Ultralytics dataset YAML; returned as-is.
    * ``--roboflow-*`` — workspace/project/version + API key; the dataset is
      downloaded via the (lazily imported) ``roboflow`` package in YOLOv8 format
      and the path to its ``data.yaml`` is returned.

    Raises:
        ValueError:    If neither or both sources are specified, or the Roboflow
                       group is incomplete.
        FileNotFoundError: If ``--data`` is given but does not exist.
        ImportError:   If a Roboflow download is requested but the package is
                       not installed.
    """
    roboflow_args = (
        args.roboflow_api_key, args.roboflow_workspace,
        args.roboflow_project, args.roboflow_version,
    )
    using_roboflow = any(a is not None for a in roboflow_args)

    if args.data is not None and using_roboflow:
        raise ValueError("Pass either --data or the --roboflow-* group, not both.")
    if args.data is None and not using_roboflow:
        raise ValueError(
            "No dataset source. Pass --data path/to/data.yaml, or the "
            "--roboflow-api-key/--roboflow-workspace/--roboflow-project/"
            "--roboflow-version group."
        )

    if args.data is not None:
        if not args.data.exists():
            raise FileNotFoundError(f"--data not found: {args.data}")
        return args.data

    if not all(a is not None for a in roboflow_args):
        raise ValueError(
            "Incomplete Roboflow args — all of --roboflow-api-key, "
            "--roboflow-workspace, --roboflow-project, --roboflow-version "
            "are required together."
        )
    try:
        from roboflow import Roboflow  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "The 'roboflow' package is required for --roboflow-* downloads. "
            "Install via `pip install roboflow`."
        ) from exc

    logger.info("Downloading dataset from Roboflow (%s/%s v%s)...",
                args.roboflow_workspace, args.roboflow_project,
                args.roboflow_version)
    rf = Roboflow(api_key=args.roboflow_api_key)
    project = rf.workspace(args.roboflow_workspace).project(args.roboflow_project)
    dataset = project.version(args.roboflow_version).download("yolov8")
    data_yaml = Path(dataset.location) / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(
            f"Roboflow download did not produce a data.yaml at {data_yaml}"
        )
    return data_yaml


def finetune(args: argparse.Namespace, data_yaml: Path) -> Path:
    """Fine-tune the base checkpoint and return the path to ``best.pt``.

    Args:
        args:      Parsed CLI args (hyper-parameters + output config).
        data_yaml: Ultralytics dataset YAML resolved by :func:`resolve_dataset`.

    Returns:
        Path to the best checkpoint produced by training.
    """
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "ultralytics is not installed. Install via "
            "`pip install ultralytics==8.3.0`."
        ) from exc

    _warn_if_no_gpu(args.device)
    logger.info(
        "Fine-tuning %s on %s — epochs=%d imgsz=%d batch=%d freeze=%d "
        "patience=%d cache=%s",
        args.base_weights, data_yaml, args.epochs, args.imgsz, args.batch,
        args.freeze, args.patience, args.cache or False,
    )
    model = YOLO(args.base_weights)  # transfer-learns from the pretrained backbone
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        freeze=args.freeze,
        patience=args.patience,
        cache=(args.cache if args.cache else False),
        project=args.project,
        name=args.name,
    )

    # ultralytics records the best checkpoint path on the trainer after training.
    best = getattr(getattr(model, "trainer", None), "best", None)
    best_path = Path(best) if best else Path(args.project) / args.name / "weights" / "best.pt"
    if not best_path.exists():
        raise FileNotFoundError(
            f"Training finished but best.pt was not found at {best_path}. "
            "Check the ultralytics run directory."
        )
    return best_path


def summarize(best_path: Path, output: Path) -> None:
    """Copy the best checkpoint to *output* and print integration notes.

    Loads the trained model purely to read its class table so the exact
    ``allowed_classes`` tuple can be printed — that is the one value the runtime
    integration needs that only training can supply.
    """
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, output)
    logger.info("Copied best checkpoint -> %s", output)

    class_names: tuple[str, ...] = ()
    try:
        from ultralytics import YOLO  # type: ignore[import-not-found]
        names = getattr(YOLO(str(output)), "names", {}) or {}
        class_names = tuple(str(v) for v in names.values())
    except (ImportError, OSError, RuntimeError) as exc:
        logger.warning("Could not read class names from the model (%s); "
                       "inspect data.yaml's 'names:' instead.", exc)

    allowlist = ", ".join(f'"{c}"' for c in class_names) or '"dumbbell", "..."'
    logger.info(
        "\n%s\nINTEGRATION — make the runtime use these weights\n%s\n"
        "Trained classes: %s\n\n"
        "1. Copy %s to the repo at:\n"
        "       %s\n"
        "   (or keep its own name and pass weights_path=... to "
        "Yolov8NanoDetector).\n\n"
        "2. In api_backend.py lifespan, set the allowlist so /api/equipment "
        "only forwards gym classes:\n"
        "       app.state.equipment_detector = EquipmentDetector(\n"
        "           Yolov8NanoDetector(weights_path=YOLO_WEIGHTS_PATH),\n"
        "           allowed_classes=(%s),\n"
        "       )\n\n"
        "3. (Optional) Update EquipmentDetection's docstring in core/schemas.py "
        "so the documented label set matches the trained classes.\n%s",
        "=" * 70, "=" * 70,
        list(class_names) or "(unknown — read from model/data.yaml)",
        output, _RUNTIME_WEIGHTS, allowlist, "=" * 70,
    )


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    _apply_speed_preset(args)
    try:
        data_yaml = resolve_dataset(args)
        best_path = finetune(args, data_yaml)
        summarize(best_path, args.output)
    except (ValueError, FileNotFoundError, ImportError) as exc:
        logger.error("%s", exc)
        return 1
    logger.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
