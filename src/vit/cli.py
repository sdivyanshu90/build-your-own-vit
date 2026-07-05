"""Command-line interface.

A single ``vit`` entry point with subcommands covering the full lifecycle:
train, evaluate, predict, export, and serve — plus ``init-config`` to scaffold a
config file and ``info`` for environment diagnostics.

Examples:
    vit info
    vit init-config --preset vit_tiny --dataset cifar10 -o configs/my.yaml
    vit train --config configs/vit_tiny_cifar10.yaml --smoke
    vit evaluate --config configs/vit_tiny_cifar10.yaml --checkpoint outputs/best.pt
    vit predict --checkpoint outputs/best.pt --image cat.jpg --top-k 3
    vit export --checkpoint outputs/best.pt --output model.onnx
    vit serve --checkpoint outputs/best.pt --port 8080
"""

from __future__ import annotations

import argparse
import sys
from typing import Any

import yaml

from vit import __version__
from vit.utils.logging import configure_logging, get_logger

logger = get_logger("vit.cli")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """Construct the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="vit",
        description="Build Your Own ViT — train, evaluate, and serve a Vision Transformer.",
    )
    parser.add_argument("--version", action="version", version=f"vit {__version__}")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO).")
    parser.add_argument("--json-logs", action="store_true", help="Emit structured JSON logs.")
    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train a model from a config file.")
    p_train.add_argument("--config", "-c", required=True, help="Path to a YAML config.")
    p_train.add_argument("--resume", help="Checkpoint to resume from.")
    p_train.add_argument("--device", help="Override the compute device.")
    p_train.add_argument("--epochs", type=int, help="Override number of epochs.")
    p_train.add_argument(
        "--smoke",
        action="store_true",
        help="Fast synthetic run (few steps, no download) for CI/demo.",
    )
    p_train.add_argument(
        "--set",
        nargs="*",
        default=[],
        metavar="KEY=VALUE",
        help="Dotted config overrides, e.g. --set train.epochs=5 optim.lr=1e-3",
    )
    p_train.set_defaults(func=_cmd_train)

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate a checkpoint on a split.")
    p_eval.add_argument("--config", "-c", required=True)
    p_eval.add_argument("--checkpoint", "-k", required=True)
    p_eval.add_argument("--split", choices=["val", "test"], default="test", help="Which split.")
    p_eval.add_argument("--device", help="Override the compute device.")
    p_eval.set_defaults(func=_cmd_evaluate)

    # predict
    p_pred = sub.add_parser("predict", help="Classify an image with a checkpoint.")
    p_pred.add_argument("--checkpoint", "-k", required=True)
    p_pred.add_argument("--image", "-i", required=True)
    p_pred.add_argument("--top-k", type=int, default=5)
    p_pred.add_argument("--device", default="auto")
    p_pred.add_argument("--no-ema", action="store_true", help="Ignore EMA weights.")
    p_pred.set_defaults(func=_cmd_predict)

    # export
    p_exp = sub.add_parser("export", help="Export a checkpoint to ONNX.")
    p_exp.add_argument("--checkpoint", "-k", required=True)
    p_exp.add_argument("--output", "-o", required=True)
    p_exp.add_argument("--opset", type=int, default=17)
    p_exp.add_argument("--no-ema", action="store_true")
    p_exp.set_defaults(func=_cmd_export)

    # serve
    p_srv = sub.add_parser("serve", help="Run the inference HTTP API.")
    p_srv.add_argument("--checkpoint", "-k", help="Checkpoint to serve.")
    p_srv.add_argument("--host", default=None)
    p_srv.add_argument("--port", type=int, default=None)
    p_srv.add_argument("--device", default=None)
    p_srv.set_defaults(func=_cmd_serve)

    # init-config
    p_init = sub.add_parser("init-config", help="Generate a starter config file.")
    p_init.add_argument("--preset", default="vit_tiny")
    p_init.add_argument(
        "--dataset", default="cifar10", choices=["cifar10", "cifar100", "imagefolder", "fake"]
    )
    p_init.add_argument("--image-size", type=int, default=32)
    p_init.add_argument("--patch-size", type=int, default=4)
    p_init.add_argument("--num-classes", type=int, default=10)
    p_init.add_argument("--output", "-o", required=True)
    p_init.set_defaults(func=_cmd_init_config)

    # info
    p_info = sub.add_parser("info", help="Print environment and version info.")
    p_info.set_defaults(func=_cmd_info)

    return parser


def _parse_overrides(pairs: list[str]) -> dict[str, Any]:
    """Parse ``key=value`` strings into a dict, decoding values as YAML scalars."""
    overrides: dict[str, Any] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Invalid --set entry {pair!r}; expected KEY=VALUE.")
        key, _, raw = pair.partition("=")
        overrides[key.strip()] = _coerce_scalar(raw)
    return overrides


def _coerce_scalar(raw: str) -> Any:
    """Decode a CLI value, tolerating YAML 1.1's quirk with ``1e-3``-style floats.

    ``yaml.safe_load`` parses ``1e-3`` (exponent without a sign) as a string, so
    we retry an explicit float conversion before falling back to the raw value.
    """
    value = yaml.safe_load(raw)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return value
    return value


# ---------------------------------------------------------------------------
# Command implementations (each returns a process exit code)
# ---------------------------------------------------------------------------
def _cmd_train(args: argparse.Namespace) -> int:
    from vit.config import load_config
    from vit.data import DataModule
    from vit.training import Trainer

    overrides = _parse_overrides(args.set)
    if args.epochs is not None:
        overrides["train.epochs"] = args.epochs
    if args.device:
        overrides["train.device"] = args.device
    if args.smoke:
        # Synthetic, tiny, fast: no downloads, a handful of steps.
        overrides.update(
            {
                "data.dataset": "fake",
                "data.num_workers": 0,
                "train.epochs": 1,
                "train.precision": "fp32",
                "train.compile_model": False,
            }
        )
    config = load_config(args.config, **overrides)
    logger.info("Loaded config", extra={"run_name": config.run_name})

    datamodule = DataModule(config.data, seed=config.train.seed)
    trainer = Trainer(config, datamodule, progress=not args.smoke)
    if args.resume:
        trainer.resume(args.resume)
    state = trainer.fit(max_steps=4 if args.smoke else None)
    best = trainer.output_dir / "best.pt"
    logger.info(
        "Training finished",
        extra={"best_metric": state.best_metric, "checkpoint": str(best)},
    )
    print(f"Best {config.train.checkpoint_metric}: {state.best_metric:.4f}")
    print(f"Checkpoints written to: {trainer.output_dir}")
    return 0


def _cmd_evaluate(args: argparse.Namespace) -> int:
    from vit.config import load_config
    from vit.data import DataModule
    from vit.training import Trainer

    overrides = {"train.device": args.device} if args.device else {}
    config = load_config(args.config, **overrides)
    datamodule = DataModule(config.data, seed=config.train.seed)
    trainer = Trainer(config, datamodule, progress=False)
    trainer.resume(args.checkpoint)
    loader = datamodule.test_dataloader() if args.split == "test" else datamodule.val_dataloader()
    if loader is None:
        print(f"No '{args.split}' split available for this dataset.", file=sys.stderr)
        return 2
    metrics = trainer.evaluate(loader, use_ema=trainer.ema is not None)
    for key, value in metrics.items():
        print(f"{key}: {value:.4f}")
    return 0


def _cmd_predict(args: argparse.Namespace) -> int:
    from PIL import Image

    from vit.inference import Predictor

    predictor = Predictor.from_checkpoint(
        args.checkpoint, device=args.device, use_ema=not args.no_ema
    )
    image = Image.open(args.image)
    results = predictor.predict(image, top_k=args.top_k)[0]
    print(f"Predictions for {args.image}:")
    for rank, pred in enumerate(results, start=1):
        print(f"  {rank}. {pred.label:<20} {pred.probability * 100:6.2f}%")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    from vit.inference import export_checkpoint_to_onnx

    out = export_checkpoint_to_onnx(
        args.checkpoint, args.output, use_ema=not args.no_ema, opset=args.opset
    )
    print(f"Exported ONNX model to: {out}")
    return 0


def _cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from vit.serving.settings import ServingSettings

    overrides: dict[str, Any] = {}
    if args.checkpoint:
        overrides["model_checkpoint"] = args.checkpoint
    if args.host:
        overrides["host"] = args.host
    if args.port:
        overrides["port"] = args.port
    if args.device:
        overrides["device"] = args.device
    settings = ServingSettings(**overrides)
    logger.info("Starting server", extra={"host": settings.host, "port": settings.port})
    # Import string keeps reload-friendliness; factory reads the same env/settings.
    uvicorn.run(
        "vit.serving.app:create_app",
        factory=True,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )
    return 0


def _cmd_init_config(args: argparse.Namespace) -> int:
    from vit.config import (
        Config,
        DataConfig,
        ModelConfig,
        save_config,
    )

    model = ModelConfig.from_preset(
        args.preset,
        num_classes=args.num_classes,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    data = DataConfig(dataset=args.dataset, image_size=args.image_size)
    config = Config(run_name=f"{args.preset}-{args.dataset}", model=model, data=data)
    path = save_config(config, args.output)
    print(f"Wrote config to: {path}")
    return 0


def _cmd_info(args: argparse.Namespace) -> int:
    import torch

    from vit.config import PRESETS
    from vit.utils.device import resolve_device

    print(f"vit version:      {__version__}")
    print(f"torch version:    {torch.__version__}")
    print(f"CUDA available:   {torch.cuda.is_available()}")
    print(f"resolved device:  {resolve_device('auto')}")
    print(f"available presets: {', '.join(sorted(PRESETS))}")
    return 0


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    configure_logging(args.log_level, json_logs=args.json_logs)
    try:
        exit_code: int = args.func(args)
        return exit_code
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Command failed", extra={"error": str(exc)})
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
