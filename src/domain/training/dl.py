import logging
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from ignite.engine import Events
from ignite.metrics import Average

from src.engine.dl.ignite_builder import EngineBuilder
from src.engine.dl.builders import (
    create_dataloader,
    create_dataset,
    create_loss,
    create_optimizer,
    create_scheduler,
)
from src.engine.dl.engine import eval_step, train_step
from src.engine.dl.infer import df_to_tensors, run_model
from src.engine.dl.model.checkpoint import load_best_checkpoint
from src.engine.dl.model import DLClassifierFactory

logger = logging.getLogger(__name__)


def _create_model(name: str, params: dict, device: torch.device) -> nn.Module:
    """Instantiate a DL classifier via factory and move to `device`."""
    return DLClassifierFactory.create(name, params).to(device)


def _make_loader(df, num_cols, cat_cols, label_col, dataloader_cfg):
    return create_dataloader(
        create_dataset(df, num_cols, cat_cols, label_col=[label_col]), dataloader_cfg
    )


def _build_trainer(model, loss_fn, optimizer, scheduler, device, max_grad_norm):
    """Engine that trains for one epoch and collects per-step loss into history."""
    builder = (
        EngineBuilder(train_step)
        .with_state(
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_history(output_transform=lambda x: {"loss": x["loss"]})
    )
    return builder.build(), builder.history


def _build_validator(model, loss_fn, device, trainer, patience, min_delta, models_path):
    """Validator engine with early stopping + best-loss checkpointing."""
    return (
        EngineBuilder(eval_step)
        .with_state(model=model, loss_fn=loss_fn, device=device)
        .with_metric("loss", Average(output_transform=lambda x: x["loss"]))
        .with_early_stopping(
            trainer=trainer, metric="loss", patience=patience, min_delta=min_delta
        )
        .with_checkpointing(
            trainer=trainer,
            checkpoint_dir=models_path,
            objects_to_save={"model": model},
            metric="loss",
        )
        .build()
    )


def fit_classifier(
    name: str,
    params: dict,
    X: pd.DataFrame,
    y: object = None,
    *,
    X_val: pd.DataFrame | None = None,
    y_val: object = None,
    context: dict | None = None,
) -> tuple[nn.Module, dict]:
    """Fit a DL classifier and return ``(model, fit_summary)``.

    `X` and `X_val` are DataFrames; the label column is read internally via
    ``context['label_col']``. `y`/`y_val` are accepted (interface parity with
    ML) but ignored.

    `context` is required and must contain:
      device, df_meta, num_cols, cat_cols, label_col,
      loss_cfg, optimizer_cfg, scheduler_cfg, loops_cfg, models_path.

    Returns ``(model, {"history": {scalar_name: [values_per_step, ...]}})``.
    """
    if context is None:
        raise ValueError("DL fit_classifier requires `context`.")
    if X_val is None:
        raise ValueError("DL fit_classifier requires `X_val`.")

    device = context["device"]
    df_meta = context["df_meta"]
    num_cols = context["num_cols"]
    cat_cols = context["cat_cols"]
    label_col = context["label_col"]
    loss_cfg = context["loss_cfg"]
    optimizer_cfg = context["optimizer_cfg"]
    scheduler_cfg = context["scheduler_cfg"]
    loops_cfg = context["loops_cfg"]
    models_path = Path(context["models_path"])

    loss_params = dict(loss_cfg.params)
    loss_params.setdefault("class_weight", df_meta["class_weights"])

    model = _create_model(name, params, device)
    loss_fn = create_loss(loss_cfg.name, loss_params, device)

    train_loader = _make_loader(
        X, num_cols, cat_cols, label_col, loops_cfg.training.dataloader
    )
    val_loader = _make_loader(
        X_val, num_cols, cat_cols, label_col, loops_cfg.validation.dataloader
    )

    optimizer = create_optimizer(
        optimizer_cfg.name, optimizer_cfg.params, model, loss_fn=loss_fn
    )
    scheduler = create_scheduler(
        scheduler_cfg.name, scheduler_cfg.params, optimizer, train_loader
    )

    trainer, history = _build_trainer(
        model,
        loss_fn,
        optimizer,
        scheduler,
        device,
        loops_cfg.training.max_grad_norm,
    )
    validator = _build_validator(
        model,
        loss_fn,
        device,
        trainer,
        loops_cfg.training.early_stopping.patience,
        loops_cfg.training.early_stopping.min_delta,
        models_path,
    )

    @trainer.on(Events.EPOCH_COMPLETED)
    def _run_validation(engine):
        logger.info(
            "Epoch [%d] Train Loss: %.6f",
            engine.state.epoch,
            engine.state.metrics["loss"],
        )
        validator.run(val_loader)
        logger.info(
            "Epoch [%d] Val Loss: %.6f",
            engine.state.epoch,
            validator.state.metrics["loss"],
        )

    trainer.run(train_loader, max_epochs=loops_cfg.training.epochs)

    load_best_checkpoint(models_path, model, device)
    logger.info("Best checkpoint reloaded after training.")

    return model, {"history": history}


def grid_search_classifier(*args, **kwargs):
    """DL grid search is not implemented in this PR."""
    raise NotImplementedError(
        "DL grid search is not implemented. Train a single configuration instead."
    )


def predict_with_proba(
    model: nn.Module,
    X: pd.DataFrame,
    *,
    context: dict | None = None,
    return_embedding: bool = False,
) -> tuple:
    """Return ``(y_pred, y_proba)`` for a DL model on a DataFrame.

    With ``return_embedding=True`` also returns the latent embedding ``z`` from
    the same forward pass (``None`` when the model exposes none).
    `context` must contain ``device``, ``num_cols``, ``cat_cols``.
    """
    if context is None:
        raise ValueError("DL predict_with_proba requires `context`.")
    device = context["device"]
    num_cols = context["num_cols"]
    cat_cols = context["cat_cols"]

    inputs = df_to_tensors(
        X,
        [num_cols, cat_cols],
        dtypes=[torch.float32, torch.long],
    )
    output = run_model(model, inputs, device)
    probs = F.softmax(output["logits"].cpu(), dim=1)
    y_pred = probs.argmax(dim=1).numpy()
    y_proba = probs.numpy()
    if return_embedding:
        z = output["z"].cpu().numpy() if "z" in output else None
        return y_pred, y_proba, z
    return y_pred, y_proba


def save_model(
    model: nn.Module,
    path: Path,
    *,
    name: str = "",
    params: dict | None = None,
    suffix: str = "",
) -> None:
    """Save state dict + metadata to ``path / f'model{suffix}.pt'``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"state_dict": model.state_dict(), "name": name, "params": params or {}},
        path / f"model{suffix}.pt",
    )


def load_model(
    path: Path, *, context: dict | None = None, suffix: str = ""
) -> nn.Module:
    """Load model from ``path / f'model{suffix}.pt'``. `context` must provide ``device``."""
    if context is None:
        raise ValueError("DL load_model requires `context` with `device`.")
    device = context["device"]
    ckpt = torch.load(
        Path(path) / f"model{suffix}.pt", map_location="cpu", weights_only=True
    )
    model = _create_model(ckpt["name"], ckpt["params"], device)
    model.load_state_dict(ckpt["state_dict"])
    return model
