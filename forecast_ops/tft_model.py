from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import pandas as pd
import torch

from .data_utils import ForecastConfig


def train_tft(
    panel: pd.DataFrame,
    config: ForecastConfig,
    known_reals: Optional[Iterable[str]],
    observed_reals: Optional[Iterable[str]],
    outdir: Path,
) -> Tuple[pd.DataFrame, Path, Optional[pd.DataFrame]]:
    from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
    from pytorch_forecasting.data.encoders import GroupNormalizer
    from pytorch_forecasting.metrics import QuantileLoss
    from lightning.pytorch import Trainer, seed_everything
    from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

    seed_everything(config.seed, workers=True)

    known_reals = [c for c in (known_reals or []) if c in panel.columns]
    observed_reals = [c for c in (observed_reals or []) if c in panel.columns if c != "y"]

    panel = panel.copy()
    max_idx_map = panel.groupby("series_id")["time_idx"].max().to_dict()
    panel["max_time_idx"] = panel["series_id"].map(max_idx_map)

    train_mask = panel["time_idx"] <= panel["max_time_idx"] - config.val_size
    val_support_mask = panel["time_idx"] >= panel["max_time_idx"] - (config.val_size + config.max_encoder_len)

    train_df = panel[train_mask].copy()
    val_df = panel[val_support_mask].copy()
    if train_df.empty:
        raise RuntimeError("Training dataframe is empty after applying validation split; reduce val_size or max_encoder_len")

    train_dataset = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="y",
        group_ids=["series_id"],
        max_encoder_length=config.max_encoder_len,
        max_prediction_length=config.horizon,
        time_varying_known_reals=["time_idx"] + known_reals,
        time_varying_unknown_reals=["y"] + observed_reals,
        static_categoricals=["series_id"],
        allow_missing_timesteps=True,
        add_relative_time_idx=True,
        add_target_scales=True,
        add_encoder_length=True,
        target_normalizer=GroupNormalizer(groups=["series_id"]),
    )

    val_dataset = TimeSeriesDataSet.from_dataset(train_dataset, val_df, predict=True, stop_randomization=True)

    train_loader = train_dataset.to_dataloader(train=True, batch_size=config.batch_size, num_workers=config.num_workers)
    val_loader = val_dataset.to_dataloader(train=False, batch_size=config.batch_size, num_workers=config.num_workers)

    tft = TemporalFusionTransformer.from_dataset(
        train_dataset,
        learning_rate=config.learning_rate,
        hidden_size=config.hidden_size,
        attention_head_size=config.attention_heads,
        dropout=config.dropout,
        loss=QuantileLoss(),
        log_interval=-1,
        log_val_interval=-1,
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")
    checkpoint_cb = ModelCheckpoint(dirpath=str(outdir), monitor="val_loss", mode="min", filename="tft-best")

    trainer = Trainer(
        max_epochs=config.epochs,
        accelerator=accelerator,
        devices=1,
        enable_model_summary=False,
        gradient_clip_val=0.1,
        deterministic=True,
        enable_progress_bar=True,
        logger=False,
        default_root_dir=str(outdir),
        callbacks=[early_stop, checkpoint_cb],
    )

    trainer.fit(tft, train_loader, val_loader)

    best_path = Path(checkpoint_cb.best_model_path) if checkpoint_cb.best_model_path else None
    if best_path and best_path.exists():
        tft = TemporalFusionTransformer.load_from_checkpoint(best_path)

    torch.save(tft.state_dict(), outdir / "tft.pt")

    raw_predictions, x = tft.predict(val_loader, mode="raw", return_x=True, trainer=trainer)
    pred_df = tft.to_prediction_df(raw_predictions, x=x)
    pred_df = pred_df[pred_df["quantile"] == 0.5].copy()
    pred_df = pred_df.rename(columns={"prediction": "yhat_tft"})

    val_actual = panel.merge(
        pred_df[["series_id"] + [col for col in pred_df.columns if "time_idx" in col] + ["yhat_tft"]],
        how="inner",
        left_on=["series_id", "time_idx"],
        right_on=["series_id", "time_idx"],
    )
    val_actual = val_actual[val_actual["time_idx"] > val_actual["max_time_idx"] - config.val_size].copy()
    val_actual = val_actual.drop(columns=["max_time_idx"])

    importance_df: Optional[pd.DataFrame] = None
    try:
        interpretation = tft.interpret_output(raw_predictions, reduction="mean")
        encoder_imp = interpretation.get("encoder_variable_importance")
        decoder_imp = interpretation.get("decoder_variable_importance")
        parts: List[pd.DataFrame] = []
        real_names = getattr(train_dataset, "reals", None)
        if real_names is None and hasattr(train_dataset, "input_names"):
            real_names = train_dataset.input_names.get("reals", [])
        if encoder_imp is not None and real_names:
            enc_series = pd.Series(encoder_imp.mean(axis=0).detach().cpu().numpy(), index=real_names)
            parts.append(pd.DataFrame({"variable": enc_series.index, "importance": enc_series.values, "stage": "encoder"}))
        if decoder_imp is not None and real_names:
            dec_series = pd.Series(decoder_imp.mean(axis=0).detach().cpu().numpy(), index=real_names)
            parts.append(pd.DataFrame({"variable": dec_series.index, "importance": dec_series.values, "stage": "decoder"}))
        if parts:
            importance_df = pd.concat(parts, ignore_index=True)
    except Exception:  # pragma: no cover - interpretation may fail on some versions
        importance_df = None

    return val_actual[["series_id", "time", "y", "yhat_tft"]], outdir / "tft.pt", importance_df
