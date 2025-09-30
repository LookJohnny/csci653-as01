"""
Configuration management for DarkHorse system.

Centralizes all hyperparameters, paths, and settings.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import json


@dataclass
class DataConfig:
    """Data processing configuration."""
    input_dir: Path = Path("data/raw")
    output_dir: Path = Path("data/processed")
    asin_mapping_path: Path = Path("asin2category.json")
    num_shards: int = 64
    min_reviews: int = 1
    top_quantile: float = 0.95
    frequency: str = "W"  # Weekly
    seed: int = 2025


@dataclass
class TransformerConfig:
    """Transformer model configuration."""
    # Model architecture
    n_features: int = 4
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 3
    dim_feedforward: int = 128
    dropout: float = 0.1
    seq_len: int = 32

    # Training
    batch_size: int = 256
    epochs: int = 10
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0

    # Data split
    val_split_weeks: int = 26
    min_weeks: int = 12

    # Output
    output_dir: Path = Path("transformer_output")
    save_best_only: bool = True


@dataclass
class ForecastConfig:
    """Forecast ensemble configuration."""
    # Time series settings
    horizon: int = 12
    val_size: int = 12
    max_encoder_len: int = 64

    # Model selection
    enable_tft: bool = True
    enable_autots: bool = True

    # TFT hyperparameters
    hidden_size: int = 64
    attention_heads: int = 4
    dropout: float = 0.1
    learning_rate: float = 1e-3

    # Training
    batch_size: int = 128
    epochs: int = 30
    num_workers: int = 0
    gradient_clip_val: float = 0.1

    # Features
    known_reals: List[str] = field(default_factory=list)
    observed_reals: List[str] = field(default_factory=list)
    static_categoricals: List[str] = field(default_factory=list)

    # Output
    output_dir: Path = Path("forecast_output")
    save_predictions: bool = True
    generate_plots: bool = True


@dataclass
class LoadBalancerConfig:
    """Load balancing configuration."""
    # Thresholds
    hot_seller_threshold: float = 0.5  # Probability threshold
    demand_spike_multiplier: float = 2.0  # How much demand increase qualifies as spike

    # Resource allocation
    min_instances: int = 1
    max_instances: int = 10
    scale_up_threshold: float = 0.7
    scale_down_threshold: float = 0.3

    # Monitoring
    check_interval_minutes: int = 5
    history_window_hours: int = 24

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    enable_cors: bool = True


@dataclass
class DarkHorseConfig:
    """Master configuration for DarkHorse system."""
    data: DataConfig = field(default_factory=DataConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    load_balancer: LoadBalancerConfig = field(default_factory=LoadBalancerConfig)

    # System settings
    device: str = "cuda"  # or "cpu"
    log_level: str = "INFO"
    seed: int = 2025

    @classmethod
    def from_json(cls, path: str) -> "DarkHorseConfig":
        """Load configuration from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: str) -> None:
        """Save configuration to JSON file."""
        def convert_paths(obj):
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_paths(v) for v in obj]
            else:
                return obj

        data = {
            "data": self.data.__dict__,
            "transformer": self.transformer.__dict__,
            "forecast": self.forecast.__dict__,
            "load_balancer": self.load_balancer.__dict__,
            "device": self.device,
            "log_level": self.log_level,
            "seed": self.seed,
        }

        data = convert_paths(data)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def from_env(cls) -> "DarkHorseConfig":
        """Load configuration from environment variables."""
        config = cls()

        # Override with environment variables if present
        if "DARKHORSE_DEVICE" in os.environ:
            config.device = os.environ["DARKHORSE_DEVICE"]

        if "DARKHORSE_SEED" in os.environ:
            config.seed = int(os.environ["DARKHORSE_SEED"])

        if "DARKHORSE_LOG_LEVEL" in os.environ:
            config.log_level = os.environ["DARKHORSE_LOG_LEVEL"]

        # Data config
        if "DARKHORSE_INPUT_DIR" in os.environ:
            config.data.input_dir = Path(os.environ["DARKHORSE_INPUT_DIR"])

        if "DARKHORSE_OUTPUT_DIR" in os.environ:
            config.data.output_dir = Path(os.environ["DARKHORSE_OUTPUT_DIR"])

        # Transformer config
        if "DARKHORSE_TRANSFORMER_EPOCHS" in os.environ:
            config.transformer.epochs = int(os.environ["DARKHORSE_TRANSFORMER_EPOCHS"])

        if "DARKHORSE_TRANSFORMER_LR" in os.environ:
            config.transformer.learning_rate = float(os.environ["DARKHORSE_TRANSFORMER_LR"])

        # Forecast config
        if "DARKHORSE_FORECAST_HORIZON" in os.environ:
            config.forecast.horizon = int(os.environ["DARKHORSE_FORECAST_HORIZON"])

        if "DARKHORSE_FORECAST_EPOCHS" in os.environ:
            config.forecast.epochs = int(os.environ["DARKHORSE_FORECAST_EPOCHS"])

        return config


# Default configuration instance
default_config = DarkHorseConfig()


def get_config(config_path: Optional[str] = None) -> DarkHorseConfig:
    """
    Get configuration with the following precedence:
    1. From file if path provided
    2. From environment variables
    3. Default configuration
    """
    if config_path and Path(config_path).exists():
        return DarkHorseConfig.from_json(config_path)
    else:
        return DarkHorseConfig.from_env()


if __name__ == "__main__":
    # Example: Generate default config file
    config = DarkHorseConfig()
    config.to_json("config_default.json")
    print("Default configuration saved to config_default.json")