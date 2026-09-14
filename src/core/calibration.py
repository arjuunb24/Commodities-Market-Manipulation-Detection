"""
src/core/calibration.py
========================
CalibrationEngine — fits Ornstein-Uhlenbeck (OU) process parameters
from real commodity futures price series.

The OU parameters are then used to configure the ABM's FundamentalValueProcess
and mean-reversion traders. Mean reversion is an EMERGENT property of agents
responding to the fundamental value — the OU process is the underlying generative
model for the fundamental, not the price.

Method:
  The OU process:  dX_t = κ(θ - X_t)dt + σ dW_t
  Discretised (Euler) as AR(1) on log prices:
      log(P_{t+1}) = α + β log(P_t) + ε_t
  where:
      κ (mean-reversion speed) = -log(β) / Δt
      θ (long-run mean)        = α / (1 - β)  (in log-price space)
      σ (volatility)           = std(ε_t) / sqrt(Δt)

  The fit uses OLS on the AR(1) regression of log prices.
  This is the standard discrete-time OU estimator (see Vasicek 1977).

Fallback:
  If data is unavailable (yfinance failure), uses pre-calibrated constants
  from published historical statistics. The simulation runs correctly either way.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ===========================================================================
# Parameter dataclasses
# ===========================================================================


@dataclass
class OUParams:
    """
    Ornstein-Uhlenbeck process parameters (daily scale).

    Attributes:
        mean_reversion_speed: κ — rate of pull toward long-run mean. Higher = faster reversion.
        long_run_mean: θ — the equilibrium value (in log-price space, normalised to 1.0).
        volatility: σ — daily diffusion coefficient.
        drift: μ — small additional directional drift.
    """
    mean_reversion_speed: float
    long_run_mean: float
    volatility: float
    drift: float

    def to_dict(self) -> dict:
        return {
            "mean_reversion_speed": float(self.mean_reversion_speed),
            "long_run_mean": float(self.long_run_mean),
            "volatility": float(self.volatility),
            "drift": float(self.drift),
        }


@dataclass
class VolumeParams:
    """Volume distribution parameters."""
    mean_daily_volume: float
    volume_cv: float  # coefficient of variation (std / mean)

    def to_dict(self) -> dict:
        return {
            "mean_daily_volume": float(self.mean_daily_volume),
            "volume_cv": float(self.volume_cv),
        }


# ===========================================================================
# Pre-calibrated fallback constants
# (sources: Erb & Harvey 2006; Geman 2005; Deaton & Laroque 1996)
# ===========================================================================

FALLBACK_OU_PARAMS = {
    "crude_oil_wti": OUParams(mean_reversion_speed=0.035, long_run_mean=1.0, volatility=0.022, drift=0.00005),
    "brent_crude": OUParams(mean_reversion_speed=0.033, long_run_mean=1.0, volatility=0.021, drift=0.00005),
    "gold": OUParams(mean_reversion_speed=0.05, long_run_mean=1.0, volatility=0.0094, drift=0.0001),
    "silver": OUParams(mean_reversion_speed=0.06, long_run_mean=1.0, volatility=0.0150, drift=0.0001),
    "natural_gas": OUParams(mean_reversion_speed=0.10, long_run_mean=1.0, volatility=0.040, drift=-0.0001),
    "copper": OUParams(mean_reversion_speed=0.04, long_run_mean=1.0, volatility=0.012, drift=0.0002),
    "wheat": OUParams(mean_reversion_speed=0.08, long_run_mean=1.0, volatility=0.016, drift=0.00002),
    "corn": OUParams(mean_reversion_speed=0.08, long_run_mean=1.0, volatility=0.014, drift=0.00002),
}

FALLBACK_VOLUME_PARAMS = {
    "crude_oil_wti": VolumeParams(mean_daily_volume=8000, volume_cv=0.55),
    "brent_crude": VolumeParams(mean_daily_volume=7500, volume_cv=0.50),
    "gold": VolumeParams(mean_daily_volume=5000, volume_cv=0.40),
    "silver": VolumeParams(mean_daily_volume=3000, volume_cv=0.45),
    "natural_gas": VolumeParams(mean_daily_volume=9000, volume_cv=0.60),
    "copper": VolumeParams(mean_daily_volume=4000, volume_cv=0.35),
    "wheat": VolumeParams(mean_daily_volume=3000, volume_cv=0.45),
    "corn": VolumeParams(mean_daily_volume=3500, volume_cv=0.45),
}


# ===========================================================================
# CalibrationEngine
# ===========================================================================


class CalibrationEngine:
    """
    Fits OU parameters from a commodity price series and persists them
    to the agent configuration YAML.
    """

    def fit_ou_process(
        self,
        price_series: pd.Series,
        dt: float = 1.0,
    ) -> OUParams:
        """
        Fit Ornstein-Uhlenbeck parameters from a daily price series.

        Args:
            price_series: Raw price series (not log-transformed). Must have
                          at least 30 observations with no leading/trailing NaN.
            dt: Time step in days (1.0 for daily data).

        Returns:
            OUParams fitted to the series.

        Raises:
            ValueError: If the series has fewer than 30 non-NaN observations,
                        or if the AR(1) coefficient β is not in (0, 1) (suggesting
                        the series is not stationary or the data has quality issues).
        """
        # Drop NaN — never forward-fill before computing log returns (Implementation Plan)
        clean = price_series.dropna()

        if len(clean) < 30:
            raise ValueError(
                f"Price series has only {len(clean)} non-NaN observations — "
                "minimum 30 required for reliable OU parameter estimation."
            )

        log_prices = np.log(clean.values.astype(float))

        # AR(1) OLS: log(P_{t+1}) = α + β log(P_t)
        y = log_prices[1:]
        x = log_prices[:-1]

        # OLS via normal equations: [β, α] = (X'X)^{-1} X'y
        X = np.column_stack([x, np.ones_like(x)])
        coeffs, residuals, _, _ = np.linalg.lstsq(X, y, rcond=None)
        beta, alpha = coeffs[0], coeffs[1]

        # Validate stationarity: β should be in (0, 1) for mean-reverting process
        if not (0 < beta < 1):
            logger.warning(
                "AR(1) coefficient β=%.4f is outside (0, 1) — the series may not be "
                "stationary. Using pre-calibrated fallback for mean_reversion_speed.",
                beta,
            )
            beta = max(0.01, min(0.99, beta))

        # Compute OU parameters from AR(1) coefficients
        kappa = -np.log(beta) / dt  # mean-reversion speed
        theta = alpha / (1 - beta)  # long-run mean in log-price space

        # Residual std → sigma
        y_hat = X @ coeffs
        residuals = y - y_hat
        sigma = float(np.std(residuals)) / np.sqrt(dt)

        # Small drift (mean of log returns)
        drift = float(np.mean(log_prices[1:] - log_prices[:-1]))

        params = OUParams(
            mean_reversion_speed=float(max(kappa, 1e-6)),
            long_run_mean=float(np.exp(theta)),  # convert back to price space
            volatility=float(sigma),
            drift=float(drift),
        )

        logger.info(
            "OU fit: κ=%.4f, θ=%.4f, σ=%.4f, μ=%.6f (n=%d)",
            params.mean_reversion_speed,
            params.long_run_mean,
            params.volatility,
            params.drift,
            len(clean),
        )
        return params

    def estimate_volume_distribution(
        self, volume_series: pd.Series
    ) -> VolumeParams:
        """
        Estimate volume distribution parameters from a daily volume series.

        Returns:
            VolumeParams with mean and coefficient of variation (std/mean).
        """
        clean = volume_series.dropna()
        if len(clean) < 10:
            logger.warning("Insufficient volume observations; using fallback volume params.")
            return FALLBACK_VOLUME_PARAMS.get("gold", VolumeParams(5000, 0.4))

        mean_vol = float(clean.mean())
        cv = float(clean.std() / mean_vol) if mean_vol > 0 else 0.4
        return VolumeParams(mean_daily_volume=mean_vol, volume_cv=cv)

    def store_to_yaml(
        self,
        ou_params: OUParams,
        volume_params: VolumeParams,
        commodity: str,
        config_path: Path,
    ) -> None:
        """
        Update configs/agent_config.yaml with freshly fitted parameters.

        Args:
            ou_params: Fitted OU parameters.
            volume_params: Fitted volume parameters.
            commodity: Name of the commodity (e.g., 'gold').
            config_path: Path to agent_config.yaml.
        """
        import yaml

        if config_path.exists():
            with config_path.open("r") as f:
                config: dict[str, Any] = yaml.safe_load(f) or {}
        else:
            config = {}

        config.setdefault("calibration", {})
        config["calibration"]["source"] = "yfinance"
        config["calibration"]["commodity"] = commodity
        config["calibration"]["ou_params"] = ou_params.to_dict()
        config["calibration"]["volume_params"] = volume_params.to_dict()

        with config_path.open("w") as f:
            yaml.safe_dump(config, f, default_flow_style=False, sort_keys=False)

        logger.info("Updated %s with fitted OU params for '%s'.", config_path, commodity)

    @staticmethod
    def get_fallback(commodity: str = "gold") -> tuple[OUParams, VolumeParams]:
        """
        Return pre-calibrated fallback parameters for a commodity.

        Used when yfinance is unavailable.
        """
        ou = FALLBACK_OU_PARAMS.get(commodity, FALLBACK_OU_PARAMS["gold"])
        vol = FALLBACK_VOLUME_PARAMS.get(commodity, FALLBACK_VOLUME_PARAMS["gold"])
        return ou, vol
