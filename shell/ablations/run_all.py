from dataclasses import dataclass
from typing import Any, Callable

from shell.ablations.risk_constraint import RiskAblationConfig, run_risk_constraint_ablation
from shell.ablations.temperature import TemperatureAblationConfig, run_temperature_ablation
from shell.ablations.warm_start import WarmStartAblationConfig, run_warm_start_ablation


@dataclass
class AllAblationsConfig:
    seeds: int
    episodes: int
    risk_lambda_on: float = 1.0

def run_all_ablations(*, train_fn: Callable, args: Any, cfg: AllAblationsConfig) -> None:
    """
    Runs all ablations back-to-back.
    Produces separate CSV/PNG outputs per ablation.
    """

    # Warm-start
    run_warm_start_ablation(
        train_fn=train_fn,
        args=args,
        cfg=WarmStartAblationConfig(
            seeds=cfg.seeds,
            episodes=cfg.episodes,
            out_csv="warm_start_ablation.csv",
            out_png="warm_start_ablation.png",
        ),
    )

    # Risk constraint
    run_risk_constraint_ablation(
        train_fn=train_fn,
        args=args,
        cfg=RiskAblationConfig(
            seeds=cfg.seeds,
            episodes=cfg.episodes,
            lambda_on=cfg.risk_lambda_on,
            out_csv="risk_ablation.csv",
            out_png="risk_ablation.png",
        ),
    )

    # Temperature schedule
    run_temperature_ablation(
        train_fn=train_fn,
        args=args,
        cfg=TemperatureAblationConfig(
            seeds=cfg.seeds,
            episodes=cfg.episodes,
            out_csv="temperature_ablation.csv",
            out_png="temperature_ablation.png",
        ),
    )