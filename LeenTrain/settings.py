from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, field_validator
from loguru import logger

class FormattedBase(BaseModel):
    def __str__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

    def __repr__(self) -> str:
        return "\n".join(f"{k}: {v}" for k, v in self.__dict__.items())

class TrainerSettings(FormattedBase):
    epochs: int
    metrics: List[Callable]
    logdir: Path
    train_steps: int
    valid_steps: int
    save_model: bool = False
    optimizer_kwargs: Dict[str, Any] = {"lr": 1e-3, "weight_decay": 1e-5}
    scheduler_kwargs: Optional[Dict[str, Any]] = {"factor": 0.1, "patience": 10}
    earlystop_kwargs: Optional[Dict[str, Any]] = None  # Initialize as None for customization

    @field_validator("logdir")
    @classmethod
    def check_path(cls, logdir: Path) -> Path:  # noqa: N805
        if isinstance(logdir, str):
            logdir = Path(logdir)
        if not logdir.resolve().exists():  # type: ignore
            logdir.mkdir(parents=True)
            logger.info(f"Created logdir {logdir.absolute()}")
        return logdir

    def __init__(self, **data: Any):
        super().__init__(**data)
        
        # If-else logic for earlystop_kwargs based on save_model or another condition
        self.earlystop_kwargs = {
            "save": self.save_model,
            "verbose": True,
            "patience": 10,
            "early_stop": True if self.save_model else False  # Logic here
        }

class ModelSettings(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class TuneSettings(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
