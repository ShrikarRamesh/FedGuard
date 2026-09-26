"""Config loading: YAML composition with OmegaConf, validation with pydantic.

Composition rules (deliberately small, no Hydra):

* A config file may list ``defaults: [data, model/patchtst, train/centralized, ...]``. Each entry is a
  file under ``configs/`` and is merged under the key given by its first path component
  (``model/patchtst.yaml`` -> ``cfg.model``, ``data.yaml`` -> ``cfg.data``). The file's own keys are
  merged afterwards, so experiments override groups.
* Any mapping may contain a ``_fast`` block. With ``fast=True`` it is merged into its parent;
  otherwise it is dropped. This is how every command gets a ``--fast`` mode.
* ``overrides`` are OmegaConf dotlist strings, e.g. ``["train.lr=3e-4", "seed=1"]``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fedguard.utils.io import configs_dir

FAST_KEY = "_fast"


def _load_file(ref: str | Path, base: Path) -> DictConfig:
    p = Path(ref)
    if p.suffix != ".yaml":
        p = p.with_suffix(".yaml")
    if not p.is_absolute():
        cand = base / p
        p = cand if cand.exists() else configs_dir() / p
    if not p.exists():
        raise FileNotFoundError(f"config file not found: {ref} (looked in {base} and {configs_dir()})")
    cfg = OmegaConf.load(p)
    if not isinstance(cfg, DictConfig):
        raise ValueError(f"{p} must contain a mapping at the top level")
    return cfg


def _group_of(ref: str | Path) -> str:
    """Group key of a config reference: ``model/patchtst`` -> ``model``; ``data`` -> ``data``."""
    return Path(ref).parts[0].removesuffix(".yaml")


def _compose(
    path: str | Path, base: Path, seen: tuple[Path, ...] = (), own_group: str | None = None
) -> DictConfig:
    """Load ``path`` and its ``defaults``. A default in the same group as the including file is merged
    flat (inheritance, e.g. ``fl/fedprox`` <- ``fl/fedavg``); other groups are nested under their key."""
    cfg = _load_file(path, base)
    here = Path(path)
    if here in seen:
        raise ValueError(f"circular defaults: {' -> '.join(map(str, (*seen, here)))}")
    defaults = cfg.pop("defaults", None) or []
    merged = OmegaConf.create({})
    for ref in defaults:
        group = _group_of(ref)
        sub = _compose(ref, configs_dir(), (*seen, here), own_group=group)
        merged = OmegaConf.merge(merged, sub if group == own_group else {group: sub})
    return OmegaConf.merge(merged, cfg)


def _apply_fast(node: Any, fast: bool) -> Any:
    """Recursively merge (fast) or drop (not fast) every ``_fast`` block."""
    if isinstance(node, DictConfig):
        block = node.pop(FAST_KEY, None) if FAST_KEY in node else None
        for k in list(node.keys()):
            node[k] = _apply_fast(node[k], fast)
        if fast and block is not None:
            node = OmegaConf.merge(node, block)
        return node
    return node


def load_config(*paths: str | Path, overrides: list[str] | None = None, fast: bool = False) -> DictConfig:
    """Compose one or more config files, apply ``--fast`` blocks and dotlist overrides."""
    cfg = OmegaConf.create({})
    for p in paths:
        own = _group_of(p) if len(Path(p).parts) > 1 else None
        cfg = OmegaConf.merge(cfg, _compose(p, configs_dir(), own_group=own))
    cfg = _apply_fast(cfg, fast)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    cfg.fast = fast
    return cfg


# ---------------------------------------------------------------------------------------------
# Validated schemas (extended milestone by milestone)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class FeatureConfig(_Strict):
    use_masks: bool = True
    use_deltas: bool = True
    delta_cap_hours: float = Field(48.0, gt=0)
    static: list[Literal["Age", "Gender", "ICULOS", "HospAdmTime"]] = [
        "Age",
        "Gender",
        "ICULOS",
        "HospAdmTime",
    ]
    hospadm_clip_hours: float = Field(720.0, gt=0)


class DataConfig(_Strict):
    raw_subdir: str = "raw/physionet2019"
    processed_subdir: str = "processed"
    unk_policy: Literal["exclude", "separate", "merge_into_hospital"] = "exclude"
    partition: Literal["unit", "hospital", "dirichlet"] = "unit"
    dirichlet_k: int = Field(8, ge=2)
    dirichlet_alpha: float = Field(0.5, gt=0)
    split_fracs: tuple[float, float, float] = (0.70, 0.15, 0.15)
    split_seed: int = 42
    lookback: int = Field(24, ge=1)
    features: FeatureConfig = FeatureConfig()
    subset_per_hospital: int | None = Field(None, ge=10, description="fast mode: patients per hospital")
    expected_counts: dict[str, int] = {"training_setA": 20336, "training_setB": 20000}

    @model_validator(mode="after")
    def _fracs_sum_to_one(self) -> DataConfig:
        if abs(sum(self.split_fracs) - 1.0) > 1e-9:
            raise ValueError(f"split_fracs must sum to 1, got {self.split_fracs}")
        return self


def load_data_config(fast: bool = False, overrides: list[str] | None = None) -> DataConfig:
    """Load and validate ``configs/data.yaml`` on its own (overrides use bare keys, e.g. ``lookback=12``)."""
    cfg = load_config("data", fast=fast, overrides=overrides)
    d = OmegaConf.to_container(cfg, resolve=True)
    d.pop("fast", None)  # type: ignore[union-attr]
    return DataConfig(**d)  # type: ignore[arg-type]


def validate_data(cfg: DictConfig) -> DataConfig:
    """Validate ``cfg.data`` and return a typed object (raises pydantic.ValidationError with details)."""
    return DataConfig(**OmegaConf.to_container(cfg.data, resolve=True))  # type: ignore[arg-type]
