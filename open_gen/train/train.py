from __future__ import annotations

from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from open_gen.train.engine import train


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = value
    return out


def resolve_config(cfg: DictConfig) -> dict[str, Any]:
    payload = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(payload, dict)

    model = payload["model"]
    size = model.pop("size")
    presets = model.pop("presets")
    if size not in presets:
        raise ValueError(f"unknown model.size {size!r}; choices: {sorted(presets)}")
    payload["model"] = _deep_merge(model, presets[size])
    payload["model"]["_size"] = size

    dims = payload["dims"]
    payload["model"]["action_dim"] = dims["action"]
    payload["model"]["proprio_dim"] = dims["proprio"]
    payload["model"]["force_dim"] = dims["force"]
    return payload


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    train(resolve_config(cfg))


if __name__ == "__main__":
    main()
