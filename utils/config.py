import os
import shutil
from typing import Any, Dict, Optional

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a YAML config file.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if cfg is None:
        cfg = {}

    return cfg


def save_config(cfg: Dict[str, Any], save_path: str) -> None:
    """
    Save config as YAML.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            cfg,
            f,
            sort_keys=False,
            allow_unicode=True,
            default_flow_style=False,
        )


def copy_config(config_path: str, out_dir: str, filename: Optional[str] = None) -> str:
    """
    Copy the used config file into the output directory.
    """
    os.makedirs(out_dir, exist_ok=True)

    if filename is None:
        filename = os.path.basename(config_path)

    dst = os.path.join(out_dir, filename)
    shutil.copy2(config_path, dst)
    return dst


def recursive_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively update a nested dict.
    """
    for key, value in updates.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            recursive_update(base[key], value)
        else:
            base[key] = value

    return base


def get_output_dir(cfg: Dict[str, Any]) -> str:
    """
    Return output directory from config.
    """
    out_cfg = cfg.get("out", {})
    return out_cfg.get("out_dir", "outputs/mapnet_3d")


def ensure_output_dirs(cfg: Dict[str, Any]) -> Dict[str, str]:
    """
    Create and return common output directories.
    """
    out_cfg = cfg.setdefault("out", {})

    out_dir = out_cfg.get("out_dir", "outputs/mapnet_3d")
    ckpt_dir = out_cfg.get("ckpt_dir", os.path.join(out_dir, "ckpt"))
    vis_dir = out_cfg.get("vis_dir", os.path.join(out_dir, "vis"))
    log_dir = out_cfg.get("log_dir", os.path.join(out_dir, "logs"))
    eval_dir = out_cfg.get("eval_dir", os.path.join(out_dir, "eval"))

    out_cfg["out_dir"] = out_dir
    out_cfg["ckpt_dir"] = ckpt_dir
    out_cfg["vis_dir"] = vis_dir
    out_cfg["log_dir"] = log_dir
    out_cfg["eval_dir"] = eval_dir

    for d in [out_dir, ckpt_dir, vis_dir, log_dir, eval_dir]:
        os.makedirs(d, exist_ok=True)

    return {
        "out_dir": out_dir,
        "ckpt_dir": ckpt_dir,
        "vis_dir": vis_dir,
        "log_dir": log_dir,
        "eval_dir": eval_dir,
    }