import csv
import os
import time
from collections import defaultdict
from typing import Dict, Iterable, Optional

import torch


def as_float(x, default: float = 0.0) -> float:
    """
    Convert tensor or numeric object to float.
    """
    if x is None:
        return float(default)

    if torch.is_tensor(x):
        if x.numel() == 0:
            return float(default)
        return float(x.detach().cpu().item())

    try:
        return float(x)
    except Exception:
        return float(default)


class AverageMeter:
    """
    Track average value.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, value, n: int = 1):
        value = as_float(value)
        n = int(n)

        self.val = value
        self.sum += value * n
        self.count += n
        self.avg = self.sum / max(1, self.count)


class MetricTracker:
    """
    Track multiple averaged metrics.
    """

    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, metrics: Dict, n: int = 1):
        for key, value in metrics.items():
            self.meters[key].update(value, n=n)

    def averages(self) -> Dict[str, float]:
        return {key: meter.avg for key, meter in self.meters.items()}

    def reset(self):
        self.meters.clear()


class CSVLogger:
    """
    Lightweight CSV logger.

    Example:
        logger = CSVLogger("train.csv", fieldnames=["epoch", "iter", "loss"])
        logger.write({"epoch": 0, "iter": 10, "loss": 1.23})
    """

    def __init__(
        self,
        path: str,
        fieldnames: Iterable[str],
        append: bool = True,
    ):
        self.path = path
        self.fieldnames = list(fieldnames)
        self.append = bool(append)

        os.makedirs(os.path.dirname(path), exist_ok=True)

        file_exists = os.path.exists(path)
        should_write_header = (not file_exists) or (not append)

        if should_write_header:
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def write(self, row: Dict):
        safe_row = {}

        for key in self.fieldnames:
            value = row.get(key, "")
            if torch.is_tensor(value):
                value = as_float(value)
            safe_row[key] = value

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(safe_row)


def format_mask_stats(mask_stats: Optional[Dict]) -> str:
    """
    Format mask debug statistics for console logging.
    """
    if not mask_stats:
        return ""

    return (
        f" | ml_mean:{mask_stats.get('ml_mean', 0.0):.3f}"
        f" ml_max:{mask_stats.get('ml_max', 0.0):.3f}"
        f" pf_mean:{mask_stats.get('pf_mean', 0.0):.3f}"
        f" pf_max:{mask_stats.get('pf_max', 0.0):.3f}"
        f" pf_p50:{mask_stats.get('pf_p50', 0.0):.3f}"
        f" pf_p90:{mask_stats.get('pf_p90', 0.0):.3f}"
        f" pf_p99:{mask_stats.get('pf_p99', 0.0):.3f}"
        f" pf>0.5:{mask_stats.get('pf_pos>0.5(%)', 0.0):.1f}%"
    )


def format_train_message(
    epoch: int,
    iteration: int,
    d_loss,
    g_loss,
    glog: Dict,
    dlog: Dict,
    adv_weight: Optional[float] = None,
    mask_stats: Optional[Dict] = None,
) -> str:
    """
    Build a standardized training log message.
    """
    adv_part = ""
    if adv_weight is not None:
        adv_part = f" AdvW:{float(adv_weight):.4f}"

    msg = (
        f"[E{epoch:03d} I{iteration:05d}] "
        f"D:{as_float(d_loss):.3f} "
        f"G:{as_float(g_loss):.3f}"
        f"{adv_part} "
        f"REC:{as_float(glog.get('rec_loss')):.3f} "
        f"ADV:{as_float(glog.get('adv_loss')):.3f} "
        f"MTV:{as_float(glog.get('mtv')):.3f} "
        f"SUM:{as_float(glog.get('sum_loss')):.3f} "
        f"SP:{as_float(glog.get('sparse')):.3f} "
        f"GR:{as_float(glog.get('grad')):.3f} "
        f"AUX:{as_float(glog.get('aux_loss', glog.get('quant_loss'))):.3f} "
        f"Dr:{as_float(dlog.get('logits_real')):.3f} "
        f"Df:{as_float(dlog.get('logits_fake')):.3f} "
        f"mDice:{as_float(glog.get('mask_dice')):.3f} "
        f"mCE:{as_float(glog.get('mask_ce')):.3f} "
        f"mCons:{as_float(glog.get('mask_cons')):.3f} "
        f"mSparse:{as_float(glog.get('mask_sparse')):.3f}"
    )

    msg += format_mask_stats(mask_stats)
    return msg


def make_train_log_row(
    epoch: int,
    iteration: int,
    d_loss,
    g_loss,
    glog: Dict,
    dlog: Dict,
    lr_g: float,
    lr_d: float,
    adv_weight: Optional[float] = None,
) -> Dict:
    """
    Build a row for train_log.csv.
    """
    return {
        "epoch": int(epoch),
        "iter": int(iteration),
        "lr_g": float(lr_g),
        "lr_d": float(lr_d),
        "adv_weight": float(adv_weight) if adv_weight is not None else "",
        "D": as_float(d_loss),
        "G": as_float(g_loss),
        "REC": as_float(glog.get("rec_loss")),
        "ADV": as_float(glog.get("adv_loss")),
        "AUX": as_float(glog.get("aux_loss", glog.get("quant_loss"))),
        "MTV": as_float(glog.get("mtv")),
        "TLG": as_float(glog.get("tlg")),
        "SUM": as_float(glog.get("sum_loss")),
        "SP": as_float(glog.get("sparse")),
        "GR": as_float(glog.get("grad")),
        "Dr": as_float(dlog.get("logits_real")),
        "Df": as_float(dlog.get("logits_fake")),
        "mask_dice": as_float(glog.get("mask_dice")),
        "mask_ce": as_float(glog.get("mask_ce")),
        "mask_cons": as_float(glog.get("mask_cons")),
        "mask_sparse": as_float(glog.get("mask_sparse")),
    }


TRAIN_LOG_FIELDS = [
    "epoch",
    "iter",
    "lr_g",
    "lr_d",
    "adv_weight",
    "D",
    "G",
    "REC",
    "ADV",
    "AUX",
    "MTV",
    "TLG",
    "SUM",
    "SP",
    "GR",
    "Dr",
    "Df",
    "mask_dice",
    "mask_ce",
    "mask_cons",
    "mask_sparse",
]


class Timer:
    """
    Simple wall-clock timer.
    """

    def __init__(self):
        self.start_time = time.time()

    def reset(self):
        self.start_time = time.time()

    def elapsed(self) -> float:
        return time.time() - self.start_time