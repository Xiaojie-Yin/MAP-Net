import os
from typing import Dict, Optional

import torch

from engine.checkpoint import (
    get_checkpoint_best_val,
    get_checkpoint_epoch,
    load_checkpoint,
    save_checkpoint,
)
from engine.evaluator import SlidingWindowEvaluator
from utils import (
    CSVLogger,
    TRAIN_LOG_FIELDS,
    ensure_output_dirs,
    format_train_message,
    make_pet_threshold_onehot,
    make_train_log_row,
    mask_debug_stats,
    maybe_unpack_batch,
    pet_from_norm,
    save_train_val_panel,
)


def set_requires_grad(module, requires_grad: bool):
    """
    Enable or disable gradients for a module.
    """
    if module is None:
        return

    for p in module.parameters():
        p.requires_grad_(requires_grad)


class Trainer:
    """
    Trainer for MAP-Net.

    It handles:
        - G/D optimization
        - high-uptake mask losses
        - validation L1 checkpoint selection
        - periodic visualization
        - optional sliding-window validation evaluation
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        cfg: Dict,
        device,
        resume: Optional[str] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device

        self.dirs = ensure_output_dirs(cfg)
        self.ckpt_dir = self.dirs["ckpt_dir"]
        self.vis_dir = self.dirs["vis_dir"]
        self.log_dir = self.dirs["log_dir"]
        self.eval_dir = self.dirs["eval_dir"]

        self.seed = int(cfg.get("seed", 42))

        self.train_cfg = cfg.get("train", {})
        self.out_cfg = cfg.get("out", {})
        self.loss_cfg = cfg.get("loss", {})
        self.mask_cfg = cfg.get("mask", {})
        self.dataset_cfg = cfg.get("dataset", {})
        self.data_cfg = cfg.get("data", {})

        self.epochs = int(self.train_cfg.get("epochs", 200))
        self.start_epoch = 0
        self.best_val = float("inf")

        self.enable_mask = bool(
            self.mask_cfg.get(
                "enable",
                cfg.get("model", {}).get("enable_mask", True),
            )
        )

        self.use_prior = bool(
            self.data_cfg.get(
                "use_prior",
                self.dataset_cfg.get("use_prior", True),
            )
        )

        self.suv_thr = float(
            self.mask_cfg.get("threshold", {}).get(
                "suv",
                self.dataset_cfg.get("suv_thr", 2.5),
            )
        )

        self.patch_size = tuple(self.dataset_cfg.get("patch_size", [32, 96, 96]))
        self.stride = tuple(self.dataset_cfg.get("stride", [16, 48, 48]))

        self.log_interval = int(self.out_cfg.get("log_interval", 200))
        self.vis_interval = int(self.out_cfg.get("vis_interval", 1000))
        self.eval_interval = int(self.out_cfg.get("eval_interval", 0))
        self.save_every = int(self.out_cfg.get("save_every", 10))
        self.nrow = int(self.out_cfg.get("nrow", 6))

        # Useful for smoke tests. Omit or set <=0 for full training.
        self.max_train_iters_per_epoch = int(
            self.train_cfg.get("max_train_iters_per_epoch", 0)
        )

        self.mask_loss_cfg = self._build_mask_loss_cfg()

        self.optimizer_g, self.optimizer_d = self._build_optimizers()
        self.scheduler_g = self._build_scheduler()

        self.train_logger = CSVLogger(
            path=os.path.join(self.log_dir, "train_log.csv"),
            fieldnames=TRAIN_LOG_FIELDS,
            append=True,
        )

        if resume is not None:
            self.resume(resume)

    def _build_optimizers(self):
        lr_g = float(self.train_cfg.get("lr", 2.0e-4))
        lr_d = float(self.train_cfg.get("lr_d", lr_g))

        beta1 = float(self.train_cfg.get("beta1", 0.5))
        beta2 = float(self.train_cfg.get("beta2", 0.999))

        g_params = [
            p for name, p in self.model.named_parameters()
            if not name.startswith("discriminator.")
        ]

        d_params = list(self.model.discriminator.parameters())

        optimizer_g = torch.optim.Adam(
            g_params,
            lr=lr_g,
            betas=(beta1, beta2),
        )

        optimizer_d = torch.optim.Adam(
            d_params,
            lr=lr_d,
            betas=(beta1, beta2),
        )

        return optimizer_g, optimizer_d

    def _build_scheduler(self):
        use_scheduler = bool(self.train_cfg.get("use_scheduler", False))

        if not use_scheduler:
            return None

        scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer_g,
            step_size=int(self.train_cfg.get("lr_step", 50)),
            gamma=float(self.train_cfg.get("lr_gamma", 0.5)),
        )

        return scheduler

    def _build_mask_loss_cfg(self):
        raw = self.mask_cfg.get("loss", {})

        return {
            "kappa": float(raw.get("kappa", 0.01)),
            "w_bg": float(raw.get("w_bg", 0.2)),
            "w_fg": float(raw.get("w_fg", 0.8)),
            "lambda_dice": float(raw.get("lambda_dice", 0.25)),
            "lambda_ce_all": float(raw.get("lambda_ce_all", 0.25)),
            "w_bg_ce": float(raw.get("w_bg_ce", 0.1)),
            "w_fg_ce": float(raw.get("w_fg_ce", 0.9)),
            "lambda_consistency": float(raw.get("lambda_consistency", 0.2)),
            "lambda_sparse": float(raw.get("lambda_sparse", 0.0)),
            "target_fg_ratio": float(raw.get("target_fg_ratio", 0.01)),
        }

    def _get_adv_weight(self, epoch: int) -> float:
        """
        Optional adversarial warmup.

        If loss.adv_warmup is absent, return lambda_adv directly.
        """
        adv_final = float(self.loss_cfg.get("lambda_adv", 0.0))
        sched = self.loss_cfg.get("adv_warmup", None)

        if sched is None or adv_final <= 0:
            return adv_final

        start = int(sched.get("start_epoch", 0))
        end = int(sched.get("end_epoch", start))

        if epoch < start:
            return 0.0

        if epoch >= end:
            return adv_final

        t = (epoch - start) / max(1, end - start)
        return adv_final * float(t)

    def _build_mtv_cfg(self):
        weight = float(self.loss_cfg.get("lambda_mtv", 0.0))
        if weight <= 0:
            return None

        return {
            "thr": self.loss_cfg.get(
                "mtv_thrs",
                self.loss_cfg.get("mtv_thr", [4.0]),
            ),
            "tau": float(self.loss_cfg.get("mtv_tau", 0.6)),
            "mode": self.loss_cfg.get("mtv_mode", "relative"),
            "weight": weight,
            "weights": self.loss_cfg.get("mtv_weights", None),
        }

    def _build_tlg_cfg(self):
        weight = float(self.loss_cfg.get("lambda_tlg", 0.0))
        if weight <= 0:
            return None

        return {
            "thr": self.loss_cfg.get(
                "tlg_thrs",
                self.loss_cfg.get("tlg_thr", [4.0]),
            ),
            "tau": float(self.loss_cfg.get("tlg_tau", 0.6)),
            "mode": self.loss_cfg.get("tlg_mode", "relative"),
            "weight": weight,
            "weights": self.loss_cfg.get("tlg_weights", None),
        }

    def _build_sparse_cfg(self):
        return {
            "thr": float(self.loss_cfg.get("sparse_thr", 2.5)),
            "tau": float(self.loss_cfg.get("sparse_tau", 0.5)),
            "weight": float(self.loss_cfg.get("lambda_sparse", 0.0)),
            "reduce": self.loss_cfg.get("sparse_reduce", "mean_over_mask"),
        }

    def _current_lr(self, optimizer):
        return float(optimizer.param_groups[0]["lr"])

    def resume(self, path: str):
        result = load_checkpoint(
            path=path,
            model=self.model,
            optimizer_g=self.optimizer_g,
            optimizer_d=self.optimizer_d,
            scheduler_g=self.scheduler_g,
            map_location=self.device,
            strict=True,
        )

        checkpoint = result["checkpoint"]

        self.start_epoch = get_checkpoint_epoch(checkpoint) + 1
        self.best_val = get_checkpoint_best_val(checkpoint)

        print(
            f"[Resume] Loaded checkpoint from {path} | "
            f"start_epoch={self.start_epoch} | best_val={self.best_val:.6f}"
        )

    def fit(self):
        for epoch in range(self.start_epoch, self.epochs):
            adv_weight = self._get_adv_weight(epoch)
            self.model.lambda_adv = adv_weight

            print(f"[Epoch {epoch:03d}] lambda_adv={adv_weight:.6f}")

            self.train_one_epoch(epoch)

            val_l1 = self.validate_l1(epoch)
            print(f"==> Epoch {epoch} | Val L1: {val_l1:.6f}")

            is_best = val_l1 < self.best_val
            if is_best:
                self.best_val = val_l1
                save_checkpoint(
                    path=os.path.join(self.ckpt_dir, "best.pth"),
                    model=self.model,
                    optimizer_g=self.optimizer_g,
                    optimizer_d=self.optimizer_d,
                    scheduler_g=self.scheduler_g,
                    epoch=epoch,
                    best_val=self.best_val,
                    cfg=self.cfg,
                )
                print(f"[Checkpoint] Saved best.pth | best_val={self.best_val:.6f}")

            if (epoch + 1) % self.save_every == 0:
                save_path = os.path.join(self.ckpt_dir, f"epoch{epoch + 1}.pth")
                save_checkpoint(
                    path=save_path,
                    model=self.model,
                    optimizer_g=self.optimizer_g,
                    optimizer_d=self.optimizer_d,
                    scheduler_g=self.scheduler_g,
                    epoch=epoch,
                    best_val=self.best_val,
                    cfg=self.cfg,
                )
                print(f"[Checkpoint] Saved {save_path}")

            if self.scheduler_g is not None:
                self.scheduler_g.step()

    def train_one_epoch(self, epoch: int):
        self.model.train()

        mtv_cfg = self._build_mtv_cfg()
        tlg_cfg = self._build_tlg_cfg()
        sparse_cfg = self._build_sparse_cfg()

        sum_weight = float(self.loss_cfg.get("lambda_sum", 0.0))
        grad_weight = float(self.loss_cfg.get("lambda_grad", 0.0))

        for iteration, batch in enumerate(self.train_loader):
            if self.max_train_iters_per_epoch > 0 and iteration >= self.max_train_iters_per_epoch:
                break

            src, pet, *_ = maybe_unpack_batch(batch)
            src = src.to(self.device, non_blocking=True)
            pet = pet.to(self.device, non_blocking=True)

            if self.enable_mask:
                pred, mask_logits, aux_loss = self.model.forward_with_mask(src)
            else:
                pred, aux_loss = self.model(src)
                mask_logits = None

            pred_raw = pet_from_norm(pred)
            gt_raw = pet_from_norm(pet)

            if self.enable_mask:
                target_onehot_real = make_pet_threshold_onehot(
                    pet,
                    suv_thr=self.suv_thr,
                )
                target_onehot_spet = make_pet_threshold_onehot(
                    pred,
                    suv_thr=self.suv_thr,
                )
            else:
                target_onehot_real = None
                target_onehot_spet = None

            # ----------------
            # G-step
            # ----------------
            set_requires_grad(self.model.discriminator, False)
            self.optimizer_g.zero_grad(set_to_none=True)

            g_loss, glog = self.model.g_loss(
                real=pet,
                fake=pred,
                aux_loss=aux_loss,
                pred_raw=pred_raw,
                gt_raw=gt_raw,
                mtv_cfg=mtv_cfg,
                tlg_cfg=tlg_cfg,
                sum_weight=sum_weight,
                sparse_cfg=sparse_cfg,
                grad_weight=grad_weight,
                mask_logits=mask_logits,
                mask_target_onehot=target_onehot_real,
                mask_spet_onehot=target_onehot_spet,
                mask_cfg=self.mask_loss_cfg if self.enable_mask else None,
            )

            g_loss.backward()
            self.optimizer_g.step()

            # ----------------
            # D-step
            # ----------------
            set_requires_grad(self.model.discriminator, True)

            if float(self.model.lambda_adv) > 0:
                self.optimizer_d.zero_grad(set_to_none=True)
                d_loss, dlog = self.model.d_loss(pet, pred.detach())
                d_loss.backward()
                self.optimizer_d.step()
            else:
                d_loss = torch.zeros((), device=self.device)
                dlog = {
                    "logits_real": torch.zeros((), device=self.device),
                    "logits_fake": torch.zeros((), device=self.device),
                }

            # ----------------
            # Logging
            # ----------------
            if iteration % self.log_interval == 0:
                mstats = mask_debug_stats(mask_logits) if self.enable_mask else None

                msg = format_train_message(
                    epoch=epoch,
                    iteration=iteration,
                    d_loss=d_loss,
                    g_loss=g_loss,
                    glog=glog,
                    dlog=dlog,
                    adv_weight=float(self.model.lambda_adv),
                    mask_stats=mstats,
                )
                print(msg)

            if iteration % int(self.out_cfg.get("csv_interval", self.log_interval)) == 0:
                row = make_train_log_row(
                    epoch=epoch,
                    iteration=iteration,
                    d_loss=d_loss,
                    g_loss=g_loss,
                    glog=glog,
                    dlog=dlog,
                    lr_g=self._current_lr(self.optimizer_g),
                    lr_d=self._current_lr(self.optimizer_d),
                    adv_weight=float(self.model.lambda_adv),
                )
                self.train_logger.write(row)

            # ----------------
            # Visualization
            # ----------------
            if self.vis_interval > 0 and iteration % self.vis_interval == 0:
                save_train_val_panel(
                    model=self.model,
                    src=src.detach(),
                    pet=pet.detach(),
                    pred=pred.detach(),
                    val_dataset=self.val_loader.dataset,
                    device=self.device,
                    save_dir=self.vis_dir,
                    epoch=epoch,
                    iteration=iteration,
                    mask_logits=mask_logits.detach() if mask_logits is not None else None,
                    suv_thr=self.suv_thr,
                    nrow=self.nrow,
                    enable_mask=self.enable_mask,
                )

            # ----------------
            # Optional sliding-window validation evaluation
            # ----------------
            if self.eval_interval > 0 and iteration > 0 and iteration % self.eval_interval == 0:
                self.run_periodic_eval(epoch, iteration)

    @torch.no_grad()
    def validate_l1(self, epoch: int):
        self.model.eval()

        total_l1 = 0.0
        total_n = 0

        for batch in self.val_loader:
            src, pet, *_ = maybe_unpack_batch(batch)

            src = src.to(self.device, non_blocking=True)
            pet = pet.to(self.device, non_blocking=True)

            if self.enable_mask:
                pred, _, _ = self.model.forward_with_mask(src)
            else:
                pred, _ = self.model(src)

            batch_l1 = torch.abs(pet - pred).mean()
            total_l1 += float(batch_l1.item()) * src.shape[0]
            total_n += src.shape[0]

        return total_l1 / max(1, total_n)

    @torch.no_grad()
    def run_periodic_eval(self, epoch: int, iteration: int):
        eval_cfg = self.cfg.get("evaluation", {})

        evaluator = SlidingWindowEvaluator(
            model=self.model,
            device=self.device,
            patch_size=self.patch_size,
            stride=self.stride,
            suv_thr=self.suv_thr,
            enable_mask=self.enable_mask,
            use_prior=self.use_prior,
            compute_perceptual=bool(eval_cfg.get("compute_perceptual", False)),
        )

        rows = evaluator.evaluate_dataset(
            self.val_loader.dataset,
            max_patients=eval_cfg.get("max_patients", None),
            shuffle=False,
        )

        summary = evaluator.summarize(rows)

        evaluator.append_summary_csv(
            summary=summary,
            csv_path=os.path.join(self.eval_dir, "val_summary.csv"),
            epoch=epoch,
            iteration=iteration,
            split="val",
        )

        print(
            f"[Eval][Val] E{epoch} I{iteration} | "
            f"SSIM={summary.get('SSIM', 0.0):.4f} "
            f"PSNR={summary.get('PSNR', 0.0):.4f} "
            f"MAE={summary.get('MAE', 0.0):.4f} "
            f"Dice={summary.get('Dice', 0.0):.4f}"
        )