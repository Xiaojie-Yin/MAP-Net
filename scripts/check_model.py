import torch

from models.build import build_model


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main():
    cfg = {
        "model": {
            "in_ch": 2,
            "out_ch": 1,
            "base_ch": 16,  # 测试时先用小一点，避免显存不够
            "enable_mask": True,
            "enable_mgca": True,
            "mgca_window": [4, 4, 4],
            "mgca_heads": 2,
            "mgca_alpha": 0.05,
            "output_activation": "none",
        },
        "discriminator": {
            "type": "fdd",
            "base_ch": 16,
            "r0_ratio": 0.25,
            "alpha": 1.0,
            "beta": 1.0,
        },
        "loss": {
            "lambda_rec": 1.0,
            "lambda_adv": 0.01,
        },
        "mask": {
            "enable": True,
        },
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(cfg).to(device)
    model.train()

    print("Model:", model.__class__.__name__)
    print("Trainable parameters:", count_params(model))

    # 假输入：CT + SDM，两通道
    # D/H/W 最好能被 8 整除，因为 U-Net 下采样 3 次
    x = torch.randn(2, 2, 32, 96, 96, device=device)
    pet = torch.randn(2, 1, 32, 96, 96, device=device)

    # forward with mask
    pred, mask_logits, aux_loss = model.forward_with_mask(x)

    print("Input:", x.shape)
    print("Pred PET:", pred.shape)
    print("Mask logits:", mask_logits.shape)
    print("Aux loss:", aux_loss.item())

    assert pred.shape == pet.shape, f"pred shape mismatch: {pred.shape} vs {pet.shape}"
    assert mask_logits.shape == (2, 2, 32, 96, 96), f"mask shape mismatch: {mask_logits.shape}"

    # SUV domain
    pred_raw = (pred + 1.0) * 0.5 * 20.0
    gt_raw = (pet + 1.0) * 0.5 * 20.0

    suv_thr = 5.0
    m_fg_real = (gt_raw > suv_thr).float()
    m_bg_real = 1.0 - m_fg_real
    target_onehot_real = torch.cat([m_bg_real, m_fg_real], dim=1)

    m_fg_spet = (pred_raw > suv_thr).float()
    m_bg_spet = 1.0 - m_fg_spet
    target_onehot_spet = torch.cat([m_bg_spet, m_fg_spet], dim=1)

    mtv_cfg = {
        "thr": [2.5, 4.0, 5.0],
        "tau": 0.6,
        "mode": "relative",
        "weight": 0.1,
        "weights": None,
    }

    sparse_cfg = {
        "thr": 2.5,
        "tau": 0.5,
        "weight": 0.01,
        "reduce": "mean_over_mask",
    }

    mask_cfg = {
        "kappa": 0.01,
        "w_bg": 0.2,
        "w_fg": 0.8,
        "lambda_dice": 1.0,
        "lambda_ce_all": 0.2,
        "w_bg_ce": 0.1,
        "w_fg_ce": 0.9,
        "lambda_consistency": 0.2,
        "lambda_sparse": 0.02,
        "target_fg_ratio": 0.01,
    }

    # generator loss
    g_loss, glog = model.g_loss(
        real=pet,
        fake=pred,
        aux_loss=aux_loss,
        pred_raw=pred_raw,
        gt_raw=gt_raw,
        mtv_cfg=mtv_cfg,
        tlg_cfg=None,
        sum_weight=0.0,
        sparse_cfg=sparse_cfg,
        grad_weight=0.01,
        mask_logits=mask_logits,
        mask_target_onehot=target_onehot_real,
        mask_spet_onehot=target_onehot_spet,
        mask_cfg=mask_cfg,
    )

    print("G loss:", float(g_loss.item()))
    print("G log keys:", sorted(glog.keys()))

    g_loss.backward()
    print("G backward: OK")

    # discriminator loss
    model.zero_grad(set_to_none=True)
    d_loss, dlog = model.d_loss(pet, pred.detach())
    print("D loss:", float(d_loss.item()))
    print("D log:", {k: float(v.item()) for k, v in dlog.items()})

    d_loss.backward()
    print("D backward: OK")

    # Check whether MGCA exists
    mgca_keys = [name for name, _ in model.named_parameters() if "mgca" in name.lower()]
    print("MGCA parameter examples:")
    for k in mgca_keys[:10]:
        print("  ", k)

    assert len(mgca_keys) > 0, "MGCA parameters not found. enable_mgca may be False."

    # Check whether FDD exists
    fdd_keys = [name for name, _ in model.named_parameters() if "spectral" in name.lower()]
    print("FDD spectral parameter examples:")
    for k in fdd_keys[:10]:
        print("  ", k)

    assert len(fdd_keys) > 0, "FDD spectral branch not found. discriminator type may not be fdd."

    print("\nAll model checks passed.")


if __name__ == "__main__":
    main()