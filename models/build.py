from .mapnet import MAPNet3D


def build_model(cfg):
    """
    Build MAP-Net from config.

    Expected config example:

    model:
      in_ch: 2
      out_ch: 1
      base_ch: 64
      enable_mask: true
      enable_mgca: true
      mgca_window: [4, 4, 4]
      mgca_heads: 2
      mgca_alpha: 0.05

    discriminator:
      type: fdd
      base_ch: 64
      r0_ratio: 0.25
      alpha: 1.0
      beta: 1.0

    loss:
      lambda_rec: 1.0
      lambda_adv: 0.01
    """
    model_cfg = cfg.get("model", {})
    loss_cfg = cfg.get("loss", {})
    mask_cfg = cfg.get("mask", {})
    disc_cfg = cfg.get("discriminator", {})

    enable_mask = bool(
        model_cfg.get(
            "enable_mask",
            mask_cfg.get("enable", True),
        )
    )

    enable_mgca = bool(
        model_cfg.get(
            "enable_mgca",
            model_cfg.get("enable_cross_attn", True),
        )
    )

    model = MAPNet3D(
        in_ch=int(model_cfg.get("in_ch", 2)),
        out_ch=int(model_cfg.get("out_ch", 1)),
        base_ch=int(model_cfg.get("base_ch", 64)),

        lambda_rec=float(loss_cfg.get("lambda_rec", 1.0)),
        lambda_adv=float(loss_cfg.get("lambda_adv", 0.01)),

        enable_mask=enable_mask,
        enable_mgca=enable_mgca,
        mgca_window=tuple(model_cfg.get("mgca_window", model_cfg.get("attn_window", [4, 4, 4]))),
        mgca_heads=int(model_cfg.get("mgca_heads", model_cfg.get("attn_heads", 2))),
        mgca_alpha=float(model_cfg.get("mgca_alpha", model_cfg.get("attn_alpha", 0.05))),
        output_activation=str(model_cfg.get("output_activation", "none")),

        discriminator_type=str(disc_cfg.get("type", "fdd")),
        disc_base_ch=int(disc_cfg.get("base_ch", 64)),
        disc_r0_ratio=float(disc_cfg.get("r0_ratio", 0.25)),
        disc_alpha=float(disc_cfg.get("alpha", 1.0)),
        disc_beta=float(disc_cfg.get("beta", 1.0)),
    )

    return model