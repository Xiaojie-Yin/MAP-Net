import os
import sys
import yaml

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT_DIR)

from datasets.build import build_dataloaders


def main():
    with open("configs/mapnet_3d.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_loader, val_loader, test_loader = build_dataloaders(cfg, include_test=True)

    print("Train batches:", len(train_loader))
    print("Val batches:", len(val_loader))
    print("Test batches:", len(test_loader))

    batch = next(iter(train_loader))
    src, pet = batch[:2]

    print("src:", src.shape)
    print("pet:", pet.shape)
    print("src min/max:", float(src.min()), float(src.max()))
    print("pet min/max:", float(pet.min()), float(pet.max()))

    assert src.dim() == 5
    assert pet.dim() == 5
    assert src.shape[1] == 2
    assert pet.shape[1] == 1

    print("Dataset check passed.")


if __name__ == "__main__":
    main()