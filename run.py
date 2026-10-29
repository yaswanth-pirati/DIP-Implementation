import os
import torch
from engine.dehaze import train, train_one_epoch, valid
from data.uieb import UIEBTrain, UIEBValid
from torch.utils.data import DataLoader
from timm.optim import AdamW
from timm.scheduler import CosineLRScheduler
from model.base import CLCC
from utils.common_utils import *
from torch.amp import GradScaler
from tqdm import tqdm


def configuration_dataloader(hparams, stage_index):
    train_dataset = UIEBTrain(
        folder=hparams["data"]["train_path"],
        size=hparams["data"]["train_img_size"][stage_index],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=hparams["data"]["train_batch_size"][stage_index],
        shuffle=True,
        num_workers=hparams["data"]["num_workers"],
        pin_memory=hparams["data"]["pin_memory"],
    )

    valid_dataset = UIEBValid(folder=hparams["data"]["valid_path"], size=256)

    valid_loader = DataLoader(valid_dataset, batch_size=1)

    return train_loader, valid_loader


def configuration_optimizer(model, hparams):
    optimizer = AdamW(model.parameters(), lr=hparams["optim"]["lr_init"])

    scheduler = CosineLRScheduler(
        optimizer,
        t_initial=sum(hparams["train"]["stage_epochs"]),
        lr_min=hparams["optim"]["lr_min"],
    )

    return optimizer, scheduler


if __name__ == "__main__":
    args = parse_yaml("./config.yaml")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args["train"]["device"] = device

    model = CLCC(64, 3, 3).to(device)

    scaler = GradScaler(enabled=args["train"]["use_amp"])

    logger = Logger("./log")

    optimizer, scheduler = configuration_optimizer(model, args)

    total_epochs = sum(args["train"]["stage_epochs"])

    best_metric = {
        "ssim": {"value": 0, "epoch": 0},
        "psnr": {"value": 0, "epoch": 0},
    }

    epoch_counter = 1

    print(f"\n==========> TOTAL EPOCHS: {total_epochs} <==========\n")

    for stage in range(len(args["train"]["stage_epochs"])):

        train_loader, valid_loader = configuration_dataloader(args, stage)

        stage_epochs = args["train"]["stage_epochs"][stage]

        print(
            f"\n\033[92m====== Stage {stage+1} ({stage_epochs} epochs) ======\033[0m\n"
        )

        for _ in range(stage_epochs):

            print(f"\nEpoch [{epoch_counter}/{total_epochs}]")

            # -------- TRAIN --------
            train_result = train_one_epoch(
                args, model, scaler, optimizer, scheduler, train_loader
            )

            scheduler.step(epoch_counter)

            # -------- VALID --------
            valid_result = valid(args, model, valid_loader)

            # -------- UPDATE BEST --------
            if valid_result["ssim"] > best_metric["ssim"]["value"]:
                best_metric["ssim"] = {
                    "value": valid_result["ssim"],
                    "epoch": epoch_counter,
                }
                save_all(
                    epoch_counter,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    args,
                    best_metric,
                    "best_ssim",
                )

            if valid_result["psnr"] > best_metric["psnr"]["value"]:
                best_metric["psnr"] = {
                    "value": valid_result["psnr"],
                    "epoch": epoch_counter,
                }
                save_all(
                    epoch_counter,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    args,
                    best_metric,
                    "best_psnr",
                )

            # -------- SAVE LAST --------
            save_all(
                epoch_counter,
                model,
                optimizer,
                scheduler,
                scaler,
                args,
                best_metric,
                "last",
            )

            # -------- PRINT CLEAN OUTPUT --------
            print(
                f"Train Loss: {train_result['train_loss']:.4f} | "
                f"Valid Loss: {valid_result['valid_loss']:.4f}"
            )

            print(
                f"SSIM: {valid_result['ssim']:.4f} | "
                f"PSNR: {valid_result['psnr']:.4f}"
            )

            print(
                f"BEST SSIM: {best_metric['ssim']['value']:.4f} "
                f"(Epoch {best_metric['ssim']['epoch']})"
            )

            print(
                f"BEST PSNR: {best_metric['psnr']['value']:.4f} "
                f"(Epoch {best_metric['psnr']['epoch']})"
            )

            print("-" * 60)

            epoch_counter += 1
