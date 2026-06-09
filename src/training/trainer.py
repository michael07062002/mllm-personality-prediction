from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.training.losses import combined_loss
from src.training.metrics import build_ccc_metrics, infer_target_names
from src.training.utils import (
    AverageMeter,
    build_output_dirs,
    count_parameters,
    detach_to_cpu,
    move_batch_to_device,
    plot_training_curves,
    save_json,
    save_training_history,
)


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        outputs_root: str | Path,
        model_name: str,
        layers: List[int],
        scheduler: Optional[Any] = None,
        max_grad_norm: Optional[float] = None,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.max_grad_norm = max_grad_norm

        self.output_dirs = build_output_dirs(
            outputs_root=outputs_root,
            model_name=model_name,
            layers=layers,
        )

        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "test_loss": [],
            "train_mean_ccc": [],
            "test_mean_ccc": [],
        }

        self.best_mean_ccc = float("-inf")
        self.best_epoch = -1

        self.model.to(self.device)

    def save_run_config(self, config: Dict[str, Any], filename: str = "run_config.json") -> None:
        save_json(config, self.output_dirs["logs_dir"] / filename)

    def save_model_info(self, extra: Optional[Dict[str, Any]] = None) -> None:
        info = count_parameters(self.model)
        if extra is not None:
            info.update(extra)
        save_json(info, self.output_dirs["logs_dir"] / "model_info.json")

    def _forward_step(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        x = batch["x"]
        mask = batch["mask"]
        y = batch["y"]

        outputs = self.model(x=x, mask=mask)
        preds = outputs["preds"]

        if preds.shape != y.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: preds={preds.shape}, y={y.shape}"
            )

        loss = combined_loss(preds, y, alpha=0.2)

        outputs["loss"] = loss
        return outputs

    def _get_target_names_from_batch_or_tensor(
        self,
        batch: Dict[str, Any] | None,
        y: torch.Tensor,
    ) -> List[str]:
        if batch is not None:
            names = batch.get("target_names")
            if isinstance(names, list) and len(names) == y.shape[1]:
                return [str(x) for x in names]

        return infer_target_names(int(y.shape[1]))

    def _run_one_epoch(
        self,
        loader: DataLoader,
        train: bool,
        epoch: int,
    ) -> Dict[str, Any]:
        mode = "train" if train else "test"
        self.model.train(train)

        loss_meter = AverageMeter()
        all_preds: List[torch.Tensor] = []
        all_targets: List[torch.Tensor] = []
        all_video_ids: List[str] = []
        target_names: List[str] | None = None

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [{mode}]", leave=False)

        for batch in pbar:
            batch = move_batch_to_device(batch, self.device)

            if train:
                self.optimizer.zero_grad(set_to_none=True)

            outputs = self._forward_step(batch)
            loss = outputs["loss"]

            if train:
                loss.backward()
                if self.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            bs = batch["y"].shape[0]
            loss_meter.update(loss.item(), n=bs)

            preds_cpu = detach_to_cpu(outputs["preds"])
            y_cpu = detach_to_cpu(batch["y"])

            all_preds.append(preds_cpu)
            all_targets.append(y_cpu)
            all_video_ids.extend(batch["video_id"])

            if target_names is None:
                target_names = self._get_target_names_from_batch_or_tensor(batch, y_cpu)

            pbar.set_postfix(loss=f"{loss_meter.avg:.6f}")

        y_pred = torch.cat(all_preds, dim=0)
        y_true = torch.cat(all_targets, dim=0)

        if target_names is None:
            target_names = infer_target_names(int(y_true.shape[1]))

        metrics = build_ccc_metrics(
            y_true=y_true,
            y_pred=y_pred,
            target_names=target_names,
        )

        result = {
            "loss": float(loss_meter.avg),
            "metrics": metrics,
            "y_true": y_true,
            "y_pred": y_pred,
            "video_ids": all_video_ids,
            "target_names": target_names,
        }
        return result

    def _append_history_value(self, key: str, value: float) -> None:
        if key not in self.history:
            self.history[key] = []
        self.history[key].append(float(value))

    def _update_history(self, train_result: Dict[str, Any], test_result: Dict[str, Any]) -> None:
        self.history["train_loss"].append(float(train_result["loss"]))
        self.history["test_loss"].append(float(test_result["loss"]))

        self.history["train_mean_ccc"].append(float(train_result["metrics"]["mean_ccc"]))
        self.history["test_mean_ccc"].append(float(test_result["metrics"]["mean_ccc"]))

        for key, value in train_result["metrics"].items():
            if key == "mean_ccc":
                continue
            self._append_history_value(f"train_{key}", float(value))

        for key, value in test_result["metrics"].items():
            if key == "mean_ccc":
                continue
            self._append_history_value(f"test_{key}", float(value))

    def _save_epoch_predictions(
        self,
        result: Dict[str, Any],
        split: str,
        epoch: int,
    ) -> None:
        y_true = result["y_true"].numpy()
        y_pred = result["y_pred"].numpy()
        video_ids = result["video_ids"]

        target_names = result.get("target_names")
        if not target_names:
            target_names = infer_target_names(int(y_true.shape[1]))

        rows = []
        for i, vid in enumerate(video_ids):
            row = {"video_id": vid}
            for j, name in enumerate(target_names):
                row[f"true_{name}"] = float(y_true[i, j])
                row[f"pred_{name}"] = float(y_pred[i, j])
            rows.append(row)

        df = pd.DataFrame(rows)
        out_path = self.output_dirs["predictions_dir"] / f"{split}_epoch_{epoch:03d}.csv"
        df.to_csv(out_path, index=False)

    def _save_checkpoint(self, epoch: int, mean_ccc: float, is_best: bool) -> None:
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_mean_ccc": self.best_mean_ccc,
            "current_mean_ccc": mean_ccc,
            "history": self.history,
        }

        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        last_path = self.output_dirs["checkpoints_dir"] / "last.pt"
        torch.save(checkpoint, last_path)

        if is_best:
            best_path = self.output_dirs["checkpoints_dir"] / "best.pt"
            torch.save(checkpoint, best_path)

    def _format_target_line(self, prefix: str, metrics: Dict[str, float]) -> str:
        parts = []
        for key, value in metrics.items():
            if not key.startswith("ccc_"):
                continue
            name = key.replace("ccc_", "", 1)
            parts.append(f"{name}={value:.4f}")

        if not parts:
            return f"{prefix} no target CCC metrics"

        return f"{prefix} " + "  ".join(parts)

    def _print_epoch_summary(
        self,
        epoch: int,
        train_result: Dict[str, Any],
        test_result: Dict[str, Any],
    ) -> None:
        print(f"\nEpoch {epoch:03d}")
        print(f"train_loss:     {train_result['loss']:.6f}")
        print(f"test_loss:      {test_result['loss']:.6f}")
        print(f"train_mean_ccc: {train_result['metrics']['mean_ccc']:.4f}")
        print(f"test_mean_ccc:  {test_result['metrics']['mean_ccc']:.4f}")
        print(self._format_target_line("train_ccc:", train_result["metrics"]))
        print(self._format_target_line("test_ccc: ", test_result["metrics"]))
        print(f"best_mean_ccc:  {self.best_mean_ccc:.4f}")
        print(f"best_epoch:     {self.best_epoch}")

    def _safe_plot_training_curves(self) -> None:
        try:
            plot_training_curves(
                self.history,
                self.output_dirs["plots_dir"],
            )
        except Exception as e:
            print(f"[warn] plot_training_curves failed, training continues: {e}")

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        num_epochs: int,
        save_every_epoch_predictions: bool = False,
    ) -> Dict[str, Any]:
        for epoch in range(1, num_epochs + 1):
            train_result = self._run_one_epoch(train_loader, train=True, epoch=epoch)
            test_result = self._run_one_epoch(test_loader, train=False, epoch=epoch)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(test_result["metrics"]["mean_ccc"])
                else:
                    self.scheduler.step()

            self._update_history(train_result, test_result)

            current_mean_ccc = float(test_result["metrics"]["mean_ccc"])
            is_best = current_mean_ccc > self.best_mean_ccc

            if is_best:
                self.best_mean_ccc = current_mean_ccc
                self.best_epoch = epoch

            self._save_checkpoint(
                epoch=epoch,
                mean_ccc=current_mean_ccc,
                is_best=is_best,
            )

            if save_every_epoch_predictions:
                self._save_epoch_predictions(train_result, split="train", epoch=epoch)
                self._save_epoch_predictions(test_result, split="test", epoch=epoch)

            save_training_history(
                self.history,
                self.output_dirs["logs_dir"] / "history.json",
            )

            self._safe_plot_training_curves()

            self._print_epoch_summary(
                epoch=epoch,
                train_result=train_result,
                test_result=test_result,
            )

        final_summary = {
            "best_mean_ccc": self.best_mean_ccc,
            "best_epoch": self.best_epoch,
            "history": self.history,
            "output_dirs": {k: str(v) for k, v in self.output_dirs.items()},
        }
        save_json(final_summary, self.output_dirs["logs_dir"] / "final_summary.json")
        return final_summary