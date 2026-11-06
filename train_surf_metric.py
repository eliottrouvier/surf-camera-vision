"""
Entraînement Approfondi Surf Vision avec Callback d'Évaluation Métrique en Continu.
Mesure l'erreur en centimètres sur la taille du surfeur et de la planche à chaque epoch
et retient le meilleur modèle selon une fitness combinée (mAP + précision métrique).
"""

import sys
import glob
import shutil
from pathlib import Path
import numpy as np
import torch
from ultralytics import YOLO


class MetricEvaluationCallback:
    """
    Callback d'évaluation métrique à chaque fin d'epoch.
    Évalue la précision de mesure physique (erreur en centimètres sur le surfeur et la planche)
    sur le set de validation et sauvegarde le meilleur modèle selon le score combiné.
    """
    def __init__(
        self,
        val_img_dir: Path,
        target_dest: Path,
        nominal_surfer_cm: float = 185.0,
        nominal_board_cm: float = 173.0,
    ):
        self.val_img_dir = val_img_dir
        self.target_dest = target_dest
        self.nominal_surfer_cm = nominal_surfer_cm
        self.nominal_board_cm = nominal_board_cm
        self.best_fitness = 0.0
        self.val_images = sorted(glob.glob(str(val_img_dir / "*.jpg")))

    def on_fit_epoch_end(self, trainer):
        epoch = getattr(trainer, "epoch", 0) + 1
        metrics = getattr(trainer, "metrics", {})

        map50_95 = float(metrics.get("metrics/mAP50-95(B)", 0.0))
        map50 = float(metrics.get("metrics/mAP50(B)", 0.0))

        errors_surfer = []
        errors_ratio = []

        # Échantillon de validation pour une évaluation rapide et stable
        sample_imgs = self.val_images[:30]
        save_dir = Path(trainer.save_dir) if hasattr(trainer, "save_dir") else Path("runs/surf_metric_deep/m4_100epochs")
        last_pt = save_dir / "weights" / "last.pt"

        if last_pt.exists() and sample_imgs:
            try:
                eval_model = YOLO(str(last_pt))
                results = eval_model.predict(sample_imgs, conf=0.25, verbose=False)
                for res in results:
                    if res.boxes is None or len(res.boxes) == 0:
                        continue
                    boxes = res.boxes.xyxy.cpu().numpy()
                    classes = res.boxes.cls.cpu().numpy().astype(int)

                    surfers = [b for b, c in zip(boxes, classes) if c == 0]
                    boards = [b for b, c in zip(boxes, classes) if c == 1]

                    if surfers and boards:
                        s = surfers[0]
                        b = boards[0]
                        s_h = max(1.0, float(s[3] - s[1]))
                        b_len = max(1.0, float(np.sqrt((b[2] - b[0])**2 + (b[3] - b[1])**2)))

                        # Hauteur estimée du surfeur en centimètres (étalonnée sur planche 173 cm)
                        pred_surfer_cm = (s_h / b_len) * self.nominal_board_cm
                        err_cm = abs(pred_surfer_cm - self.nominal_surfer_cm)
                        errors_surfer.append(err_cm)

                        # Écart au ratio physique nominal
                        pred_ratio = s_h / b_len
                        real_ratio = self.nominal_surfer_cm / self.nominal_board_cm
                        errors_ratio.append(abs(pred_ratio - real_ratio))
            except Exception as e:
                print(f"[Callback Warning] Évaluation métrique ignorée pour l'epoch {epoch}: {e}")

        mean_err_cm = float(np.mean(errors_surfer)) if errors_surfer else 15.0
        mean_ratio_err = float(np.mean(errors_ratio)) if errors_ratio else 0.15

        # Score métrique normalisé entre 0 et 1 (1.0 = erreur nulle, 0.5 = 10 cm d'erreur)
        metric_fitness = 10.0 / (10.0 + mean_err_cm)

        # Fitness combinée pondérée (60% détection mAP50-95, 40% précision physique)
        combined_fitness = 0.60 * map50_95 + 0.40 * metric_fitness

        print(f"\n📏 [ÉVALUATION MÉTRIQUE PHYSIQUE — EPOCH {epoch}]")
        print(f"   • mAP50-95(B)     : {map50_95:.4f} (mAP50: {map50:.4f})")
        print(f"   • Erreur Moyenne  : {mean_err_cm:.2f} cm (écart ratio: {mean_ratio_err:.3f})")
        print(f"   • Score Métrique  : {metric_fitness:.4f}")
        print(f"   • Score Combiné   : {combined_fitness:.4f} (Record actuel: {self.best_fitness:.4f})")

        if combined_fitness > self.best_fitness:
            self.best_fitness = combined_fitness
            self.target_dest.parent.mkdir(parents=True, exist_ok=True)
            best_pt = save_dir / "weights" / "best.pt"
            source_pt = best_pt if best_pt.exists() else last_pt
            if source_pt.exists():
                shutil.copy(source_pt, self.target_dest)
                print(f"   🎯 NOUVEAU RECORD MÉTRIQUE SAUVEGARDÉ -> {self.target_dest}")


def train(epochs: int = 100, batch_size: int = 16):
    yaml_path = Path("dataset_surf/dataset.yaml").resolve()
    if not yaml_path.exists():
        print(f"❌ Erreur : Fichier {yaml_path} introuvable.")
        sys.exit(1)

    val_img_dir = Path("dataset_surf/images/val").resolve()
    target_metric_weights = Path("weights/yolov8n_surf_metric_best.pt").resolve()

    # Détection de l'accélération matérielle Apple Silicon M4
    if torch.backends.mps.is_available():
        device = "mps"
        device_name = "Apple Silicon GPU (MPS Metal Acceleration)"
    elif torch.cuda.is_available():
        device = "cuda"
        device_name = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
    else:
        device = "cpu"
        device_name = "CPU"

    print("=" * 70)
    print("  🏄 Entraînement Approfondi Surf Vision avec Contrôle Métrique")
    print("=" * 70)
    print(f"• Matériel GPU     : {device_name}")
    print(f"• Dataset          : {yaml_path}")
    print(f"• Époques prévues  : {epochs}")
    print(f"• Taille de batch  : {batch_size}")
    print(f"• Poids de sortie  : {target_metric_weights}")
    print("=" * 70)

    # Initialisation du modèle YOLO
    model = YOLO("yolov8n.pt")

    # Enregistrement du callback personnalisé
    metric_cb = MetricEvaluationCallback(
        val_img_dir=val_img_dir,
        target_dest=target_metric_weights,
        nominal_surfer_cm=185.0,
        nominal_board_cm=173.0,
    )
    model.add_callback("on_fit_epoch_end", metric_cb.on_fit_epoch_end)

    # Lancement de l'entraînement
    results = model.train(
        data=str(yaml_path),
        epochs=epochs,
        imgsz=640,
        batch=batch_size,
        device=device,
        freeze=10,             # Geler les couches backbone initiales
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        dropout=0.10,
        cos_lr=True,           # Décroissance cosinusoïdale du taux d'apprentissage
        project="runs/surf_metric_deep",
        name="m4_100epochs",
        save=True,
        verbose=True,
    )

    print("\n🎉 Entraînement approfondi terminé avec succès !")
    if target_metric_weights.exists():
        print(f"💾 Modèle optimal métrique prêt à l'emploi : {target_metric_weights}")

    return results


if __name__ == "__main__":
    epochs_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 100
    batch_arg = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    train(epochs=epochs_arg, batch_size=batch_arg)
