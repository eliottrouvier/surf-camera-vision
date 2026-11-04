"""
Local Fine-Tuning Script for Surf AI v2 on Apple Silicon (MPS).
Fine-tunes YOLOv8 on the custom surf dataset extracted from ~/Desktop/videos surf.
"""

import sys
from pathlib import Path
import torch
from ultralytics import YOLO


def train():
    yaml_path = Path("dataset_surf/dataset.yaml")
    if not yaml_path.exists():
        print(f"❌ Erreur : Fichier {yaml_path} introuvable. Exécutez d'abord dataset_extractor.py !")
        sys.exit(1)

    # Detect Apple Silicon hardware acceleration
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
    print("  🏄 Entraînement Local Surf AI v2 (YOLOv8 Surf Custom)")
    print("=" * 70)
    print(f"• Matériel détecté : {device_name}")
    print(f"• Dataset config   : {yaml_path.resolve()}")
    print("=" * 70)

    # Load baseline model
    model = YOLO("yolov8n.pt")

    # Start fine-tuning
    results = model.train(
        data=str(yaml_path.resolve()),
        epochs=30,
        imgsz=640,
        batch=16,
        device=device,
        freeze=10,             # Freeze early backbone layers
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.01,
        dropout=0.10,
        cos_lr=True,           # Cosine learning rate scheduler
        project="runs/surf_finetuning",
        name="surf_v2_m4",
        save=True,
        verbose=True,
    )

    print("\n🎉 Entraînement terminé avec succès !")
    best_weights = Path("runs/surf_finetuning/surf_v2_m4/weights/best.pt")
    if best_weights.exists():
        target_dest = Path("weights/yolov8n_surf_v2_best.pt")
        target_dest.parent.mkdir(parents=True, exist_ok=True)
        import shutil
        shutil.copy(best_weights, target_dest)
        print(f"💾 Poids optimisés exportés : {target_dest.resolve()}")

    return results


if __name__ == "__main__":
    train()
