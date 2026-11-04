"""
Automated Dataset Extractor & Marine Data Augmentation Engine.
Extracts sharp keyframes from personal surf footage (~/Desktop/videos surf),
applies pseudo-labeling with YOLOv8, performs marine data augmentations
(sunset glare, foam spray occlusion, horizontal mirroring), and formats
a clean YOLO training dataset split 90% Train / 10% Val.
"""

import os
import glob
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import cv2
import numpy as np
import yaml
from ultralytics import YOLO


class SurfDatasetExtractor:
    """
    Extracts, filters, annotates, and augments surf footage into a compact YOLO dataset.
    """
    def __init__(
        self,
        video_dir: str = "/Users/eliottrouvier/Desktop/videos surf",
        output_dir: str = "dataset_surf",
        model_name: str = "yolov8n-seg.pt",
        target_clean_frames: int = 150,
        train_val_split: float = 0.90,
    ):
        self.video_dir = Path(video_dir)
        self.output_dir = Path(output_dir)
        self.target_clean_frames = target_clean_frames
        self.train_val_split = train_val_split
        self.model = YOLO(model_name)

        # Classes mapping:
        # COCO person (0) -> surfer (0)
        # COCO surfboard (37) -> surfboard (1)
        self.CLASS_MAP = {0: 0, 37: 1}
        self.CLASS_NAMES = ["surfer", "surfboard"]

    def extract_and_build(self) -> Dict:
        """Runs the complete extraction, labeling, augmentation, and split pipeline."""
        print("=" * 70)
        print("  🏄 Extraction & Génération du Dataset Surf Personnalisé")
        print("=" * 70)

        # 1. Discover videos
        video_paths = sorted(
            glob.glob(str(self.video_dir / "*.mov")) +
            glob.glob(str(self.video_dir / "*.mp4")) +
            glob.glob(str(self.video_dir / "video-surf1" / "*.mp4"))
        )
        print(f"📹 {len(video_paths)} vidéos détectées dans {self.video_dir}")
        if not video_paths:
            raise ValueError(f"Aucune vidéo trouvée dans {self.video_dir}")

        # Setup directory structure
        img_train_dir = self.output_dir / "images" / "train"
        img_val_dir = self.output_dir / "images" / "val"
        lbl_train_dir = self.output_dir / "labels" / "train"
        lbl_val_dir = self.output_dir / "labels" / "val"

        for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # 2. Extract sharp frames with pseudo-labels
        samples = []  # [(image_bgr, labels_list), ...]
        frames_per_video = max(2, self.target_clean_frames // len(video_paths) + 1)

        random.seed(42)
        shuffled_videos = video_paths.copy()
        random.shuffle(shuffled_videos)

        for v_path in shuffled_videos:
            if len(samples) >= self.target_clean_frames:
                break

            cap = cv2.VideoCapture(v_path)
            if not cap.isOpened():
                continue

            total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
            if total_f < 15:
                cap.release()
                continue

            # Sample spaced frames
            step = max(10, total_f // (frames_per_video + 2))
            frame_indices = list(range(step, total_f - step, step))[:frames_per_video]

            for f_idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
                ret, frame = cap.read()
                if not ret or frame is None:
                    continue

                # Filter blur using Laplacian variance
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
                if lap_var < 45.0:  # Skip excessively blurry frames
                    continue

                # Detect surfers and boards
                labels = self._pseudo_label_frame(frame)
                if len(labels) > 0:
                    samples.append((frame, labels))
                    if len(samples) >= self.target_clean_frames:
                        break

            cap.release()

        print(f"✅ {len(samples)} frames nettes annotées extraites.")

        # 3. Apply Marine Data Augmentations
        augmented_dataset = []
        for orig_img, orig_labels in samples:
            augmented_dataset.append((orig_img, orig_labels, "orig"))

            # Augmentation A: Horizontal Flip (Right / Left wave symmetry)
            h_img, h_lbls = self._augment_horizontal_flip(orig_img, orig_labels)
            augmented_dataset.append((h_img, h_lbls, "flip"))

            # Augmentation B: Sunset / Sun Glare (Warm contrast shift)
            s_img, s_lbls = self._augment_sunset_glare(orig_img, orig_labels)
            augmented_dataset.append((s_img, s_lbls, "sunset"))

            # Augmentation C: Water Foam & Spray Occlusion
            f_img, f_lbls = self._augment_spray_foam(orig_img, orig_labels)
            augmented_dataset.append((f_img, f_lbls, "spray"))

        print(f"⚡ Dataset enrichi avec augmentations marines : {len(augmented_dataset)} images au total.")

        # 4. Split 90% Train / 10% Val
        random.shuffle(augmented_dataset)
        split_idx = int(len(augmented_dataset) * self.train_val_split)
        train_data = augmented_dataset[:split_idx]
        val_data = augmented_dataset[split_idx:]

        self._save_subset(train_data, img_train_dir, lbl_train_dir, prefix="train")
        self._save_subset(val_data, img_val_dir, lbl_val_dir, prefix="val")

        # 5. Generate dataset.yaml
        yaml_content = {
            "path": str(self.output_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "names": {
                0: "surfer",
                1: "surfboard",
            }
        }
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, "w", encoding="utf-8") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

        # 6. Generate Preview Mosaic
        mosaic_path = self.output_dir / "dataset_preview.jpg"
        self._generate_preview_mosaic(train_data[:9], mosaic_path)

        total_disk_mb = sum(
            f.stat().st_size for f in self.output_dir.rglob("*") if f.is_file()
        ) / (1024 * 1024)

        print(f"\n🎉 Dataset Surf AI v2 prêt !")
        print(f"• Images d'entraînement : {len(train_data)} (90%)")
        print(f"• Images de validation   : {len(val_data)} (10%)")
        print(f"• Espace disque total    : {total_disk_mb:.1f} Mo (ultra-léger)")
        print(f"• Fichier de config      : {yaml_path}")
        print(f"• Aperçu des annotations : {mosaic_path}")

        return {
            "total_images": len(augmented_dataset),
            "train_images": len(train_data),
            "val_images": len(val_data),
            "yaml_path": str(yaml_path),
            "preview_path": str(mosaic_path),
            "disk_mb": round(total_disk_mb, 1),
        }

    def _pseudo_label_frame(self, frame: np.ndarray) -> List[List[float]]:
        """Runs YOLO to extract bounding boxes in normalized format [cls, cx, cy, w, h]."""
        results = self.model.predict(
            source=frame,
            conf=0.25,
            classes=[0, 37],
            verbose=False,
            device="cpu",
        )
        if not results or len(results) == 0:
            return []

        res = results[0]
        if res.boxes is None or len(res.boxes) == 0:
            return []

        h, w = frame.shape[:2]
        labels = []

        coords = res.boxes.xywhn.cpu().numpy()
        classes = res.boxes.cls.cpu().numpy().astype(int)

        for bbox_norm, cls_id in zip(coords, classes):
            if cls_id in self.CLASS_MAP:
                new_cls = self.CLASS_MAP[cls_id]
                cx, cy, bw, bh = bbox_norm
                labels.append([new_cls, float(cx), float(cy), float(bw), float(bh)])

        return labels

    def _augment_horizontal_flip(
        self, img: np.ndarray, labels: List[List[float]]
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Mirrors the frame horizontally and adjusts bounding boxes."""
        flipped_img = cv2.flip(img, 1)
        flipped_labels = []
        for cls_id, cx, cy, bw, bh in labels:
            flipped_labels.append([cls_id, 1.0 - cx, cy, bw, bh])
        return flipped_img, flipped_labels

    def _augment_sunset_glare(
        self, img: np.ndarray, labels: List[List[float]]
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Applies warm golden sunset tone and slight highlights boost."""
        out = img.astype(np.float32)
        # Boost Red & Green slightly (warmth)
        out[:, :, 2] = np.clip(out[:, :, 2] * 1.15 + 10, 0, 255)  # R
        out[:, :, 1] = np.clip(out[:, :, 1] * 1.05 + 5, 0, 255)   # G
        out[:, :, 0] = np.clip(out[:, :, 0] * 0.90, 0, 255)       # B
        return out.astype(np.uint8), [l.copy() for l in labels]

    def _augment_spray_foam(
        self, img: np.ndarray, labels: List[List[float]]
    ) -> Tuple[np.ndarray, List[List[float]]]:
        """Procedurally draws realistic white foam splashes over part of the surfer/board."""
        out = img.copy()
        h, w = out.shape[:2]

        for cls_id, cx, cy, bw, bh in labels:
            px = int(cx * w)
            py = int((cy + bh * 0.25) * h)
            rx = max(5, int(bw * w * 0.4))
            ry = max(4, int(bh * h * 0.2))

            # Draw 2-3 foam droplets
            for _ in range(3):
                ox = px + random.randint(-rx, rx)
                oy = py + random.randint(-ry, ry)
                spray_r = random.randint(3, max(4, rx // 2))
                cv2.circle(out, (ox, oy), spray_r, (250, 250, 255), -1)

        return out, [l.copy() for l in labels]

    def _save_subset(
        self, data: List[Tuple[np.ndarray, List[List[float]], str]],
        img_dir: Path, lbl_dir: Path, prefix: str
    ):
        """Saves images as high quality JPGs and labels as YOLO format .txt files."""
        for idx, (img, labels, aug_type) in enumerate(data):
            base_name = f"{prefix}_{idx:04d}_{aug_type}"
            img_path = img_dir / f"{base_name}.jpg"
            lbl_path = lbl_dir / f"{base_name}.txt"

            cv2.imwrite(str(img_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 92])

            with open(lbl_path, "w", encoding="utf-8") as f:
                for l in labels:
                    cls_id, cx, cy, bw, bh = l
                    f.write(f"{int(cls_id)} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n")

    def _generate_preview_mosaic(
        self, samples: List[Tuple[np.ndarray, List[List[float]], str]], out_path: Path
    ):
        """Generates a 3x3 visual contact sheet with bounding boxes overlaid."""
        if not samples:
            return

        mosaic_tiles = []
        target_size = (384, 216)

        for img, labels, aug_type in samples[:9]:
            vis = img.copy()
            h, w = vis.shape[:2]

            for cls_id, cx, cy, bw, bh in labels:
                x1 = int((cx - bw / 2.0) * w)
                y1 = int((cy - bh / 2.0) * h)
                x2 = int((cx + bw / 2.0) * w)
                y2 = int((cy + bh / 2.0) * h)

                color = (0, 255, 120) if cls_id == 0 else (0, 220, 255)
                lbl_name = "Surfer" if cls_id == 0 else "Board"

                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, lbl_name, (x1, max(15, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

            vis_resized = cv2.resize(vis, target_size, interpolation=cv2.INTER_AREA)
            cv2.putText(vis_resized, f"Aug: {aug_type}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
            mosaic_tiles.append(vis_resized)

        # Fill up to 9 tiles
        while len(mosaic_tiles) < 9:
            mosaic_tiles.append(np.zeros((target_size[1], target_size[0], 3), dtype=np.uint8))

        row1 = np.hstack(mosaic_tiles[0:3])
        row2 = np.hstack(mosaic_tiles[3:6])
        row3 = np.hstack(mosaic_tiles[6:9])
        mosaic = np.vstack([row1, row2, row3])

        cv2.imwrite(str(out_path), mosaic, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


if __name__ == "__main__":
    extractor = SurfDatasetExtractor()
    extractor.extract_and_build()
