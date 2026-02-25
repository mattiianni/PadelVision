"""
Player tracking con YOLOv8 — ottimizzato per CPU multi-core.

Strategia:
  - Singolo processo con torch.set_num_threads(N_CORES) → usa tutti i core
  - Inference in batch su più frame contemporaneamente
  - model.predict() con stream=True (più veloce di .track() per heatmap)
  - Assegnazione giocatori per posizione campo (nessun ByteTrack necessario)
"""

import cv2
import numpy as np
import torch
import multiprocessing
import os
from ultralytics import YOLO


# Usa tutti i core logici disponibili per l'inference PyTorch/OpenCV
_N_CORES = multiprocessing.cpu_count()
torch.set_num_threads(_N_CORES)
cv2.setNumThreads(_N_CORES)


class PlayerTracker:
    def __init__(
        self,
        model_name: str = "yolov8n.pt",
        conf: float = 0.35,
        batch_size: int = 8,
    ):
        """
        model_name : 'yolov8n.pt' (veloce) o 'yolov8s.pt' (preciso)
        conf       : soglia confidenza detection
        batch_size : frame per batch (aumenta se hai RAM sufficiente)
        """
        print(f"  Caricamento modello {model_name} — {_N_CORES} core attivi...")
        os.environ["YOLO_VERBOSE"] = "False"
        self.model = YOLO(model_name)
        self.conf = conf
        self.batch_size = batch_size

    def track_video(
        self,
        video_path: str,
        calibrator,
        sample_every: int = 2,
        progress_callback=None,
    ) -> tuple[dict, float, int]:
        """
        Processa il video in batch per massimizzare l'uso dei core.

        Ritorna:
          tracks       — dict {player_id: [(cx, cy, frame_num), ...]}
          fps          — frame rate
          total_frames — frame totali
        """
        cap = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        all_detections = []  # lista di (cx, cy, frame_num)
        frame_num  = 0
        batch      = []
        batch_idxs = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % sample_every == 0:
                batch.append(frame)
                batch_idxs.append(frame_num)

            # Inference quando batch è pieno o fine video
            if len(batch) >= self.batch_size or (not ret and batch):
                self._infer_batch(batch, batch_idxs, calibrator, all_detections)
                batch.clear()
                batch_idxs.clear()

            frame_num += 1
            if progress_callback and frame_num % 30 == 0:
                progress_callback(frame_num, total_frames)

        # Eventuale batch residuo
        if batch:
            self._infer_batch(batch, batch_idxs, calibrator, all_detections)

        cap.release()

        if progress_callback:
            progress_callback(total_frames, total_frames)

        tracks = self._assign_player_ids(all_detections)
        return tracks, fps, total_frames

    # ------------------------------------------------------------------

    def _infer_batch(self, frames, frame_nums, calibrator, out_list):
        """Esegue YOLO su un batch di frame e trasforma le posizioni."""
        results = self.model.predict(
            frames,
            classes=[0],      # solo persone
            conf=self.conf,
            verbose=False,
            imgsz=640,
        )
        for result, frame_num in zip(results, frame_nums):
            if result.boxes is None or len(result.boxes) == 0:
                continue
            boxes = result.boxes.xyxy.cpu().numpy()
            # Piede = bottom-center bbox
            feet = np.column_stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,
                boxes[:, 3],
            ])
            court_pts = calibrator.transform_points_batch(feet)
            for (cx, cy) in court_pts:
                out_list.append((float(cx), float(cy), frame_num))

    # ------------------------------------------------------------------
    # Assegnazione ID giocatori tramite k-means sul campo
    # ------------------------------------------------------------------

    @staticmethod
    def _assign_player_ids(
        all_positions: list,
        k: int = 4,
        court_length: float = 20.0,
    ) -> dict:
        """
        Raggruppa le detection in k cluster (=giocatori) con k-means 2D.
        Non serve tracking ID cross-frame: per le heatmap conta la posizione.
        """
        if not all_positions:
            return {}

        pts = np.array([(cx, cy) for cx, cy, _ in all_positions], dtype=np.float32)

        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
        _, labels, centers = cv2.kmeans(
            pts, k, None, criteria, 10, cv2.KMEANS_PP_CENTERS
        )
        labels = labels.flatten()

        # Ordina i cluster per posizione y (squadra A prima, squadra B dopo)
        order = np.argsort(centers[:, 1])
        remap = {old: new + 1 for new, old in enumerate(order)}

        tracks = {}
        for old_pid in range(k):
            new_pid = remap[old_pid]
            mask = labels == old_pid
            tracks[new_pid] = [all_positions[i] for i in range(len(all_positions)) if mask[i]]

        return tracks

    @staticmethod
    def filter_players(
        tracks: dict,
        min_frames: int = 30,
        max_players: int = 4,
    ) -> dict:
        filtered = {
            pid: pos for pid, pos in tracks.items()
            if len(pos) >= min_frames
        }
        return dict(
            sorted(filtered.items(), key=lambda x: len(x[1]), reverse=True)[:max_players]
        )
