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
        clip: float = 1.0,
        start_s: float = 0.0,
        end_s: float = None,
        progress_callback=None,
    ) -> tuple[dict, float, int]:
        """
        Processa il video in batch per massimizzare l'uso dei core.

        start_s / end_s : analizza solo la finestra temporale [start_s, end_s].
                          Se end_s è None usa clip per calcolare la fine.

        Ritorna:
          tracks       — dict {player_id: [(cx, cy, frame_num), ...]}
          fps          — frame rate
          total_frames — frame totali
        """
        cap = cv2.VideoCapture(video_path)
        fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        start_frame = int(start_s * fps)
        if end_s is not None:
            stop_frame = int(end_s * fps)
        else:
            stop_frame = int(total_frames * max(0.01, min(1.0, clip)))
        stop_frame = min(stop_frame, total_frames)

        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        seg_length = max(stop_frame - start_frame, 1)

        all_detections = []  # lista di (cx, cy, frame_num)
        frame_num  = start_frame
        batch      = []
        batch_idxs = []

        while True:
            ret, frame = cap.read()
            if not ret or frame_num >= stop_frame:
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
                # Progresso relativo al segmento (0..seg_length)
                progress_callback(frame_num - start_frame, seg_length)

        # Eventuale batch residuo
        if batch:
            self._infer_batch(batch, batch_idxs, calibrator, all_detections)

        cap.release()

        if progress_callback:
            progress_callback(seg_length, seg_length)

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

    def extract_player_crops(
        self,
        video_path: str,
        tracks: dict,
        calibrator,
        output_dir: str,
        n_frames: int = 20,
    ) -> dict:
        """
        Per ogni player ID estrae il miglior crop dal video.

        Strategia: seleziona n_frames frame distribuiti nel video,
        abbina ogni detection YOLO al player più vicino per posizione
        sul campo (coordinate omografiche), salva il crop più grande trovato.

        Ritorna: {player_id: path_png}
        """
        if not tracks:
            return {}

        # Posizione media sul campo per ogni player
        player_avg_pos = {
            pid: np.mean([(cx, cy) for cx, cy, _ in positions], axis=0)
            for pid, positions in tracks.items()
            if positions
        }

        # Frame dove tutti i player sono presenti (intersezione)
        player_frame_sets = {
            pid: set(fn for _, _, fn in pos) for pid, pos in tracks.items()
        }
        common = set.intersection(*player_frame_sets.values()) if player_frame_sets else set()
        if not common:
            common = set().union(*player_frame_sets.values())

        frame_candidates = sorted(common)
        if len(frame_candidates) > n_frames:
            step = max(1, len(frame_candidates) // n_frames)
            frame_candidates = frame_candidates[::step][:n_frames]

        best_crops = {}   # pid -> (img_bgr, area)

        cap = cv2.VideoCapture(video_path)
        for fn in frame_candidates:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fn)
            ret, frame = cap.read()
            if not ret:
                continue

            results = self.model.predict(
                frame, classes=[0], conf=self.conf, verbose=False, imgsz=640
            )
            if not results or results[0].boxes is None or len(results[0].boxes) == 0:
                continue

            boxes = results[0].boxes.xyxy.cpu().numpy()
            feet = np.column_stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,
                boxes[:, 3],
            ])
            court_pts = calibrator.transform_points_batch(feet)

            h, w = frame.shape[:2]
            for i, (cx, cy) in enumerate(court_pts):
                # Abbina al player con la posizione media più vicina
                best_pid, best_dist = None, float("inf")
                for pid, avg_pos in player_avg_pos.items():
                    d = float(np.hypot(cx - avg_pos[0], cy - avg_pos[1]))
                    if d < best_dist:
                        best_dist = d
                        best_pid = pid

                if best_pid is None or best_dist > 2.5:
                    continue   # detection troppo lontana da qualunque cluster

                x1, y1, x2, y2 = boxes[i].astype(int)
                # Padding +10% larghezza, +5% altezza per contesto
                pad_x = int((x2 - x1) * 0.10)
                pad_y = int((y2 - y1) * 0.05)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                crop = frame[y1:y2, x1:x2].copy()
                if crop.size == 0:
                    continue

                area = (x2 - x1) * (y2 - y1)
                if best_pid not in best_crops or area > best_crops[best_pid][1]:
                    best_crops[best_pid] = (crop, area)

        cap.release()

        os.makedirs(output_dir, exist_ok=True)
        crop_paths = {}
        for pid, (crop_img, _) in best_crops.items():
            path = os.path.join(output_dir, f"crop_player_{pid}.png")
            cv2.imwrite(path, crop_img)
            crop_paths[pid] = path

        return crop_paths

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
