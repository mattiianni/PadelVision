"""
Player tracking con YOLOv8 + ByteTrack.

Per ogni frame del video rileva le persone (classe 0),
le traccia con ID consistenti e ne ricava la posizione sul campo
tramite la calibrazione.

Output: dict {player_id: [(court_x, court_y, frame_num), ...]}
"""

import cv2
import numpy as np
from collections import defaultdict
from ultralytics import YOLO
from court_calibration import CourtCalibrator


class PlayerTracker:
    def __init__(self, model_name: str = "yolov8n.pt", conf: float = 0.35):
        """
        model_name: 'yolov8n.pt' (veloce), 'yolov8s.pt' (più accurato)
        conf: soglia di confidenza per il rilevamento
        """
        print(f"  Caricamento modello {model_name}...")
        self.model = YOLO(model_name)
        self.conf = conf

    def track_video(
        self,
        video_path: str,
        calibrator: CourtCalibrator,
        sample_every: int = 1,
        progress_callback=None,
    ) -> tuple[dict, float, int]:
        """
        Traccia i giocatori in tutto il video.

        sample_every: analizza 1 frame ogni N (1 = tutti, 3 = 1/3 dei frame)
                      Usare 2-3 per video lunghi per risparmiare tempo.

        Ritorna:
          tracks    - dict {player_id: [(cx, cy, frame_num), ...]}
          fps       - frame rate del video
          total_frames - numero totale di frame
        """
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        tracks: dict[int, list] = defaultdict(list)
        frame_num = 0

        results = self.model.track(
            source=video_path,
            classes=[0],              # solo persone
            conf=self.conf,
            tracker="bytetrack.yaml",
            stream=True,
            verbose=False,
            imgsz=640,
        )

        for result in results:
            if frame_num % sample_every != 0:
                frame_num += 1
                if progress_callback:
                    progress_callback(frame_num, total_frames)
                continue

            if result.boxes is None or result.boxes.id is None:
                frame_num += 1
                if progress_callback:
                    progress_callback(frame_num, total_frames)
                continue

            boxes = result.boxes.xyxy.cpu().numpy()      # (N, 4) x1 y1 x2 y2
            ids   = result.boxes.id.cpu().numpy().astype(int)  # (N,)

            # Batch transform dei piedi (bottom-center di ogni bbox)
            foot_pixels = np.column_stack([
                (boxes[:, 0] + boxes[:, 2]) / 2,   # x center
                boxes[:, 3],                        # y bottom
            ])
            court_pts = calibrator.transform_points_batch(foot_pixels)

            for pid, (cx, cy) in zip(ids, court_pts):
                tracks[int(pid)].append((float(cx), float(cy), frame_num))

            frame_num += 1
            if progress_callback:
                progress_callback(frame_num, total_frames)

        return dict(tracks), fps, total_frames

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def filter_players(
        tracks: dict,
        min_frames: int = 30,
        max_players: int = 4,
    ) -> dict:
        """
        Filtra i track mantenendo solo i giocatori rilevati in almeno
        `min_frames` frame, tenendo i `max_players` più presenti.
        Serve a eliminare arbitri, spettatori o falsi positivi.
        """
        filtered = {
            pid: pos
            for pid, pos in tracks.items()
            if len(pos) >= min_frames
        }
        # Tieni i più presenti
        sorted_tracks = sorted(
            filtered.items(), key=lambda x: len(x[1]), reverse=True
        )
        return dict(sorted_tracks[:max_players])
