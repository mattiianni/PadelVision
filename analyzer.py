"""
Orchestratore principale dell'analisi video padel.
Coordina: calibrazione → tracking → heatmap → output.
"""

import json
import os
import time

from court_calibration import CourtCalibrator
from player_tracker import PlayerTracker
from heatmap import generate_heatmaps


class PadelAnalyzer:
    def __init__(
        self,
        video_path: str,
        output_dir: str = "output",
        sample_every: int = 2,
        min_player_frames: int = 50,
        max_players: int = 4,
    ):
        """
        video_path       : path al video da analizzare
        output_dir       : cartella di output per immagini e stats
        sample_every     : analizza 1 frame ogni N (2 = metà frame, più veloce)
        min_player_frames: frame minimi per considerare un ID come giocatore
        max_players      : numero massimo di giocatori da tenere
        """
        self.video_path = video_path
        self.output_dir = output_dir
        self.sample_every = sample_every
        self.min_player_frames = min_player_frames
        self.max_players = max_players

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        self.calib_path = os.path.join(output_dir, f"calibration_{video_name}.json")

    # ------------------------------------------------------------------

    def run(self, recalibrate: bool = False) -> tuple[dict, list[str]]:
        self._header()
        os.makedirs(self.output_dir, exist_ok=True)

        # ---- Calibrazione ----
        self._step(1, "Calibrazione campo")
        calibrator = CourtCalibrator(self.video_path, self.calib_path)
        if recalibrate and os.path.exists(self.calib_path):
            os.remove(self.calib_path)
        calibrator.calibrate()
        self._ok()

        # ---- Tracking ----
        self._step(2, "Rilevamento e tracking giocatori")
        tracker = PlayerTracker()

        t0 = time.time()
        tracks, fps, total_frames = tracker.track_video(
            self.video_path,
            calibrator,
            sample_every=self.sample_every,
            progress_callback=self._progress,
        )
        elapsed = time.time() - t0
        print()  # newline dopo la progress bar

        # Filtra track spuri (arbitri, spettatori, falsi positivi)
        tracks = PlayerTracker.filter_players(
            tracks,
            min_frames=self.min_player_frames,
            max_players=self.max_players,
        )
        self._ok(f"{len(tracks)} giocatori  ·  {total_frames} frame  ·  {elapsed:.0f}s")

        # ---- Heatmap ----
        self._step(3, "Generazione heatmap e statistiche")
        images, stats = generate_heatmaps(tracks, self.output_dir, fps)
        self._ok()

        # ---- Riepilogo ----
        self._summary(stats, images)

        return stats, images

    # ------------------------------------------------------------------
    # Progress e output helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _progress(frame: int, total: int):
        pct = int(frame / max(total, 1) * 100)
        if frame % 30 == 0:
            filled = pct // 5
            bar = "█" * filled + "░" * (20 - filled)
            print(f"\r  [{bar}] {pct:3d}%  frame {frame}/{total}", end="", flush=True)

    @staticmethod
    def _header():
        print()
        print("=" * 54)
        print("  PadelVision  —  Analisi Video Padel")
        print("=" * 54)

    @staticmethod
    def _step(n: int, label: str):
        print(f"\n[{n}/3] {label}...")

    @staticmethod
    def _ok(detail: str = ""):
        msg = "  ✓ OK"
        if detail:
            msg += f"  ({detail})"
        print(msg)

    def _summary(self, stats: dict, images: list[str]):
        print()
        print("─" * 54)
        print("  RISULTATI")
        print("─" * 54)
        for pid, s in stats.items():
            print(f"\n  Player {pid}  ·  {s['time_s']}s a schermo")
            print(f"    Zona rete : {s['zone_net_pct']:5.1f}%")
            print(f"    Zona medio: {s['zone_mid_pct']:5.1f}%")
            print(f"    Zona fondo: {s['zone_back_pct']:5.1f}%")
            print(f"    Sin/Dx    : {s['side_left_pct']:.0f}% / {s['side_right_pct']:.0f}%")

        print()
        print("  Output:")
        for p in images:
            print(f"    → {p}")
        print()
