"""
Orchestratore principale dell'analisi video padel.
Coordina: calibrazione → tracking → heatmap → output.
"""

import json
import os
import time

from court_calibration import CourtCalibrator
from player_tracker import PlayerTracker
from player_namer import name_players
from heatmap import generate_heatmaps
from report import generate_report


class PadelAnalyzer:
    def __init__(
        self,
        video_path: str,
        output_dir: str = "output",
        sample_every: int = 2,
        min_player_frames: int = 50,
        max_players: int = 4,
        clip: float = 1.0,
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
        self.clip = max(0.01, min(1.0, clip))

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
            clip=self.clip,
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

        # ---- Crop + Naming giocatori ----
        self._step(3, "Estrazione crop giocatori")
        print("  Cerco i migliori frame per ogni giocatore...")
        crop_paths = tracker.extract_player_crops(
            self.video_path, tracks, calibrator, self.output_dir
        )
        self._ok(f"{len(crop_paths)} crop salvati")

        player_ids = sorted(tracks.keys())
        player_names = name_players(crop_paths, player_ids)
        print()
        for pid, name in player_names.items():
            print(f"  Player {pid} → {name}")

        # ---- Heatmap ----
        self._step(4, "Generazione heatmap e statistiche")
        images, stats = generate_heatmaps(tracks, self.output_dir, fps,
                                           player_names=player_names)
        self._ok()

        # ---- Report HTML ----
        video_name = os.path.splitext(os.path.basename(self.video_path))[0]
        report_path = generate_report(
            images={
                "players": images.get("players"),
                "teams":   images.get("teams"),
                "zones":   images.get("zones"),
            },
            stats=stats,
            video_name=video_name,
            output_dir=self.output_dir,
            player_names=player_names,
            crop_paths=crop_paths,
        )
        self._ok(f"Report aperto nel browser → {report_path}")

        # ---- Riepilogo terminale ----
        self._summary(stats, player_names, list(images.values()) + [report_path])

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
        print(f"\n[{n}/4] {label}...")

    @staticmethod
    def _ok(detail: str = ""):
        msg = "  ✓ OK"
        if detail:
            msg += f"  ({detail})"
        print(msg)

    def _summary(self, stats: dict, player_names: dict, images: list[str]):
        print()
        print("─" * 54)
        print("  RISULTATI  (Team A = vicino camera · Team B = lontano)")
        print("─" * 54)
        # Team A prima, Team B dopo; dentro ogni squadra per player_id
        ordered = sorted(stats.keys(),
                         key=lambda pid: (0 if stats[pid].get("team") == "A" else 1, pid))
        for pid in ordered:
            s = stats[pid]
            name = player_names.get(pid, f"Player {pid}")
            team = s.get("team", "?")
            print(f"\n  {name}  (Player {pid} · Team {team})  ·  {s['time_s']}s a schermo")
            print(f"    Zona rete : {s['zone_net_pct']:5.1f}%")
            print(f"    Zona medio: {s['zone_mid_pct']:5.1f}%")
            print(f"    Zona fondo: {s['zone_back_pct']:5.1f}%")
            print(f"    Sin/Dx    : {s['side_left_pct']:.0f}% / {s['side_right_pct']:.0f}%")

        print()
        print("  Output:")
        for p in images:
            print(f"    → {p}")
        print()
