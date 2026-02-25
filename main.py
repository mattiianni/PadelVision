#!/usr/bin/env python3
"""
PadelVision — CLI entry point

Uso:
  python main.py video.mp4
  python main.py video.mp4 --output risultati/ --sample 1
  python main.py video.mp4 --recalibrate
"""

import argparse
import sys
import os


def main():
    parser = argparse.ArgumentParser(
        prog="padelvision",
        description="Analizza un video di padel: heatmap giocatori, "
                    "statistiche di zona e molto altro.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "video",
        help="Path al file video (mp4, mov, avi, ...)",
    )
    parser.add_argument(
        "--output", "-o",
        default="output",
        metavar="DIR",
        help="Cartella di output (default: ./output)",
    )
    parser.add_argument(
        "--sample", "-s",
        type=int,
        default=2,
        metavar="N",
        help="Analizza 1 frame ogni N (1=tutti, 2=metà, 3=un terzo). "
             "Default: 2  (consigliato per velocità)",
    )
    parser.add_argument(
        "--recalibrate",
        action="store_true",
        help="Forza una nuova calibrazione del campo anche se esiste già",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=50,
        metavar="N",
        help="Frame minimi per considerare un track come giocatore (default: 50)",
    )
    parser.add_argument(
        "--max-players",
        type=int,
        default=4,
        metavar="N",
        help="Numero massimo di giocatori da tenere (default: 4)",
    )

    args = parser.parse_args()

    # Controlla che il video esista
    if not os.path.exists(args.video):
        print(f"Errore: file non trovato → {args.video}")
        sys.exit(1)

    from analyzer import PadelAnalyzer

    analyzer = PadelAnalyzer(
        video_path=args.video,
        output_dir=args.output,
        sample_every=args.sample,
        min_player_frames=args.min_frames,
        max_players=args.max_players,
    )

    try:
        analyzer.run(recalibrate=args.recalibrate)
    except KeyboardInterrupt:
        print("\n\nInterrotto dall'utente.")
        sys.exit(0)
    except Exception as e:
        print(f"\nErrore: {e}")
        raise


if __name__ == "__main__":
    main()
