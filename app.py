"""
PadelVision — Interfaccia Web (Gradio)

Avvio:
    python app.py          → http://localhost:7860
    python app.py --share  → link pubblico temporaneo
    python app.py --port 8080
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from court_calibration import CourtCalibrator, CALIB_POINTS, COLORS
from player_tracker import PlayerTracker
from heatmap import generate_heatmaps
from report import generate_report

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_web")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SEGS = 3  # numero massimo di segmenti (set) supportati nell'UI


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def extract_calib_frame(video_path: str):
    """Estrae un frame buono per la calibrazione (a ~1 min o 10%)."""
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for t in [int(60 * fps), int(total * 0.10), int(total * 0.02), 0]:
        t = max(0, min(t, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ret, f = cap.read()
        if ret and np.mean(f) > 20:
            cap.release()
            return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()
    return None


def draw_calib_overlay(frame_rgb: np.ndarray, pts: list) -> np.ndarray:
    """Disegna cerchi colorati numerati + HUD con prossimo punto da cliccare."""
    canvas = frame_rgb.copy()
    h, w   = canvas.shape[:2]

    for i, (px, py) in enumerate(pts):
        c     = COLORS[i % len(COLORS)]
        color = (int(c[2]), int(c[1]), int(c[0]))
        cv2.circle(canvas, (int(px), int(py)), 12, color, -1)
        cv2.circle(canvas, (int(px), int(py)), 15, (255, 255, 255), 2)
        cv2.putText(canvas, str(i + 1), (int(px) + 18, int(py) + 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2, cv2.LINE_AA)

    n = len(pts)
    ov = canvas.copy()
    cv2.rectangle(ov, (0, 0), (w, 58), (18, 18, 18), -1)
    cv2.addWeighted(ov, 0.75, canvas, 0.25, 0, canvas)

    if n < len(CALIB_POINTS):
        label, _ = CALIB_POINTS[n]
        opt      = " [opzionale]" if n >= 4 else ""
        c        = COLORS[n % len(COLORS)]
        col      = (int(c[2]), int(c[1]), int(c[0]))
        cv2.putText(canvas, f"Punto {n + 1}: {label}{opt}", (16, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2, cv2.LINE_AA)
    elif n >= 4:
        cv2.putText(canvas, "Calibrazione pronta — clicca Conferma",
                    (16, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.9,
                    (80, 220, 100), 2, cv2.LINE_AA)
    return canvas


def calib_instruction_html(n_pts: int) -> str:
    items = []
    for i, (label, _) in enumerate(CALIB_POINTS):
        if i < n_pts:
            ico, col, fw = "✅", "#3fb950", "normal"
        elif i == n_pts:
            ico, col, fw = "👉", "#f0f6fc", "bold"
        else:
            ico, col, fw = "⬜", "#484f58", "normal"
        opt = " <em style='color:#484f58'>(opzionale)</em>" if i >= 4 else ""
        items.append(
            f'<li style="color:{col};font-weight:{fw};line-height:1.9">'
            f'{ico} <b>{i+1}.</b> {label}{opt}</li>'
        )
    note = ""
    if n_pts >= 4:
        note = ('<p style="color:#58a6ff;margin:8px 0 0;font-size:.85rem">'
                '✓ Minimi raggiunti! Puoi confermare ora.</p>')
    elif n_pts == 0:
        note = ('<p style="color:#8b949e;margin:8px 0 0;font-size:.82rem">'
                'Prima premi <b>Estrai Frame</b>, poi clicca sul campo.</p>')
    return (f'<ol style="margin:0;padding-left:18px;font-size:.88rem">'
            f'{"".join(items)}</ol>{note}')


def parse_segment_times(text: str) -> list:
    """Converte '720, 24:00' → [720.0, 1440.0] in secondi."""
    if not text or not text.strip():
        return []
    result = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        try:
            if ":" in part:
                mm, ss = part.split(":", 1)
                result.append(int(mm) * 60 + float(ss))
            else:
                result.append(float(part))
        except (ValueError, IndexError):
            pass
    return sorted(result)


def stats_to_html(stats: dict, player_names: dict) -> str:
    if not stats:
        return "<p style='color:#8b949e'>Nessuna statistica disponibile.</p>"

    def sort_key(pid):
        return (0 if stats[pid].get("team", "B") == "A" else 1, pid)

    rows = ""
    for pid in sorted(stats.keys(), key=sort_key):
        s    = stats[pid]
        team = s.get("team", "?")
        bc   = "#3b82f6" if team == "A" else "#ef4444"
        name = player_names.get(pid, f"Player {pid}")
        rows += (
            f'<tr><td><span style="background:{bc};color:#fff;padding:2px 8px;'
            f'border-radius:12px;font-size:.75rem;font-weight:700">Team {team}</span>'
            f'&nbsp;{name}</td>'
            f'<td style="text-align:center">{s["time_s"]}s</td>'
            f'<td style="text-align:center;color:#00aaff">{s["zone_net_pct"]}%</td>'
            f'<td style="text-align:center;color:#ffaa00">{s["zone_mid_pct"]}%</td>'
            f'<td style="text-align:center;color:#ff4444">{s["zone_back_pct"]}%</td>'
            f'<td style="text-align:center">{s["side_left_pct"]:.0f}% / {s["side_right_pct"]:.0f}%</td></tr>'
        )

    return (
        '<table style="width:100%;border-collapse:collapse;font-size:.88rem;color:#e6edf3">'
        '<thead style="background:#21262d">'
        '<tr>'
        '<th style="padding:10px 14px;text-align:left;color:#8b949e;font-size:.75rem;text-transform:uppercase">Giocatore</th>'
        '<th style="padding:10px 14px;color:#8b949e;font-size:.75rem">Tempo</th>'
        '<th style="padding:10px 14px;color:#00aaff;font-size:.75rem">Rete</th>'
        '<th style="padding:10px 14px;color:#ffaa00;font-size:.75rem">Medio</th>'
        '<th style="padding:10px 14px;color:#ff4444;font-size:.75rem">Fondo</th>'
        '<th style="padding:10px 14px;color:#8b949e;font-size:.75rem;text-transform:uppercase">Sin / Dx</th>'
        '</tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Event handlers
# ─────────────────────────────────────────────────────────────────────────────

def on_video_upload(path: str):
    return path or ""


def load_frame(state_video: str):
    if not state_video:
        raise gr.Error("Carica prima un video.")
    frame = extract_calib_frame(state_video)
    if frame is None:
        raise gr.Error("Impossibile leggere il frame dal video.")
    pts     = []
    overlay = draw_calib_overlay(frame, pts)
    return frame, overlay, pts, calib_instruction_html(0), gr.update(interactive=False)


def on_calib_click(evt: gr.SelectData, state_frame, state_pts: list):
    if state_frame is None:
        return gr.update(), state_pts, calib_instruction_html(len(state_pts)), gr.update()
    n = len(state_pts)
    if n >= len(CALIB_POINTS):
        return gr.update(), state_pts, calib_instruction_html(n), gr.update(interactive=n >= 4)
    new_pts = state_pts + [[int(evt.index[0]), int(evt.index[1])]]
    overlay = draw_calib_overlay(state_frame, new_pts)
    return overlay, new_pts, calib_instruction_html(len(new_pts)), gr.update(interactive=len(new_pts) >= 4)


def undo_calib(state_frame, state_pts: list):
    if not state_pts:
        return gr.update(), state_pts, calib_instruction_html(0), gr.update(interactive=False)
    new_pts = state_pts[:-1]
    overlay = draw_calib_overlay(state_frame, new_pts) if state_frame is not None else gr.update()
    return overlay, new_pts, calib_instruction_html(len(new_pts)), gr.update(interactive=len(new_pts) >= 4)


def reset_calib(state_frame):
    pts     = []
    overlay = draw_calib_overlay(state_frame, pts) if state_frame is not None else gr.update()
    return overlay, pts, calib_instruction_html(0), gr.update(interactive=False)


def confirm_calib(state_pts: list):
    if len(state_pts) < 4:
        raise gr.Error("Clicca almeno 4 punti prima di confermare.")
    dst_pts = [CALIB_POINTS[i][1] for i in range(len(state_pts))]
    try:
        cal = CourtCalibrator.from_click_points(state_pts, dst_pts)
    except ValueError as e:
        raise gr.Error(str(e))
    status = (
        '<div style="color:#3fb950;font-weight:bold;font-size:.95rem">'
        f'✅ Calibrazione OK — {len(state_pts)} punti · Vai al Tab 2 → Tracking</div>'
    )
    return cal.H.tolist(), status


def run_tracking(video_path: str, clip_val: float, segment_str: str,
                 H_state, progress=gr.Progress()):
    """Tracking per tutti i segmenti. Ritorna state + gallery images + vis."""
    if not video_path:
        raise gr.Error("Carica un video (Tab 1).")
    if H_state is None:
        raise gr.Error("Completa la calibrazione (Tab 1).")

    calibrator    = CourtCalibrator.from_homography(H_state)
    segment_times = parse_segment_times(segment_str or "")
    tracker       = PlayerTracker()

    cap          = cv2.VideoCapture(video_path)
    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    clip_end_s   = (total_frames / fps) * clip_val

    segment_times = [t for t in segment_times if 0 < t < clip_end_s]
    boundaries    = [0.0] + segment_times + [clip_end_s]
    intervals     = [(boundaries[i], boundaries[i + 1])
                     for i in range(len(boundaries) - 1)]
    n_segs        = len(intervals)

    all_seg_tracks     = []
    all_seg_crop_paths = []
    fps_v              = fps

    for seg_i, (start_s, end_s) in enumerate(intervals):

        def make_cb(si, ns):
            def cb(frame, total):
                overall = (si + frame / max(total, 1)) / ns
                progress(overall, desc=f"Seg {si+1}/{ns}: frame {frame}/{total}")
            return cb

        progress(seg_i / n_segs, desc=f"Segmento {seg_i+1}/{n_segs}: tracking...")
        tracks, fps_v, _ = tracker.track_video(
            video_path, calibrator,
            start_s=start_s, end_s=end_s,
            progress_callback=make_cb(seg_i, n_segs),
        )
        tracks = PlayerTracker.filter_players(tracks, min_frames=50, max_players=4)

        seg_dir    = os.path.join(OUTPUT_DIR, f"seg{seg_i + 1}")
        progress((seg_i + 0.85) / n_segs,
                 desc=f"Segmento {seg_i+1}/{n_segs}: estrazione crop...")
        crop_paths = tracker.extract_player_crops(
            video_path, tracks, calibrator, seg_dir
        )
        all_seg_tracks.append(tracks)
        all_seg_crop_paths.append(crop_paths)

    progress(1.0, desc="Completato!")

    tracking_state = {
        "all_seg_tracks":     all_seg_tracks,
        "all_seg_crop_paths": all_seg_crop_paths,
        "fps":                fps_v,
        "n_segments":         n_segs,
        "video_path":         video_path,
    }

    # Prepara le gallery per il tab 3: lista di (image_rgb, caption) per segmento
    galleries = []
    for seg_i in range(MAX_SEGS):
        items = []
        if seg_i < n_segs:
            tracks_s = all_seg_tracks[seg_i]
            crops_s  = all_seg_crop_paths[seg_i]
            sorted_pids = sorted(tracks_s.keys())
            for slot in range(4):
                pid     = sorted_pids[slot] if slot < len(sorted_pids) else None
                path    = crops_s.get(pid) if pid else None
                team_lbl = "Team B" if slot < 2 else "Team A"
                caption  = f"Player {slot+1} · {team_lbl}"
                if path and os.path.exists(path):
                    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                    items.append((img, caption))
                else:
                    items.append((np.zeros((200, 150, 3), dtype=np.uint8), caption + " (crop non disponibile)"))
        galleries.append(items if items else [])

    status_html = (
        f'<div style="color:#3fb950;font-weight:bold;font-size:.95rem">'
        f'✅ Tracking completato — {n_segs} segmento/i &nbsp;·&nbsp; FPS: {fps_v:.1f}</div>'
        f'<p style="color:#8b949e;font-size:.85rem;margin-top:4px">'
        f'Vai al <b>Tab 3 → Giocatori</b> per assegnare i nomi.</p>'
    )

    return (
        tracking_state,
        status_html,
        galleries[0],                                     # gallery seg 1
        galleries[1],                                     # gallery seg 2
        galleries[2],                                     # gallery seg 3
        gr.update(visible=n_segs >= 2),                   # gruppo seg 2
        gr.update(visible=n_segs >= 3),                   # gruppo seg 3
    )


def generate_results(tracking_state,
                     n1_1, n1_2, n1_3, n1_4,
                     n2_1, n2_2, n2_3, n2_4,
                     n3_1, n3_2, n3_3, n3_4):
    """Unisce i track per nome e genera heatmap + report HTML."""
    if tracking_state is None:
        raise gr.Error("Esegui il tracking (Tab 2) prima.")

    all_seg_tracks     = tracking_state["all_seg_tracks"]
    all_seg_crop_paths = tracking_state["all_seg_crop_paths"]
    fps                = tracking_state["fps"]
    n_segs             = tracking_state["n_segments"]
    video_path         = tracking_state["video_path"]

    all_names_flat = [
        [n1_1, n1_2, n1_3, n1_4],
        [n2_1, n2_2, n2_3, n2_4],
        [n3_1, n3_2, n3_3, n3_4],
    ]

    all_seg_names = []
    for seg_i in range(n_segs):
        tracks      = all_seg_tracks[seg_i]
        sorted_pids = sorted(tracks.keys())
        seg_names   = {}
        for slot, pid in enumerate(sorted_pids):
            raw = all_names_flat[seg_i][slot] if slot < 4 else ""
            seg_names[pid] = raw.strip() if raw and raw.strip() else f"Player {pid}"
        all_seg_names.append(seg_names)

    # Merge per nome attraverso i segmenti
    merged_by_name: dict = {}
    for seg_i in range(n_segs):
        for pid, positions in all_seg_tracks[seg_i].items():
            name = all_seg_names[seg_i].get(pid, f"Player {pid}")
            merged_by_name.setdefault(name, []).extend(positions)

    sorted_names  = sorted(merged_by_name.keys())
    merged_tracks = {i + 1: merged_by_name[n] for i, n in enumerate(sorted_names)}
    player_names  = {i + 1: n for i, n in enumerate(sorted_names)}

    # Crop: primo disponibile per nome
    crop_paths_final: dict = {}
    for seg_i in range(n_segs):
        for pid, path in all_seg_crop_paths[seg_i].items():
            name     = all_seg_names[seg_i].get(pid)
            final_id = (sorted_names.index(name) + 1
                        if name and name in sorted_names else None)
            if final_id and final_id not in crop_paths_final:
                crop_paths_final[final_id] = path

    images, stats = generate_heatmaps(merged_tracks, OUTPUT_DIR, fps,
                                      player_names=player_names)

    video_name  = os.path.splitext(os.path.basename(video_path))[0]
    report_path = generate_report(
        images=images, stats=stats, video_name=video_name,
        output_dir=OUTPUT_DIR, player_names=player_names,
        crop_paths=crop_paths_final, open_browser=False,
    )

    report_link = (
        f'<div style="margin-top:16px">'
        f'<a href="/file={report_path}" target="_blank" '
        f'style="background:#238636;color:#fff;padding:9px 22px;'
        f'border-radius:6px;text-decoration:none;font-weight:700;font-size:.9rem">'
        f'📄 Apri Report Completo</a>&nbsp;&nbsp;'
        f'<span style="color:#8b949e;font-size:.82rem">{report_path}</span>'
        f'</div>'
    )

    return (
        images.get("players"),
        images.get("teams"),
        images.get("zones"),
        stats_to_html(stats, player_names),
        report_link,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Gradio UI
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
footer { display: none !important; }
.gr-padded { padding: 16px !important; }
"""

with gr.Blocks(title="PadelVision", css=CSS,
               theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate")) as app:

    # Stato sessione
    state_video    = gr.State("")
    state_frame    = gr.State(None)
    state_pts      = gr.State([])
    state_H        = gr.State(None)
    state_tracking = gr.State(None)

    # Header
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#161b22,#1c2333);
                border-bottom:1px solid #30363d;padding:18px 28px;
                margin-bottom:4px">
      <div style="display:inline-block;background:#21262d;border:1px solid #30363d;
                  border-radius:20px;padding:2px 12px;font-size:.75rem;color:#8b949e;
                  margin-bottom:6px">🎾 PadelVision</div>
      <h2 style="margin:0;font-size:1.4rem;font-weight:700;color:#e6edf3">
        Analisi Video Padel
      </h2>
      <p style="margin:3px 0 0;color:#8b949e;font-size:.83rem">
        Calibra · Tracking · Nomi · Risultati
      </p>
    </div>
    """)

    with gr.Tabs():

        # ══════════════════════════════════════════════════
        # TAB 1 — CALIBRAZIONE
        # ══════════════════════════════════════════════════
        with gr.Tab("1 · Calibrazione"):
            with gr.Row():
                with gr.Column(scale=1, min_width=270):
                    gr.Markdown("### 📹 Video")
                    video_upload = gr.File(
                        file_types=["video"],
                        label="Carica video (.mp4 .mov .avi ...)",
                        type="filepath",
                    )
                    btn_load = gr.Button("📷 Estrai Frame", variant="secondary", size="sm")
                    gr.Markdown("---")
                    gr.Markdown("### 🎯 Punti di calibrazione")
                    calib_instr = gr.HTML(calib_instruction_html(0))
                    with gr.Row():
                        btn_undo  = gr.Button("↩ Annulla", size="sm")
                        btn_reset = gr.Button("🗑 Reset",   size="sm")
                    btn_calib_ok = gr.Button(
                        "✅ Conferma Calibrazione",
                        variant="primary", interactive=False,
                    )
                    calib_status = gr.HTML("")

                with gr.Column(scale=2):
                    gr.HTML(
                        '<p style="color:#8b949e;font-size:.83rem;margin-bottom:6px">'
                        'Carica il video → <b>Estrai Frame</b> → clicca i punti sul campo '
                        'nell\'ordine della lista a sinistra.</p>'
                    )
                    calib_image = gr.Image(
                        label="Frame — clicca per calibrare",
                        type="numpy", interactive=True, height=520,
                    )

        # ══════════════════════════════════════════════════
        # TAB 2 — TRACKING
        # ══════════════════════════════════════════════════
        with gr.Tab("2 · Tracking"):
            with gr.Row():
                with gr.Column(scale=1, min_width=270):
                    gr.Markdown("### ⚙️ Impostazioni analisi")
                    clip_slider = gr.Slider(
                        minimum=0.05, maximum=1.0, value=0.1, step=0.05,
                        label="Porzione video da analizzare",
                        info="0.1 = primo 10% (test rapido) · 1.0 = video completo",
                    )
                    segment_input = gr.Textbox(
                        label="Cambi campo — secondi o MM:SS separati da virgola",
                        placeholder="Es: 720, 1440  oppure  12:00, 24:00  (lascia vuoto = partita singola)",
                        lines=1,
                    )
                    gr.HTML(
                        '<p style="color:#8b949e;font-size:.82rem;margin-top:4px">'
                        'Inserisci i momenti in cui i giocatori cambiano lato '
                        '(fine set). PadelVision analizzerà ogni segmento separatamente '
                        'e unirà le statistiche per nome.</p>'
                    )
                    btn_track = gr.Button("▶️ Avvia Analisi", variant="primary")
                with gr.Column(scale=2):
                    tracking_status = gr.HTML(
                        '<div style="color:#8b949e;padding:12px">In attesa...</div>'
                    )

        # ══════════════════════════════════════════════════
        # TAB 3 — NOMI GIOCATORI
        # ══════════════════════════════════════════════════
        with gr.Tab("3 · Giocatori"):
            gr.HTML(
                '<p style="color:#8b949e;margin-bottom:10px">'
                'Guarda i crop e scrivi il nome di ogni giocatore. '
                'Per partite con cambio campo compaiono più sezioni.</p>'
            )

            # ── Segmento 1 (sempre visibile) ──────────────────────────────
            gr.HTML('<div style="color:#58a6ff;font-weight:700;font-size:.95rem;'
                    'margin:4px 0 8px">📹 Segmento 1</div>')
            gallery_s1 = gr.Gallery(
                label="Crop giocatori — Segmento 1",
                columns=4, rows=1, height=280, show_label=False,
            )
            with gr.Row():
                n1_1 = gr.Textbox(label="Player 1 (Team B)", placeholder="Nome...")
                n1_2 = gr.Textbox(label="Player 2 (Team B)", placeholder="Nome...")
                n1_3 = gr.Textbox(label="Player 3 (Team A)", placeholder="Nome...")
                n1_4 = gr.Textbox(label="Player 4 (Team A)", placeholder="Nome...")

            # ── Segmento 2 (visibile solo se ci sono 2+ segmenti) ─────────
            with gr.Group(visible=False) as seg2_group:
                gr.HTML('<div style="color:#58a6ff;font-weight:700;font-size:.95rem;'
                        'margin:16px 0 8px">📹 Segmento 2 — dopo cambio campo</div>')
                gallery_s2 = gr.Gallery(
                    label="Crop giocatori — Segmento 2",
                    columns=4, rows=1, height=280, show_label=False,
                )
                with gr.Row():
                    n2_1 = gr.Textbox(label="Player 1 (Team B)", placeholder="Nome...")
                    n2_2 = gr.Textbox(label="Player 2 (Team B)", placeholder="Nome...")
                    n2_3 = gr.Textbox(label="Player 3 (Team A)", placeholder="Nome...")
                    n2_4 = gr.Textbox(label="Player 4 (Team A)", placeholder="Nome...")

            # ── Segmento 3 ────────────────────────────────────────────────
            with gr.Group(visible=False) as seg3_group:
                gr.HTML('<div style="color:#58a6ff;font-weight:700;font-size:.95rem;'
                        'margin:16px 0 8px">📹 Segmento 3 — dopo secondo cambio campo</div>')
                gallery_s3 = gr.Gallery(
                    label="Crop giocatori — Segmento 3",
                    columns=4, rows=1, height=280, show_label=False,
                )
                with gr.Row():
                    n3_1 = gr.Textbox(label="Player 1 (Team B)", placeholder="Nome...")
                    n3_2 = gr.Textbox(label="Player 2 (Team B)", placeholder="Nome...")
                    n3_3 = gr.Textbox(label="Player 3 (Team A)", placeholder="Nome...")
                    n3_4 = gr.Textbox(label="Player 4 (Team A)", placeholder="Nome...")

            btn_results = gr.Button(
                "📊 Genera Heatmap e Report", variant="primary"
            )

        # ══════════════════════════════════════════════════
        # TAB 4 — RISULTATI
        # ══════════════════════════════════════════════════
        with gr.Tab("4 · Risultati"):
            result_img_players = gr.Image(label="Heatmap Giocatori", interactive=False)
            with gr.Row():
                result_img_teams = gr.Image(label="Heatmap Squadre",      interactive=False)
                result_img_zones = gr.Image(label="Zone per Giocatore",   interactive=False)
            gr.Markdown("### 📊 Statistiche")
            result_stats  = gr.HTML('<p style="color:#8b949e">Genera i risultati per vedere le statistiche.</p>')
            result_report = gr.HTML("")

    # ─────────────────────────────────────────────────────────────────────
    # Event wiring
    # ─────────────────────────────────────────────────────────────────────

    # Tab 1
    video_upload.change(fn=on_video_upload, inputs=[video_upload], outputs=[state_video])

    btn_load.click(
        fn=load_frame,
        inputs=[state_video],
        outputs=[state_frame, calib_image, state_pts, calib_instr, btn_calib_ok],
    )
    calib_image.select(
        fn=on_calib_click,
        inputs=[state_frame, state_pts],
        outputs=[calib_image, state_pts, calib_instr, btn_calib_ok],
    )
    btn_undo.click(
        fn=undo_calib,
        inputs=[state_frame, state_pts],
        outputs=[calib_image, state_pts, calib_instr, btn_calib_ok],
    )
    btn_reset.click(
        fn=reset_calib,
        inputs=[state_frame],
        outputs=[calib_image, state_pts, calib_instr, btn_calib_ok],
    )
    btn_calib_ok.click(
        fn=confirm_calib,
        inputs=[state_pts],
        outputs=[state_H, calib_status],
    )

    # Tab 2
    btn_track.click(
        fn=run_tracking,
        inputs=[state_video, clip_slider, segment_input, state_H],
        outputs=[
            state_tracking,
            tracking_status,
            gallery_s1,
            gallery_s2,
            gallery_s3,
            seg2_group,
            seg3_group,
        ],
    )

    # Tab 3 → Tab 4
    btn_results.click(
        fn=generate_results,
        inputs=[
            state_tracking,
            n1_1, n1_2, n1_3, n1_4,
            n2_1, n2_2, n2_3, n2_4,
            n3_1, n3_2, n3_3, n3_4,
        ],
        outputs=[
            result_img_players,
            result_img_teams,
            result_img_zones,
            result_stats,
            result_report,
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PadelVision Web UI")
    parser.add_argument("--share", action="store_true", help="Link pubblico Gradio")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--host",  default="127.0.0.1")
    args = parser.parse_args()

    print()
    print("=" * 52)
    print("  PadelVision — Web Interface")
    print("=" * 52)
    print(f"  → http://{args.host}:{args.port}")
    print()

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[OUTPUT_DIR],
        show_error=True,
    )
