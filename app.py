"""
PadelVision — Interfaccia Web (Gradio)

Avvio:
    python app.py          → http://localhost:7860
    python app.py --share  → link pubblico temporaneo
"""

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gradio as gr
from court_calibration import CourtCalibrator, CALIB_POINTS
from player_tracker import PlayerTracker
from heatmap import generate_heatmaps, COURT_W
from report import generate_report

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output_web")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_SEG_ROWS = 4

SEG_TYPES = [
    "Cambio Campo (Fine Set, Stessi Giocatori)",
    "Altra Partita (Cambio Giocatori)",
    "Analisi Parziale",
]

# Colori overlay calibrazione — alto contrasto, no verde (mimetizza col campo)
OVERLAY_COLORS = [
    (255,  60,  60),
    ( 60, 130, 255),
    (255, 165,   0),
    (180,  60, 255),
    (  0, 200, 200),
    (255, 230,   0),
    (255, 100, 160),
    (120, 240, 120),
]


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def fmt_time(s: float) -> str:
    s = int(s)
    h, m, ss = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{ss:02d}" if h > 0 else f"{m:02d}:{ss:02d}"


def time_label_html(in_val: float, out_val: float) -> str:
    return (f'<p style="color:#58a6ff;text-align:center;font-weight:bold;'
            f'font-size:.95rem;margin:4px 0">'
            f'▶ {fmt_time(in_val)} &nbsp;→&nbsp; {fmt_time(out_val)} ◀</p>')


def video_duration(path: str):
    cap   = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total, total / fps


def extract_calib_frame(video_path: str, at_s: float):
    """Estrae il frame esatto al secondo `at_s`."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_n = max(0, min(int(at_s * fps), total - 1))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_n)
    ret, f = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    return None


def draw_calib_overlay(frame_rgb: np.ndarray, pts: list) -> np.ndarray:
    canvas = frame_rgb.copy()
    h, w   = canvas.shape[:2]

    # HUD opaco in cima
    cv2.rectangle(canvas, (0, 0), (w, 66), (22, 22, 24), -1)
    cv2.rectangle(canvas, (0, 65), (w, 66), (60, 60, 60), -1)

    n = len(pts)
    if n < len(CALIB_POINTS):
        label_txt, _ = CALIB_POINTS[n]
        opt = "  [opzionale]" if n >= 4 else ""
        c   = OVERLAY_COLORS[n % len(OVERLAY_COLORS)]
        cv2.circle(canvas, (22, 33), 11, c, -1)
        cv2.circle(canvas, (22, 33), 14, (255, 255, 255), 2)
        cv2.putText(canvas, f"Punto {n + 1}:  {label_txt}{opt}",
                    (44, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.88,
                    (255, 255, 255), 2, cv2.LINE_AA)
    elif n >= 4:
        cv2.circle(canvas, (22, 33), 11, (50, 220, 90), -1)
        cv2.putText(canvas, f"OK  {n} punti — premi Conferma Calibrazione",
                    (44, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.88,
                    (255, 255, 255), 2, cv2.LINE_AA)

    for i, (px, py) in enumerate(pts):
        c = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        cv2.circle(canvas, (int(px), int(py)), 13, c, -1)
        cv2.circle(canvas, (int(px), int(py)), 16, (255, 255, 255), 2)
        num = str(i + 1)
        (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tx, ty = int(px) + 20, int(py) + 8
        cv2.rectangle(canvas, (tx - 3, ty - th - 2), (tx + tw + 3, ty + 2), (0, 0, 0), -1)
        cv2.putText(canvas, num, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    return canvas


def calib_instruction_html(n_pts: int) -> str:
    items = []
    for i, (label, _) in enumerate(CALIB_POINTS):
        if i < n_pts:
            ico, col = "✅", "#3fb950"
        elif i == n_pts:
            c   = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
            col = "#{:02x}{:02x}{:02x}".format(*c)
            ico = "👉"
        else:
            ico, col = "⬜", "#484f58"
        opt = " <em style='color:#484f58'>(opz)</em>" if i >= 4 else ""
        items.append(
            f'<li style="color:{col};line-height:1.9;font-size:.87rem">'
            f'{ico} <b>{i+1}.</b> {label}{opt}</li>'
        )
    note = ""
    if n_pts >= 4:
        note = '<p style="color:#58a6ff;font-size:.82rem;margin:6px 0 0">✓ Minimi raggiunti!</p>'
    elif n_pts == 0:
        note = '<p style="color:#8b949e;font-size:.82rem;margin:6px 0 0">Clicca i punti sul frame a destra.</p>'
    return f'<ol style="margin:0;padding-left:18px">{"".join(items)}</ol>{note}'


def stats_to_html(stats: dict, player_names: dict) -> str:
    if not stats:
        return "<p style='color:#8b949e'>Nessuna statistica.</p>"

    rows = ""
    for pid in sorted(stats.keys(),
                      key=lambda p: (0 if stats[p].get("team","B")=="A" else 1, p)):
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
            f'<td style="text-align:center">'
            f'{s["side_left_pct"]:.0f}% / {s["side_right_pct"]:.0f}%</td></tr>'
        )
    return (
        '<table style="width:100%;border-collapse:collapse;font-size:.88rem;color:#e6edf3">'
        '<thead style="background:#21262d"><tr>'
        '<th style="padding:10px 14px;text-align:left;color:#8b949e;font-size:.75rem;'
        'text-transform:uppercase">Giocatore</th>'
        '<th style="padding:10px;color:#8b949e;font-size:.75rem">Tempo</th>'
        '<th style="padding:10px;color:#00aaff;font-size:.75rem">Rete</th>'
        '<th style="padding:10px;color:#ffaa00;font-size:.75rem">Medio</th>'
        '<th style="padding:10px;color:#ff4444;font-size:.75rem">Fondo</th>'
        '<th style="padding:10px;color:#8b949e;font-size:.75rem;text-transform:uppercase">'
        'Sin / Dx</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    )


# ─────────────────────────────────────────────────────────────────────────────
# Handlers
# ─────────────────────────────────────────────────────────────────────────────

def on_video_upload(path: str):
    """Carica video: estrae frame iniziale, aggiorna slider e info."""
    if not path:
        empty_sliders = [gr.update()] * (MAX_SEG_ROWS * 2)
        return (
            "", 3600.0,
            gr.update(),                            # video_info
            gr.update(value=None, visible=False),   # frame_preview (Tab 1)
            gr.update(visible=False),               # frame_sl (Tab 1)
            gr.update(visible=False),               # btn_load (Tab 1)
            gr.update(visible=False),               # calib_image (Tab 1)
            gr.update(value=None, visible=False),   # frame_preview_2 (Tab 2)
            gr.update(visible=False),               # scrub_sl_2 (Tab 2)
            *empty_sliders,
        )

    fps, total, dur = video_duration(path)
    mm, ss = int(dur // 60), int(dur % 60)
    info = (
        f'<p style="color:#58a6ff;font-size:.9rem;margin:4px 0">'
        f'⏱ <b>{mm:02d}:{ss:02d}</b> ({dur:.0f}s) &nbsp;·&nbsp; {fps:.1f} fps</p>'
        f'<p style="color:#e6edf3;font-size:.85rem;margin:6px 0 2px">'
        f'<b>Passo 1:</b> trascina lo slider — il frame si aggiorna in tempo reale.</p>'
        f'<p style="color:#e6edf3;font-size:.85rem;margin:2px 0">'
        f'<b>Passo 2:</b> quando il frame è buono (tutti e 4 i giocatori visibili), '
        f'premi <b style="color:#3fb950">Conferma Frame</b>.</p>'
    )

    # Estrai frame iniziale a 1/3 del video
    frame_default = dur / 3
    initial_frame = extract_calib_frame(path, frame_default)

    # Slider segmenti (Tab 2)
    s_ins  = [gr.update(maximum=dur, value=0)   for _ in range(MAX_SEG_ROWS)]
    s_outs = [gr.update(maximum=dur, value=dur) for _ in range(MAX_SEG_ROWS)]

    return (
        path, dur, info,
        gr.update(value=initial_frame, visible=True),                        # frame_preview (Tab 1)
        gr.update(maximum=dur, value=frame_default, step=1, visible=True),   # frame_sl (Tab 1)
        gr.update(visible=True),                                             # btn_load (Tab 1)
        gr.update(visible=False),                                            # calib_image (Tab 1)
        gr.update(value=initial_frame, visible=True),                        # frame_preview_2 (Tab 2)
        gr.update(maximum=dur, value=frame_default, step=1, visible=True),   # scrub_sl_2 (Tab 2)
        *s_ins, *s_outs,
    )


def update_preview(state_video: str, sl_val: float):
    """Aggiorna il frame preview mentre si muove lo slider."""
    if not state_video:
        return gr.update()
    frame = extract_calib_frame(state_video, float(sl_val or 0))
    if frame is None:
        return gr.update()
    return frame


def load_frame(state_video: str, frame_sl_val: float):
    if not state_video:
        raise gr.Error("Carica prima un video.")
    frame = extract_calib_frame(state_video, float(frame_sl_val or 0))
    if frame is None:
        raise gr.Error("Impossibile leggere il frame.")
    pts     = []
    overlay = draw_calib_overlay(frame, pts)
    return (
        frame,
        gr.update(value=overlay, visible=True),  # calib_image: mostra con immagine
        pts,
        calib_instruction_html(0),
        gr.update(interactive=False),            # btn_calib_ok
        gr.update(visible=False),                # frame_preview: nascondi
        gr.update(visible=False),                # frame_sl: nascondi
        gr.update(visible=False),                # btn_load: nascondi
    )


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
    pts = []
    overlay = draw_calib_overlay(state_frame, pts) if state_frame is not None else gr.update()
    return overlay, pts, calib_instruction_html(0), gr.update(interactive=False)


def confirm_calib(state_pts: list):
    if len(state_pts) < 4:
        raise gr.Error("Clicca almeno 4 punti.")
    dst_pts = [CALIB_POINTS[i][1] for i in range(len(state_pts))]
    try:
        cal = CourtCalibrator.from_click_points(state_pts, dst_pts)
    except ValueError as e:
        raise gr.Error(str(e))
    status = (
        '<div style="color:#3fb950;font-weight:bold;font-size:.93rem">'
        f'✅ Calibrazione OK — {len(state_pts)} punti · Vai al Tab 2 → Analisi</div>'
    )
    return cal.H.tolist(), status


def add_seg_row(n: int, dur: float):
    new_n = min(n + 1, MAX_SEG_ROWS)
    vis      = [gr.update(visible=i < new_n) for i in range(MAX_SEG_ROWS)]
    out_vals = [gr.update(value=dur) if i == new_n - 1 else gr.update()
                for i in range(MAX_SEG_ROWS)]
    lbl_vals = [time_label_html(0, dur) if i == new_n - 1 else gr.update()
                for i in range(MAX_SEG_ROWS)]
    return [new_n] + vis + out_vals + lbl_vals


def reset_seg_rows():
    vis = [gr.update(visible=False)] * MAX_SEG_ROWS
    return [0] + vis


def run_tracking(
    state_video,
    si0, si1, si2, si3,
    so0, so1, so2, so3,
    st0, st1, st2, st3,
    n_seg_rows, H_state,
    progress=gr.Progress(),
):
    if not state_video:
        raise gr.Error("Carica un video (Tab 1).")
    if H_state is None:
        raise gr.Error("Completa la calibrazione (Tab 1).")

    calibrator = CourtCalibrator.from_homography(H_state)
    tracker    = PlayerTracker()

    fps_v, total_frames, total_dur = video_duration(state_video)

    # Costruisce intervalli dai segmenti aggiunti con +
    raw = list(zip([si0,si1,si2,si3], [so0,so1,so2,so3], [st0,st1,st2,st3]))
    intervals = []
    for i in range(int(n_seg_rows)):
        t_in  = float(raw[i][0] or 0)
        t_out = float(raw[i][1] or total_dur)
        stype = raw[i][2] or SEG_TYPES[0]
        if t_out > t_in:
            intervals.append((t_in, t_out, stype))
    # Fallback: nessun segmento → video intero
    if not intervals:
        intervals = [(0.0, total_dur, "Analisi completa")]

    n_segs = len(intervals)
    all_seg_tracks     = []
    all_seg_crop_paths = []
    fps_result         = fps_v

    for seg_i, (start_s, end_s, seg_label) in enumerate(intervals):
        def make_cb(si, ns):
            def cb(frame, total):
                progress((si + frame / max(total, 1)) / ns,
                         desc=f"Seg {si+1}/{ns}: frame {frame}/{total}")
            return cb

        progress(seg_i / n_segs, desc=f"Segmento {seg_i+1}/{n_segs}…")
        tracks, fps_result, _ = tracker.track_video(
            state_video, calibrator,
            start_s=start_s, end_s=end_s,
            progress_callback=make_cb(seg_i, n_segs),
        )
        tracks = PlayerTracker.filter_players(tracks, min_frames=50, max_players=4)

        seg_dir = os.path.join(OUTPUT_DIR, f"seg{seg_i + 1}")
        progress((seg_i + 0.85) / n_segs, desc=f"Seg {seg_i+1}: crop…")
        crop_paths = tracker.extract_player_crops(
            state_video, tracks, calibrator, seg_dir
        )
        all_seg_tracks.append(tracks)
        all_seg_crop_paths.append(crop_paths)

    progress(1.0, desc="Completato!")

    tracking_state = {
        "all_seg_tracks":     all_seg_tracks,
        "all_seg_crop_paths": all_seg_crop_paths,
        "fps":                fps_result,
        "n_segments":         n_segs,
        "intervals":          intervals,
        "video_path":         state_video,
    }

    # Gallery per Tab 3
    galleries = []
    for seg_i in range(MAX_SEG_ROWS):
        items = []
        if seg_i < n_segs:
            tracks_s    = all_seg_tracks[seg_i]
            crops_s     = all_seg_crop_paths[seg_i]
            sorted_pids = sorted(tracks_s.keys())
            for slot in range(4):
                pid     = sorted_pids[slot] if slot < len(sorted_pids) else None
                path    = crops_s.get(pid) if pid else None
                caption = f"P{slot+1} · {'Team B' if slot < 2 else 'Team A'}"
                if path and os.path.exists(path):
                    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                    items.append((img, caption))
                else:
                    items.append((np.zeros((200, 150, 3), dtype=np.uint8),
                                  caption + " (n/d)"))
        galleries.append(items or [])

    seg_info = "".join(
        f'<li><b>Seg {i+1}:</b> {fmt_time(intervals[i][0])} → '
        f'{fmt_time(intervals[i][1])} · <em>{intervals[i][2]}</em></li>'
        for i in range(n_segs)
    )
    status_html = (
        f'<div style="color:#3fb950;font-weight:bold;font-size:.93rem">'
        f'✅ Tracking completato — {n_segs} segmento/i &nbsp;·&nbsp; '
        f'FPS: {fps_result:.1f}</div>'
        f'<ul style="color:#8b949e;font-size:.82rem;margin:8px 0 0;'
        f'padding-left:18px">{seg_info}</ul>'
        f'<p style="color:#8b949e;font-size:.82rem;margin-top:8px">'
        f'Vai al <b>Tab 3 → Giocatori</b> per assegnare i nomi.</p>'
    )
    return (
        tracking_state, status_html,
        galleries[0], galleries[1], galleries[2], galleries[3],
        gr.update(visible=n_segs >= 2),
        gr.update(visible=n_segs >= 3),
        gr.update(visible=n_segs >= 4),
    )


def generate_results(tracking_state,
                     n1_1, n1_2, n1_3, n1_4,
                     n2_1, n2_2, n2_3, n2_4,
                     n3_1, n3_2, n3_3, n3_4,
                     n4_1, n4_2, n4_3, n4_4):
    if tracking_state is None:
        raise gr.Error("Esegui il tracking (Tab 2).")

    all_seg_tracks     = tracking_state["all_seg_tracks"]
    all_seg_crop_paths = tracking_state["all_seg_crop_paths"]
    fps                = tracking_state["fps"]
    n_segs             = tracking_state["n_segments"]
    video_path         = tracking_state["video_path"]

    all_names_flat = [
        [n1_1, n1_2, n1_3, n1_4],
        [n2_1, n2_2, n2_3, n2_4],
        [n3_1, n3_2, n3_3, n3_4],
        [n4_1, n4_2, n4_3, n4_4],
    ]

    all_seg_names = []
    for seg_i in range(n_segs):
        sorted_pids = sorted(all_seg_tracks[seg_i].keys())
        seg_names   = {}
        for slot, pid in enumerate(sorted_pids):
            raw = all_names_flat[seg_i][slot] if slot < 4 else ""
            seg_names[pid] = raw.strip() if raw and raw.strip() else f"Player {pid}"
        all_seg_names.append(seg_names)

    # Merge per nome
    merged_by_name: dict = {}
    for seg_i in range(n_segs):
        for pid, positions in all_seg_tracks[seg_i].items():
            name = all_seg_names[seg_i].get(pid, f"Player {pid}")
            merged_by_name.setdefault(name, []).extend(positions)

    sorted_names  = sorted(merged_by_name.keys())
    merged_tracks = {i + 1: merged_by_name[n] for i, n in enumerate(sorted_names)}
    player_names  = {i + 1: n for i, n in enumerate(sorted_names)}

    # Inverti sempre destra/sinistra (correzione omografia telecamera)
    merged_tracks = {
        pid: [(COURT_W - cx, cy, fn) for (cx, cy, fn) in pos]
        for pid, pos in merged_tracks.items()
    }

    # Crop
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
        f'style="background:#238636;color:#fff;padding:9px 22px;border-radius:6px;'
        f'text-decoration:none;font-weight:700;font-size:.9rem">'
        f'📄 Apri Report Completo</a></div>'
    )

    return (
        images.get("players"),
        images.get("teams"),
        images.get("zones"),
        stats_to_html(stats, player_names),
        report_link,
    )


# ─────────────────────────────────────────────────────────────────────────────
# UI
# ─────────────────────────────────────────────────────────────────────────────

CSS = "footer { display: none !important; }"

with gr.Blocks(title="PadelVision") as app:

    state_video     = gr.State("")
    state_video_dur = gr.State(3600.0)
    state_frame     = gr.State(None)
    state_pts       = gr.State([])
    state_H         = gr.State(None)
    state_tracking  = gr.State(None)
    state_n_rows    = gr.State(0)

    gr.HTML("""
    <div style="background:linear-gradient(135deg,#161b22,#1c2333);
                border-bottom:1px solid #30363d;padding:18px 28px;margin-bottom:4px">
      <div style="display:inline-block;background:#21262d;border:1px solid #30363d;
                  border-radius:20px;padding:2px 12px;font-size:.75rem;color:#8b949e;
                  margin-bottom:6px">🎾 PadelVision</div>
      <h2 style="margin:0;font-size:1.4rem;font-weight:700;color:#e6edf3">
        Analisi Video Padel</h2>
      <p style="margin:3px 0 0;color:#8b949e;font-size:.83rem">
        1: Calibra &nbsp;·&nbsp; 2: Imposta analisi &nbsp;·&nbsp;
        3: Assegna nomi &nbsp;·&nbsp; 4: Risultati</p>
    </div>""")

    with gr.Tabs():

        # ══════════════════════════════════════════════════
        # TAB 1 — VIDEO & CALIBRAZIONE
        # ══════════════════════════════════════════════════
        with gr.Tab("1 · Video & Calibrazione"):
            with gr.Row():

                # ── Colonna sinistra: upload + lista punti + conferma ────────
                with gr.Column(scale=1, min_width=280):

                    gr.Markdown("### 📹 Video")
                    video_upload = gr.File(
                        file_types=["video"],
                        label="Carica video (.mp4 .mov .m4v .avi ...)",
                        type="filepath",
                    )
                    video_info = gr.HTML(
                        '<p style="color:#8b949e;font-size:.83rem">Nessun video.</p>'
                    )

                    gr.Markdown("---")
                    gr.Markdown("### 🎯 Calibrazione campo")
                    gr.HTML(
                        '<p style="color:#8b949e;font-size:.81rem;margin-bottom:6px">'
                        'Dopo aver confermato il frame, clicca i punti sul campo '
                        'nell\'ordine indicato qui sotto.</p>'
                    )
                    calib_instr = gr.HTML(calib_instruction_html(0))
                    with gr.Row():
                        btn_undo  = gr.Button("↩ Annulla ultimo", size="sm")
                        btn_reset = gr.Button("🗑 Reset",          size="sm")
                    btn_calib_ok = gr.Button(
                        "✅ Conferma Calibrazione",
                        variant="primary", interactive=False,
                    )
                    calib_status = gr.HTML("")

                # ── Colonna destra: preview frame live → slider → calibrazione ──
                with gr.Column(scale=2):

                    # 1. Preview frame — si aggiorna live mentre muovi lo slider
                    frame_preview = gr.Image(
                        label="Anteprima frame (trascina lo slider qui sotto)",
                        interactive=False,
                        height=420,
                        visible=False,
                    )

                    # 2. Slider secondi — appare dopo upload, scompare dopo "Conferma Frame"
                    frame_sl = gr.Slider(
                        minimum=0, maximum=3600, value=0, step=1,
                        label="⏱  Secondo del frame",
                        visible=False,
                    )

                    # 3. Bottone conferma frame
                    btn_load = gr.Button(
                        "🎯  Conferma Frame e Avvia Calibrazione",
                        variant="primary",
                        visible=False,
                    )

                    # 4. Immagine calibrazione — appare dopo "Conferma Frame"
                    calib_image = gr.Image(
                        label="Clicca i punti sul campo (inizia dalla rete)",
                        type="numpy", interactive=True, height=460,
                        visible=False,
                    )

        # ══════════════════════════════════════════════════
        # TAB 2 — ANALISI
        # ══════════════════════════════════════════════════
        with gr.Tab("2 · Analisi"):

            # Preview frame live — per trovare i timestamp IN/OUT
            frame_preview_2 = gr.Image(
                label="Anteprima video — trascina lo slider per esplorare",
                interactive=False,
                height=380,
                visible=False,
            )
            scrub_sl_2 = gr.Slider(
                minimum=0, maximum=3600, value=0, step=1,
                label="⏱  Scorri il video per trovare i secondi di INIZIO e FINE",
                visible=False,
            )

            gr.HTML('<div style="border-top:1px solid #30363d;margin:16px 0"></div>')
            gr.HTML(
                '<div style="color:#e6edf3;font-weight:700;font-size:.9rem;'
                'margin-bottom:6px">Range di analisi</div>'
                '<p style="color:#8b949e;font-size:.81rem;margin-bottom:10px">'
                'Premi <b>+</b> per aggiungere un range. '
                'Ogni range ha IN, OUT e tipo. '
                'I giocatori con lo stesso nome in range diversi vengono uniti.</p>'
            )

            seg_groups    = []
            seg_in_sls    = []
            seg_out_sls   = []
            seg_type_dds  = []
            seg_time_lbls = []

            for i in range(MAX_SEG_ROWS):
                with gr.Group(visible=False) as sg:
                    gr.HTML(f'<div style="color:#58a6ff;font-size:.82rem;'
                            f'font-weight:700;margin:8px 0 4px">Segmento {i+1}</div>')
                    with gr.Row():
                        si = gr.Slider(0, 3600, value=0,    step=1, label="▶ IN",  scale=4)
                        so = gr.Slider(0, 3600, value=3600, step=1, label="OUT ◀", scale=4)
                        st = gr.Dropdown(choices=SEG_TYPES, value=SEG_TYPES[0],
                                         label="Tipo", scale=3)
                    lbl = gr.HTML(time_label_html(0, 3600))

                seg_groups.append(sg)
                seg_in_sls.append(si)
                seg_out_sls.append(so)
                seg_type_dds.append(st)
                seg_time_lbls.append(lbl)

            with gr.Row():
                btn_add_seg   = gr.Button("+ Aggiungi Segmento",
                                           variant="secondary", size="sm")
                btn_reset_seg = gr.Button("Rimuovi tutti", variant="stop", size="sm")

            gr.HTML('<div style="border-top:1px solid #30363d;margin:16px 0"></div>')
            btn_track = gr.Button("▶️ Avvia Analisi", variant="primary")
            tracking_status = gr.HTML(
                '<div style="color:#8b949e;padding:8px">In attesa...</div>'
            )

        # ══════════════════════════════════════════════════
        # TAB 3 — NOMI GIOCATORI
        # ══════════════════════════════════════════════════
        with gr.Tab("3 · Giocatori"):
            gr.HTML(
                '<p style="color:#8b949e;margin-bottom:10px">'
                'Guarda i crop e scrivi il nome di ogni giocatore. '
                'Stessi nomi in segmenti diversi = dati uniti automaticamente.</p>'
            )
            name_groups  = []
            galleries    = []
            all_name_tbs = []

            for seg_i in range(MAX_SEG_ROWS):
                with gr.Group(visible=(seg_i == 0)) as ng:
                    gr.HTML(
                        f'<div style="color:#58a6ff;font-weight:700;font-size:.93rem;'
                        f'margin:4px 0 8px">📹 Segmento {seg_i + 1}'
                        + (" · <em style='color:#8b949e;font-weight:normal'>"
                           "dopo cambio campo</em>" if seg_i > 0 else "")
                        + "</div>"
                    )
                    gal = gr.Gallery(columns=4, rows=1, height=260, show_label=False)
                    galleries.append(gal)
                    with gr.Row():
                        row_tbs = []
                        for slot in range(4):
                            tb = gr.Textbox(
                                label=f"P{slot+1} — {'Team B' if slot < 2 else 'Team A'}",
                                placeholder="Nome...",
                            )
                            row_tbs.append(tb)
                        all_name_tbs.extend(row_tbs)
                name_groups.append(ng)

            btn_results = gr.Button("📊 Genera Heatmap e Report", variant="primary")

        # ══════════════════════════════════════════════════
        # TAB 4 — RISULTATI
        # ══════════════════════════════════════════════════
        with gr.Tab("4 · Risultati"):
            result_img_players = gr.Image(label="Heatmap Giocatori",    interactive=False)
            with gr.Row():
                result_img_teams = gr.Image(label="Heatmap Squadre",    interactive=False)
                result_img_zones = gr.Image(label="Zone per Giocatore", interactive=False)
            gr.Markdown("### 📊 Statistiche")
            result_stats  = gr.HTML(
                '<p style="color:#8b949e">Genera i risultati per vedere le statistiche.</p>'
            )
            result_report = gr.HTML("")

    # ─────────────────────────────────────────────────────────────────────
    # Wiring
    # ─────────────────────────────────────────────────────────────────────

    # Tab 1 — upload video
    video_upload.change(
        fn=on_video_upload,
        inputs=[video_upload],
        outputs=[
            state_video, state_video_dur, video_info,
            frame_preview, frame_sl, btn_load, calib_image,
            frame_preview_2, scrub_sl_2,
            *seg_in_sls, *seg_out_sls,
        ],
    )

    # Tab 1 — slider muove il frame preview in tempo reale
    frame_sl.change(
        fn=update_preview,
        inputs=[state_video, frame_sl],
        outputs=[frame_preview],
    )

    # Tab 1 — conferma frame → avvia calibrazione
    btn_load.click(
        fn=load_frame,
        inputs=[state_video, frame_sl],
        outputs=[
            state_frame, calib_image, state_pts, calib_instr,
            btn_calib_ok, frame_preview, frame_sl, btn_load,
        ],
    )

    # Tab 2 — slider scrub muove il frame preview in tempo reale
    scrub_sl_2.change(
        fn=update_preview,
        inputs=[state_video, scrub_sl_2],
        outputs=[frame_preview_2],
    )

    # Tab 1 — click sui punti del campo
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

    # Tab 2 — slider segmenti → label tempo
    for i in range(MAX_SEG_ROWS):
        gr.on(
            triggers=[seg_in_sls[i].change, seg_out_sls[i].change],
            fn=time_label_html,
            inputs=[seg_in_sls[i], seg_out_sls[i]],
            outputs=[seg_time_lbls[i]],
        )

    # Tab 2 — + / reset segmenti
    btn_add_seg.click(
        fn=add_seg_row,
        inputs=[state_n_rows, state_video_dur],
        outputs=[state_n_rows] + seg_groups + seg_out_sls + seg_time_lbls,
    )
    btn_reset_seg.click(
        fn=reset_seg_rows,
        inputs=[],
        outputs=[state_n_rows] + seg_groups,
    )

    # Tab 2 — tracking
    btn_track.click(
        fn=run_tracking,
        inputs=[
            state_video,
            seg_in_sls[0], seg_in_sls[1], seg_in_sls[2], seg_in_sls[3],
            seg_out_sls[0], seg_out_sls[1], seg_out_sls[2], seg_out_sls[3],
            seg_type_dds[0], seg_type_dds[1], seg_type_dds[2], seg_type_dds[3],
            state_n_rows, state_H,
        ],
        outputs=[
            state_tracking, tracking_status,
            galleries[0], galleries[1], galleries[2], galleries[3],
            name_groups[1], name_groups[2], name_groups[3],
        ],
    )

    # Tab 3 → Tab 4
    btn_results.click(
        fn=generate_results,
        inputs=[state_tracking] + all_name_tbs,
        outputs=[
            result_img_players, result_img_teams, result_img_zones,
            result_stats, result_report,
        ],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--host",  default="127.0.0.1")
    args = parser.parse_args()

    print(f"\n{'='*52}\n  PadelVision — Web Interface\n{'='*52}")
    print(f"  → http://{args.host}:{args.port}\n")

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[OUTPUT_DIR],
        show_error=True,
        inbrowser=True,
        css=CSS,
        theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate"),
    )
