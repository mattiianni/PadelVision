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

# Colori ad alto contrasto per l'overlay di calibrazione (RGB)
# Volutamente evitato il verde puro — mimetizza con il manto del campo
OVERLAY_COLORS = [
    (255,  60,  60),   # 1 — rosso vivo
    ( 60, 130, 255),   # 2 — blu brillante
    (255, 165,   0),   # 3 — arancio
    (180,  60, 255),   # 4 — viola
    (  0, 200, 200),   # 5 — teal  [opzionale]
    (255, 230,   0),   # 6 — giallo [opzionale]
    (255, 100, 160),   # 7 — rosa   [opzionale]
    (120, 240, 120),   # 8 — verde  [opzionale]
]


# ─────────────────────────────────────────────────────────────────────────────
# Utility
# ─────────────────────────────────────────────────────────────────────────────

def parse_time(s: str):
    """Converte 'MM:SS' o '720' → float secondi, o None se vuoto/invalido."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        if ":" in s:
            parts = s.split(":")
            return int(parts[0]) * 60 + float(parts[1])
        return float(s)
    except (ValueError, IndexError):
        return None


def video_duration(path: str):
    """Ritorna (fps, total_frames, duration_s) dal video."""
    cap   = cv2.VideoCapture(path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return fps, total, total / fps


def extract_calib_frame(video_path: str, in_s: float = None):
    """Estrae un frame adatto alla calibrazione (vicino all'IN, poi a 1 min, poi 10%)."""
    cap   = cv2.VideoCapture(video_path)
    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    candidates = []
    if in_s is not None:
        candidates.append(int((in_s + 5) * fps))   # 5s dopo l'IN
    candidates += [int(60 * fps), int(total * 0.10), int(total * 0.02), 0]

    for t in candidates:
        t = max(0, min(t, total - 1))
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ret, f = cap.read()
        if ret and np.mean(f) > 20:
            cap.release()
            return cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    cap.release()
    return None


def draw_calib_overlay(frame_rgb: np.ndarray, pts: list) -> np.ndarray:
    """
    Overlay per la calibrazione con:
    - HUD barra in cima completamente opaca + testo BIANCO + pallino colorato
    - Cerchi colorati ad alto contrasto con bordo bianco + numero su sfondo scuro
    """
    canvas = frame_rgb.copy()
    h, w   = canvas.shape[:2]

    # ── HUD barra in cima ────────────────────────────────────────────────
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
        cv2.putText(canvas, f"✓ {n} punti — premi  Conferma Calibrazione",
                    (44, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.88,
                    (255, 255, 255), 2, cv2.LINE_AA)

    # ── Punti già cliccati ────────────────────────────────────────────────
    for i, (px, py) in enumerate(pts):
        c = OVERLAY_COLORS[i % len(OVERLAY_COLORS)]
        cv2.circle(canvas, (int(px), int(py)), 13, c, -1)
        cv2.circle(canvas, (int(px), int(py)), 16, (255, 255, 255), 2)
        # Numero su sfondo scuro
        num = str(i + 1)
        (tw, th), _ = cv2.getTextSize(num, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        tx, ty = int(px) + 20, int(py) + 8
        cv2.rectangle(canvas, (tx - 3, ty - th - 2), (tx + tw + 3, ty + 2),
                      (0, 0, 0), -1)
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
            hex_c = "#{:02x}{:02x}{:02x}".format(*c)
            ico, col = "👉", hex_c
        else:
            ico, col = "⬜", "#484f58"
        opt = " <em style='color:#484f58'>(opzionale)</em>" if i >= 4 else ""
        items.append(
            f'<li style="color:{col};line-height:1.9;font-size:.87rem">'
            f'{ico} <b>{i+1}.</b> {label}{opt}</li>'
        )
    note = ""
    if n_pts >= 4:
        note = ('<p style="color:#58a6ff;margin:8px 0 0;font-size:.83rem">'
                '✓ Minimi raggiunti! Puoi confermare ora.</p>')
    elif n_pts == 0:
        note = ('<p style="color:#8b949e;margin:8px 0 0;font-size:.82rem">'
                'Prima premi <b>Estrai Frame</b>, poi clicca sul campo '
                'nell\'ordine indicato.</p>')
    return (f'<ol style="margin:0;padding-left:18px">{"".join(items)}</ol>{note}')


def stats_to_html(stats: dict, player_names: dict) -> str:
    if not stats:
        return "<p style='color:#8b949e'>Nessuna statistica disponibile.</p>"

    def sk(pid):
        return (0 if stats[pid].get("team", "B") == "A" else 1, pid)

    rows = ""
    for pid in sorted(stats.keys(), key=sk):
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
# Event handlers
# ─────────────────────────────────────────────────────────────────────────────

def on_video_upload(path: str):
    if not path:
        return "", '<p style="color:#8b949e">Nessun video caricato.</p>'
    _, _, dur = video_duration(path)
    mm, ss = int(dur // 60), int(dur % 60)
    info = (f'<p style="color:#58a6ff;font-size:.9rem;margin:4px 0">'
            f'⏱ Durata: <b>{mm:02d}:{ss:02d}</b> ({dur:.0f}s)</p>'
            f'<p style="color:#8b949e;font-size:.82rem">'
            f'Imposta IN/OUT se vuoi analizzare solo una parte, '
            f'poi clicca <b>Estrai Frame</b>.</p>')
    return path, info


def load_frame(state_video: str, in_tb: str):
    if not state_video:
        raise gr.Error("Carica prima un video.")
    in_s  = parse_time(in_tb)
    frame = extract_calib_frame(state_video, in_s)
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
        f'✅ Calibrazione OK — {len(state_pts)} punti · Vai al Tab 2 → Analisi</div>'
    )
    return cal.H.tolist(), status


def add_seg_row(n):
    new_n = min(n + 1, MAX_SEG_ROWS)
    updates = [gr.update(visible=i < new_n) for i in range(MAX_SEG_ROWS)]
    return [new_n] + updates


def reset_seg_rows():
    updates = [gr.update(visible=False)] * MAX_SEG_ROWS
    return [0] + updates


def run_tracking(
    state_video, clip_in_str, clip_out_str,
    si0, si1, si2, si3,     # segment IN times
    so0, so1, so2, so3,     # segment OUT times
    st0, st1, st2, st3,     # segment types
    n_seg_rows, mirror_x, H_state,
    progress=gr.Progress(),
):
    if not state_video:
        raise gr.Error("Carica un video (Tab 1).")
    if H_state is None:
        raise gr.Error("Completa la calibrazione (Tab 1).")

    calibrator = CourtCalibrator.from_homography(H_state)
    tracker    = PlayerTracker()

    fps_v, total_frames, total_dur = video_duration(state_video)

    # ── Calcola clip globale ──────────────────────────────────────────────
    global_in  = parse_time(clip_in_str)  or 0.0
    global_out = parse_time(clip_out_str) or total_dur

    # ── Costruisce lista intervalli ───────────────────────────────────────
    raw_segs = list(zip(
        [si0, si1, si2, si3],
        [so0, so1, so2, so3],
        [st0, st1, st2, st3],
    ))

    intervals = []  # [(start_s, end_s, label)]
    if n_seg_rows == 0:
        # Nessun segmento → analisi unica sull'intervallo globale
        intervals = [(global_in, global_out, "Analisi")]
    else:
        for i in range(n_seg_rows):
            t_in  = parse_time(raw_segs[i][0])
            t_out = parse_time(raw_segs[i][1])
            seg_type = raw_segs[i][2] or SEG_TYPES[0]
            if t_in is not None and t_out is not None and t_out > t_in:
                # Clip all'intervallo globale
                t_in  = max(t_in,  global_in)
                t_out = min(t_out, global_out)
                intervals.append((t_in, t_out, seg_type))
        if not intervals:
            intervals = [(global_in, global_out, "Analisi")]

    n_segs = len(intervals)
    all_seg_tracks     = []
    all_seg_crop_paths = []
    fps_result         = fps_v

    for seg_i, (start_s, end_s, seg_label) in enumerate(intervals):
        def make_cb(si, ns):
            def cb(frame, total):
                overall = (si + frame / max(total, 1)) / ns
                progress(overall, desc=f"Seg {si+1}/{ns}: frame {frame}/{total}")
            return cb

        progress(seg_i / n_segs, desc=f"Segmento {seg_i+1}/{n_segs}: tracking…")
        tracks, fps_result, _ = tracker.track_video(
            state_video, calibrator,
            start_s=start_s, end_s=end_s,
            progress_callback=make_cb(seg_i, n_segs),
        )
        tracks = PlayerTracker.filter_players(tracks, min_frames=50, max_players=4)

        seg_dir    = os.path.join(OUTPUT_DIR, f"seg{seg_i + 1}")
        progress((seg_i + 0.85) / n_segs,
                 desc=f"Segmento {seg_i+1}/{n_segs}: estrazione crop…")
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
        "mirror_x":           mirror_x,
    }

    # Gallery per tab 3 — già ordinati per avg_y: P1-P2=TeamB, P3-P4=TeamA
    galleries = []
    for seg_i in range(MAX_SEG_ROWS):
        items = []
        if seg_i < n_segs:
            tracks_s    = all_seg_tracks[seg_i]
            crops_s     = all_seg_crop_paths[seg_i]
            sorted_pids = sorted(tracks_s.keys())
            _, _, seg_label = intervals[seg_i]
            for slot in range(4):
                pid      = sorted_pids[slot] if slot < len(sorted_pids) else None
                path     = crops_s.get(pid) if pid else None
                team_lbl = "Team B" if slot < 2 else "Team A"
                caption  = f"P{slot+1} · {team_lbl}"
                if path and os.path.exists(path):
                    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                    items.append((img, caption))
                else:
                    items.append((np.zeros((200, 150, 3), dtype=np.uint8),
                                  caption + " (non disponibile)"))
        galleries.append(items or [])

    dur_str = f"{int(total_dur//60):02d}:{int(total_dur%60):02d}"
    seg_labels_html = "".join(
        f'<li><b>Seg {i+1}:</b> {intervals[i][0]:.0f}s → {intervals[i][1]:.0f}s '
        f'· <em>{intervals[i][2]}</em></li>'
        for i in range(n_segs)
    )
    status_html = (
        f'<div style="color:#3fb950;font-weight:bold;font-size:.95rem">'
        f'✅ Tracking completato — {n_segs} segmento/i &nbsp;·&nbsp; '
        f'FPS: {fps_result:.1f} &nbsp;·&nbsp; Durata: {dur_str}</div>'
        f'<ul style="color:#8b949e;font-size:.83rem;margin:8px 0 0;'
        f'padding-left:20px">{seg_labels_html}</ul>'
        f'<p style="color:#8b949e;font-size:.83rem;margin-top:8px">'
        f'Vai al <b>Tab 3 → Giocatori</b> per assegnare i nomi.</p>'
    )
    mirror_note = ""
    if mirror_x:
        mirror_note = ('<p style="color:#ffaa00;font-size:.82rem">'
                       '⚠️ Mirror X attivo: destra ↔ sinistra invertita nel report.</p>')

    return (
        tracking_state,
        status_html + mirror_note,
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
        raise gr.Error("Esegui il tracking (Tab 2) prima.")

    all_seg_tracks     = tracking_state["all_seg_tracks"]
    all_seg_crop_paths = tracking_state["all_seg_crop_paths"]
    fps                = tracking_state["fps"]
    n_segs             = tracking_state["n_segments"]
    video_path         = tracking_state["video_path"]
    mirror_x           = tracking_state.get("mirror_x", False)

    all_names_flat = [
        [n1_1, n1_2, n1_3, n1_4],
        [n2_1, n2_2, n2_3, n2_4],
        [n3_1, n3_2, n3_3, n3_4],
        [n4_1, n4_2, n4_3, n4_4],
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

    # Merge per nome
    merged_by_name: dict = {}
    for seg_i in range(n_segs):
        for pid, positions in all_seg_tracks[seg_i].items():
            name = all_seg_names[seg_i].get(pid, f"Player {pid}")
            merged_by_name.setdefault(name, []).extend(positions)

    sorted_names  = sorted(merged_by_name.keys())
    merged_tracks = {i + 1: merged_by_name[n] for i, n in enumerate(sorted_names)}
    player_names  = {i + 1: n for i, n in enumerate(sorted_names)}

    # Applica mirror X se richiesto (sin ↔ dx)
    if mirror_x:
        merged_tracks = {
            pid: [(COURT_W - cx, cy, fn) for (cx, cy, fn) in pos]
            for pid, pos in merged_tracks.items()
        }

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
        f'📄 Apri Report Completo</a>'
        f'&nbsp;&nbsp;<span style="color:#484f58;font-size:.8rem">{report_path}</span>'
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
# UI
# ─────────────────────────────────────────────────────────────────────────────

CSS = "footer { display: none !important; }"

with gr.Blocks(title="PadelVision", css=CSS,
               theme=gr.themes.Base(primary_hue="blue", neutral_hue="slate")) as app:

    state_video    = gr.State("")
    state_frame    = gr.State(None)
    state_pts      = gr.State([])
    state_H        = gr.State(None)
    state_tracking = gr.State(None)
    state_n_rows   = gr.State(0)      # righe segmento visibili

    # Header
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#161b22,#1c2333);
                border-bottom:1px solid #30363d;padding:18px 28px;margin-bottom:4px">
      <div style="display:inline-block;background:#21262d;border:1px solid #30363d;
                  border-radius:20px;padding:2px 12px;font-size:.75rem;color:#8b949e;
                  margin-bottom:6px">🎾 PadelVision</div>
      <h2 style="margin:0;font-size:1.4rem;font-weight:700;color:#e6edf3">
        Analisi Video Padel</h2>
      <p style="margin:3px 0 0;color:#8b949e;font-size:.83rem">
        Tab 1: Calibra · Tab 2: Analisi · Tab 3: Nomi · Tab 4: Risultati</p>
    </div>""")

    with gr.Tabs():

        # ══════════════════════════════════════════════════
        # TAB 1 — VIDEO & CALIBRAZIONE
        # ══════════════════════════════════════════════════
        with gr.Tab("1 · Video & Calibrazione"):
            with gr.Row():

                # Colonna sinistra
                with gr.Column(scale=1, min_width=280):
                    gr.Markdown("### 📹 Video")
                    video_upload = gr.File(
                        file_types=["video"],
                        label="Carica video (.mp4 .mov .m4v .avi ...)",
                        type="filepath",
                    )
                    video_info = gr.HTML(
                        '<p style="color:#8b949e;font-size:.83rem">Nessun video caricato.</p>'
                    )
                    with gr.Row():
                        clip_in_tb  = gr.Textbox(
                            label="Analizza DA (MM:SS o sec)",
                            placeholder="00:00  (default: inizio)",
                        )
                        clip_out_tb = gr.Textbox(
                            label="Analizza FINO A (MM:SS o sec)",
                            placeholder="fine  (default: fine video)",
                        )
                    btn_load = gr.Button("📷 Estrai Frame per Calibrazione",
                                         variant="secondary", size="sm")

                    gr.Markdown("---")
                    gr.Markdown("### 🎯 Punti di calibrazione")
                    gr.HTML(
                        '<p style="color:#8b949e;font-size:.82rem;margin-bottom:6px">'
                        '<b>Sinistro/Destro</b> = dalla prospettiva della telecamera '
                        '(stai guardando il campo dalla telecamera verso la rete).</p>'
                    )
                    calib_instr = gr.HTML(calib_instruction_html(0))
                    with gr.Row():
                        btn_undo  = gr.Button("↩ Annulla", size="sm")
                        btn_reset = gr.Button("🗑 Reset",   size="sm")
                    btn_calib_ok = gr.Button(
                        "✅ Conferma Calibrazione",
                        variant="primary", interactive=False,
                    )
                    calib_status = gr.HTML("")

                # Colonna destra — frame interattivo
                with gr.Column(scale=2):
                    gr.HTML(
                        '<p style="color:#8b949e;font-size:.83rem;margin-bottom:6px">'
                        'Clicca sul campo nell\'ordine indicato a sinistra. '
                        'Il numero e il colore corrispondono alla lista.</p>'
                    )
                    calib_image = gr.Image(
                        label="Frame — clicca per calibrare",
                        type="numpy", interactive=True, height=540,
                    )

        # ══════════════════════════════════════════════════
        # TAB 2 — ANALISI
        # ══════════════════════════════════════════════════
        with gr.Tab("2 · Analisi"):
            with gr.Row():

                # Colonna sinistra — impostazioni
                with gr.Column(scale=1, min_width=300):
                    gr.Markdown("### 🎬 Segmenti da analizzare")
                    gr.HTML(
                        '<p style="color:#8b949e;font-size:.82rem;margin-bottom:10px">'
                        'Aggiungi uno o più range IN→OUT. Se non aggiungi nulla, '
                        'viene analizzato l\'intero video (secondo i limiti di Tab 1).<br>'
                        '<b>Cambio Campo</b> = stessi giocatori, lato opposto (fine set).<br>'
                        '<b>Altra Partita</b> = giocatori diversi.<br>'
                        '<b>Analisi Parziale</b> = vuoi vedere solo quella porzione.</p>'
                    )

                    # Righe segmenti dinamiche (max 4)
                    seg_groups = []
                    seg_in_tbs   = []
                    seg_out_tbs  = []
                    seg_type_dds = []

                    for i in range(MAX_SEG_ROWS):
                        with gr.Group(visible=False) as sg:
                            gr.HTML(f'<div style="color:#58a6ff;font-size:.82rem;'
                                    f'font-weight:700;margin:6px 0 2px">'
                                    f'Segmento {i+1}</div>')
                            with gr.Row():
                                si = gr.Textbox(label="IN  (MM:SS o sec)",
                                                placeholder="es: 0:00", scale=1)
                                so = gr.Textbox(label="OUT (MM:SS o sec)",
                                                placeholder="es: 25:00", scale=1)
                            st = gr.Dropdown(
                                label="Tipo segmento",
                                choices=SEG_TYPES,
                                value=SEG_TYPES[0],
                            )
                        seg_groups.append(sg)
                        seg_in_tbs.append(si)
                        seg_out_tbs.append(so)
                        seg_type_dds.append(st)

                    with gr.Row():
                        btn_add_seg   = gr.Button("+ Aggiungi Segmento",
                                                   variant="secondary", size="sm")
                        btn_reset_seg = gr.Button("Rimuovi tutti",
                                                   variant="stop", size="sm")

                    gr.Markdown("---")
                    gr.Markdown("### ⚙️ Opzioni")
                    mirror_x_cb = gr.Checkbox(
                        label="Specchia asse Sin/Dx (se destra e sinistra sono invertite nell'analisi)",
                        value=False,
                    )
                    gr.HTML(
                        '<p style="color:#8b949e;font-size:.81rem;margin-top:2px">'
                        'Attiva solo se nell\'analisi un giocatore risulta sul lato '
                        'opposto rispetto a dove gioca realmente.</p>'
                    )

                    btn_track = gr.Button("▶️ Avvia Analisi", variant="primary")

                # Colonna destra — status
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
                'Guarda i crop di ogni segmento e scrivi il nome di ogni giocatore. '
                'I giocatori con lo stesso nome in segmenti diversi vengono uniti '
                'automaticamente nelle statistiche.</p>'
            )

            # Righe naming: MAX_SEG_ROWS segmenti × 4 giocatori
            name_groups  = []
            galleries    = []
            all_name_tbs = []   # flat: [s0p0, s0p1, s0p2, s0p3, s1p0, ...]

            for seg_i in range(MAX_SEG_ROWS):
                with gr.Group(visible=(seg_i == 0)) as ng:
                    gr.HTML(f'<div style="color:#58a6ff;font-weight:700;'
                            f'font-size:.95rem;margin:4px 0 8px">'
                            f'📹 Segmento {seg_i + 1}'
                            + (" &nbsp;·&nbsp; <em style='color:#8b949e;"
                               "font-weight:normal'>dopo cambio campo</em>"
                               if seg_i > 0 else "")
                            + "</div>")
                    gal = gr.Gallery(
                        columns=4, rows=1, height=260, show_label=False,
                    )
                    galleries.append(gal)
                    with gr.Row():
                        row_tbs = []
                        for slot in range(4):
                            team_lbl = "Team B" if slot < 2 else "Team A"
                            tb = gr.Textbox(
                                label=f"P{slot+1} — {team_lbl}",
                                placeholder="Nome giocatore...",
                            )
                            row_tbs.append(tb)
                        all_name_tbs.extend(row_tbs)
                name_groups.append(ng)

            btn_results = gr.Button("📊 Genera Heatmap e Report", variant="primary")

        # ══════════════════════════════════════════════════
        # TAB 4 — RISULTATI
        # ══════════════════════════════════════════════════
        with gr.Tab("4 · Risultati"):
            result_img_players = gr.Image(label="Heatmap Giocatori",   interactive=False)
            with gr.Row():
                result_img_teams = gr.Image(label="Heatmap Squadre",   interactive=False)
                result_img_zones = gr.Image(label="Zone per Giocatore",interactive=False)
            gr.Markdown("### 📊 Statistiche")
            result_stats  = gr.HTML(
                '<p style="color:#8b949e">Genera i risultati per vedere le statistiche.</p>'
            )
            result_report = gr.HTML("")

    # ─────────────────────────────────────────────────────────────────────
    # Event wiring
    # ─────────────────────────────────────────────────────────────────────

    # Tab 1
    video_upload.change(
        fn=on_video_upload,
        inputs=[video_upload],
        outputs=[state_video, video_info],
    )
    btn_load.click(
        fn=load_frame,
        inputs=[state_video, clip_in_tb],
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

    # Tab 2 — segment rows
    btn_add_seg.click(
        fn=add_seg_row,
        inputs=[state_n_rows],
        outputs=[state_n_rows] + seg_groups,
    )
    btn_reset_seg.click(
        fn=reset_seg_rows,
        inputs=[],
        outputs=[state_n_rows] + seg_groups,
    )

    # Tab 2 — run tracking
    btn_track.click(
        fn=run_tracking,
        inputs=[
            state_video, clip_in_tb, clip_out_tb,
            seg_in_tbs[0], seg_in_tbs[1], seg_in_tbs[2], seg_in_tbs[3],
            seg_out_tbs[0], seg_out_tbs[1], seg_out_tbs[2], seg_out_tbs[3],
            seg_type_dds[0], seg_type_dds[1], seg_type_dds[2], seg_type_dds[3],
            state_n_rows, mirror_x_cb, state_H,
        ],
        outputs=[
            state_tracking,
            tracking_status,
            galleries[0], galleries[1], galleries[2], galleries[3],
            name_groups[1], name_groups[2], name_groups[3],
        ],
    )

    # Tab 3 → Tab 4
    # all_name_tbs = 16 textbox (4 segs × 4 players)
    btn_results.click(
        fn=generate_results,
        inputs=[state_tracking] + all_name_tbs,
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
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port",  type=int, default=7860)
    parser.add_argument("--host",  default="127.0.0.1")
    args = parser.parse_args()

    print("\n" + "=" * 52)
    print("  PadelVision — Web Interface")
    print("=" * 52)
    print(f"  → http://{args.host}:{args.port}\n")

    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        allowed_paths=[OUTPUT_DIR],
        show_error=True,
        inbrowser=True,
    )
