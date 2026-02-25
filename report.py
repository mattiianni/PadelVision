"""
Genera un report HTML self-contained con le heatmap e le statistiche.
Le immagini vengono embeddate come base64 — nessuna dipendenza esterna.
"""

import base64
import os
import subprocess
from pathlib import Path
from datetime import datetime


def _img_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def generate_report(
    images: dict,    # {"players": path, "teams": path, "zones": path}
    stats: dict,     # {player_id: {...}}
    video_name: str,
    output_dir: str,
) -> str:
    """Genera report HTML e lo apre nel browser. Ritorna il path del file."""

    # --- Immagini base64 ---
    img_players = _img_b64(images["players"]) if images.get("players") else None
    img_teams   = _img_b64(images["teams"])   if images.get("teams")   else None
    img_zones   = _img_b64(images["zones"])   if images.get("zones")   else None

    # --- Tabella statistiche ---
    team_label = {pid: ("A" if i < 2 else "B") for i, pid in enumerate(sorted(stats.keys()))}

    rows = ""
    for pid in sorted(stats.keys()):
        s = stats[pid]
        team = team_label.get(pid, "?")
        badge_color = "#3b82f6" if team == "A" else "#ef4444"
        rows += f"""
        <tr>
          <td><span class="badge" style="background:{badge_color}">Team {team}</span> Giocatore {pid}</td>
          <td>{s['time_s']}s</td>
          <td>
            <div class="bar-wrap">
              <div class="bar net"  style="width:{s['zone_net_pct']}%"  title="Rete {s['zone_net_pct']}%"></div>
              <div class="bar mid"  style="width:{s['zone_mid_pct']}%"  title="Medio {s['zone_mid_pct']}%"></div>
              <div class="bar back" style="width:{s['zone_back_pct']}%" title="Fondo {s['zone_back_pct']}%"></div>
            </div>
            <div class="bar-labels">
              <span>Rete {s['zone_net_pct']}%</span>
              <span>Medio {s['zone_mid_pct']}%</span>
              <span>Fondo {s['zone_back_pct']}%</span>
            </div>
          </td>
          <td>{s['side_left_pct']}% / {s['side_right_pct']}%</td>
        </tr>"""

    players_img_html = f'<img src="data:image/png;base64,{img_players}" alt="Heatmap giocatori">' if img_players else "<p>N/A</p>"
    teams_img_html   = f'<img src="data:image/png;base64,{img_teams}"   alt="Heatmap squadre">'   if img_teams   else "<p>N/A</p>"
    zones_img_html   = f'<img src="data:image/png;base64,{img_zones}"   alt="Zone chart">'         if img_zones   else "<p>N/A</p>"

    date_str = datetime.now().strftime("%d/%m/%Y %H:%M")

    html = f"""<!DOCTYPE html>
<html lang="it">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PadelVision — {video_name}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}

    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #0d1117;
      color: #e6edf3;
      min-height: 100vh;
    }}

    header {{
      background: linear-gradient(135deg, #161b22 0%, #1c2333 100%);
      border-bottom: 1px solid #30363d;
      padding: 24px 40px;
      display: flex;
      align-items: center;
      gap: 16px;
    }}
    header h1 {{
      font-size: 1.6rem;
      font-weight: 700;
      letter-spacing: -0.5px;
    }}
    header h1 span {{ color: #58a6ff; }}
    header .meta {{ color: #8b949e; font-size: 0.85rem; margin-left: auto; }}

    .pill {{
      background: #21262d;
      border: 1px solid #30363d;
      border-radius: 20px;
      padding: 4px 14px;
      font-size: 0.78rem;
      color: #8b949e;
    }}

    main {{ padding: 32px 40px; max-width: 1600px; margin: 0 auto; }}

    section {{ margin-bottom: 48px; }}
    section h2 {{
      font-size: 1rem;
      font-weight: 600;
      color: #8b949e;
      text-transform: uppercase;
      letter-spacing: 1px;
      margin-bottom: 16px;
      padding-bottom: 8px;
      border-bottom: 1px solid #21262d;
    }}

    .img-card {{
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      overflow: hidden;
      padding: 16px;
    }}
    .img-card img {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 8px;
    }}

    .grid-2 {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 24px;
    }}

    /* Tabella statistiche */
    table {{
      width: 100%;
      border-collapse: collapse;
      background: #161b22;
      border: 1px solid #30363d;
      border-radius: 12px;
      overflow: hidden;
    }}
    thead {{ background: #21262d; }}
    th {{
      padding: 12px 16px;
      text-align: left;
      font-size: 0.8rem;
      text-transform: uppercase;
      letter-spacing: 0.5px;
      color: #8b949e;
      font-weight: 600;
    }}
    td {{
      padding: 14px 16px;
      border-top: 1px solid #21262d;
      font-size: 0.9rem;
      vertical-align: middle;
    }}
    tr:hover td {{ background: #1c2333; }}

    .badge {{
      display: inline-block;
      padding: 2px 10px;
      border-radius: 20px;
      font-size: 0.72rem;
      font-weight: 700;
      color: white;
      margin-right: 8px;
    }}

    .bar-wrap {{
      display: flex;
      height: 10px;
      border-radius: 5px;
      overflow: hidden;
      background: #21262d;
      width: 260px;
      margin-bottom: 4px;
    }}
    .bar       {{ height: 100%; transition: width 0.3s; }}
    .bar.net   {{ background: #00aaff; }}
    .bar.mid   {{ background: #ffaa00; }}
    .bar.back  {{ background: #ff4444; }}
    .bar-labels {{
      display: flex;
      gap: 12px;
      font-size: 0.72rem;
      color: #8b949e;
    }}
    .bar-labels span:nth-child(1) {{ color: #00aaff; }}
    .bar-labels span:nth-child(2) {{ color: #ffaa00; }}
    .bar-labels span:nth-child(3) {{ color: #ff4444; }}

    footer {{
      text-align: center;
      padding: 24px;
      color: #484f58;
      font-size: 0.8rem;
      border-top: 1px solid #21262d;
    }}
  </style>
</head>
<body>

<header>
  <div>
    <div class="pill" style="margin-bottom:8px">🎾 PadelVision</div>
    <h1><span>{video_name}</span></h1>
  </div>
  <div class="meta">
    4 giocatori · 2 squadre<br>
    Analisi del {date_str}
  </div>
</header>

<main>

  <section>
    <h2>Heatmap Giocatori</h2>
    <div class="img-card">
      {players_img_html}
    </div>
  </section>

  <section>
    <h2>Confronto Squadre</h2>
    <div class="grid-2">
      <div class="img-card">{teams_img_html}</div>
      <div class="img-card">{zones_img_html}</div>
    </div>
  </section>

  <section>
    <h2>Statistiche per Giocatore</h2>
    <table>
      <thead>
        <tr>
          <th>Giocatore</th>
          <th>Tempo a schermo</th>
          <th>Distribuzione zone</th>
          <th>Sinistra / Destra</th>
        </tr>
      </thead>
      <tbody>
        {rows}
      </tbody>
    </table>
  </section>

</main>

<footer>PadelVision · Generato il {date_str}</footer>

</body>
</html>"""

    report_path = os.path.join(output_dir, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)

    # Apri nel browser di default (macOS)
    subprocess.Popen(["open", report_path])

    return report_path
