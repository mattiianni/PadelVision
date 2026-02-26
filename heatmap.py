"""
Generazione heatmap e statistiche di zona per i giocatori.

Produce:
  - heatmap_players.png   : heatmap individuale per ogni giocatore
  - heatmap_teams.png     : heatmap aggregata per squadra
  - zone_chart.png        : grafico a barre con % per zona (rete / medio / fondo)
  - stats.json            : dati grezzi per l'API
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless, nessuna finestra
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from scipy.ndimage import gaussian_filter

# -----------------------------------------------------------------------
# Costanti campo padel
# -----------------------------------------------------------------------

COURT_W = 10.0    # larghezza (m)
COURT_L = 20.0    # lunghezza (m)

# Risoluzione griglia heatmap: pixel per metro
PPM = 50
GRID_W = int(COURT_W * PPM)
GRID_H = int(COURT_L * PPM)

# Linea di servizio: 6.95 m dalla rete (regolamento FIP ufficiale)
# La linea divide ogni metà campo in: zona vicina rete (6.95m) + zona fondo (3.05m)
SERVICE_LINE_FROM_NET = 6.95

# Zone (distanza dalla rete verso il fondo, per ogni metà campo)
ZONE_NET  = (0.0, 3.0)              # 0-3 m dalla rete  (zona d'attacco)
ZONE_MID  = (3.0, SERVICE_LINE_FROM_NET)   # 3-6.95 m (fino alla linea di servizio)
ZONE_BACK = (SERVICE_LINE_FROM_NET, 10.0)  # 6.95-10 m (vetro di fondo)

PLAYER_CMAPS = [
    ["#000000", "#ff2222", "#ffff00"],   # rosso → giallo
    ["#000000", "#2255ff", "#00ffff"],   # blu   → ciano
    ["#000000", "#ff8800", "#ffff00"],   # arancio → giallo
    ["#000000", "#aa00ff", "#ff88ff"],   # viola → rosa
]

PLAYER_COLORS_SOLID = ["#ff4444", "#4488ff", "#ff8800", "#cc44ff"]


# -----------------------------------------------------------------------
# Disegno campo
# -----------------------------------------------------------------------

def _draw_court(ax, alpha: float = 1.0):
    """Disegna le linee di un campo padel su axes matplotlib."""
    ax.set_xlim(0, COURT_W)
    ax.set_ylim(0, COURT_L)
    ax.set_aspect("equal")
    ax.set_facecolor("#0a2a50")   # blu campo padel

    kw = dict(color="white", linewidth=1.5, alpha=alpha, solid_capstyle="round")

    # Bordo esterno
    for x0, y0, x1, y1 in [
        (0, 0, COURT_W, 0), (COURT_W, 0, COURT_W, COURT_L),
        (COURT_W, COURT_L, 0, COURT_L), (0, COURT_L, 0, 0),
    ]:
        ax.plot([x0, x1], [y0, y1], color="white", linewidth=2.5, alpha=alpha, solid_capstyle="round")

    net_y = COURT_L / 2

    # Rete
    ax.plot([0, COURT_W], [net_y, net_y], color="white", linewidth=3, alpha=alpha)

    # Linee di servizio: 6.95 m dalla rete (regolamento FIP)
    # Lasciano ~3.05 m fino al vetro di fondo
    sd = SERVICE_LINE_FROM_NET
    ax.plot([0, COURT_W], [net_y - sd, net_y - sd], **kw)
    ax.plot([0, COURT_W], [net_y + sd, net_y + sd], **kw)

    # Linea centrale zona servizio (dalla linea servizio a quella opposta)
    ax.plot([COURT_W / 2, COURT_W / 2], [net_y - sd, net_y + sd], **kw)

    # Label squadre
    for y_pos, label in [(COURT_L * 0.25, "TEAM A"), (COURT_L * 0.75, "TEAM B")]:
        ax.text(COURT_W / 2, y_pos, label, ha="center", va="center",
                color="white", fontsize=9, alpha=0.35, fontweight="bold")

    ax.set_xlabel("Larghezza (m)", color="white", fontsize=8)
    ax.set_ylabel("Lunghezza (m)", color="white", fontsize=8)
    ax.tick_params(colors="white", labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")


# -----------------------------------------------------------------------
# Calcolo griglia
# -----------------------------------------------------------------------

def _build_grid(positions: list[tuple[float, float]], sigma_m: float = 0.6) -> np.ndarray:
    """
    Costruisce e liscita una griglia di densità 2D dalle posizioni sul campo.
    sigma_m: deviazione standard del filtro gaussiano in metri.
    """
    grid = np.zeros((GRID_H, GRID_W), dtype=np.float32)
    for cx, cy in positions:
        px = int(np.clip(cx * PPM, 0, GRID_W - 1))
        py = int(np.clip(cy * PPM, 0, GRID_H - 1))
        grid[py, px] += 1.0
    if grid.max() > 0:
        grid = gaussian_filter(grid, sigma=sigma_m * PPM)
        grid /= grid.max()   # normalizza 0-1
    return grid


def _make_cmap(colors: list[str]) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("_c", colors, N=256)


# -----------------------------------------------------------------------
# Assegnazione squadre
# -----------------------------------------------------------------------

def _assign_teams(tracks: dict) -> tuple[list, list]:
    """
    Divide i giocatori in Team A (metà campo y < 10) e Team B (y >= 10)
    in base alla posizione media.
    """
    team_a, team_b = [], []
    for pid, positions in tracks.items():
        avg_y = np.mean([p[1] for p in positions]) if positions else COURT_L / 2
        (team_a if avg_y < COURT_L / 2 else team_b).append(pid)
    return team_a, team_b


# -----------------------------------------------------------------------
# Statistiche zona
# -----------------------------------------------------------------------

def compute_zone_stats(tracks: dict, fps: float) -> dict:
    """
    Per ogni giocatore calcola:
      - tempo a schermo (secondi)
      - % tempo in zona rete / medio / fondo (relativa alla propria metà campo)
      - % tempo lato sinistro / destro
    """
    stats = {}
    for pid, positions in tracks.items():
        total = len(positions)
        if total == 0:
            continue

        xs = np.array([p[0] for p in positions])
        ys_raw = np.array([p[1] for p in positions])

        # Distanza dalla rete: sempre abs(y - 10), uguale per entrambi i team
        # Team A (y 10→20, camera): dist = y - 10
        # Team B (y 0→10, lontano):  dist = 10 - y
        # → formula unica: abs(y - net)
        ys = np.abs(ys_raw - (COURT_L / 2))

        net_pct  = float(np.sum(ys < ZONE_NET[1])                           / total * 100)
        mid_pct  = float(np.sum((ys >= ZONE_MID[0]) & (ys < ZONE_MID[1])) / total * 100)
        back_pct = float(np.sum(ys >= ZONE_BACK[0])                        / total * 100)

        left_pct  = float(np.sum(xs < COURT_W / 2)  / total * 100)
        right_pct = float(np.sum(xs >= COURT_W / 2) / total * 100)

        stats[pid] = {
            "frames":          total,
            "time_s":          round(total / fps, 1),
            "zone_net_pct":    round(net_pct,  1),
            "zone_mid_pct":    round(mid_pct,  1),
            "zone_back_pct":   round(back_pct, 1),
            "side_left_pct":   round(left_pct,  1),
            "side_right_pct":  round(right_pct, 1),
            "avg_x":           round(float(np.mean(xs)), 2),
            "avg_y":           round(float(np.mean(ys_raw)), 2),
        }
    return stats


# -----------------------------------------------------------------------
# Plot principale: heatmap individuali
# -----------------------------------------------------------------------

def _plot_individual_heatmaps(tracks: dict, team_a: list, team_b: list,
                               fps: float, output_dir: str,
                               player_names: dict = None) -> str:
    players = list(tracks.items())[:4]
    n = len(players)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(4.5 * n, 10))
    fig.patch.set_facecolor("#0d1117")
    if n == 1:
        axes = [axes]

    extent = [0, COURT_W, 0, COURT_L]

    for i, (pid, pos_list) in enumerate(players):
        ax = axes[i]
        positions_2d = [(p[0], p[1]) for p in pos_list]
        grid = _build_grid(positions_2d)

        _draw_court(ax, alpha=0.5)
        cmap = _make_cmap(PLAYER_CMAPS[i % len(PLAYER_CMAPS)])
        ax.imshow(grid, extent=extent, origin="lower", cmap=cmap,
                  alpha=0.80, aspect="auto", interpolation="bilinear",
                  vmin=0, vmax=1)

        team = "A" if pid in team_a else "B"
        color = PLAYER_COLORS_SOLID[i % len(PLAYER_COLORS_SOLID)]
        time_s = len(pos_list) / fps
        pname = (player_names or {}).get(pid, f"Player {pid}")
        ax.set_title(
            f"{pname}  ·  Team {team}\n{len(pos_list)} frame  ·  {time_s:.0f}s",
            color=color, fontsize=11, fontweight="bold", pad=10,
        )
        ax.set_xlabel("Larghezza (m)", color="#aaa", fontsize=8)
        ax.set_ylabel("Lunghezza (m)", color="#aaa", fontsize=8)
        ax.tick_params(colors="#aaa", labelsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    plt.suptitle("Heatmap Giocatori", color="white", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "heatmap_players.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    return path


# -----------------------------------------------------------------------
# Plot: heatmap per squadra
# -----------------------------------------------------------------------

def _plot_team_heatmaps(tracks: dict, team_a: list, team_b: list,
                         fps: float, output_dir: str,
                         player_names: dict = None) -> str:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 10))
    fig.patch.set_facecolor("#0d1117")

    team_cmap = _make_cmap(["#00000000", "#001f7f", "#0055ff",
                             "#00ffff", "#ffff00", "#ff0000"])

    for ax, ids, name in [(ax1, team_a, "Team A"), (ax2, team_b, "Team B")]:
        all_pos = [(p[0], p[1]) for pid in ids for p in tracks.get(pid, [])]
        _draw_court(ax, alpha=0.45)
        if all_pos:
            grid = _build_grid(all_pos)
            ax.imshow(grid, extent=[0, COURT_W, 0, COURT_L], origin="lower",
                      cmap=team_cmap, alpha=0.85, aspect="auto",
                      interpolation="bilinear", vmin=0, vmax=1)
        pnames = [str((player_names or {}).get(pid, f"P{pid}")) for pid in ids]
        ax.set_title(f"{name}  ({' · '.join(pnames)})", color="white",
                     fontsize=12, fontweight="bold")
        ax.set_xlabel("Larghezza (m)", color="#aaa", fontsize=9)
        ax.set_ylabel("Lunghezza (m)", color="#aaa", fontsize=9)
        ax.tick_params(colors="#aaa")

    plt.suptitle("Heatmap per Squadra", color="white", fontsize=14,
                 fontweight="bold", y=1.01)
    plt.tight_layout()
    path = os.path.join(output_dir, "heatmap_teams.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    return path


# -----------------------------------------------------------------------
# Plot: grafico zone per giocatore
# -----------------------------------------------------------------------

def _plot_zone_chart(stats: dict, output_dir: str, player_names: dict = None) -> str:
    pids = list(stats.keys())
    if not pids:
        return None

    net_vals  = [stats[p]["zone_net_pct"]  for p in pids]
    mid_vals  = [stats[p]["zone_mid_pct"]  for p in pids]
    back_vals = [stats[p]["zone_back_pct"] for p in pids]

    x = np.arange(len(pids))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(8, len(pids) * 2.5), 6))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#161b22")

    b1 = ax.bar(x - width, net_vals,  width, label="Zona Rete (0-3m)",      color="#00aaff", alpha=0.85)
    b2 = ax.bar(x,          mid_vals,  width, label="Zona Medio (3-6.95m)", color="#ffaa00", alpha=0.85)
    b3 = ax.bar(x + width,  back_vals, width, label="Fondo (>6.95m)",       color="#ff4444", alpha=0.85)

    def _label_bars(bars):
        for bar in bars:
            h = bar.get_height()
            if h > 3:
                ax.annotate(f"{h:.0f}%",
                            xy=(bar.get_x() + bar.get_width() / 2, h),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", color="white",
                            fontsize=8, fontweight="bold")

    _label_bars(b1)
    _label_bars(b2)
    _label_bars(b3)

    ax.set_xticks(x)
    xlabels = [(player_names or {}).get(p, f"Player {p}") for p in pids]
    ax.set_xticklabels(xlabels, color="white", fontsize=10)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([f"{v}%" for v in range(0, 101, 10)], color="#aaa", fontsize=8)
    ax.set_ylabel("% tempo", color="#aaa")
    ax.set_title("Distribuzione per Zona di Campo", color="white",
                 fontsize=13, fontweight="bold", pad=12)
    ax.legend(facecolor="#1e2530", edgecolor="#333", labelcolor="white",
              fontsize=9, loc="upper right")
    ax.spines["bottom"].set_color("#333")
    ax.spines["left"].set_color("#333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#333", linewidth=0.5, alpha=0.5)

    plt.tight_layout()
    path = os.path.join(output_dir, "zone_chart.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close()
    return path


# -----------------------------------------------------------------------
# Entry point pubblico
# -----------------------------------------------------------------------

def generate_heatmaps(tracks: dict, output_dir: str = "output",
                       fps: float = 30.0,
                       player_names: dict = None) -> tuple[dict, dict]:
    """
    Genera tutti i grafici e le statistiche.
    Padel: sempre 4 giocatori, 2 squadre.

    Ritorna:
      images — dict {"players": path, "teams": path, "zones": path}
      stats  — dict con statistiche per giocatore
    """
    os.makedirs(output_dir, exist_ok=True)

    if not tracks:
        raise ValueError("Nessun giocatore tracciato. Controlla il video o la calibrazione.")

    # Sempre 4 giocatori ordinati per presenza (più frame = più affidabile)
    sorted_tracks = dict(
        sorted(tracks.items(), key=lambda x: len(x[1]), reverse=True)[:4]
    )

    # Team A = i 2 con avg_y più ALTO (lato vicino alla camera, y=10→20)
    # Team B = i 2 con avg_y più BASSO (lato lontano dalla camera, y=0→10)
    by_y = sorted(sorted_tracks.keys(),
                  key=lambda pid: np.mean([p[1] for p in sorted_tracks[pid]]) if sorted_tracks[pid] else 10)
    team_b = by_y[:2]   # avg_y basso = lontano camera
    team_a = by_y[2:]   # avg_y alto  = vicino camera

    stats = compute_zone_stats(sorted_tracks, fps)

    # Aggiunge campo "team" nelle stats per uso in report e summary
    for pid in team_a:
        if pid in stats:
            stats[pid]["team"] = "A"
    for pid in team_b:
        if pid in stats:
            stats[pid]["team"] = "B"

    images = {}

    p = _plot_individual_heatmaps(sorted_tracks, team_a, team_b, fps, output_dir,
                                   player_names=player_names)
    if p:
        images["players"] = p

    p = _plot_team_heatmaps(sorted_tracks, team_a, team_b, fps, output_dir,
                             player_names=player_names)
    if p:
        images["teams"] = p

    p = _plot_zone_chart(stats, output_dir, player_names=player_names)
    if p:
        images["zones"] = p

    # Salva stats JSON
    stats_path = os.path.join(output_dir, "stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    return images, stats
