# PadelVision

Analisi video di partite di padel: heatmap giocatori, velocità palla, rilevamento rally.

## Stack
- Python 3.12 + FastAPI (backend analisi)
- YOLOv8 (rilevamento giocatori e palla)
- OpenCV (elaborazione video)
- React + Vite (frontend — prossimamente)

## Setup

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Avvio

```bash
source .venv/bin/activate
python main.py
```

## Features
- [x] Setup ambiente
- [ ] Rilevamento giocatori + heatmap
- [ ] Tracking palla + velocità
- [ ] Rilevamento rally
- [ ] API FastAPI
- [ ] Frontend React
