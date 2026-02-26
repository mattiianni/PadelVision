#!/bin/bash
# ============================================================
#  PadelVision — Launcher
#  Fai doppio clic da Finder per avviare l'analisi
# ============================================================

# Vai sempre nella cartella del progetto (indipendente da dove si trova il .command)
PADELVISION_DIR="$HOME/Desktop/PadelVision"
cd "$PADELVISION_DIR"

# Attiva il virtual environment
source "$PADELVISION_DIR/.venv/bin/activate"

echo ""
echo "============================================================"
echo "  PadelVision — Analisi Video Padel"
echo "============================================================"

# ---- Selezione video tramite dialog macOS ----
VIDEO=$(osascript <<'EOF'
tell application "Finder"
    try
        set theFile to choose file ¬
            with prompt "Seleziona il video da analizzare:" ¬
            of type {"mp4", "m4v", "mov", "avi", "mkv"}
        return POSIX path of theFile
    on error
        return ""
    end try
end tell
EOF
)

if [ -z "$VIDEO" ]; then
    echo ""
    echo "  Nessun video selezionato. Uscita."
    echo ""
    read -p "Premi INVIO per chiudere..."
    exit 0
fi

echo ""
echo "  Video : $VIDEO"

# ---- Clip ----
echo ""
echo "  Quanta parte del video analizzare?"
echo "  Esempi: 0.1 = primo 10%  |  0.5 = primo 50%  |  1.0 = tutto"
read -p "  Clip [0.1]: " CLIP
CLIP=${CLIP:-0.1}

# ---- Ricalibrazione ----
echo ""
read -p "  Ricalibra il campo da zero? [s/N]: " RECAL
RECAL_FLAG=""
if [[ "$RECAL" == "s" || "$RECAL" == "S" || "$RECAL" == "si" || "$RECAL" == "SI" ]]; then
    RECAL_FLAG="--recalibrate"
fi

# ---- Cartella output ----
VIDEO_BASENAME=$(basename "$VIDEO" | sed 's/\.[^.]*$//')
OUTPUT_DIR="output_${VIDEO_BASENAME}"

echo ""
echo "  Cartella output: $OUTPUT_DIR"
echo ""
echo "============================================================"
echo "  Avvio analisi..."
echo "============================================================"
echo ""

python3 main.py \
    --clip "$CLIP" \
    --output "$OUTPUT_DIR" \
    $RECAL_FLAG \
    "$VIDEO"

EXIT_CODE=$?

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "  Analisi completata!"
    echo "  Output in: $(pwd)/$OUTPUT_DIR"
else
    echo "  Errore durante l'analisi (codice: $EXIT_CODE)"
fi
echo "============================================================"
echo ""
read -p "Premi INVIO per chiudere..."
