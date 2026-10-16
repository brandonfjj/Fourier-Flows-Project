#!/usr/bin/env bash
# correr_corrida.sh -- Corrida completa del pipeline de Fourier Flows, en orden y en su propia carpeta.
#
# Crea results/runs/<id>/ con id = s<semilla>_e<epocas>_lr<lr>_p<paso>_<etiqueta> y ejecuta:
#   1) pmu_pipeline.py      -> pmu_sintetico.npz/.json, pmu_sintetico_canales/, pipeline.log
#   2) validar_salida.py    -> validacion.csv, validacion.log
#   3) graficas_comparacion -> figs/, figuras.log
#   4) listar_corridas.py   -> results/corridas.csv
# Se detiene en el primer paso que falle (set -e). NO sobrescribe: aborta si la carpeta ya tiene contenido.
# Todo lo que imprime tambien queda en <id>/corrida.log. Parametros usados: <id>/comando.txt.
#
# Uso (desde cualquier carpeta):
#   experimentos/correr_corrida.sh --etiqueta rep                       # semilla 0, 2000 epocas, lr 1e-4, paso 5
#   experimentos/correr_corrida.sh --etiqueta s1 --semilla 1 --dry-run  # solo muestra lo que haria
#   nohup experimentos/correr_corrida.sh --etiqueta rep > /dev/null 2>&1 &   # en segundo plano (~23 min)
set -euo pipefail

PY="$HOME/miniconda3/envs/py36_ff/bin/python"
RAIZ="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

SEMILLA=0; EPOCAS=2000; LR=1e-4; PASO=5; WORKERS=6; NMUESTRAS=19953; ETIQUETA=""; DRY=0

while [ $# -gt 0 ]; do
  case "$1" in
    --semilla)    SEMILLA="$2"; shift 2;;
    --epocas)     EPOCAS="$2"; shift 2;;
    --lr)         LR="$2"; shift 2;;
    --paso)       PASO="$2"; shift 2;;
    --workers)    WORKERS="$2"; shift 2;;
    --n_muestras) NMUESTRAS="$2"; shift 2;;
    --etiqueta)   ETIQUETA="$2"; shift 2;;
    --dry-run)    DRY=1; shift;;
    -h|--help)    sed -n '2,19p' "$0"; exit 0;;
    *) echo "Argumento desconocido: $1 (ver --help)" >&2; exit 2;;
  esac
done

[ -n "$ETIQUETA" ] || { echo "Falta --etiqueta (p. ej. orig, rep, s1)." >&2; exit 2; }
case "$ETIQUETA" in *[!A-Za-z0-9_-]*) echo "La etiqueta solo admite letras, numeros, _ y -." >&2; exit 2;; esac

ID="s${SEMILLA}_e${EPOCAS}_lr${LR}_p${PASO}_${ETIQUETA}"
R="results/runs/$ID"
cd "$RAIZ"

if [ -d "$R" ] && [ -n "$(ls -A "$R")" ]; then
  echo "ABORTO: $R ya tiene contenido (no se sobrescribe). Usa otra --etiqueta." >&2
  exit 1
fi

CMD_PIPE=("$PY" -u experimentos/pmu_pipeline.py --semilla "$SEMILLA" --epocas "$EPOCAS" --lr "$LR"
          --paso "$PASO" --workers "$WORKERS" --n_muestras "$NMUESTRAS" --salida "$R/pmu_sintetico.npz")
CMD_VAL=("$PY" experimentos/validar_salida.py --npz "$R/pmu_sintetico.npz" --csv data/PMU_AlexyDonald2.csv)
CMD_FIG=("$PY" experimentos/graficas_comparacion.py --npz "$R/pmu_sintetico.npz" --validacion "$R/validacion.csv" --salida "$R/figs")
CMD_TAB=("$PY" experimentos/listar_corridas.py)

if [ "$DRY" -eq 1 ]; then
  echo "[dry-run] carpeta: $R"
  for c in "CMD_PIPE" "CMD_VAL" "CMD_FIG" "CMD_TAB"; do
    declare -n ref="$c"; echo "  ${ref[*]}"; unset -n ref
  done
  exit 0
fi

mkdir -p "$R"
exec > >(tee -a "$R/corrida.log") 2>&1
{ date '+%F %T'; for c in CMD_PIPE CMD_VAL CMD_FIG CMD_TAB; do declare -n ref="$c"; echo "${ref[*]}"; unset -n ref; done; } > "$R/comando.txt"

T0=$(date +%s)
echo "== [$(date +%T)] Corrida $ID"
echo "== 1/4 pipeline (entrenamiento, ~23 min con 6 workers)"
"${CMD_PIPE[@]}" > "$R/pipeline.log" 2>&1 || { echo "FALLO el pipeline; ver $R/pipeline.log" >&2; tail -5 "$R/pipeline.log" >&2; exit 1; }
tail -2 "$R/pipeline.log"
echo "== 2/4 validacion"
"${CMD_VAL[@]}" | tee "$R/validacion.log" || { echo "FALLO la validacion (codigo != 0: NaN o forma incorrecta); no se grafica." >&2; exit 1; }
echo "== 3/4 figuras"
"${CMD_FIG[@]}" > "$R/figuras.log" 2>&1 || { echo "FALLO las figuras; ver $R/figuras.log" >&2; exit 1; }
tail -3 "$R/figuras.log"
echo "== 4/4 tabla de corridas"
"${CMD_TAB[@]}"
echo "== Listo en $(( $(date +%s) - T0 )) s: $R"
