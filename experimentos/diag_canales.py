import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# Diagnóstico de los 20 canales de señal (no entrena, no escribe archivos).
# Parte 1: picos por bloque de 400 muestras, como diag_picos.py.
# Parte 2: fasores (bin 1, ventanas de 401) de los grupos trifásicos en el tramo 4600-10400.

df = pd.read_csv(str(ROOT / "data" / "PMU_AlexyDonald2.csv"))
df.columns = df.columns.str.strip()
CANALES = [c for c in df.columns if c not in ("Time", "EstadoEtiqueta1", "Etiqueta")]
print("Canales de señal: %d" % len(CANALES))

BLOQ = 400
UMBRAL = 0.03   # igual que diag_picos.py
UMBRAL2 = 0.10

# ---------------- Parte 1 ----------------
def agrupar(idx):
    # agrupa índices de bloque consecutivos en (primero, último)
    grupos = []
    for k in idx:
        if grupos and k == grupos[-1][1] + 1:
            grupos[-1][1] = k
        else:
            grupos.append([k, k])
    return grupos

eventos = {}  # canal -> lista de (muestra_ini, muestra_fin) con desvío > 3 %
print("\n=== Parte 1: pico de |x| por bloque de %d muestras ===" % BLOQ)
print("%-10s | %12s | %8s | %8s | rangos de muestras (bloques con desvío > %.0f %%; entre paréntesis: desvío extremo)"
      % ("canal", "mediana pico", "n>3%", "n>10%", 100 * UMBRAL))
for c in CANALES:
    x = df[c].values.astype(np.float64)
    picos = np.array([np.abs(x[a:a + BLOQ]).max() for a in range(0, len(x) - BLOQ, BLOQ)])
    med = np.median(picos)
    if med == 0:
        print("%-10s | %12.4g | mediana 0, desvío relativo no definido" % (c, med))
        eventos[c] = []
        continue
    dev = (picos - med) / med
    idx3 = [k for k in range(len(picos)) if abs(dev[k]) > UMBRAL]
    n10 = int((np.abs(dev) > UMBRAL2).sum())
    rangos = []
    eventos[c] = []
    for a, b in agrupar(idx3):
        ext = dev[a:b + 1][np.argmax(np.abs(dev[a:b + 1]))]
        rangos.append("%d-%d (%+.1f%%)" % (a * BLOQ, (b + 1) * BLOQ, 100 * ext))
        eventos[c].append((a * BLOQ, (b + 1) * BLOQ))
    print("%-10s | %12.5g | %8d | %8d | %s"
          % (c, med, len(idx3), n10, "; ".join(rangos) if rangos else "-"))

# ---------------- Parte 2 ----------------
INI, FIN, N = 4600, 10400, 401
starts = list(range(INI, FIN - N + 1, N))
print("\n=== Parte 2: ventanas de %d muestras, paso %d, tramo %d-%d: %d ventanas (inicios %d ... %d) ==="
      % (N, N, INI, FIN, len(starts), starts[0], starts[-1]))

GRUPOS = [  # (nombre, tipo, [fase a, fase b, fase c])
    ("BUS10", "VOLTAJE", ["BUS10Va", "BUS10Vb", "BUS10Vc"]),
    ("BUS14", "VOLTAJE", ["BUS14Va", "BUS14Vb", "BUS14Vc"]),
    ("L2V_FLT", "VOLTAJE", ["L2Va_FLT", "L2Vb_FLT", "L2Vc_FLT"]),
    ("L4V_FLT", "VOLTAJE", ["L4Va_FLT", "L4Vb_FLT", "L4Vc_FLT"]),
    ("L2I_FLT", "CORRIENTE", ["L2Ia_FLT", "L2Ib_FLT", "L2Ic_FLT"]),
    ("L4I_FLT", "CORRIENTE", ["L4Ia_FLT", "L4Ib_FLT", "L4Ic_FLT"]),
]

def envolver(deg, centro):
    return centro + np.degrees(np.angle(np.exp(1j * np.radians(deg - centro))))

def mediana_circular(deg):
    c = np.degrees(np.angle(np.exp(1j * np.radians(deg)).mean()))
    return np.median(envolver(deg, c))

a_op = np.exp(1j * 2 * np.pi / 3)

for nombre, tipo, fases in GRUPOS:
    print("\n--- %s [%s] %s ---" % (nombre, tipo, ", ".join(fases)))
    # aviso de eventos de la Parte 1 que tocan el tramo
    avisos = []
    for c in fases:
        for (a, b) in eventos[c]:
            if a < FIN and b > INI:
                avisos.append("%s: bloques %d-%d" % (c, a, b))
    if avisos:
        print("AVISO: la Parte 1 marca eventos (>3 %%) que tocan el tramo %d-%d -> %s"
              % (INI, FIN, "; ".join(avisos)))
    else:
        print("Parte 1: sin bloques marcados (>3 %%) dentro de %d-%d en estas fases." % (INI, FIN))

    W = {c: np.array([df[c].values[s:s + N].astype(np.float64) for s in starts]) for c in fases}
    F = {c: np.fft.rfft(W[c], axis=1)[:, 1] for c in fases}

    for c in fases:
        rms_t = np.sqrt((W[c] ** 2).mean(axis=1))
        rms_f = np.sqrt(2) * np.abs(F[c]) / N
        print("RMS %-9s temporal %.5g +- %.3g | fundamental %.5g +- %.3g"
              % (c, rms_t.mean(), rms_t.std(), rms_f.mean(), rms_f.std()))

    Fa = F[fases[0]]
    for c, et in [(fases[1], "b"), (fases[2], "c")]:
        d = np.degrees(np.angle(F[c] * np.conj(Fa)))
        d = envolver(d, mediana_circular(d))
        print("Desfase %s - %s: %.3f +- %.3f grados (min %.3f, max %.3f)"
              % (c, fases[0], d.mean(), d.std(), d.min(), d.max()))

    V0 = (F[fases[0]] + F[fases[1]] + F[fases[2]]) / 3
    V1 = (F[fases[0]] + a_op * F[fases[1]] + a_op ** 2 * F[fases[2]]) / 3
    V2 = (F[fases[0]] + a_op ** 2 * F[fases[1]] + a_op * F[fases[2]]) / 3
    q0, q2 = np.abs(V0) / np.abs(V1), np.abs(V2) / np.abs(V1)
    print("|V0|/|V1| = %.4f +- %.4f | |V2|/|V1| = %.4f +- %.4f"
          % (q0.mean(), q0.std(), q2.mean(), q2.std()))
