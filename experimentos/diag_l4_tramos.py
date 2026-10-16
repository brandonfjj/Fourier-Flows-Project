import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

# Repite la Parte 2 de diag_canales.py solo para L4V_FLT y L4I_FLT, con dos tramos:
# 4600-10400 (original) y 5700-10400 (sin el final del evento de la falla, que llega a 5600).
# No entrena ni escribe archivos.

df = pd.read_csv(str(ROOT / "data" / "PMU_AlexyDonald2.csv"))
df.columns = df.columns.str.strip()

N = 401
FIN = 10400
TRAMOS = [4600, 5700]
GRUPOS = [
    ("L4V_FLT", "VOLTAJE", ["L4Va_FLT", "L4Vb_FLT", "L4Vc_FLT"]),
    ("L4I_FLT", "CORRIENTE", ["L4Ia_FLT", "L4Ib_FLT", "L4Ic_FLT"]),
]
a_op = np.exp(1j * 2 * np.pi / 3)

def envolver(deg, centro):
    return centro + np.degrees(np.angle(np.exp(1j * np.radians(deg - centro))))

def mediana_circular(deg):
    c = np.degrees(np.angle(np.exp(1j * np.radians(deg)).mean()))
    return np.median(envolver(deg, c))

def metricas(fases, ini):
    starts = list(range(ini, FIN - N + 1, N))
    W = {c: np.array([df[c].values[s:s + N].astype(np.float64) for s in starts]) for c in fases}
    F = {c: np.fft.rfft(W[c], axis=1)[:, 1] for c in fases}
    out = {"n": len(starts), "ini": starts[0], "fin": starts[-1] + N}
    for c in fases:
        t = np.sqrt((W[c] ** 2).mean(axis=1))
        f = np.sqrt(2) * np.abs(F[c]) / N
        out["rms_t_" + c] = (t.mean(), t.std())
        out["rms_f_" + c] = (f.mean(), f.std())
    for c in fases[1:]:
        d = np.degrees(np.angle(F[c] * np.conj(F[fases[0]])))
        d = envolver(d, mediana_circular(d))
        out["fase_" + c] = (d.mean(), d.std())
    V0 = (F[fases[0]] + F[fases[1]] + F[fases[2]]) / 3
    V1 = (F[fases[0]] + a_op * F[fases[1]] + a_op ** 2 * F[fases[2]]) / 3
    V2 = (F[fases[0]] + a_op ** 2 * F[fases[1]] + a_op * F[fases[2]]) / 3
    out["q0"] = ((np.abs(V0) / np.abs(V1)).mean(), (np.abs(V0) / np.abs(V1)).std())
    out["q2"] = ((np.abs(V2) / np.abs(V1)).mean(), (np.abs(V2) / np.abs(V1)).std())
    return out

for nombre, tipo, fases in GRUPOS:
    A, B = [metricas(fases, ini) for ini in TRAMOS]
    print("\n--- %s [%s] ---" % (nombre, tipo))
    print("Tramo A: %d-%d (%d ventanas) | Tramo B: %d-%d (%d ventanas)"
          % (A["ini"], A["fin"], A["n"], B["ini"], B["fin"], B["n"]))
    print("%-34s | %-26s | %-26s" % ("métrica (media +- desv. entre ventanas)", "A: 4600-10400", "B: 5700-10400"))
    filas = []
    for c in fases:
        filas.append(("RMS temporal " + c, "rms_t_" + c, "%.5g +- %.3g"))
        filas.append(("RMS fundamental " + c, "rms_f_" + c, "%.5g +- %.3g"))
    for c in fases[1:]:
        filas.append(("Desfase %s - %s (grados)" % (c, fases[0]), "fase_" + c, "%.3f +- %.3f"))
    filas.append(("|V0|/|V1|", "q0", "%.4f +- %.4f"))
    filas.append(("|V2|/|V1|", "q2", "%.4f +- %.4f"))
    for et, k, fmt in filas:
        print("%-34s | %-26s | %-26s" % (et, fmt % A[k], fmt % B[k]))
