import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

# Análisis de las 3 fases sobre la salida de pmu_3fases.py (no entrena nada).
# Las ventanas ya están en unidades físicas (pmu_3fases.py desnormaliza antes de guardar).
d = np.load(str(ROOT / "results" / "exploratorios" / "04_tres_fases" / "pmu_3fases_out.npz"))
CANALES = ["BUS10Va", "BUS10Vb", "BUS10Vc"]
real = {c: d["real_" + c].astype(np.float64) for c in CANALES}
synth = {c: d["synth_" + c].astype(np.float64) for c in CANALES}
print("Ventanas: real %d, sint %d, longitud %d"
      % (len(real["BUS10Va"]), len(synth["BUS10Va"]), real["BUS10Va"].shape[1]))

# Real: la fila k de Va, Vb y Vc es el mismo instante (mismos starts), el desfase es físico.
# Sintético: la fila k de cada canal es una muestra INDEPENDIENTE (un flujo por canal).
# Emparejar synth_Va[k] con synth_Vb[k] es arbitrario: el desfase sintético solo mide si
# cada flujo aprendió por separado su fase respecto al cruce por cero de Va.

def f1(w): return np.fft.rfft(w, axis=1)[:, 1]  # fasor: bin 1 de cada ventana

def envolver(deg, centro):
    # lleva cada ángulo al intervalo centro +- 180 grados
    return centro + np.degrees(np.angle(np.exp(1j * np.radians(deg - centro))))

def mediana_circular(deg):
    # mediana sin el artefacto de +-180: primero se envuelve alrededor de la media circular
    c = np.degrees(np.angle(np.exp(1j * np.radians(deg)).mean()))
    return np.median(envolver(deg, c))

Fr = {c: f1(real[c]) for c in CANALES}
Fs = {c: f1(synth[c]) for c in CANALES}

# 1) Desfase de Vb y Vc respecto a Va, envuelto alrededor de la mediana real
print("\n--- 1) Desfase respecto a Va (grados), envuelto alrededor de la mediana real ---")
for otra in ["BUS10Vb", "BUS10Vc"]:
    dr = np.degrees(np.angle(Fr[otra] * np.conj(Fr["BUS10Va"])))
    ds = np.degrees(np.angle(Fs[otra] * np.conj(Fs["BUS10Va"])))  # pareja arbitraria
    m = mediana_circular(dr)
    dr, ds = envolver(dr, m), envolver(ds, m)
    print("%s - Va: mediana real %.2f | real %.2f +- %.2f | sint %.2f +- %.2f"
          % (otra, m, dr.mean(), dr.std(), ds.mean(), ds.std()))

# 5) Fase absoluta del bin 1 por canal, envuelta alrededor de la mediana real de ese canal
print("\n--- 5) Fase absoluta del bin 1 por canal (grados) ---")
for c in CANALES:
    pr = np.degrees(np.angle(Fr[c]))
    ps = np.degrees(np.angle(Fs[c]))
    m = mediana_circular(pr)
    pr, ps = envolver(pr, m), envolver(ps, m)
    print("%s: mediana real %.2f | real %.2f +- %.2f | sint %.2f +- %.2f"
          % (c, m, pr.mean(), pr.std(), ps.mean(), ps.std()))

# 2) Componentes simétricas por ventana
a = np.exp(1j * 2 * np.pi / 3)

def simetricas(F):
    Va, Vb, Vc = F["BUS10Va"], F["BUS10Vb"], F["BUS10Vc"]
    V0 = (Va + Vb + Vc) / 3
    V1 = (Va + a * Vb + a ** 2 * Vc) / 3
    V2 = (Va + a ** 2 * Vb + a * Vc) / 3
    return V0, V1, V2

print("\n--- 2) Componentes simétricas (cocientes por ventana, media +- desviación) ---")
for nombre, F in [("real", Fr), ("sint", Fs)]:
    V0, V1, V2 = simetricas(F)
    q0, q2 = np.abs(V0) / np.abs(V1), np.abs(V2) / np.abs(V1)
    print("%s: |V0|/|V1| = %.4f +- %.4f | |V2|/|V1| = %.4f +- %.4f"
          % (nombre, q0.mean(), q0.std(), q2.mean(), q2.std()))

# Comprobación cruzada en el real: 3|V0|/|Va| = |Va+Vb+Vc|/|Va| (del bin 1)
V0r = simetricas(Fr)[0]
x = 3 * np.abs(V0r) / np.abs(Fr["BUS10Va"])
print("\nComprobación real: 3|V0|/|Va| = %.4f +- %.4f (esperado ~0.73)" % (x.mean(), x.std()))
