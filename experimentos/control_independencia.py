import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

# Control de independencia (no entrena nada): con las ventanas REALES se deja Va igual y se
# mezclan las filas de Vb y Vc con permutaciones independientes. Así se rompe la relación
# conjunta entre canales pero se conserva la distribución de cada canal, igual que pasa con
# el sintético de pmu_3fases.py (un flujo por canal, filas independientes).
d = np.load(str(ROOT / "results" / "exploratorios" / "04_tres_fases" / "pmu_3fases_out.npz"))
CANALES = ["BUS10Va", "BUS10Vb", "BUS10Vc"]
real = {c: d["real_" + c].astype(np.float64) for c in CANALES}
synth = {c: d["synth_" + c].astype(np.float64) for c in CANALES}
N = len(real["BUS10Va"])
SEMILLAS = range(20)

# --- Funciones copiadas de analisis_3fases.py (no se importa: ese script corre al importarse) ---
def f1(w): return np.fft.rfft(w, axis=1)[:, 1]  # fasor: bin 1 de cada ventana

def envolver(deg, centro):
    # lleva cada ángulo al intervalo centro +- 180 grados
    return centro + np.degrees(np.angle(np.exp(1j * np.radians(deg - centro))))

def mediana_circular(deg):
    # mediana sin el artefacto de +-180: primero se envuelve alrededor de la media circular
    c = np.degrees(np.angle(np.exp(1j * np.radians(deg)).mean()))
    return np.median(envolver(deg, c))

a = np.exp(1j * 2 * np.pi / 3)

def simetricas(F):
    Va, Vb, Vc = F["BUS10Va"], F["BUS10Vb"], F["BUS10Vc"]
    V0 = (Va + Vb + Vc) / 3
    V1 = (Va + a * Vb + a ** 2 * Vc) / 3
    V2 = (Va + a ** 2 * Vb + a * Vc) / 3
    return V0, V1, V2

# Medianas de referencia: una sola vez, sobre el real sin mezclar
Fr = {c: f1(real[c]) for c in CANALES}
def desfase(F, otra): return np.degrees(np.angle(F[otra] * np.conj(F["BUS10Va"])))
MED_DES = {o: mediana_circular(desfase(Fr, o)) for o in ["BUS10Vb", "BUS10Vc"]}
MED_FASE = {c: mediana_circular(np.degrees(np.angle(Fr[c]))) for c in CANALES}

# Cada métrica: (media, desviación) sobre las ventanas
NOMBRES = ["Vb-Va", "Vc-Va", "fase Va", "fase Vb", "fase Vc", "|V0|/|V1|", "|V2|/|V1|", "3|V0|/|Va|"]

def metricas(F):
    vals = []
    for o in ["BUS10Vb", "BUS10Vc"]:
        vals.append(envolver(desfase(F, o), MED_DES[o]))
    for c in CANALES:
        vals.append(envolver(np.degrees(np.angle(F[c])), MED_FASE[c]))
    V0, V1, V2 = simetricas(F)
    vals += [np.abs(V0) / np.abs(V1), np.abs(V2) / np.abs(V1), 3 * np.abs(V0) / np.abs(F["BUS10Va"])]
    return {n: (v.mean(), v.std()) for n, v in zip(NOMBRES, vals)}

M_real = metricas(Fr)
M_sint = metricas({c: f1(synth[c]) for c in CANALES})

# --- Verificación contra la salida conocida de analisis_3fases.py ---
# (métrica, campo 0=media 1=desv, valor conocido, decimales con que se imprimió)
CONOCIDOS = [
    ("real", M_real, "Vb-Va", 0, -101.55, 2), ("real", M_real, "Vb-Va", 1, 0.05, 2),
    ("real", M_real, "Vc-Va", 0, 144.48, 2), ("real", M_real, "Vc-Va", 1, 0.02, 2),
    ("real", M_real, "|V0|/|V1|", 0, 0.1999, 4), ("real", M_real, "|V2|/|V1|", 0, 0.1050, 4),
    ("sint", M_sint, "Vb-Va", 0, -104.71, 2), ("sint", M_sint, "Vb-Va", 1, 10.43, 2),
    ("sint", M_sint, "Vc-Va", 0, 148.40, 2), ("sint", M_sint, "Vc-Va", 1, 13.62, 2),
]
print("--- Verificación contra analisis_3fases.py ---")
ok = True
for grupo, M, n, k, conocido, dec in CONOCIDOS:
    x = M[n][k]
    coincide = round(x, dec) == round(conocido, dec)
    ok &= coincide
    print("%s %-10s %-5s: calculado %.*f | conocido %.*f | %s"
          % (grupo, n, ["media", "desv"][k], dec, x, dec, conocido, "OK" if coincide else "NO COINCIDE"))
if not ok:
    sys.exit("Algún valor no coincide con analisis_3fases.py: se detiene antes del control.")

# --- Control: Va igual, filas de Vb y Vc mezcladas con permutaciones independientes ---
print("\n--- Control mezclado, %d semillas (por semilla: media y desv sobre %d ventanas) ---"
      % (len(SEMILLAS), N))
por_semilla = []
for s in SEMILLAS:
    rng = np.random.RandomState(s)
    pb, pc = rng.permutation(N), rng.permutation(N)
    Fc = {"BUS10Va": Fr["BUS10Va"], "BUS10Vb": Fr["BUS10Vb"][pb], "BUS10Vc": Fr["BUS10Vc"][pc]}
    M = metricas(Fc)
    por_semilla.append(M)
    print("semilla %2d: Vb-Va %.2f +- %.2f | Vc-Va %.2f +- %.2f | |V0|/|V1| %.4f +- %.4f | "
          "|V2|/|V1| %.4f +- %.4f | 3|V0|/|Va| %.4f +- %.4f"
          % (s, M["Vb-Va"][0], M["Vb-Va"][1], M["Vc-Va"][0], M["Vc-Va"][1],
             M["|V0|/|V1|"][0], M["|V0|/|V1|"][1], M["|V2|/|V1|"][0], M["|V2|/|V1|"][1],
             M["3|V0|/|Va|"][0], M["3|V0|/|Va|"][1]))

# Media y desviación ENTRE semillas, de la media y de la desviación de cada métrica
ctrl = {}
for n in NOMBRES:
    medias = np.array([M[n][0] for M in por_semilla])
    desvs = np.array([M[n][1] for M in por_semilla])
    ctrl[n] = (medias.mean(), medias.std(), desvs.mean(), desvs.std())

print("\n--- Control: media +- desviación entre semillas ---")
for n in NOMBRES:
    m, sm, dv, sd = ctrl[n]
    print("%-10s: media %.4f +- %.4f | desv %.4f +- %.4f" % (n, m, sm, dv, sd))

# --- Tabla final ---
print("\n--- Tabla: real | control (promedio de %d semillas) | sintético | sint - control ---"
      % len(SEMILLAS))
print("%-17s %12s %12s %12s %14s" % ("Métrica", "Real", "Control", "Sintético", "Sint - Ctrl"))
for n in NOMBRES:
    for k, campo in enumerate(["media", "desv"]):
        r, c, s = M_real[n][k], ctrl[n][2 * k], M_sint[n][k]
        print("%-17s %12.4f %12.4f %12.4f %14.4f" % (n + " " + campo, r, c, s, s - c))
