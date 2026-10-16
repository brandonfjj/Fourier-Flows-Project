# -*- coding: utf-8 -*-
"""Revisión de la prueba a 48 muestras: rango, ventanas de ejemplo y PCA.
Uso: python experimentos/fig_ff48.py [CANAL]"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

canal = sys.argv[1] if len(sys.argv) > 1 else "BUS10Va"
d = np.load(str(ROOT / "results" / "exploratorios" / "06_ff48_prueba" / ("ff48_%s_out.npz" % canal)))
print("Claves en el .npz:", d.files)
kr = [k for k in d.files if "real" in k.lower()]
ks = [k for k in d.files if any(s in k.lower() for s in ("sint", "synth", "fake", "gen"))]
if not kr or not ks:
    print("No identifiqué las claves; pégame la línea de arriba.")
    sys.exit(1)
real = np.asarray(d[kr[0]], dtype=float)[:, :48]
sint = np.asarray(d[ks[0]], dtype=float)[:, :48]
print("Ventanas: real", real.shape, "| sintético", sint.shape)

fuera = (sint < 0) | (sint > 1)
print("Valores sintéticos fuera de [0,1]: %.2f %%" % (100 * fuera.mean()))
print("Ventanas con algún valor fuera de [0,1]: %.2f %%" % (100 * fuera.any(axis=1).mean()))
pc = [0.1, 1, 99, 99.9]
print("Percentiles %s  real: %s" % (pc, np.round(np.percentile(real, pc), 3)))
print("Percentiles %s  sint: %s" % (pc, np.round(np.percentile(sint, pc), 3)))
rug = np.abs(np.diff(sint, axis=1)).mean() / np.abs(np.diff(real, axis=1)).mean()
print("Rugosidad sintético/real: %.2f" % rug)

mu = real.mean(0)
U, S, Vt = np.linalg.svd(real - mu, full_matrices=False)
P = Vt[:2].T
pr, ps = (real - mu).dot(P), (sint - mu).dot(P)
var = S ** 2 / np.sum(S ** 2)
print("Varianza explicada por las 2 primeras componentes: %.1f %%" % (100 * var[:2].sum()))

rng = np.random.RandomState(0)
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.6))
a1.scatter(pr[:, 0], pr[:, 1], s=6, alpha=0.35, color="tab:red", label="Real")
a1.scatter(ps[:, 0], ps[:, 1], s=6, alpha=0.35, color="tab:blue", label="Sintético")
a1.set_title("PCA ajustado sobre las ventanas reales")
a1.set_xlabel("Componente 1")
a1.set_ylabel("Componente 2")
a1.legend()
a1.grid(alpha=0.3)

for i in rng.choice(len(real), 5, replace=False):
    a2.plot(real[i], color="tab:red", lw=1, alpha=0.8)
for i in rng.choice(len(sint), 5, replace=False):
    a2.plot(sint[i], color="tab:blue", lw=1, alpha=0.8)
a2.axhline(0, color="gray", ls="--", lw=0.8)
a2.axhline(1, color="gray", ls="--", lw=0.8)
a2.plot([], [], color="tab:red", label="Real")
a2.plot([], [], color="tab:blue", label="Sintético")
a2.set_title("Cinco ventanas al azar de cada tipo")
a2.set_xlabel("Muestra dentro de la ventana")
a2.set_ylabel("Valor normalizado")
a2.legend()
a2.grid(alpha=0.3)

fig.suptitle("%s: Fourier Flows con ventanas de 48 muestras" % canal)
fig.tight_layout()
salida = ROOT / "results" / "exploratorios" / "06_ff48_prueba" / ("fig_ff48_%s.png" % canal)
fig.savefig(str(salida), dpi=200)
print("Guardada:", salida)