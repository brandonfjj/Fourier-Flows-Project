# -*- coding: utf-8 -*-
"""Figura: BUS10Va completo con el hundimiento (falla) y la sobretensión marcados."""
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANAL = "BUS10Va"
FALLA = (2001, 4082)     # muestras
SOBRE = (10800, 13200)   # muestras
BLOQUE = 400             # ~1 ciclo

df = pd.read_csv(str(ROOT / "data" / "PMU_AlexyDonald2.csv"))
df.columns = df.columns.str.strip()
t = df["Time"].values.astype(float)
v = df[CANAL].values.astype(float)

nb = len(v) // BLOQUE
picos = np.abs(v[:nb * BLOQUE]).reshape(nb, BLOQUE).max(axis=1)
inicios = np.arange(nb) * BLOQUE
centros = inicios + BLOQUE // 2
med = np.median(picos)
rel = (picos / med - 1.0) * 100.0


def solapa(ini, fin):
    return (inicios < fin) & (inicios + BLOQUE > ini)


min_falla = rel[solapa(*FALLA)].min()
max_sobre = rel[solapa(*SOBRE)].max()
print("Mediana del pico por bloque: %.0f" % med)
print("Hundimiento: pico mínimo %.1f %% respecto a la mediana" % min_falla)
print("Sobretensión: pico máximo %+.1f %% respecto a la mediana" % max_sobre)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})

ax1.plot(t, v, lw=0.4, color="tab:blue")
ax1.set_ylabel("BUS10Va (unidades del archivo)")
ax1.set_title("BUS10Va: hundimiento de tensión y sobretensión")

ax2.plot(t[centros], rel, "o-", ms=3, lw=0.8, color="tab:blue")
ax2.axhline(0, color="gray", lw=0.8)
ax2.set_ylabel("Pico por ciclo\nrespecto a la mediana (%)")
ax2.set_xlabel("Tiempo (s)")

for ax in (ax1, ax2):
    ax.axvspan(t[FALLA[0]], t[FALLA[1]], color="tab:red", alpha=0.15,
               label="Hundimiento (hasta %.0f %%)" % min_falla if ax is ax1 else None)
    ax.axvspan(t[SOBRE[0]], t[SOBRE[1]], color="tab:orange", alpha=0.20,
               label="Sobretensión (hasta %+.1f %%)" % max_sobre if ax is ax1 else None)
    ax.grid(alpha=0.3)

ax1.legend(loc="upper right", fontsize=9)
fig.tight_layout()
salida = ROOT / "results" / "exploratorios" / "07_datos" / "fig_eventos_BUS10Va.png"
fig.savefig(str(salida), dpi=200)
print("Guardada:", salida)