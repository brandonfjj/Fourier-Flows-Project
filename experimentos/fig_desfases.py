# -*- coding: utf-8 -*-
"""Figura: desfases Vb-Va y Vc-Va (real, sintético y control barajado). Lee results/exploratorios/04_tres_fases/pmu_3fases_out.npz."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUTA = ROOT / "results" / "exploratorios" / "04_tres_fases" / "pmu_3fases_out.npz"
CANALES = ["BUS10Va", "BUS10Vb", "BUS10Vc"]
N_SEMILLAS = 20
SIN = {"real": ["real", "orig"],
       "sint": ["sint", "synth", "fake", "gen", "muestra"]}

d = np.load(str(RUTA), allow_pickle=True)
print("Claves en el .npz:")
for k in d.files:
    print("   %-30s %s" % (k, d[k].shape))


def clave_canal(tipo, canal):
    for k in d.files:
        kl = k.lower()
        if canal.lower() in kl and any(s in kl for s in SIN[tipo]):
            return k
    return None


def clave_apilada(tipo):
    for k in d.files:
        kl = k.lower()
        a = d[k]
        if any(s in kl for s in SIN[tipo]) and a.ndim == 3 and 3 in a.shape[:2]:
            return k
    return None


def cargar(tipo):
    claves = [clave_canal(tipo, c) for c in CANALES]
    if all(k is not None for k in claves):
        return [np.asarray(d[k], dtype=float).reshape(d[k].shape[0], -1) for k in claves]
    k = clave_apilada(tipo)
    if k is not None:
        a = np.asarray(d[k], dtype=float)
        if a.shape[0] != 3:
            a = np.transpose(a, (1, 0, 2))
        return [a[0], a[1], a[2]]
    return None


real = cargar("real")
sint = cargar("sint")
if real is None or sint is None:
    print("\nNo pude identificar las ventanas reales/sintéticas por canal.")
    print("Pégame la lista de claves de arriba y ajusto el script.")
    sys.exit(1)


def fase_deg(w):
    return np.degrees(np.angle(np.fft.fft(w, axis=1)[:, 1]))


def centrar(dif, ref):
    return (dif - ref + 180.0) % 360.0 - 180.0 + ref


fr = [fase_deg(w) for w in real]
fs = [fase_deg(w) for w in sint]
rng = np.random.RandomState(0)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
print("\n%-10s %-22s %-22s %-22s" % ("", "Real", "Sintético", "Control barajado"))
for ax, nombre, j in zip(axes, ["Vb - Va", "Vc - Va"], [1, 2]):
    dr = fr[j] - fr[0]
    ref = np.degrees(np.angle(np.mean(np.exp(1j * np.radians(dr)))))
    dr = centrar(dr, ref)
    ds = centrar(fs[j] - fs[0], ref)
    dc = np.concatenate([centrar(fr[j][rng.permutation(len(fr[j]))] - fr[0], ref)
                         for _ in range(N_SEMILLAS)])

    print("%-10s %8.2f +- %-10.2f %8.2f +- %-10.2f %8.2f +- %-10.2f"
          % (nombre, dr.mean(), dr.std(), ds.mean(), ds.std(), dc.mean(), dc.std()))

    bins = np.arange(ref - 60, ref + 60.01, 1.5)
    ax.hist(dc, bins=bins, density=True, alpha=0.40, color="tab:gray",
            label="Control barajado: %.1f$^\\circ$ $\\pm$ %.1f$^\\circ$" % (dc.mean(), dc.std()))
    ax.hist(ds, bins=bins, density=True, alpha=0.55, color="tab:blue",
            label="Sintético: %.1f$^\\circ$ $\\pm$ %.1f$^\\circ$" % (ds.mean(), ds.std()))
    ax.axvline(dr.mean(), color="tab:red", lw=2,
               label="Real: %.2f$^\\circ$ $\\pm$ %.2f$^\\circ$" % (dr.mean(), dr.std()))
    ax.set_title("Desfase %s" % nombre)
    ax.set_xlabel("Desfase (grados)")
    ax.set_ylabel("Densidad")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

fig.suptitle("Relación de fase entre canales: un flujo por canal frente al real y al control")
fig.tight_layout()
salida = ROOT / "results" / "exploratorios" / "04_tres_fases" / "fig_desfases_3fases.png"
fig.savefig(str(salida), dpi=200)
print("\nGuardada:", salida)