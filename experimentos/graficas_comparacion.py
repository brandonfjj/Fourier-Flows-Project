#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
graficas_comparacion.py -- Figuras de comparacion real vs Fourier Flows (y TimeGAN, opcional).

Genera PNG (dpi 200, fondo claro) en --salida (por defecto results/figs/):
  fig1_ventanas.png      ventanas de 48 muestras: real vs sintetico, por canal
  fig2_pca_canales.png   PCA por canal (ajustado sobre el real, el sintetico se proyecta)
  fig3_pca_conjunta.png  PCA de las ventanas de los 20 canales juntos (48x20 aplanado), con un
                         control: real con cada canal barajado por separado (aisla el efecto de
                         la independencia entre canales)
  fig4_espectro.png      espectro de potencia medio por canal (eje y logaritmico)
  fig5_histogramas.png   distribucion de valores normalizados (eje y logaritmico)
  fig6_metricas.png      metricas por canal leidas de results/validacion_s0.csv

Todo se normaliza con el min-max del canal sobre la serie completa (el mismo esquema que usa
el MinMaxScaler del TimeGAN, que opera por columna). Asi los tres conjuntos son comparables.

TimeGAN (opcional): --timegan archivo.npy con forma (N, 48, 20).
  - Se asume que los canales siguen el orden de columnas del CSV (columnas 1..20, como en
    data_loading.py del TimeGAN). Si no, el orden en las figuras sera incorrecto.
  - Por defecto se asume que esta en unidades fisicas; si ya viene en [0,1] usar
    --timegan_normalizado.
  - Esta ruta NO esta probada con datos reales del TimeGAN (aun no existen); solo con datos falsos.

Los PNG se derivan de datos confidenciales: no van a git (results/ esta ignorado).
Requiere solo numpy y matplotlib.
"""
import argparse
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CANALES = [
    "BUS10Va", "BUS10Vb", "BUS10Vc",
    "BUS14Va", "BUS14Vb", "BUS14Vc",
    "L2Ia_FLT", "L2Ib_FLT", "L2Ic_FLT", "L2In_FLT",
    "L2Va_FLT", "L2Vb_FLT", "L2Vc_FLT",
    "L4Ia_FLT", "L4Ib_FLT", "L4Ic_FLT", "L4In_FLT",
    "L4Va_FLT", "L4Vb_FLT", "L4Vc_FLT",
]
VENTANA = 48
FS_KHZ = 20.0  # 20 kHz

# Paleta validada (categorica, slots 1-3, modo claro) y tintas
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA2, APAGADO, REJILLA, EJE, SUPERFICIE = (
    "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb")


def estilo():
    plt.rcParams.update({
        "figure.facecolor": SUPERFICIE, "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE, "axes.edgecolor": EJE,
        "axes.labelcolor": TINTA2, "xtick.color": APAGADO, "ytick.color": APAGADO,
        "text.color": TINTA, "axes.titlecolor": TINTA, "font.size": 12,
        "axes.titlesize": 13, "axes.labelsize": 12, "legend.fontsize": 12,
        "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
    })


def leer_csv(ruta):
    with open(ruta, "r", newline="", encoding="utf-8") as f:
        primera = f.readline()
        delim = ";" if primera.count(";") > primera.count(",") else ","
        f.seek(0)
        lector = csv.reader(f, delimiter=delim)
        enc = [c.strip() for c in next(lector)]
        filas = [fila for fila in lector if fila]
    return enc, filas


def cargar_real(ruta):
    """Devuelve (datos_en_orden_CANALES, orden_columnas_csv_1a20)."""
    enc, filas = leer_csv(ruta)
    faltan = [c for c in CANALES if c not in enc]
    if faltan:
        raise ValueError("Columnas ausentes en el CSV: %s" % faltan)
    idx = [enc.index(c) for c in CANALES]
    datos = np.asarray([[float(f[i]) for i in idx] for f in filas], dtype=np.float64)
    return datos, enc[1:21]  # el TimeGAN usa las columnas 1..20 (sin 0, 21, 22)


def normalizar(x, mn, mx):
    return (x - mn) / (mx - mn)


def muestrear_ventanas(norm, n, rs):
    """n ventanas reales de 48 muestras (paso 1, inicios al azar). norm: (T, C) -> (n, 48, C)."""
    inicios = rs.randint(0, norm.shape[0] - VENTANA + 1, size=n)
    return np.stack([norm[i:i + VENTANA] for i in inicios])


def submuestra(w, n, rs):
    if len(w) <= n:
        return w
    return w[rs.choice(len(w), size=n, replace=False)]


def pca_ajustar(X, k=2):
    mu = X.mean(axis=0)
    _, S, Vt = np.linalg.svd(X - mu, full_matrices=False)
    var = S ** 2
    return mu, Vt[:k], var[:k] / var.sum()


def proyectar(X, mu, comps):
    return (X - mu).dot(comps.T)


def leyenda(fig, series):
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markeredgecolor=c,
                markersize=9, label=n) for n, c, _ in series]
    fig.legend(handles=h, loc="upper center", ncol=len(series) if len(series) <= 3 else 2,
               frameon=False)


def guardar(fig, ruta):
    fig.savefig(ruta, dpi=200)
    plt.close(fig)
    print("Guardado:", ruta)


def fig_ventanas(series, canales, out, rs, n_ver=6):
    nr, nc = len(canales), len(series)
    fig, axs = plt.subplots(nr, nc, figsize=(4.4 * nc, 2.4 * nr + 0.6), sharex=True, squeeze=False)
    t = np.arange(VENTANA) / FS_KHZ  # ms
    for i, (cj, cn) in enumerate(canales):
        for j, (nombre, color, w) in enumerate(series):
            ax = axs[i][j]
            sel = w[rs.choice(len(w), size=n_ver, replace=False), :, cj]
            for v in sel:
                ax.plot(t, v, color=color, lw=1.2, alpha=0.9)
            ax.axhline(0, color=APAGADO, lw=0.8, ls="--")
            ax.axhline(1, color=APAGADO, lw=0.8, ls="--")
            ax.set_ylim(-0.4, 1.4)
            if i == 0:
                ax.set_title(nombre)
            if j == 0:
                ax.set_ylabel(cn + "\n(normalizado)")
            if i == nr - 1:
                ax.set_xlabel("tiempo dentro de la ventana (ms)")
    fig.suptitle("Ventanas de 48 muestras (líneas punteadas: mínimo y máximo reales del canal)",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    guardar(fig, os.path.join(out, "fig1_ventanas.png"))


def fig_pca_canales(series, canales, out):
    n = len(canales)
    nc = 2
    nr = (n + 1) // 2
    fig, axs = plt.subplots(nr, nc, figsize=(6.2 * nc, 4.8 * nr + 0.6), squeeze=False)
    for k, (cj, cn) in enumerate(canales):
        ax = axs[k // nc][k % nc]
        real = series[0][2][:, :, cj]
        mu, comps, ratio = pca_ajustar(real)
        for nombre, color, w in reversed(series):  # el real queda encima
            p = proyectar(w[:, :, cj], mu, comps)
            ax.scatter(p[:, 0], p[:, 1], s=5, color=color, alpha=0.35 if nombre != series[0][0] else 0.6,
                       linewidths=0)
        ax.set_title("%s  (PC1+PC2 del real: %.1f %%)" % (cn, 100 * ratio.sum()))
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    for k in range(n, nr * nc):
        axs[k // nc][k % nc].axis("off")
    leyenda(fig, series)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    guardar(fig, os.path.join(out, "fig2_pca_canales.png"))


def fig_pca_conjunta(series, out):
    X = [w.reshape(len(w), -1) for _, _, w in series]
    mu, comps, ratio = pca_ajustar(X[0])
    fig, ax = plt.subplots(figsize=(7.5, 6.2))
    for (nombre, color, _), x in reversed(list(zip(series, X))):
        p = proyectar(x, mu, comps)
        ax.scatter(p[:, 0], p[:, 1], s=5, color=color, alpha=0.35 if nombre != series[0][0] else 0.6,
                   linewidths=0)
    ax.set_title("PCA conjunta de los 20 canales (ventanas 48×20 aplanadas)\n"
                 "PC1+PC2 del real: %.1f %%" % (100 * ratio.sum()))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    leyenda(fig, series)
    fig.tight_layout(rect=(0, 0, 1, 0.86))
    guardar(fig, os.path.join(out, "fig3_pca_conjunta.png"))


def fig_espectro(series, canales, out):
    n = len(canales)
    nc = 2
    nr = (n + 1) // 2
    fig, axs = plt.subplots(nr, nc, figsize=(6.2 * nc, 4.2 * nr + 0.6), squeeze=False)
    frec = np.fft.rfftfreq(VENTANA, d=1.0 / FS_KHZ)  # kHz
    for k, (cj, cn) in enumerate(canales):
        ax = axs[k // nc][k % nc]
        for nombre, color, w in series:
            p = (np.abs(np.fft.rfft(w[:, :, cj], axis=1)) ** 2).mean(axis=0)
            ax.semilogy(frec, p + 1e-12, color=color, lw=2, marker="o", ms=4)
        ax.set_title(cn)
        ax.set_xlabel("frecuencia (kHz)")
        ax.set_ylabel("potencia media (normalizada)")
    for k in range(n, nr * nc):
        axs[k // nc][k % nc].axis("off")
    leyenda(fig, series)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    guardar(fig, os.path.join(out, "fig4_espectro.png"))


def fig_hist(series, canales, out):
    n = len(canales)
    nc = 2
    nr = (n + 1) // 2
    fig, axs = plt.subplots(nr, nc, figsize=(6.2 * nc, 4.2 * nr + 0.6), squeeze=False)
    bins = np.linspace(-0.5, 1.5, 101)
    for k, (cj, cn) in enumerate(canales):
        ax = axs[k // nc][k % nc]
        for nombre, color, w in series:
            ax.hist(w[:, :, cj].ravel(), bins=bins, density=True, histtype="step",
                    color=color, lw=2)
        ax.set_yscale("log")
        ax.axvline(0, color=APAGADO, lw=0.8, ls="--")
        ax.axvline(1, color=APAGADO, lw=0.8, ls="--")
        ax.set_title(cn)
        ax.set_xlabel("valor normalizado (punteadas: 0 y 1 = rango real)")
        ax.set_ylabel("densidad")
    for k in range(n, nr * nc):
        axs[k // nc][k % nc].axis("off")
    leyenda(fig, series)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    guardar(fig, os.path.join(out, "fig5_histogramas.png"))


def fig_metricas(ruta_val, out):
    with open(ruta_val, newline="") as f:
        filas = list(csv.DictReader(f))
    nombres = [r["canal"] for r in filas]
    rug = [float(r["rugosidad_s/r"]) for r in filas]
    fuera = [float(r["fuera_rango_%"]) for r in filas]
    dist = [float(r["dist_logesp_dec"]) for r in filas]
    y = np.arange(len(nombres))[::-1]
    fig, axs = plt.subplots(1, 3, figsize=(15, 7.5), sharey=True)
    paneles = [
        (rug, "Rugosidad sintético / real\n(1 = igual que el real)", True),
        (fuera, "Valores fuera del rango real (%)", False),
        (dist, "Distancia log-espectral (décadas)", False),
    ]
    for ax, (v, titulo, log) in zip(axs, paneles):
        ax.barh(y, v, height=0.6, color=NARANJA)
        ax.set_title(titulo)
        if log:
            ax.set_xscale("log")
            ax.axvline(1, color=APAGADO, lw=1, ls="--")
        ax.grid(axis="y", visible=False)
        for yi, vi, nm in zip(y, v, nombres):
            if nm in ("L4Ia_FLT", "L4In_FLT"):  # anotacion selectiva
                ax.annotate("%.0f" % vi if vi >= 100 else "%.3g" % vi, (vi, yi),
                            xytext=(4, 0), textcoords="offset points",
                            va="center", color=TINTA2, fontsize=11)
    axs[0].set_yticks(y)
    axs[0].set_yticklabels(nombres)
    fig.suptitle("Fourier Flows, un flujo por canal (semilla 0): métricas por canal", fontsize=14, y=1.0)
    fig.tight_layout()
    guardar(fig, os.path.join(out, "fig6_metricas.png"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--csv", default="data/PMU_AlexyDonald2.csv")
    p.add_argument("--validacion", default=None, help="por defecto validacion.csv junto al .npz")
    p.add_argument("--salida", default=None, help="por defecto figs/ junto al .npz")
    p.add_argument("--canales", default="BUS10Va,BUS14Va,L2Ia_FLT,L4Ia_FLT")
    p.add_argument("--n", type=int, default=3000, help="ventanas por conjunto en PCA/espectro/hist")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--timegan", default=None, help="archivo .npy (N, 48, 20), opcional")
    p.add_argument("--timegan_normalizado", action="store_true")
    a = p.parse_args()

    nombres_sel = [c.strip() for c in a.canales.split(",") if c.strip()]
    malos = [c for c in nombres_sel if c not in CANALES]
    if malos:
        sys.exit("Canales desconocidos: %s" % malos)
    sel = [(CANALES.index(c), c) for c in nombres_sel]

    base = os.path.dirname(os.path.abspath(a.npz))
    a.validacion = a.validacion or os.path.join(base, "validacion.csv")
    a.salida = a.salida or os.path.join(base, "figs")
    os.makedirs(a.salida, exist_ok=True)
    estilo()
    rs = np.random.RandomState(a.semilla)

    datos, orden_tg = cargar_real(a.csv)
    mn, mx = datos.min(axis=0), datos.max(axis=0)
    norm_real = normalizar(datos, mn, mx)

    z = np.load(a.npz)
    canales_npz = [str(c) for c in z["canales"]]
    if canales_npz != CANALES:
        sys.exit("El .npz no tiene los 20 canales en el orden esperado.")
    sint = normalizar(z["sintetico"], mn, mx)  # (N, 48, 20)

    series = [("Real", AZUL, muestrear_ventanas(norm_real, a.n, rs)),
              ("Fourier Flows", NARANJA, submuestra(sint, a.n, rs))]

    if a.timegan:
        tg = np.load(a.timegan)
        if tg.ndim != 3 or tg.shape[1] != VENTANA or tg.shape[2] != 20:
            sys.exit("TimeGAN: se esperaba (N, 48, 20), llego %s" % (tg.shape,))
        if sorted(orden_tg) != sorted(CANALES):
            sys.exit("Las columnas 1..20 del CSV no son los 20 canales esperados.")
        tg = tg[:, :, [orden_tg.index(c) for c in CANALES]]  # al orden CANALES
        if not a.timegan_normalizado:
            tg = normalizar(tg, mn, mx)
        series.append(("TimeGAN", AQUA, submuestra(tg, a.n, rs)))

    fig_ventanas(series, sel, a.salida, rs)
    fig_pca_canales(series, sel, a.salida)
    # Control para la PCA conjunta: ventanas reales con cada canal barajado por separado.
    # Conserva la distribucion de cada canal y destruye solo la relacion entre canales.
    ctrl = series[0][2].copy()
    for c in range(ctrl.shape[2]):
        ctrl[:, :, c] = ctrl[rs.permutation(len(ctrl)), :, c]
    fig_pca_conjunta(series + [("Control: canales barajados", APAGADO, ctrl)], a.salida)
    fig_espectro(series, sel, a.salida)
    fig_hist(series, sel, a.salida)
    if os.path.exists(a.validacion):
        fig_metricas(a.validacion, a.salida)
    else:
        print("Sin %s: se omite fig6 (correr validar_salida.py primero)." % a.validacion)


if __name__ == "__main__":
    main()