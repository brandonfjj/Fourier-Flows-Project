#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
analisis_rep.py -- Analisis de la repeticion s0_e2000_lr1e-4_p5_rep frente a la corrida orig.
Solo lectura y calculo (no entrena). Secciones A-H; salida en <rep>/analisis/.
  A reproducibilidad (npz, validacion.csv, sha256, perdidas finales)
  B orden de canales del CSV vs CANALES
  C comparacion pixel a pixel de las figuras
  D PCA conjunta de la fig. 3 en numeros (real, Fourier Flows, control barajado)
  E rugosidad absoluta real y sintetica + perdida final, Spearman
  F diagnostico de normalizacion de los 20 canales
  G efecto de recortar el sintetico a [0,1]
  H distancia log-espectral con y sin quitar la recta de minimos cuadrados
"""
import csv
import hashlib
import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experimentos"))

import numpy as np
import matplotlib.image as mpimg
import graficas_comparacion as gc  # reutiliza carga, normalizacion, muestreo y PCA de la fig. 3

RUNS = ROOT / "results" / "runs"
ORIG = RUNS / "s0_e2000_lr1e-4_p5_orig"
REP = RUNS / "s0_e2000_lr1e-4_p5_rep"
OUT = REP / "analisis"
CSV_DATOS = ROOT / "data" / "PMU_AlexyDonald2.csv"
CANALES = gc.CANALES
V = 48
EPS = 1e-12

OUT.mkdir(exist_ok=True)
_reporte = open(str(OUT / "reporte.txt"), "w")


def p(*a):
    s = " ".join(str(x) for x in a)
    print(s)
    _reporte.write(s + "\n")


def titulo(t):
    p("\n" + "=" * 100 + "\n" + t + "\n" + "=" * 100)


def guardar_csv(nombre, cab, filas):
    with open(str(OUT / nombre), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cab)
        w.writerows(filas)
    p("(tabla guardada en %s)" % (OUT / nombre).relative_to(ROOT))


def sha256(ruta):
    h = hashlib.sha256()
    with open(str(ruta), "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def rangos(x):
    """Rangos con empates promediados (para Spearman)."""
    x = np.asarray(x, dtype=float)
    orden = np.argsort(x, kind="mergesort")
    r = np.empty(len(x))
    r[orden] = np.arange(1, len(x) + 1)
    for v in np.unique(x):
        m = x == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    return r


def spearman(a, b):
    ra, rb = rangos(a), rangos(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def ventanas_todas(norm):
    """Todas las ventanas reales de 48 muestras con paso 1: (T, C) -> (N, 48, C)."""
    n = norm.shape[0] - V + 1
    idx = np.arange(V)[None, :] + np.arange(n)[:, None]
    return norm[idx]


def rug(w):
    """Media de |segunda diferencia| por canal. w: (N, 48, C) -> (C,)"""
    d2 = w[:, 2:, :] - 2.0 * w[:, 1:-1, :] + w[:, :-2, :]
    return np.abs(d2).mean(axis=(0, 1))


def dist_logesp(wr, ws):
    """Igual que validar_salida.py: potencia media por bin, RMS de la diferencia en log10, por canal."""
    pr = (np.abs(np.fft.rfft(wr, axis=1)) ** 2).mean(axis=0)  # (bins, C)
    ps = (np.abs(np.fft.rfft(ws, axis=1)) ** 2).mean(axis=0)
    return np.sqrt(np.mean((np.log10(pr + EPS) - np.log10(ps + EPS)) ** 2, axis=0))


def quitar_recta(w):
    """Resta a cada ventana (y canal) su recta de minimos cuadrados."""
    t = np.arange(V, dtype=float)
    A = np.stack([t, np.ones(V)], axis=1)          # (48, 2)
    P = A.dot(np.linalg.pinv(A))                   # proyector sobre {1, t}
    return w - np.einsum("ij,njc->nic", P, w)


def fuera01(w):
    return 100.0 * ((w < 0) | (w > 1)).mean(axis=(0, 1))


# ----------------------------------------------------------------------------------------------
datos, orden_tg = gc.cargar_real(str(CSV_DATOS))   # (T, 20) en orden CANALES
mn, mx = datos.min(axis=0), datos.max(axis=0)
norm_real = gc.normalizar(datos, mn, mx)

zo = np.load(str(ORIG / "pmu_sintetico.npz"))
zr = np.load(str(REP / "pmu_sintetico.npz"))
so, sr = zo["sintetico"], zr["sintetico"]
meta_rep = json.load(open(str(REP / "pmu_sintetico.json")))["por_canal"]

# ============================== A ==============================
titulo("A) Reproducibilidad orig vs rep")
p("canales orig == rep == CANALES:", [str(c) for c in zo["canales"]] == [str(c) for c in zr["canales"]] == CANALES)
p("forma orig %s | rep %s" % (so.shape, sr.shape))
igual = bool(np.array_equal(so, sr))
p("np.array_equal(sintetico):", igual)
if not igual:
    p("%-10s %14s %14s %14s" % ("canal", "max|dif|", "rango_canal", "max|dif|/rango"))
    for j, c in enumerate(CANALES):
        d = float(np.max(np.abs(so[:, :, j] - sr[:, :, j])))
        p("%-10s %14.6g %14.6g %14.3e" % (c, d, mx[j] - mn[j], d / (mx[j] - mn[j])))
p("minimos/maximos guardados iguales:",
  bool(np.array_equal(zo["minimos"], zr["minimos"]) and np.array_equal(zo["maximos"], zr["maximos"])))

vo = list(csv.reader(open(str(ORIG / "validacion.csv"))))
vr = list(csv.reader(open(str(REP / "validacion.csv"))))
p("validacion.csv identicos (texto):", vo == vr)
if vo != vr:
    cab = vo[0]
    p("Diferencias en validacion.csv (columna: max |orig - rep| sobre canales):")
    for k in range(2, len(cab)):
        a = np.array([float(f[k]) for f in vo[1:]])
        b = np.array([float(f[k]) for f in vr[1:]])
        p("  %-18s %.6g" % (cab[k], np.max(np.abs(a - b))))

p("\nsha256:")
for ruta in (ORIG / "pmu_sintetico.npz", REP / "pmu_sintetico.npz", CSV_DATOS):
    p("  %s  %s" % (sha256(ruta), ruta.relative_to(ROOT)))
p("  (un .npz puede diferir en bytes aunque el arreglo sea igual: guarda tambien config con tiempos)")

perd_orig = [float(m.group(1)) for m in
             (re.search(r"step:\s*1999\s*/\s*2000\s+loss:\s*(\S+)", l)
              for l in open(str(ORIG / "pipeline.log"))) if m]
perd_rep = {c: meta_rep[c]["perdida_final"] for c in CANALES}
p("\nPerdidas finales: orig (log, sin nombre de canal) n=%d | rep (json, por canal) n=%d"
  % (len(perd_orig), len(perd_rep)))
so_ = sorted(perd_orig)
sr_ = sorted(perd_rep.values())
p("orig ordenadas:", ["%.3f" % x for x in so_])
p("rep  ordenadas:", ["%.3f" % x for x in sr_])
mismo = len(so_) == len(sr_) and np.allclose(so_, sr_, atol=5e-4)
p("Mismo conjunto de valores (tolerancia 5e-4, el log imprime 3 decimales):", mismo)
if not mismo:
    p("max |dif| entre listas ordenadas: %.4f" % float(np.max(np.abs(np.array(so_) - np.array(sr_)))))

# ============================== B ==============================
titulo("B) Orden de canales: columnas 1..20 del CSV vs CANALES de pmu_pipeline.py")
for k, (a, b) in enumerate(zip(orden_tg, CANALES), start=1):
    p("  col %2d  %-10s  %-10s  %s" % (k, a, b, "ok" if a == b else "DISTINTO"))
p("Coinciden en orden:", orden_tg == CANALES, "| ultima columna (objetivo de la metrica predictiva):", orden_tg[-1])

# ============================== C ==============================
titulo("C) Figuras orig vs rep, pixel a pixel")
for nombre in sorted(os.listdir(str(ORIG / "figs"))):
    a = mpimg.imread(str(ORIG / "figs" / nombre))
    b = mpimg.imread(str(REP / "figs" / nombre))
    if a.shape != b.shape:
        p("  %-24s DISTINTA forma: orig %s | rep %s" % (nombre, a.shape, b.shape))
    else:
        dif = np.any(a != b, axis=-1)
        p("  %-24s %s | pixeles distintos: %d de %d (%.3f %%)" % (
            nombre, "IDENTICA" if not dif.any() else "DISTINTA", int(dif.sum()), dif.size, 100.0 * dif.mean()))

# ============================== D ==============================
titulo("D) Fig. 3 en numeros: PCA conjunta (48x20 aplanado), ajustada sobre 3000 ventanas reales")
# Se reproduce la MISMA secuencia del generador que graficas_comparacion.main (semilla 0, n=3000):
rs = np.random.RandomState(0)
sint_rep_n = gc.normalizar(sr, mn, mx)
real_w = gc.muestrear_ventanas(norm_real, 3000, rs)
ff_w = gc.submuestra(sint_rep_n, 3000, rs)
for _ in range(4):           # fig_ventanas: 4 canales x 2 series x rs.choice(3000, 6)
    for w in (real_w, ff_w):
        rs.choice(len(w), size=6, replace=False)
ctrl = real_w.copy()
for c in range(ctrl.shape[2]):
    ctrl[:, :, c] = ctrl[rs.permutation(len(ctrl)), :, c]
mu, comps, ratio = gc.pca_ajustar(real_w.reshape(len(real_w), -1))
p("Varianza explicada PC1+PC2 (real): %.1f %% (la fig. 3 indica 97.7 %%)" % (100 * ratio.sum()))
p("%-26s %10s %10s %12s %12s" % ("conjunto", "std PC1", "std PC2", "radio medio", "radio/real"))
radio_real = None
filas_d = []
for nombre, w in (("Real", real_w), ("Fourier Flows (rep)", ff_w), ("Control canales barajados", ctrl)):
    pr = gc.proyectar(w.reshape(len(w), -1), mu, comps)
    radio = float(np.mean(np.hypot(pr[:, 0], pr[:, 1])))
    radio_real = radio_real or radio
    filas_d.append([nombre, pr[:, 0].std(), pr[:, 1].std(), radio, radio / radio_real])
    p("%-26s %10.3f %10.3f %12.3f %12.3f" % tuple(filas_d[-1]))
guardar_csv("D_pca_conjunta.csv", ["conjunto", "std_pc1", "std_pc2", "radio_medio", "radio_sobre_real"], filas_d)

# ============================== E ==============================
titulo("E) Rugosidad absoluta (media |segunda diferencia|, unidades normalizadas) y perdida final (rep)")
wr_all = ventanas_todas(norm_real)            # (19953, 48, 20), paso 1, igual que validar_salida.py
rug_r = rug(wr_all)
rug_s = rug(sint_rep_n)
perd = np.array([perd_rep[c] for c in CANALES])
orden = np.argsort(rug_r)
p("%-10s %12s %12s %10s %12s" % ("canal", "rug_real", "rug_sint", "s/r", "perdida_fin"))
filas_e = []
for j in orden:
    filas_e.append([CANALES[j], rug_r[j], rug_s[j], rug_s[j] / rug_r[j], perd[j]])
    p("%-10s %12.4e %12.4e %10.1f %12.3f" % tuple(filas_e[-1]))
guardar_csv("E_rugosidad.csv", ["canal", "rug_real", "rug_sint", "cociente", "perdida_final"], filas_e)
p("Spearman(perdida_final, rugosidad real)       = %+.3f" % spearman(perd, rug_r))
p("Spearman(perdida_final, cociente sint/real)   = %+.3f" % spearman(perd, rug_s / rug_r))
p("Spearman(perdida_final, rugosidad sintetica)  = %+.3f" % spearman(perd, rug_s))
p("n = 20 canales, varios casi duplicados (corrientes = tension/1e6; L4Ia ~ L4In): exploratorio, no inferencial.")

# ============================== F ==============================
titulo("F) Diagnostico de normalizacion (unidades fisicas), orden por (max-min)/std descendente")
filas_f = []
for j, c in enumerate(CANALES):
    x = datos[:, j]
    md = np.median(x)
    r = x.max() - x.min()
    filas_f.append([c, x.min(), np.percentile(x, 1), md, np.percentile(x, 99), x.max(), x.std(),
                    r / x.std(), float(np.mean(np.abs(x - md) < 0.01 * r))])
filas_f.sort(key=lambda f: -f[7])
p("%-10s %11s %11s %11s %11s %11s %11s %9s %9s" % ("canal", "min", "p1", "p50", "p99", "max", "std",
                                                    "rango/std", "frac<1%"))
for f in filas_f:
    p("%-10s %11.4g %11.4g %11.4g %11.4g %11.4g %11.4g %9.2f %9.4f" % tuple(f))
guardar_csv("F_normalizacion.csv", ["canal", "min", "p1", "p50", "p99", "max", "std", "rango_sobre_std",
                                    "frac_cerca_mediana"], filas_f)

# ============================== G ==============================
titulo("G) Efecto de recortar el sintetico (rep) a [0,1] en unidades normalizadas")
sint_clip = np.clip(sint_rep_n, 0.0, 1.0)
rug_c = rug(sint_clip)
d_sin = dist_logesp(wr_all, sint_rep_n)
d_con = dist_logesp(wr_all, sint_clip)
f_sin = fuera01(sint_rep_n)
f_con = fuera01(sint_clip)
p("%-10s %9s %9s %8s | %8s %8s %8s | %7s %7s" % ("canal", "rug s/r", "recort.", "cambio%", "d_logesp",
                                                 "recort.", "cambio", "fuera%", "recort."))
filas_g = []
for j, c in enumerate(CANALES):
    a, b = rug_s[j] / rug_r[j], rug_c[j] / rug_r[j]
    filas_g.append([c, a, b, 100 * (b - a) / a, d_sin[j], d_con[j], d_con[j] - d_sin[j], f_sin[j], f_con[j]])
    p("%-10s %9.1f %9.1f %8.1f | %8.3f %8.3f %+8.3f | %7.2f %7.2f" % tuple(filas_g[-1]))
guardar_csv("G_recorte.csv", ["canal", "rug_sr", "rug_sr_recortado", "cambio_pct", "dlogesp",
                              "dlogesp_recortado", "cambio_dlogesp", "fuera_pct", "fuera_pct_recortado"], filas_g)
p("Nota: d_logesp aqui es en unidades normalizadas (validar_salida.py la calcula en fisicas; el bin 0 cambia).")
p("Nota: fuera% recortado es 0 por construccion.")

# ============================== H ==============================
titulo("H) Distancia log-espectral restando a cada ventana su recta de minimos cuadrados (normalizado)")
wr_dt = quitar_recta(wr_all)
ws_dt = quitar_recta(sint_rep_n)
d_dt = dist_logesp(wr_dt, ws_dt)
# Cociente de potencia en la mitad alta del espectro (bins 12..24), sin y con quitar la recta
def pot_alta(w):
    return (np.abs(np.fft.rfft(w, axis=1)) ** 2)[:, 12:, :].mean(axis=(0, 1))
q_sin = pot_alta(sint_rep_n) / pot_alta(wr_all)
q_dt = pot_alta(ws_dt) / pot_alta(wr_dt)
p("%-10s %9s %11s %8s | %13s %13s | %8s" % ("canal", "d_logesp", "sin recta", "cambio",
                                            "Palta s/r", "sin recta", "rug s/r"))
filas_h = []
for j in np.argsort(-(rug_s / rug_r)):
    filas_h.append([CANALES[j], d_sin[j], d_dt[j], d_dt[j] - d_sin[j], q_sin[j], q_dt[j], rug_s[j] / rug_r[j]])
    p("%-10s %9.3f %11.3f %+8.3f | %13.3g %13.3g | %8.1f" % tuple(filas_h[-1]))
guardar_csv("H_espectro_sin_recta.csv", ["canal", "dlogesp", "dlogesp_sin_recta", "cambio",
                                         "pot_alta_s_r", "pot_alta_s_r_sin_recta", "rug_sr"], filas_h)
p("Spearman(d_logesp sin recta, rug s/r) = %+.3f | Spearman(d_logesp original, rug s/r) = %+.3f"
  % (spearman(d_dt, rug_s / rug_r), spearman(d_sin, rug_s / rug_r)))
p("Palta = potencia media en bins 12..24 (5-10 kHz); s/r = sintetico / real.")

_reporte.close()
