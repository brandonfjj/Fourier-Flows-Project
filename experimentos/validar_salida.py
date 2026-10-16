#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
validar_salida.py -- Validacion basica del arreglo sintetico (N, 48, C) de pmu_pipeline.py.

Revisa, por canal, contra ventanas reales de 48 muestras (paso 1) del CSV:
  1. Integridad: forma esperada, sin NaN/inf, min/max guardados == min/max del CSV.
  2. Rango: fraccion de valores sinteticos fuera de [min, max] real del canal.
  3. Estadisticas globales: media, desviacion y RMS (real vs sintetico, en unidades fisicas).
  4. Rugosidad: media de |segunda diferencia| sintetico / real (>>1 = ruido de alta frecuencia).
  5. Espectro: distancia log-espectral entre espectros de potencia medios (rfft de 48 muestras,
     sin ventana), en decadas = raiz de la media de (log10 Pr - log10 Ps)^2 sobre todos los bins.
     OJO: es MI definicion; no es necesariamente la misma que usa ff48_prueba.py.
Salida: tabla en pantalla + CSV (results/validacion_<nombre>.csv). Codigo de salida 1 si hay
NaN/inf o forma incorrecta. No imprime muestras crudas; el CSV de salida contiene solo estadisticos.
"""
import argparse
import csv
import os
import sys

import numpy as np

VENTANA = 48
EPS = 1e-12


def cargar_csv(ruta, canales):
    with open(ruta, "r", newline="", encoding="utf-8") as f:
        primera = f.readline()
        delim = ";" if primera.count(";") > primera.count(",") else ","
        f.seek(0)
        lector = csv.reader(f, delimiter=delim)
        enc = [c.strip() for c in next(lector)]
        faltan = [c for c in canales if c not in enc]
        if faltan:
            raise ValueError("Columnas ausentes en el CSV: %s" % faltan)
        idx = [enc.index(c) for c in canales]
        filas = [[float(fila[i]) for i in idx] for fila in lector if fila]
    return np.asarray(filas, dtype=np.float64)


def ventanas(serie, paso=1):
    n = (len(serie) - VENTANA) // paso + 1
    s = serie.strides[0]
    return np.lib.stride_tricks.as_strided(serie, shape=(n, VENTANA), strides=(s * paso, s))


def rugosidad(w):
    d2 = w[:, 2:] - 2.0 * w[:, 1:-1] + w[:, :-2]
    return float(np.mean(np.abs(d2)))


def espectro_medio(w):
    p = np.abs(np.fft.rfft(w, axis=1)) ** 2
    return p.mean(axis=0)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--npz", required=True)
    p.add_argument("--csv", default="data/PMU_AlexyDonald2.csv")
    p.add_argument("--salida", default=None, help="CSV de resultados (por defecto junto al .npz)")
    a = p.parse_args()

    z = np.load(a.npz)
    sint = z["sintetico"]
    canales = [str(c) for c in z["canales"]]
    mins, maxs = z["minimos"], z["maximos"]
    ok = True

    if sint.ndim != 3 or sint.shape[1] != VENTANA or sint.shape[2] != len(canales):
        print("FORMA INESPERADA: %s para %d canales" % (sint.shape, len(canales)))
        sys.exit(1)
    print("Arreglo: %s | canales: %d" % (sint.shape, len(canales)))

    n_nan = int(np.sum(~np.isfinite(sint)))
    print("Valores no finitos: %d %s" % (n_nan, "(MAL)" if n_nan else "(ok)"))
    if n_nan:
        ok = False

    datos = cargar_csv(a.csv, canales)
    filas = []
    cab = ["canal", "min_ok", "fuera_rango_%", "media_r", "media_s", "std_r", "std_s",
           "rms_r", "rms_s", "rugosidad_s/r", "dist_logesp_dec"]
    for j, nombre in enumerate(canales):
        real = np.ascontiguousarray(datos[:, j])
        s = sint[:, :, j]
        wr = ventanas(real)

        mm_ok = bool(np.isclose(mins[j], real.min()) and np.isclose(maxs[j], real.max()))
        rango = real.max() - real.min()
        fuera = 100.0 * float(np.mean((s < real.min() - 1e-9 * rango) | (s > real.max() + 1e-9 * rango)))
        rug = rugosidad(s) / max(rugosidad(wr), EPS)
        pr, ps = espectro_medio(wr), espectro_medio(s)
        dist = float(np.sqrt(np.mean((np.log10(pr + EPS) - np.log10(ps + EPS)) ** 2)))
        filas.append([nombre, "si" if mm_ok else "NO",
                      fuera, float(wr.mean()), float(s.mean()), float(wr.std()), float(s.std()),
                      float(np.sqrt(np.mean(wr ** 2))), float(np.sqrt(np.mean(s ** 2))), rug, dist])

    print("\n%-10s %-6s %8s %10s %10s %10s %10s %10s %10s %8s %8s" % (
        "canal", "min/max", "fuera%", "media_r", "media_s", "std_r", "std_s",
        "rms_r", "rms_s", "rug s/r", "d_logesp"))
    for f in filas:
        print("%-10s %-6s %8.2f %10.4g %10.4g %10.4g %10.4g %10.4g %10.4g %8.1f %8.3f" % tuple(f))

    malos = [f[0] for f in filas if f[1] == "NO"]
    if malos:
        print("\nAVISO: min/max guardados no coinciden con el CSV en: %s" % malos)
    print("\nLectura: 'fuera%' alto o 'rug s/r' >> 1 indica ruido/valores fuera de rango "
          "(ya observado en 3 canales). Son diagnosticos descriptivos, no una prueba de calidad.")

    salida = a.salida or (os.path.splitext(a.npz)[0].replace("pmu_sintetico", "validacion") + ".csv")
    with open(salida, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cab)
        w.writerows(filas)
    print("Guardado:", salida)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()