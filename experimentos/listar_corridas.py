#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
listar_corridas.py -- Tabla de corridas de pmu_pipeline.py a partir de los .json que cada una escribe.

Busca recursivamente en --raiz (por defecto results/) los .json con "por_canal" y resume, por
corrida: carpeta, fecha, semilla, epocas, lr, paso, hidden, flujos, ventanas sinteticas, tiempo,
reintentos, perdida final media (si el .json la tiene), y si existen validacion y figuras junto
al .npz. Escribe tambien <raiz>/corridas.csv. Solo usa la libreria estandar.
"""
import argparse
import csv
import glob
import json
import os
import time


def resumen(ruta, raiz):
    with open(ruta) as f:
        c = json.load(f)
    if "por_canal" not in c:
        return None
    carpeta = os.path.dirname(ruta)
    base = os.path.splitext(os.path.basename(ruta))[0]
    pc = c["por_canal"].values()
    perd = [v["perdida_final"] for v in pc if v.get("perdida_final") is not None]
    forma = c.get("forma", [])
    return {
        "corrida": os.path.relpath(ruta, raiz),
        "fecha": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(ruta))),
        "semilla": c.get("semilla_base"), "epocas": c.get("epocas"), "lr": c.get("lr_inicial"),
        "paso": c.get("paso"), "hidden": c.get("hidden"), "flujos": c.get("n_flows"),
        "forma": "x".join(str(x) for x in forma),
        "tiempo_s": c.get("tiempo_esta_invocacion_s"),
        "reintentos": sum(1 for v in c["por_canal"].values() if v.get("intentos", 1) > 1),
        "perdida_final_media": round(sum(perd) / len(perd), 2) if perd else "n/d",
        "validacion": "si" if glob.glob(os.path.join(carpeta, "validacion*.csv")) else "no",
        "figuras": "si" if glob.glob(os.path.join(carpeta, "figs*")) else "no",
        "_orden": os.path.getmtime(ruta),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--raiz", default="results")
    a = p.parse_args()
    filas = []
    for ruta in glob.glob(os.path.join(a.raiz, "**", "*.json"), recursive=True):
        try:
            r = resumen(ruta, a.raiz)
        except Exception:
            r = None
        if r:
            filas.append(r)
    if not filas:
        print("No se encontraron corridas en", a.raiz)
        return
    filas.sort(key=lambda r: r["_orden"])
    cols = [k for k in filas[0] if not k.startswith("_")]
    for r in filas:
        print(" | ".join("%s=%s" % (k, r[k]) for k in cols))
    with open(os.path.join(a.raiz, "corridas.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(filas)
    print("\nGuardado:", os.path.join(a.raiz, "corridas.csv"))


if __name__ == "__main__":
    main()