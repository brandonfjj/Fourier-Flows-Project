#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pmu_pipeline.py -- Pipeline Fourier Flows para los 20 canales PMU (un flujo por canal).

Para cada canal:
  1. Ventanas de 49 muestras (paso --paso), min-max sobre TODO el canal (se guardan min y max).
  2. Entrenamiento de FourierFlowSafe (lr 1e-4, guardia contra NaN con reintento a lr/2).
  3. Muestreo, recorte 49 -> 48 y desnormalizacion con el min/max guardado.
Salida: arreglo (N, 48, 20) en un .npz, mas un .json con semilla y configuracion.

Procesos paralelos: un canal por proceso, 1 hilo por proceso (el modelo es pequeno).
Cada canal terminado se guarda aparte (checkpoint, con su propia metadata); con --reanudar se
omiten los que ya existen con la MISMA configuracion. Un canal que falla no tumba a los demas.
La semilla de cada canal es semilla_base*1000 + indice GLOBAL del canal (CANALES.index).

SUPUESTOS NO VERIFICADOS (no vi ff48_prueba.py ni ff_safe.py; revisar en entrenar_canal):
  - from ff_safe import FourierFlowSafe  (ff_safe.py importable desde la raiz del repo)
  - FourierFlowSafe(hidden, fft_size, n_flows, normalize=True)
  - .fit(X, epochs=, batch_size=, learning_rate=, display_step=)
  - .sample(n) devuelve (n, 49), tensor o ndarray
  - X en float32 (igualar con ff48_prueba.py)
El CSV y los .npz/.json generados derivan de datos confidenciales: no van a git (results/).
"""
import argparse
import contextlib
import csv
import json
import multiprocessing as mp
import os
import platform
import re
import sys
import time
import traceback

import numpy as np

CANALES = [
    "BUS10Va", "BUS10Vb", "BUS10Vc",
    "BUS14Va", "BUS14Vb", "BUS14Vc",
    "L2Ia_FLT", "L2Ib_FLT", "L2Ic_FLT", "L2In_FLT",
    "L2Va_FLT", "L2Vb_FLT", "L2Vc_FLT",
    "L4Ia_FLT", "L4Ib_FLT", "L4Ic_FLT", "L4In_FLT",
    "L4Va_FLT", "L4Vb_FLT", "L4Vc_FLT",
]
VENTANA = 49   # longitud impar requerida por el modelo
RECORTE = 48   # longitud final comparable con el TimeGAN
CLAVES_FIRMA = ("semilla", "epocas", "lr", "paso", "hidden", "n_flows", "n_muestras")


def cargar_csv(ruta, canales):
    """Lee solo las columnas pedidas; limpia espacios en los encabezados."""
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
    datos = np.asarray(filas, dtype=np.float64)
    if not np.isfinite(datos).all():
        raise ValueError("El CSV contiene valores no finitos.")
    return datos  # (filas, canales)


def hacer_ventanas(serie, paso):
    inicios = np.arange(0, len(serie) - VENTANA + 1, paso)
    return np.stack([serie[i:i + VENTANA] for i in inicios])  # (n, 49)


def ruta_checkpoint(dir_tmp, indice, nombre):
    return os.path.join(dir_tmp, "canal_%02d_%s.npz" % (indice, nombre))


def firma_de(cfg):
    return {k: cfg[k] for k in CLAVES_FIRMA}


def leer_meta(ruta):
    z = np.load(ruta)
    return json.loads(str(z["meta"]))


def checkpoint_vigente(ruta, firma):
    """True si el checkpoint existe y se hizo con la misma configuracion."""
    if not os.path.exists(ruta):
        return False
    try:
        return leer_meta(ruta)["firma"] == firma
    except Exception:
        return False


def perdidas_de_log(ruta):
    """Valores 'loss: x' del log de fit() de un canal, en orden (el ultimo intento al final)."""
    vals = []
    with open(ruta) as f:
        for linea in f:
            m = re.search(r"loss:\s*(\S+)", linea)
            if m:
                try:
                    vals.append(float(m.group(1)))
                except ValueError:
                    pass
    return vals


def entrenar_canal(args):
    """Corre en un proceso hijo. Nunca lanza: devuelve un dict con ok True/False."""
    indice, nombre, serie, cfg = args
    try:
        import torch
        torch.set_num_threads(1)
        sys.path.insert(0, cfg["raiz_repo"])
        from ff_safe import FourierFlowSafe  # SUPUESTO: ver cabecera

        t0 = time.time()
        mn, mx = float(serie.min()), float(serie.max())
        if mx - mn <= 0:
            raise ValueError("Canal %s constante: min-max indefinido." % nombre)
        X = hacer_ventanas((serie - mn) / (mx - mn), cfg["paso"]).astype(np.float32)
        n_out = cfg["n_muestras"] or len(X)

        semilla = cfg["semilla"] * 1000 + indice
        lr = cfg["lr"]
        sint = None
        intentos = 0
        ruta_log = os.path.join(cfg["dir_tmp"], "fit_%02d_%s.log" % (indice, nombre))
        open(ruta_log, "w").close()
        for intentos in range(1, cfg["max_reintentos"] + 2):
            np.random.seed(semilla)
            torch.manual_seed(semilla)
            modelo = FourierFlowSafe(hidden=cfg["hidden"], fft_size=VENTANA,
                                     n_flows=cfg["n_flows"], normalize=True)
            # La salida de fit() va a un log por canal (en el log general salian sin nombre)
            with open(ruta_log, "a") as flog:
                flog.write("# canal=%s intento=%d lr=%g semilla=%d\n" % (nombre, intentos, lr, semilla))
                with contextlib.redirect_stdout(flog):
                    modelo.fit(X, epochs=cfg["epocas"], batch_size=len(X),
                               learning_rate=lr, display_step=max(cfg["epocas"] // 5, 1))
            s = modelo.sample(n_out)
            s = s.detach().cpu().numpy() if hasattr(s, "detach") else np.asarray(s)
            if np.isfinite(s).all():
                sint = s
                break
            lr = lr / 2.0  # guardia contra NaN: reintento mas conservador
        if sint is None:
            return {"indice": indice, "nombre": nombre, "ok": False, "lr_usado": lr,
                    "intentos": intentos, "error": "NaN tras todos los reintentos"}

        sint = sint[:, :RECORTE].astype(np.float64)
        desnorm = sint * (mx - mn) + mn
        perdidas = perdidas_de_log(ruta_log)
        meta = {"firma": firma_de(cfg), "indice": indice, "nombre": nombre,
                "min": mn, "max": mx, "lr_usado": lr, "intentos": intentos,
                "perdida_inicial": perdidas[0] if perdidas else None,
                "perdida_final": perdidas[-1] if perdidas else None,
                "log_fit": os.path.basename(ruta_log),
                "semilla_canal": semilla, "n_ventanas_entrenamiento": int(len(X)),
                "n_ventanas_sinteticas": int(len(sint)),
                "segundos": round(time.time() - t0, 1)}
        np.savez(ruta_checkpoint(cfg["dir_tmp"], indice, nombre),
                 sintetico=desnorm, min=mn, max=mx, meta=json.dumps(meta))
        meta["ok"] = True
        return meta
    except Exception:
        return {"indice": indice, "nombre": nombre, "ok": False, "lr_usado": cfg["lr"],
                "intentos": 0, "error": traceback.format_exc(limit=3)}


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/PMU_AlexyDonald2.csv")
    p.add_argument("--salida", required=True,
                   help="Ruta .npz, p. ej. results/runs/<id>/pmu_sintetico.npz (sin valor por defecto)")
    p.add_argument("--semilla", type=int, default=0)
    p.add_argument("--epocas", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--paso", type=int, default=5)
    p.add_argument("--hidden", type=int, default=200)
    p.add_argument("--n_flows", type=int, default=3)
    p.add_argument("--n_muestras", type=int, default=0,
                   help="Ventanas sinteticas por canal (0 = igual que las reales de entrenamiento)")
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--max_reintentos", type=int, default=2)
    p.add_argument("--canales", default="", help="Subconjunto separado por comas (para pruebas)")
    p.add_argument("--reanudar", action="store_true")
    a = p.parse_args()

    canales = [c.strip() for c in a.canales.split(",") if c.strip()] or CANALES
    desconocidos = [c for c in canales if c not in CANALES]
    if desconocidos:
        sys.exit("Canales desconocidos: %s" % desconocidos)

    salida = a.salida
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    dir_tmp = os.path.splitext(salida)[0] + "_canales"
    os.makedirs(dir_tmp, exist_ok=True)

    # Antes de crear procesos: un hilo por proceso (spawn hereda el entorno)
    for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[v] = "1"

    datos = cargar_csv(a.csv, canales)
    print("CSV: %d filas, %d canales" % datos.shape)

    cfg = {"paso": a.paso, "lr": a.lr, "epocas": a.epocas, "hidden": a.hidden,
           "n_flows": a.n_flows, "semilla": a.semilla, "n_muestras": a.n_muestras,
           "max_reintentos": a.max_reintentos, "dir_tmp": dir_tmp,
           "raiz_repo": os.getcwd()}
    firma = firma_de(cfg)

    tareas = []
    for nombre in canales:
        i = CANALES.index(nombre)  # indice GLOBAL: semilla y checkpoint reproducibles
        if a.reanudar and checkpoint_vigente(ruta_checkpoint(dir_tmp, i, nombre), firma):
            print("Omitido (checkpoint vigente):", nombre)
            continue
        tareas.append((i, nombre, datos[:, canales.index(nombre)].copy(), cfg))

    t0 = time.time()
    if tareas:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(a.workers, len(tareas))) as pool:
            for r in pool.imap_unordered(entrenar_canal, tareas):
                if r["ok"]:
                    pf = r.get("perdida_final")
                    print("[%5.0f s] %-10s ok | lr=%g intentos=%d (%.0f s) | perdida final %s" % (
                        time.time() - t0, r["nombre"], r["lr_usado"], r["intentos"], r["segundos"],
                        "%.3f" % pf if pf is not None else "n/d"))
                else:
                    print("[%5.0f s] %-10s FALLO | %s" % (
                        time.time() - t0, r["nombre"], r.get("error", "")))

    # Ensamblado (N, 48, C), solo con checkpoints de esta configuracion
    partes, mins, maxs, por_canal = [], [], [], {}
    for nombre in canales:
        i = CANALES.index(nombre)
        ruta = ruta_checkpoint(dir_tmp, i, nombre)
        if not os.path.exists(ruta):
            sys.exit("Falta el canal %s (fallo o no se corrio). Reintentar con --reanudar." % nombre)
        meta = leer_meta(ruta)
        if meta["firma"] != firma:
            sys.exit("El checkpoint de %s es de otra configuracion (%s). "
                     "Borrar %s o cambiar --salida." % (nombre, meta["firma"], dir_tmp))
        z = np.load(ruta)
        partes.append(z["sintetico"])
        mins.append(float(z["min"]))
        maxs.append(float(z["max"]))
        por_canal[nombre] = meta
    arreglo = np.stack(partes, axis=-1)

    config = {"semilla_base": a.semilla, "semilla_canal": "semilla_base*1000 + indice_global_canal",
              "epocas": a.epocas, "lr_inicial": a.lr, "paso": a.paso, "hidden": a.hidden,
              "n_flows": a.n_flows, "n_muestras": a.n_muestras,
              "ventana_entrenamiento": VENTANA, "recorte": RECORTE,
              "normalizacion": "min-max sobre todo el canal", "flujo_por_canal": True,
              "canales": canales, "forma": list(arreglo.shape), "por_canal": por_canal,
              "python": platform.python_version(), "nproc": os.cpu_count(),
              "tiempo_esta_invocacion_s": round(time.time() - t0, 1)}
    np.savez(salida, sintetico=arreglo, canales=np.array(canales), minimos=np.array(mins),
             maximos=np.array(maxs), semilla=a.semilla, config=json.dumps(config))
    with open(os.path.splitext(salida)[0] + ".json", "w") as f:
        json.dump(config, f, indent=2)
    print("Guardado %s con forma %s" % (salida, arreglo.shape))


if __name__ == "__main__":
    main()