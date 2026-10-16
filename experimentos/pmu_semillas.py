import sys
import argparse
import time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from ff_safe import FourierFlowSafe

parser = argparse.ArgumentParser()
parser.add_argument("--canal", default="BUS10Vc")
parser.add_argument("--semillas", type=int, default=5)
parser.add_argument("--epocas", type=int, default=2000)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

df = pd.read_csv(str(ROOT / "data" / "PMU_AlexyDonald2.csv"))
df.columns = df.columns.str.strip()
ref = df["BUS10Va"].values.astype(np.float64)   # la alineación siempre es con el cruce por cero de BUS10Va
x = df[args.canal].values.astype(np.float64)
N = len(ref)

# Zona sana: sin la falla (2001-4082) ni la sobretensión (~10800-13200), con margen
sana = np.ones(N, dtype=bool)
sana[2001 - 500:4082 + 500] = False
sana[10500:13800] = False

WIN, JITTER = 401, 20
cruces = np.flatnonzero((ref[:-1] < 0) & (ref[1:] >= 0)) + 1
starts = [c + d for c in cruces for d in range(-JITTER, JITTER + 1)
          if c + d >= 0 and c + d + WIN <= N and sana[c + d:c + d + WIN].all()]
print("Canal:", args.canal, "| ventanas:", len(starts), flush=True)

xmin, xmax = x[sana].min(), x[sana].max()
xn = (x - xmin) / (xmax - xmin + 1e-7)
X = [xn[i:i + WIN] for i in starts]
real = np.array(X) * (xmax - xmin) + xmin


def f1(w): return np.fft.rfft(w, axis=1)[:, 1]
def rms(w): return np.sqrt((w ** 2).mean(1))
def envolver(d, c): return (d - c + 180) % 360 - 180 + c


fr = f1(real)
centro = np.degrees(np.angle(np.mean(fr / np.abs(fr))))   # media circular de la fase real
fase_r = envolver(np.degrees(np.angle(fr)), centro)
amp_r = np.abs(fr)
ref_vals = (rms(real).mean(), rms(real).std(), amp_r.std() / amp_r.mean(),
            fase_r.mean(), fase_r.std())
print("REAL: RMS %.0f +- %.1f | std/media |bin1| %.4f | fase %.2f +- %.2f grados" % ref_vals, flush=True)

filas, sinteticos = [], {}
for s in range(args.semillas):
    torch.manual_seed(s)
    np.random.seed(s)
    t0 = time.time()
    model = FourierFlowSafe(hidden=200, fft_size=WIN, n_flows=3, normalize=True)
    losses = model.fit(X, epochs=args.epocas, learning_rate=args.lr,
                       display_step=args.epocas)
    S = model.sample(len(X))
    synth = S * (xmax - xmin) + xmin
    fs_ = f1(synth)
    amp_s = np.abs(fs_)
    fase_s = envolver(np.degrees(np.angle(fs_)), centro)
    fila = [float(losses[-1]), rms(synth).mean(), rms(synth).std(),
            amp_s.std() / amp_s.mean(), fase_s.mean(), fase_s.std()]
    filas.append(fila)
    sinteticos["synth_semilla%d" % s] = synth.astype(np.float32)
    print("semilla %d (%.0f s): perdida %.1f | RMS %.0f +- %.1f | std/media |bin1| %.4f | "
          "fase %.2f +- %.2f grados" % tuple([s, time.time() - t0] + fila), flush=True)

filas = np.array(filas)
print("\n--- Resumen entre semillas (media +- desviación entre semillas) ---")
nombres = ["perdida final", "RMS medio", "desv. del RMS", "std/media |bin1|",
           "fase media (grados)", "desv. de la fase (grados)"]
for k, nombre in enumerate(nombres):
    print("%-26s %.4f +- %.4f" % (nombre, filas[:, k].mean(), filas[:, k].std()))
print("(real: RMS %.0f +- %.1f | std/media |bin1| %.4f | fase %.2f +- %.2f)" % ref_vals)

np.savez(str(ROOT / "results" / "exploratorios" / "05_semillas" / ("semillas_%s_out.npz" % args.canal)),
         real=real, **sinteticos)