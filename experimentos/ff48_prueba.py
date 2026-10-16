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
parser.add_argument("--canal", default="BUS10Va")
parser.add_argument("--paso", type=int, default=5)
parser.add_argument("--epocas", type=int, default=1500)
parser.add_argument("--lr", type=float, default=1e-3)
args = parser.parse_args()

df = pd.read_csv(str(ROOT / "data" / "PMU_AlexyDonald2.csv"))
df.columns = df.columns.str.strip()
x = df[args.canal].values.astype(np.float64)
N = len(x)

# Min-Max sobre TODO el canal (falla incluida), igual que el TimeGAN
xmin, xmax = x.min(), x.max()
xn = (x - xmin) / (xmax - xmin + 1e-7)

# Ventanas de 49 (longitud impar para el modelo); se recortan a 48 al comparar
WIN, CMP = 49, 48
X = [xn[i:i + WIN] for i in range(0, N - WIN + 1, args.paso)]
print("Canal:", args.canal, "| ventanas:", len(X), "| paso:", args.paso, flush=True)

torch.manual_seed(0)
np.random.seed(0)
t0 = time.time()
model = FourierFlowSafe(hidden=200, fft_size=WIN, n_flows=3, normalize=True)
losses = model.fit(X, epochs=args.epocas, learning_rate=args.lr,
                   display_step=max(1, args.epocas // 10))
t_train = time.time() - t0
S = np.asarray(model.sample(len(X)))

real = np.array(X)[:, :CMP]
synth = S[:, :CMP]


def resumen(nombre, w):
    dif = np.abs(np.diff(w, axis=1)).mean()
    nivel = w.mean(1).std()
    print("%-10s media %.4f | desv %.4f | min %.4f | max %.4f | |dif consecutiva| %.5f | "
          "desv. del nivel medio entre ventanas %.4f"
          % (nombre, w.mean(), w.std(), w.min(), w.max(), dif, nivel), flush=True)


print("\n--- Métricas (unidades normalizadas [0,1], ventanas recortadas a 48) ---")
resumen("real", real)
resumen("sintetico", synth)

esp_r = np.abs(np.fft.rfft(real, axis=1)).mean(0)
esp_s = np.abs(np.fft.rfft(synth, axis=1)).mean(0)
eps = 1e-12
lsd = np.sqrt(np.mean((np.log10(esp_r + eps) - np.log10(esp_s + eps)) ** 2))
print("Distancia log-espectral (décadas): %.4f" % lsd, flush=True)
print("Perdida final: %.3f" % float(losses[-1]))
print("Tiempo de entrenamiento: %.0f s" % t_train, flush=True)

np.savez(str(ROOT / "results" / "exploratorios" / "06_ff48_prueba" / ("ff48_%s_out.npz" % args.canal)),
         real=real.astype(np.float32), synth=synth.astype(np.float32),
         xmin=xmin, xmax=xmax)
