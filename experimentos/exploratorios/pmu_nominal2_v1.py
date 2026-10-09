import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ff_safe import FourierFlowSafe

df = pd.read_csv(str(ROOT / "data" / "PMU_AlexyDonald2.csv"))
df.columns = df.columns.str.strip()
lab = df["Etiqueta"].values
CANAL = "BUS10Va"
x = df[CANAL].values.astype(np.float64)

# Zona sana: todo menos el bloque de falla (2001, 4082) y un margen
MARGEN = 500
sana = np.ones(len(x), dtype=bool)
sana[2001 - MARGEN:4082 + MARGEN] = False

# 0) Prueba de la hipótesis: ¿Etiqueta = 1 cuando |v| cae bajo un umbral?
for c in ["BUS10Va", "BUS10Vb", "BUS10Vc", "BUS14Va", "BUS14Vb", "BUS14Vc"]:
    v = np.abs(df[c].values[sana])
    l = lab[sana]
    pico = v.max()
    print("%s: max|v| con etiqueta 1 = %.3f del pico | min|v| con etiqueta 0 = %.3f del pico"
          % (c, v[l == 1].max() / pico, v[l == 0].min() / pico))
# Si en algún canal el primer número es menor que el segundo, la etiqueta
# separa limpio por umbral de |v| en ese canal.

# 1) Ventanas de 1 ciclo (+1 muestra), solo en zona sana
WIN, STRIDE = 401, 10
xmin, xmax = x.min(), x.max()
xn = (x - xmin) / (xmax - xmin + 1e-7)
X = [xn[i:i + WIN] for i in range(0, len(xn) - WIN, STRIDE)
     if sana[i:i + WIN].all()]
print("Ventanas sanas:", len(X))

# 2) Entrenar (misma configuración de antes; solo cambió la ventana)
model = FourierFlowSafe(hidden=200, fft_size=WIN, n_flows=3, normalize=True)
losses = model.fit(X, epochs=1500, learning_rate=1e-4, display_step=250)
S = model.sample(len(X))

real = np.array(X) * (xmax - xmin) + xmin
synth = S * (xmax - xmin) + xmin

# 3) Métricas numéricas
def rms(w): return np.sqrt((w ** 2).mean(1))
def pico(w): return np.abs(w).max(1)
print("RMS  real %.0f +- %.0f | sint %.0f +- %.0f"
      % (rms(real).mean(), rms(real).std(), rms(synth).mean(), rms(synth).std()))
print("Pico real %.0f +- %.0f | sint %.0f +- %.0f"
      % (pico(real).mean(), pico(real).std(), pico(synth).mean(), pico(synth).std()))
mr = np.abs(np.fft.rfft(real, axis=1)).mean(0)
ms = np.abs(np.fft.rfft(synth, axis=1)).mean(0)
print("Error relativo medio del espectro medio: %.3f" % (np.abs(ms - mr).mean() / mr.mean()))

# 4) Gráficas
fig, ax = plt.subplots(2, 2, figsize=(13, 8))
ax[0, 0].plot(losses); ax[0, 0].set_title("Loss")
for k in range(5):
    ax[0, 1].plot(real[k * 100], alpha=0.7)
    ax[1, 0].plot(synth[k], alpha=0.7)
ax[0, 1].set_title("5 ventanas reales"); ax[1, 0].set_title("5 sintéticas")
ax[1, 1].semilogy(mr, label="Real"); ax[1, 1].semilogy(ms, label="Sintético")
ax[1, 1].legend(); ax[1, 1].set_title("Espectro medio |FFT|")
plt.tight_layout(); plt.savefig(str(ROOT / "results" / "pmu_nominal2_v1.png"), dpi=120)
print("Guardado pmu_nominal2.png")