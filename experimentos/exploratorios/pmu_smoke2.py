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

# 1) ¿Qué hay en las etiquetas y qué pasa en la ventana rara (muestras 2000-2333)?
print(df[["EstadoEtiqueta1", "Etiqueta"]].drop_duplicates())
print(df["Etiqueta"].value_counts().sort_index())
print("Etiquetas en la ventana rara:", df["Etiqueta"].iloc[2000:2333].unique())

# 2) Solo la etiqueta más frecuente (normalmente la operación nominal)
ETQ = df["Etiqueta"].value_counts().idxmax()
print("Usando etiqueta:", ETQ)

CANAL, WIN, STRIDE = "BUS10Va", 333, 10
x = df[CANAL].values.astype(np.float64)
lab = df["Etiqueta"].values
xmin, xmax = x.min(), x.max()
xn = (x - xmin) / (xmax - xmin + 1e-7)

# solo ventanas completamente dentro de esa etiqueta
X = [xn[i:i + WIN] for i in range(0, len(xn) - WIN, STRIDE)
     if (lab[i:i + WIN] == ETQ).all()]
print("Ventanas:", len(X))

model = FourierFlowSafe(hidden=200, fft_size=WIN, n_flows=3, normalize=True)
losses = model.fit(X, epochs=1500, learning_rate=1e-4, display_step=250)
S = model.sample(len(X))

real = np.array(X) * (xmax - xmin) + xmin
synth = S * (xmax - xmin) + xmin

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
ax[0].plot(losses); ax[0].set_title("Loss")
ax[1].plot(real[0], label="Real"); ax[1].plot(synth[0], label="Sintética")
ax[1].legend(); ax[1].set_title("Una de cada (no son pares)")
ax[2].semilogy(np.abs(np.fft.rfft(real, axis=1)).mean(0), label="Real")
ax[2].semilogy(np.abs(np.fft.rfft(synth, axis=1)).mean(0), label="Sintético")
ax[2].legend(); ax[2].set_title("Espectro medio |FFT|")
plt.savefig(str(ROOT / "results" / "pmu_smoke2.png"), dpi=120)