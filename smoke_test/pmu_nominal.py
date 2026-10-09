import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ff_safe import FourierFlowSafe

df = pd.read_csv("PMU_AlexyDonald2.csv")
df.columns = df.columns.str.strip()
lab = df["Etiqueta"].values
CANAL, FS = "BUS10Va", 20000
x = df[CANAL].values.astype(np.float64)

# 1) Bloque largo de falla = racha larga con etiqueta 1
cambios = np.flatnonzero(np.diff(lab)) + 1
bordes = np.r_[0, cambios, len(lab)]
runs = np.diff(bordes)
vals = lab[bordes[:-1]]
largas = [(bordes[i], bordes[i + 1]) for i in range(len(runs))
          if vals[i] == 1 and runs[i] > 1000]
print("Bloques largos de falla:", largas)

# 2) Máscara de operación sana: todo menos la falla y un margen alrededor
MARGEN = 500
sana = np.ones(len(x), dtype=bool)
for a, b in largas:
    sana[max(0, a - MARGEN):b + MARGEN] = False

# 3) Frecuencia fundamental, usando el tramo sano contiguo más largo
idx = np.flatnonzero(sana)
cortes = np.flatnonzero(np.diff(idx) > 1) + 1
tramos = np.split(idx, cortes)
mayor = max(tramos, key=len)
seg = x[mayor[0]:mayor[-1] + 1]
seg = seg - seg.mean()
espectro = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=8 * len(seg)))
f0 = np.fft.rfftfreq(8 * len(seg), 1 / FS)[espectro.argmax()]
print("Tramo sano más largo: %d muestras" % len(seg))
print("Frecuencia fundamental estimada: %.1f Hz -> %.1f muestras/ciclo" % (f0, FS / f0))

# 4) Normalizar y ventanear solo dentro de la zona sana
WIN, STRIDE = 333, 10
xmin, xmax = x.min(), x.max()
xn = (x - xmin) / (xmax - xmin + 1e-7)
X = [xn[i:i + WIN] for i in range(0, len(xn) - WIN, STRIDE)
     if sana[i:i + WIN].all()]
print("Ventanas sanas:", len(X))

# 5) Entrenar y comparar
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
plt.savefig("pmu_nominal.png", dpi=120)
print("Guardado pmu_nominal.png")