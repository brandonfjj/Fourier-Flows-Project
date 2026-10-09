import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # WSL sin pantalla: guardamos a PNG
import matplotlib.pyplot as plt
from ff_safe import FourierFlowSafe

CSV = "PMU_AlexyDonald2.csv"   
CANAL = "BUS10Va"
WIN = 333      # impar (ver nota abajo); ~1 ciclo a 60 Hz si fs = 20 kHz
STRIDE = 10    # paso entre ventanas (menos solapamiento, menos memoria)

# 1) Carga. Los encabezados traen espacios (" BUS10Va"), hay que limpiarlos
df = pd.read_csv(CSV)
df.columns = df.columns.str.strip()
print("Columnas:", list(df.columns)[:5], "...")
dt = df["Time"].diff().median()
print("dt mediano: %.3e s  ->  fs = %.1f Hz" % (dt, 1.0 / dt))

x = df[CANAL].values.astype(np.float64)

# 2) MinMax del canal (guardamos min/max para desnormalizar)
xmin, xmax = x.min(), x.max()
xn = (x - xmin) / (xmax - xmin + 1e-7)

# 3) Ventanas
X = [xn[i:i + WIN] for i in range(0, len(xn) - WIN, STRIDE)]
print("Ventanas:", len(X), "de largo", WIN)

# 4) Entrenar y muestrear
model = FourierFlowSafe(hidden=200, fft_size=WIN, n_flows=3, normalize=True)
losses = model.fit(X, epochs=300, display_step=50)
S = model.sample(200)
print("Forma de muestras:", S.shape, " nan:", np.isnan(S).any())

# 5) Desnormalizar y comparar visualmente
real = np.array(X) * (xmax - xmin) + xmin
synth = S * (xmax - xmin) + xmin

fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for k in range(5):
    ax[0].plot(real[k * 50])
    ax[1].plot(synth[k])
ax[0].set_title("Reales"); ax[1].set_title("Sintéticas")
ax[2].semilogy(np.abs(np.fft.rfft(real, axis=1)).mean(0), label="Real")
ax[2].semilogy(np.abs(np.fft.rfft(synth, axis=1)).mean(0), label="Sintético")
ax[2].set_title("Espectro medio |FFT|"); ax[2].legend()
plt.savefig("pmu_smoke.png", dpi=120)
print("Guardado pmu_smoke.png")