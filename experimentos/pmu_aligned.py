import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from ff_safe import FourierFlowSafe

df = pd.read_csv(str(ROOT / "data" / "PMU_AlexyDonald2.csv"))
df.columns = df.columns.str.strip()
x = df["BUS10Va"].values.astype(np.float64)

MARGEN = 500
sana = np.ones(len(x), dtype=bool)
sana[2001 - MARGEN:4082 + MARGEN] = False

WIN = 401
JITTER = 20   # +- muestras alrededor del cruce por cero

# Ventanas que empiezan cerca de un cruce por cero ascendente
cruces = np.flatnonzero((x[:-1] < 0) & (x[1:] >= 0)) + 1
starts = []
for c in cruces:
    for d in range(-JITTER, JITTER + 1):
        i = c + d
        if i >= 0 and i + WIN <= len(x) and sana[i:i + WIN].all():
            starts.append(i)
print("Cruces:", len(cruces), "| ventanas:", len(starts))

xmin, xmax = x.min(), x.max()
xn = (x - xmin) / (xmax - xmin + 1e-7)
X = [xn[i:i + WIN] for i in starts]

model = FourierFlowSafe(hidden=200, fft_size=WIN, n_flows=3, normalize=True)
losses = model.fit(X, epochs=2000, learning_rate=1e-3, display_step=250)
S = model.sample(len(X))

real = np.array(X) * (xmax - xmin) + xmin
synth = S * (xmax - xmin) + xmin
np.savez(str(ROOT / "results" / "pmu_aligned_out.npz"), real=real, synth=synth)

def rms(w): return np.sqrt((w ** 2).mean(1))
fr = np.fft.rfft(real, axis=1)[:, 1]
fs_ = np.fft.rfft(synth, axis=1)[:, 1]
print("RMS  real %.0f +- %.0f | sint %.0f +- %.0f"
      % (rms(real).mean(), rms(real).std(), rms(synth).mean(), rms(synth).std()))
print("Razon std(RMS) sint/real: %.1f" % (rms(synth).std() / rms(real).std()))
print("std/media de |bin 1|  real: %.3f | sint: %.3f"
      % (np.abs(fr).std() / np.abs(fr).mean(), np.abs(fs_).std() / np.abs(fs_).mean()))
print("Fase de bin 1 (grados)  real: %.0f +- %.0f | sint: %.0f +- %.0f"
      % (np.degrees(np.angle(fr)).mean(), np.degrees(np.angle(fr)).std(),
         np.degrees(np.angle(fs_)).mean(), np.degrees(np.angle(fs_)).std()))

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].scatter(fr.real, fr.imag, s=4, alpha=0.5, label="Real")
ax[0].scatter(fs_.real, fs_.imag, s=4, alpha=0.5, label="Sintético")
ax[0].set_aspect("equal"); ax[0].legend(); ax[0].set_title("Fundamental: Re vs Im")
ax[1].hist(np.abs(fr), bins=40, alpha=0.6, label="Real")
ax[1].hist(np.abs(fs_), bins=40, alpha=0.6, label="Sintético")
ax[1].legend(); ax[1].set_title("Amplitud de la fundamental")
for k in range(5):
    ax[2].plot(synth[k], alpha=0.7)
ax[2].set_title("5 sintéticas")
plt.tight_layout(); plt.savefig(str(ROOT / "results" / "pmu_aligned.png"), dpi=120)
print("Guardado pmu_aligned.png")