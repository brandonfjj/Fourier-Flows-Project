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
CANALES = ["BUS10Va", "BUS10Vb", "BUS10Vc"]
data = {c: df[c].values.astype(np.float64) for c in CANALES}
ref = data["BUS10Va"]
N = len(ref)

# Zona sana: sin la falla (2001-4082) ni la sobretensión (~10800-13200), con margen
sana = np.ones(N, dtype=bool)
sana[2001 - 500:4082 + 500] = False
sana[10500:13800] = False

WIN, JITTER = 401, 20
cruces = np.flatnonzero((ref[:-1] < 0) & (ref[1:] >= 0)) + 1
starts = [c + d for c in cruces for d in range(-JITTER, JITTER + 1)
          if c + d >= 0 and c + d + WIN <= N and sana[c + d:c + d + WIN].all()]
print("Ventanas:", len(starts))

# Un flujo por canal, mismas ventanas (mismos instantes) para los tres
real, synth = {}, {}
for c in CANALES:
    x = data[c]
    xmin, xmax = x[sana].min(), x[sana].max()
    xn = (x - xmin) / (xmax - xmin + 1e-7)
    X = [xn[i:i + WIN] for i in starts]
    print("== Entrenando", c)
    model = FourierFlowSafe(hidden=200, fft_size=WIN, n_flows=3, normalize=True)
    model.fit(X, epochs=2000, learning_rate=1e-3, display_step=500)
    S = model.sample(len(X))
    real[c] = np.array(X) * (xmax - xmin) + xmin
    synth[c] = S * (xmax - xmin) + xmin

np.savez(str(ROOT / "results" / "pmu_3fases_out.npz"),
         **{"real_" + c: real[c] for c in CANALES},
         **{"synth_" + c: synth[c] for c in CANALES})

def rms(w): return np.sqrt((w ** 2).mean(1))
def f1(w): return np.fft.rfft(w, axis=1)[:, 1]
def dif(a, b): return np.degrees(np.angle(f1(b) * np.conj(f1(a))))  # fase(b) - fase(a)

print("\n--- Por canal ---")
for c in CANALES:
    ar, as_ = np.abs(f1(real[c])), np.abs(f1(synth[c]))
    print("%s: RMS real %.0f +- %.0f | sint %.0f +- %.0f | razon std %.1f | "
          "std/media |bin1| real %.3f sint %.3f"
          % (c, rms(real[c]).mean(), rms(real[c]).std(),
             rms(synth[c]).mean(), rms(synth[c]).std(),
             rms(synth[c]).std() / rms(real[c]).std(),
             ar.std() / ar.mean(), as_.std() / as_.mean()))

print("\n--- Relación entre fases (lo que se pierde al modelar por separado) ---")
for otra in ["BUS10Vb", "BUS10Vc"]:
    dr = dif(real["BUS10Va"], real[otra])
    ds = dif(synth["BUS10Va"], synth[otra])   # pareja arbitraria: muestras independientes
    print("Desfase %s - Va: real %.1f +- %.1f grados | sint %.1f +- %.1f grados"
          % (otra, dr.mean(), dr.std(), ds.mean(), ds.std()))

sr = real["BUS10Va"] + real["BUS10Vb"] + real["BUS10Vc"]
ss = synth["BUS10Va"] + synth["BUS10Vb"] + synth["BUS10Vc"]
print("RMS(a+b+c)/RMS(a): real %.3f | sint %.3f"
      % (rms(sr).mean() / rms(real["BUS10Va"]).mean(),
         rms(ss).mean() / rms(synth["BUS10Va"]).mean()))

fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for k, otra in enumerate(["BUS10Vb", "BUS10Vc"]):
    ax[k].hist(dif(real["BUS10Va"], real[otra]), bins=40, alpha=0.6, label="Real")
    ax[k].hist(dif(synth["BUS10Va"], synth[otra]), bins=40, alpha=0.6, label="Sintético")
    ax[k].set_title("Desfase %s - Va (grados)" % otra); ax[k].legend()
for c in CANALES:
    ax[2].plot(synth[c][0], label=c)
ax[2].set_title("Una muestra sintética (3 canales)"); ax[2].legend()
plt.tight_layout(); plt.savefig(str(ROOT / "results" / "pmu_3fases.png"), dpi=120)
print("Guardado pmu_3fases.png")