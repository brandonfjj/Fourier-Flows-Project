import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = np.load(str(ROOT / "results" / "pmu_nominal2_out.npz"))
real, synth = d["real"], d["synth"]
fr = np.fft.rfft(real, axis=1)[:, 1]
fs_ = np.fft.rfft(synth, axis=1)[:, 1]

fig, ax = plt.subplots(1, 3, figsize=(16, 5))
ax[0].scatter(fr.real, fr.imag, s=4, alpha=0.5, label="Real")
ax[0].scatter(fs_.real, fs_.imag, s=4, alpha=0.5, label="Sintético")
ax[0].set_aspect("equal"); ax[0].legend()
ax[0].set_title("Fundamental (bin 1): Re vs Im")
ax[1].hist(np.abs(fr), bins=40, alpha=0.6, label="Real")
ax[1].hist(np.abs(fs_), bins=40, alpha=0.6, label="Sintético")
ax[1].legend(); ax[1].set_title("Amplitud de la fundamental")
ax[2].hist(np.angle(fr), bins=40, alpha=0.6, label="Real")
ax[2].hist(np.angle(fs_), bins=40, alpha=0.6, label="Sintético")
ax[2].legend(); ax[2].set_title("Fase de la fundamental")
plt.tight_layout(); plt.savefig(str(ROOT / "results" / "ring_test.png"), dpi=120)
print("std/media de |bin 1|  real: %.3f | sint: %.3f"
      % (np.abs(fr).std() / np.abs(fr).mean(), np.abs(fs_).std() / np.abs(fs_).mean()))