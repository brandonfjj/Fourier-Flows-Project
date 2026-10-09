import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv(str(ROOT / "data" / "PMU_AlexyDonald2.csv"))
df.columns = df.columns.str.strip()
lab = df["Etiqueta"].values

cambios = np.flatnonzero(np.diff(lab)) + 1
bordes = np.r_[0, cambios, len(lab)]
runs = np.diff(bordes)
vals = lab[bordes[:-1]]

print("Número de cambios de etiqueta:", len(cambios))
for v, r in list(zip(vals, runs))[:15]:
    print("etiqueta", v, "dura", r, "muestras")
print("Racha más larga con etiqueta 0:", runs[vals == 0].max())
print("Racha más larga con etiqueta 1:", runs[vals == 1].max())

fig, ax = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
ax[0].plot(df["BUS10Va"].values[:6000])
ax[0].set_title("BUS10Va (primeras 6000 muestras)")
ax[1].plot(lab[:6000]); ax[1].set_title("Etiqueta")
plt.savefig(str(ROOT / "results" / "diag_labels.png"), dpi=120)