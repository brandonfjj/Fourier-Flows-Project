import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

df = pd.read_csv(str(ROOT / "data" / "PMU_AlexyDonald2.csv"))
df.columns = df.columns.str.strip()
x = df["BUS10Va"].values.astype(np.float64)

picos = np.array([np.abs(x[a:a + 400]).max() for a in range(0, len(x) - 400, 400)])
med = np.median(picos)
print("Pico mediano por ciclo: %.0f" % med)
for k, p in enumerate(picos):
    if abs(p - med) / med > 0.03:
        print("muestras %d-%d: pico %.0f (%+.1f%%)" % (k * 400, k * 400 + 400, p, 100 * (p - med) / med))