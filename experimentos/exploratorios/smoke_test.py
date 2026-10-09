import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
from ff_safe import FourierFlowSafe

T = 24
t = np.arange(T + 1)
# 500 senoides con fase aleatoria, longitud T+1 como hace el repo
X = [np.sin(0.3 * t + np.random.uniform(0, 6.28)) for _ in range(500)]

model = FourierFlowSafe(hidden=50, fft_size=T + 1, n_flows=3, normalize=True)
losses = model.fit(X, epochs=200, display_step=50)

S = model.sample(5)
print("Forma de las muestras:", S.shape)