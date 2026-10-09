import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from fourier.transforms import DFT
from SequentialFlows import FourierFlow

T = 24
t = np.arange(T + 1)
X = [np.sin(0.3 * t + np.random.uniform(0, 6.28)) for _ in range(500)]

# 1) ¿Hay componentes espectrales con std = 0?
X_t = torch.from_numpy(np.array(X)).float()
S = DFT(N_fft=T + 1)(X_t)[0]
print("forma del espectro:", S.shape)
print("hay nan en el espectro:", torch.isnan(S).any().item())
print("std mínima:", S.std(0).min().item())
print("componentes con std == 0:", (S.std(0) == 0).sum().item())

# 2) ¿Funciona sin normalizar?
model = FourierFlow(hidden=50, fft_size=T + 1, n_flows=3, normalize=False)
losses = model.fit(X, epochs=200, display_step=50)