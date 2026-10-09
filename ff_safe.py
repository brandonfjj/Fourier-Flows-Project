import numpy as np
import torch
from SequentialFlows import FourierFlow


class FourierFlowSafe(FourierFlow):
    """Igual que FourierFlow, pero evita std=0 al normalizar el espectro."""

    def fit(self, X, epochs=500, batch_size=128, learning_rate=1e-3,
            display_step=100, eps=1e-6):

        X_train = torch.from_numpy(np.array(X)).float()

        X_train_spectral = self.FourierTransform(X_train)[0]
        self.fft_mean = torch.mean(X_train_spectral, dim=0)
        # ÚNICO CAMBIO respecto al original: piso mínimo a la std
        self.fft_std = torch.clamp(torch.std(X_train_spectral, dim=0), min=eps)

        self.d = X_train.shape[1]
        self.k = int(np.floor(X_train.shape[1] / 2))

        optim = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, 0.999)

        losses = []
        for step in range(epochs):
            optim.zero_grad()
            z, log_pz, log_jacob = self(X_train)
            loss = (-log_pz - log_jacob).mean()
            losses.append(loss.detach().numpy())
            loss.backward()
            optim.step()
            scheduler.step()

            if (step % display_step == 0) or (step == epochs - 1):
                print("step: %d / %d \t loss: %.3f" % (step, epochs, loss))

        print("Finished training!")
        return losses