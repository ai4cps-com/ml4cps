from __future__ import annotations

import os
import warnings
import torch

from ml4cps import examples
from ml4cps import vis
from ml4cps.debta.vis import plot_learning_curve
from ml4cps.debta.rbms.bb import BinaryBinaryRBM
from ml4cps.debta.rbms import training
from ml4cps.tools import check_latent_purity

try:
    from torchvision import datasets, transforms
except ModuleNotFoundError:
    warnings.warn("Torchvision not installed")


if __name__ == "__main__":
    torch.manual_seed(0) # For reproducibility
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, valid_loader, test_loader = examples.mnist()
    if os.path.exists("bbrbm_mnist.pt"):
        model = BinaryBinaryRBM.load("bbrbm_mnist.pt")
    else:
        model = BinaryBinaryRBM(
            n_visible=28 * 28,
            n_hidden=64,
            device=device
        )
        training.train_rbm(
            model,
            train_loader,
            valid_data=valid_loader,
            loss_name="cd",
            min_epoch=0,
            max_epoch=100,
            weight_decay=0.0,
            lr=0.001,
            shuffle=True,
            num_gibbs=5,
            verbose=1,
            early_stopping=False,
            early_stopping_patience=3,
            use_probability_last_v_update=False,
            log_mlflow=False,
            optimizer="RMSprop"
        )
        model.save(filename="bbrbm_mnist.pt")

    plot_learning_curve(model).show("browser")

    samples = model.generate(num_examples=16, num_steps=500)
    samples = samples.detach().cpu().view(-1, 28, 28)

    vis.plot_images(samples).show("browser")

    train_purity, _, _ = check_latent_purity(model, train_loader)
    test_purity, _, _ = check_latent_purity(model, test_loader)

    print(f"Train purity: {train_purity:.4f}")
    print(f"Test purity:  {test_purity:.4f}")
