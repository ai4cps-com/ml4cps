from __future__ import annotations

import os

import torch

from ml4cps import examples
from ml4cps import vis
from ml4cps.debta.vis import plot_learning_curve
from ml4cps.tools import check_latent_purity
from ml4cps.debta import DEBTA


if __name__ == "__main__":
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DEBTA(num_y=28*28, num_h=10, num_sigm_layers=2, first_hidden_size=64, sigma=0.1, device=device)

    train_loader, valid_loader, test_loader = examples.mnist()

    if os.path.exists("debta_mnist.pt"):
        model = DEBTA.load("debta_mnist.pt")
    else:
        model.pretrain_layers(train_loader,
                              valid_data=valid_loader,
                              loss_name='cd',
                              lr=[0.001, 0.01],
                              verbose=True,
                              batch_size=128,
                              min_epoch=0,
                              max_epoch=2,
                              num_gibbs=10,
                              early_stopping=False,
                              early_stopping_patience=3,
                              use_probability_last_v_update=True,
                              log_mlflow=False,
                              optimizer="RMSprop"
                              )
        model.save(filename="debta_mnist.pt")

    fig = plot_learning_curve(model)
    fig.show("browser")

    batch, _ = next(iter(test_loader))
    x = batch.to(model.device)

    samples = model.ebm.generate(num_examples=16, num_steps=500)
    samples = samples.detach().cpu().view(-1, 28, 28)

    vis.plot_images(samples).show("browser")

    train_purity, _, _ = check_latent_purity(model, train_loader)
    test_purity, _, _ = check_latent_purity(model, test_loader)

    print(f"Train purity: {train_purity:.4f}")
    print(f"Test purity:  {test_purity:.4f}")
