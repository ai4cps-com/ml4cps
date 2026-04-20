import time as tm
import mlflow
import numpy as np
from datetime import timedelta
import pprint
import torch
from torch.utils.data import DataLoader

from denta.denta_tools import mse, sparsity, bce
from denta.rbms.base import XBinaryRBM


def train_rbm(
        rbm,
        train_data,
        valid_data=None,
        loss_name="cd",
        min_epoch=0,
        max_epoch=10,
        weight_decay=0.0,
        lr=0.01,
        batch_size=128,
        shuffle=True,
        num_gibbs=1,
        verbose=True,
        early_stopping=False,
        early_stopping_patience=3,
        use_probability_last_v_update=True,
        log_mlflow=False,
        optimizer="RMSprop",
):
    """
    Train the RBM using one of the supported losses.

    Parameters
    ----------
    rbm: XBinaryRBM
    train_data : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] | torch.utils.DataLoader
        Training data. Lists/tuples are vertically stacked.
    valid_data : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] | None, optional
        Validation data.
    loss_name : str, optional
        Training objective. Supported values:
        - ``"cd"``
        - ``"pcd"``
        - ``"sm"``
        - ``"recon"``
    min_epoch : int, optional
        Minimum epoch before early stopping is allowed.
    max_epoch : int, optional
        Maximum number of epochs.
    weight_decay : float, optional
        Weight decay used by the optimizer.
    lr : float, optional
        Learning rate.
    batch_size : int, optional
        Minibatch size.
    shuffle : bool, optional
        Whether to shuffle training data.
    num_gibbs : int, optional
        Number of Gibbs steps used by CD/PCD.
    verbose : bool | int, optional
        Verbosity level. If > 1, prints batch-level stats.
    early_stopping : bool, optional
        Whether to use early stopping on validation MSE.
    early_stopping_patience : int, optional
        Patience window for early stopping.
    use_probability_last_v_update : bool, optional
        Whether to use visible probabilities in the final negative step
        for CD/PCD.
    log_mlflow : bool, optional
        Whether to log selected hyperparameters to MLflow.
    optimizer : str, optional
        Optimizer type. Supported values:
        - ``"SGD"``
        - ``"RMSprop"``

    Notes
    -----
    Validation and progress tracking are based on reconstruction metrics and
    free energy, regardless of the training loss.
    """
    if log_mlflow:
        mlflow.log_param("min_epoch", min_epoch)
        mlflow.log_param("max_epoch", max_epoch)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("shuffle", shuffle)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("early_stopping_patience", early_stopping_patience)
        mlflow.log_param("early_stopping", early_stopping)
        mlflow.log_param("lr", lr)
        mlflow.log_param("num_gibbs", num_gibbs)
        mlflow.log_param("optimizer", optimizer)
        mlflow.log_param("loss_name", loss_name)

    if isinstance(train_data, DataLoader):
        data_loader = train_data
    else:
        if isinstance(train_data, (list, tuple)):
            train_data = torch.vstack(train_data)

        train_data = rbm.prepare_input(train_data).float().to(rbm.device)
        data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle, drop_last=loss_name == "pcd")

    if valid_data is not None:
        if isinstance(valid_data, DataLoader):
            valid_data = next(iter(valid_data))
            if isinstance(valid_data, (tuple, list)):
                valid_data = valid_data[0]
        elif isinstance(valid_data, (list, tuple)):
            valid_data = torch.vstack(valid_data)
        valid_data = rbm.prepare_input(valid_data).float().to(rbm.device)

    train_batch = next(iter(data_loader))
    if isinstance(train_batch, (tuple, list)):
        train_batch = train_batch[0]
    train_batch = rbm.prepare_input(train_batch).float().to(rbm.device)

    if optimizer == "SGD":
        opt = torch.optim.SGD(rbm.parameters(), weight_decay=weight_decay, lr=lr)
    elif optimizer == "RMSprop":
        opt = torch.optim.RMSprop(rbm.parameters(), weight_decay=weight_decay, lr=lr)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {optimizer}")

    if loss_name == "pcd":
        rbm.reset_persistent_chain()

    t_start = tm.time()

    for epoch in range(1, max_epoch + 1):
        if verbose:
            print(f"Epoch {epoch} started...")

        rbm.train()
        for i, batch in enumerate(data_loader):
            if isinstance(batch, list):
                batch = batch[0]

            x0 = batch.to(rbm.device)

            opt.zero_grad()
            loss = rbm.compute_loss(
                x0,
                loss_name=loss_name,
                num_gibbs=num_gibbs,
                use_probability_last_v_update=use_probability_last_v_update,
            )
            loss.backward()
            opt.step()

            if verbose and verbose > 1:
                with torch.no_grad():
                    progress = _get_progress_rbm(rbm, train_batch)
                    print(f"\n############### BATCH {i} ###############")
                    print("Train:")
                    pprint.pp(progress)

        rbm.eval()

        with torch.no_grad():
            train_progress = _get_progress_rbm(rbm, train_batch)
            rbm.learning_curve.append(train_progress)

            if verbose:
                print(f"\n############### Epoch {epoch} ###############")
                print("Train:")
                pprint.pp(train_progress)

        if valid_data is not None:
            with torch.no_grad():
                valid_progress = _get_progress_rbm(rbm, valid_data)
                rbm.valid_curve.append(valid_progress)

            if verbose:
                print("Valid:")
                pprint.pp(valid_progress)

            if (
                    early_stopping
                    and epoch > min_epoch
                    and epoch > early_stopping_patience
            ):
                valid_metrics = np.array(
                    [v["MSE"] for v in rbm.valid_curve[-early_stopping_patience - 1:]]
                )
                if np.all(valid_metrics[1:] > valid_metrics[0]):
                    print("Early stop after valid metrics:", valid_metrics)
                    break

    rbm.eval()
    rbm.num_epoch = epoch
    print("Training finished after", timedelta(seconds=tm.time() - t_start))
    
    
def train_3way_rbm(
        rbm,
        train_data,
        valid_data=None,
        min_epoch: int = 0,
        max_epoch: int = 10,
        weight_decay: float = 0.0,
        lr: float = 0.01,
        batch_size: int = 128,
        shuffle: bool = True,
        num_gibbs: int = 1,
        verbose=True,
        early_stopping: bool = False,
        early_stopping_patience: int = 3,
        use_probability_last_v_update: bool = True,
        log_mlflow: bool = False,
        optimizer: str = "RMSprop",
):
    """
    Train the Three-Way Binary-Binary RBM.

    Parameters
    ----------
    train_data : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor]
        Training data containing both visible and auxiliary parts.
        If a list or tuple is given, tensors are stacked vertically.
    valid_data : torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor] | None, optional
        Validation data in the same format as ``train_data``.
    min_epoch : int, optional
        Minimum number of epochs before early stopping is allowed.
    max_epoch : int, optional
        Maximum number of epochs.
    weight_decay : float, optional
        Weight decay coefficient for the optimizer.
    lr : float, optional
        Learning rate.
    batch_size : int, optional
        Mini-batch size.
    shuffle : bool, optional
        Whether to shuffle the training data.
    num_gibbs : int, optional
        Number of Gibbs steps used in Contrastive Divergence.
    verbose : bool | int, optional
        If truthy, prints epoch-level progress. If greater than 1, also prints
        batch-level progress.
    early_stopping : bool, optional
        Whether to enable validation-based early stopping.
    early_stopping_patience : int, optional
        Number of epochs used in the early-stopping window.
    use_probability_last_v_update : bool, optional
        If ``True``, the final visible negative sample in CD uses visible
        probabilities instead of sampled states.
    log_mlflow : bool, optional
        Whether to log selected hyperparameters to MLflow.
    optimizer : str, optional
        Optimizer name. Supported values are ``"SGD"`` and ``"RMSprop"``.

    Returns
    -------
    None

    Notes
    -----
    This trainer is specialized for the three-way RBM and does not rely on the
    generic :class:`XBinaryRBM` training logic, because this model has a joint
    visible/auxiliary state and free energy of the form ``F(v, u)``.

    Early stopping is based on the validation visible reconstruction MSE.
    """
    if log_mlflow:
        mlflow.log_param("min_epoch", min_epoch)
        mlflow.log_param("max_epoch", max_epoch)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("shuffle", shuffle)
        mlflow.log_param("weight_decay", weight_decay)
        mlflow.log_param("early_stopping_patience", early_stopping_patience)
        mlflow.log_param("early_stopping", early_stopping)
        mlflow.log_param("lr", lr)
        mlflow.log_param("num_gibbs", num_gibbs)
        mlflow.log_param("optimizer", optimizer)

    if isinstance(train_data, (list, tuple)):
        train_data = torch.vstack(train_data)
    train_data = train_data.float().to(rbm.device)

    if valid_data is not None:
        if isinstance(valid_data, (list, tuple)):
            valid_data = torch.vstack(valid_data)
        valid_data = valid_data.float().to(rbm.device)

    data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

    if optimizer == "SGD":
        opt = torch.optim.SGD(rbm.parameters(), weight_decay=weight_decay, lr=lr)
    elif optimizer == "RMSprop":
        opt = torch.optim.RMSprop(rbm.parameters(), weight_decay=weight_decay, lr=lr)
    else:
        raise NotImplementedError(f"Unsupported optimizer: {optimizer}")

    t_start = tm.time()

    for epoch in range(1, max_epoch + 1):
        if verbose:
            print(f"Epoch {epoch} started...")

        rbm.train()
        for i, batch in enumerate(data_loader):
            batch = batch.to(rbm.device)

            opt.zero_grad()
            loss = rbm.contrastive_divergence(
                batch,
                num_gibbs=num_gibbs,
                use_probability_last_v_update=use_probability_last_v_update,
            )
            loss.backward()
            opt.step()

            if verbose and verbose > 1:
                with torch.no_grad():
                    progress = _get_progress_rbm(rbm, train_data)
                    print(f"\n############### BATCH {i} ###############")
                    print("Train:")
                    pprint.pp(progress)

        rbm.eval()

        with torch.no_grad():
            train_progress = _get_progress_rbm(rbm, train_data)
            rbm.learning_curve.append(train_progress)

            if verbose:
                print(f"\n############### Epoch {epoch} ###############")
                print("Train:")
                pprint.pp(train_progress)

        if valid_data is not None:
            with torch.no_grad():
                valid_progress = _get_progress_rbm(rbm, valid_data)
                rbm.valid_curve.append(valid_progress)

            if verbose:
                print("Valid:")
                pprint.pp(valid_progress)

            if (
                    early_stopping
                    and epoch > min_epoch
                    and epoch > early_stopping_patience
            ):
                valid_metrics = np.array(
                    [v["MSE"] for v in rbm.valid_curve[-early_stopping_patience - 1:]]
                )
                if np.all(valid_metrics[1:] > valid_metrics[0]):
                    print("Early stop after valid metrics:", valid_metrics)
                    break

    rbm.eval()
    rbm.num_epoch = epoch
    print("Training finished after", timedelta(seconds=tm.time() - t_start))
    
    
def _get_progress_rbm(rbm, d: torch.Tensor):
    """
    Compute simple progress metrics for logging.

    Parameters
    ----------
    d : torch.Tensor
        Dataset or minibatch.

    Returns
    -------
    dict
        Dictionary of summary metrics.

    Notes
    -----
    Assumes external helper functions ``mse(...)`` and ``sparsity(...)`` are
    available in scope.
    """
    with torch.no_grad():
        r, h = rbm.recon(d)
        e = rbm.free_energy(d)

        progress = dict(
            MSE=mse(d.view(d.shape[0], -1), r).item(),
            BCE=bce(d.view(d.shape[0], -1), r).item(),
            Sparsity=sparsity(h).item(),
            Energy=torch.mean(e).item(),
            Weights=torch.mean(torch.abs(rbm.weights)).item(),
            VisBias=torch.mean(torch.abs(rbm.visible_bias)).item(),
            HidBias=torch.mean(torch.abs(rbm.hidden_bias)).item(),
        )
    return progress