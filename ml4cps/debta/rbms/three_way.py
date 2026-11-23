"""Full or factored three-way RBM with binary/Gaussian auxiliary context."""

import json
from numbers import Integral
from pathlib import Path

import numpy as np

import torch
from torch import nn

from .bb import BinaryBinaryRBM


class ThreeWayBinaryBinaryRBM(BinaryBinaryRBM):
    """Add ``v_i * T_ijk * h_j * u_k`` interactions to a binary RBM.

    The default stores the full interaction tensor ``T``. Supplying
    ``n_factors`` instead represents it as
    ``T_ija = sum_f visible_factors_if * hidden_factors_jf * auxiliary_factors_af``;
    no dense interaction tensor is allocated. Both parameterizations use the
    same conditional energy and Gibbs machinery. ``auxiliary_bias=False`` omits
    the context-only bias, useful for conditionally trained subclasses.

    Auxiliary units default to Bernoulli. ``auxiliary_types`` can mark selected
    coordinates as ``"gaussian"`` with fixed positive ``auxiliary_sigma``.
    Following GaussianBinaryRBM, interactions use ``u / sigma`` and the Gaussian
    energy is ``(u - auxiliary_bias)**2 / (2 * sigma**2)``. A scalar sigma applies
    only to Gaussian coordinates; explicit vectors must use 1 for binary units.
    Standardize real configuration values using training data before fitting.

    ``vu2h``, ``hu2v`` and ``free_energy(v, u)`` accept separate visible/context
    tensors. Training methods use feature-last concatenation ``[v, u]``. The
    historical ``d[:, :, -1]`` visible / ``d[:, :, :-1]`` auxiliary layout is
    accepted by ``prepare_input`` and ``split_visible_auxiliary`` when its
    flattened auxiliary width agrees with ``n_auxiliary``. Canonical feature-last
    layout takes precedence when shapes coincide (a one-visible-unit window).

    CD and PCD reconstruct only ``v``: the observed auxiliary context stays
    fixed. ``vh2u`` and unconditional ``generate`` remain available for sampling
    the full joint model, whose auxiliary marginal is not identified by
    conditional training. The inherited ordinary ``h2v(h)`` is the zero-context
    mapping; use ``hu2v(h, u)`` for a nonzero context.
    """

    def __init__(self, n_visible, n_hidden, n_auxiliary, device="cpu", name="",
                 *, n_factors=None, auxiliary_bias=True,
                 auxiliary_types=None, auxiliary_sigma=1.):
        if (isinstance(n_auxiliary, bool) or not isinstance(n_auxiliary, Integral)
                or n_auxiliary < 1):
            raise ValueError("n_auxiliary must be a positive integer")
        if n_factors is not None and (isinstance(n_factors, bool)
                or not isinstance(n_factors, Integral) or n_factors < 1):
            raise ValueError("n_factors must be a positive integer or None")
        if not isinstance(auxiliary_bias, bool):
            raise ValueError("auxiliary_bias must be a boolean")
        super().__init__(n_visible=n_visible, n_hidden=n_hidden, device=device, name=name)
        self.n_auxiliary = int(n_auxiliary)
        self.n_factors = None if n_factors is None else int(n_factors)
        types = ("binary",) * self.n_auxiliary if auxiliary_types is None else tuple(auxiliary_types)
        if len(types) != self.n_auxiliary or any(t not in ("binary", "gaussian") for t in types):
            raise ValueError("Provide one binary/gaussian auxiliary type per coordinate")
        mask = torch.tensor([t == "gaussian" for t in types], dtype=torch.bool, device=self.device)
        sigma = torch.as_tensor(auxiliary_sigma, dtype=self.weights.dtype, device=self.device)
        if sigma.ndim == 0:
            if not torch.isfinite(sigma) or sigma <= 0:
                raise ValueError("Auxiliary sigma must be finite and positive")
            sigma = torch.where(mask, sigma, self.weights.new_ones(self.n_auxiliary))
        elif sigma.shape != (self.n_auxiliary,) or not torch.all(sigma[~mask] == 1):
            raise ValueError("Provide one sigma per auxiliary unit, using 1 for binary units")
        if not torch.isfinite(sigma).all() or not torch.all(sigma > 0):
            raise ValueError("Auxiliary sigma must be finite and positive")
        self.register_buffer("_gaussian_auxiliary", mask)
        self.register_buffer("auxiliary_sigma", sigma.detach().clone())
        if self.n_factors is None:
            self.three_way_weights = nn.Parameter(
                self.weights.new_empty(n_visible, n_hidden, self.n_auxiliary).normal_(std=.01)
            )
        else:
            for factor_name, width in (("visible_factors", n_visible),
                                       ("hidden_factors", n_hidden),
                                       ("auxiliary_factors", self.n_auxiliary)):
                self.register_parameter(factor_name, nn.Parameter(
                    self.weights.new_empty(width, self.n_factors).normal_(std=.2)))
        self.register_parameter("auxiliary_bias", nn.Parameter(
            self.weights.new_zeros(self.n_auxiliary)) if auxiliary_bias else None)

    @property
    def auxiliary_types(self):
        return tuple("gaussian" if flag else "binary"
                     for flag in self._gaussian_auxiliary.detach().cpu().tolist())

    def prepare_auxiliary(self, auxiliary, *, scale=False):
        """Validate mixed-unit rows; optionally apply the Gaussian RBM scaling."""
        values = torch.atleast_2d(torch.as_tensor(auxiliary, dtype=self.weights.dtype, device=self.device))
        if (values.ndim != 2 or values.shape[1] != self.n_auxiliary
                or not torch.isfinite(values).all()):
            raise ValueError("Expected finite auxiliary rows matching the model width")
        binary = values[:, ~self._gaussian_auxiliary]
        if not torch.all((binary == 0) | (binary == 1)):
            raise ValueError("Binary auxiliary coordinates must be 0 or 1")
        return values / self.auxiliary_sigma if scale else values

    def prepare_input(self, data):
        """Return packed feature-last samples, preserving temporal windows."""
        if data.ndim in (2, 3) and data.shape[-1] == self.n_visible + self.n_auxiliary:
            return data
        if (data.ndim == 3 and data.shape[1] == self.n_visible
                and data.shape[1] * (data.shape[2] - 1) == self.n_auxiliary):
            return torch.cat((data[:, :, -1], data[:, :, :-1].reshape(data.shape[0], -1)), dim=-1)
        raise ValueError("Expected packed [visible, auxiliary] features or the matching legacy layout")

    def split_visible_auxiliary(self, data):
        packed = self.prepare_input(data).reshape(-1, self.n_visible + self.n_auxiliary)
        return packed[:, :self.n_visible], packed[:, self.n_visible:]

    def effective_weights(self, auxiliary):
        """Return the binary RBM weight matrix for each auxiliary context."""
        auxiliary = self.prepare_auxiliary(auxiliary, scale=True)
        if getattr(self, "n_factors", None) is None:
            interaction = torch.einsum("ba,ija->bij", auxiliary, self.three_way_weights)
        else:
            interaction = torch.einsum("bf,if,jf->bij", auxiliary @ self.auxiliary_factors,
                                       self.visible_factors, self.hidden_factors)
        return self.weights.unsqueeze(0) + interaction

    def three_way_hidden(self, visible, auxiliary, *, visible_factors=None):
        """Return only the three-way contribution to hidden logits.

        Factored subclasses can supply a second visible factor matrix, e.g.
        for a clamped source code, while sharing context and hidden factors.
        """
        visible = torch.atleast_2d(torch.as_tensor(
            visible, dtype=self.weights.dtype, device=self.device))
        auxiliary = self.prepare_auxiliary(auxiliary, scale=True)
        if visible.shape[-1] != self.n_visible:
            raise ValueError("Visible width differs from the model")
        visible = visible.reshape(-1, self.n_visible)
        if getattr(self, "n_factors", None) is None:
            if visible_factors is not None:
                raise ValueError("Visible factor overrides require a factored model")
            return torch.einsum("bi,ija,ba->bj", visible, self.three_way_weights, auxiliary)
        factors = self.visible_factors if visible_factors is None else visible_factors
        if factors.shape != (self.n_visible, self.n_factors):
            raise ValueError("Visible factor matrix has the wrong shape")
        return ((visible @ factors) * (auxiliary @ self.auxiliary_factors)) @ self.hidden_factors.T

    def vu2h(self, v, u, pre_sigmoid=False):
        v = torch.atleast_2d(v).reshape(-1, self.n_visible)
        logits = super().v2h(v[:, None, :], pre_sigmoid=True,
                             weights=self.effective_weights(u), hidden_bias=self.hidden_bias).squeeze(-2)
        return logits if pre_sigmoid else logits.sigmoid()

    def v2h(self, visible, pre_sigmoid=False, *, weights=None, hidden_bias=None):
        # BinaryBinaryRBM.free_energy delegates its affine kernel back through
        # this method with explicit effective weights; avoid reparsing that input.
        if weights is not None or hidden_bias is not None:
            return super().v2h(visible, pre_sigmoid=pre_sigmoid,
                               weights=weights, hidden_bias=hidden_bias)
        v, u = self.split_visible_auxiliary(visible)
        return self.vu2h(v, u, pre_sigmoid=pre_sigmoid)

    def hu2v(self, h, u):
        h = torch.atleast_2d(h).reshape(-1, self.n_hidden)
        return super().h2v(h[:, None, :], weights=self.effective_weights(u),
                           visible_bias=self.visible_bias).squeeze(-2)

    def vh2u(self, v, h):
        v = torch.atleast_2d(v).reshape(-1, self.n_visible)
        h = torch.atleast_2d(h).reshape(-1, self.n_hidden)
        if getattr(self, "n_factors", None) is None:
            logits = torch.einsum("bi,bj,ija->ba", v, h, self.three_way_weights)
        else:
            logits = ((v @ self.visible_factors) * (h @ self.hidden_factors)) @ self.auxiliary_factors.T
        bias = 0. if self.auxiliary_bias is None else self.auxiliary_bias
        return torch.where(self._gaussian_auxiliary,
                           bias + self.auxiliary_sigma * logits, (logits + bias).sigmoid())

    def sample_u(self, moments):
        """Sample Bernoulli coordinates from probabilities and Gaussian ones from means."""
        moments = torch.as_tensor(moments, dtype=self.weights.dtype, device=self.device)
        if moments.shape[-1] != self.n_auxiliary or not torch.isfinite(moments).all():
            raise ValueError("Expected finite auxiliary conditional moments")
        mask = self._gaussian_auxiliary
        samples = moments.clone()
        samples[..., ~mask] = torch.bernoulli(moments[..., ~mask].clamp(0., 1.))
        if mask.any():
            samples[..., mask] = moments[..., mask] + self.auxiliary_sigma[mask] * torch.randn_like(moments[..., mask])
        return samples

    def free_energy(self, v, u=None, *, weights=None, visible_bias=None, hidden_bias=None):
        if weights is not None or visible_bias is not None or hidden_bias is not None:
            if u is not None:
                raise ValueError("Pass auxiliary context or explicit conditional parameters, not both")
            return super().free_energy(v, weights=weights, visible_bias=visible_bias,
                                       hidden_bias=hidden_bias)
        if u is None:
            v, u = self.split_visible_auxiliary(v)
        else:
            v = torch.atleast_2d(v).reshape(-1, self.n_visible)
        u = self.prepare_auxiliary(u)
        energy = super().free_energy(v[:, None, :], weights=self.effective_weights(u),
                                     visible_bias=self.visible_bias, hidden_bias=self.hidden_bias).squeeze(-1)
        bias = 0. if self.auxiliary_bias is None else self.auxiliary_bias
        auxiliary_energy = torch.where(self._gaussian_auxiliary,
                                       .5 * ((u - bias) / self.auxiliary_sigma).square(), -u * bias)
        return energy + auxiliary_energy.sum(-1)

    def recon(self, visible, round_hidden=False, *, auxiliary=None):
        packed = self.prepare_input(visible) if auxiliary is None else None
        v, u = self.split_visible_auxiliary(packed) if packed is not None else (visible, auxiliary)
        hidden = self.vu2h(v, u)
        reconstruction = self.hu2v(hidden, u)
        output_hidden = hidden.round() if round_hidden else hidden
        if packed is None:
            return reconstruction, output_hidden
        reconstruction = torch.cat((reconstruction, u), dim=-1).reshape(packed.shape)
        return reconstruction, output_hidden.reshape(*packed.shape[:-1], self.n_hidden)

    def gibbs_step(self, visible, sample_visible=True):
        packed = self.prepare_input(visible)
        v, u = self.split_visible_auxiliary(packed)
        hidden_probability = self.vu2h(v, u)
        hidden = self.sample_h(hidden_probability)
        probability = self.hu2v(hidden, u)
        state = self.sample_v(probability) if sample_visible else probability
        return (torch.cat((state, u), -1).reshape(packed.shape),
                torch.cat((probability, u), -1).reshape(packed.shape), hidden, hidden_probability)

    def gibbs_chain(self, v_init, num_gibbs=1, use_probability_last_v_update=False):
        if isinstance(num_gibbs, bool) or not isinstance(num_gibbs, Integral) or num_gibbs < 1:
            raise ValueError("num_gibbs must be a positive integer")
        visible = self.prepare_input(v_init)
        for step in range(num_gibbs):
            visible, _, _, _ = self.gibbs_step(
                visible, sample_visible=not (use_probability_last_v_update and step == num_gibbs - 1))
        return visible

    def contrastive_divergence(self, data, num_gibbs=1, use_probability_last_v_update=False):
        return self.cd_loss(self.prepare_input(data), num_gibbs=num_gibbs,
                            use_probability_last_v_update=use_probability_last_v_update)

    def pcd_loss(self, v0, num_gibbs=1, use_probability_last_v_update=False):
        packed = self.prepare_input(v0).reshape(-1, self.n_visible + self.n_auxiliary)
        if self.persistent_visible is None or self.persistent_visible.shape != packed.shape:
            self._init_persistent_chain(batch_size=packed.shape[0], like=packed)
        # Fantasy destinations persist, but every minibatch supplies its own
        # clamped context. Never train against an unrelated fantasy context.
        self.persistent_visible[:, self.n_visible:] = packed[:, self.n_visible:].detach()
        return super().pcd_loss(packed, num_gibbs=num_gibbs,
                                use_probability_last_v_update=use_probability_last_v_update)

    @torch.no_grad()
    def generate(self, num_examples, num_steps=10, *, auxiliary=None):
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1
               for n in (num_examples, num_steps)):
            raise ValueError("num_examples and num_steps must be positive integers")
        v = torch.randint(0, 2, (num_examples, self.n_visible), device=self.device).to(self.weights.dtype)
        if auxiliary is not None:
            u = torch.as_tensor(auxiliary, dtype=self.weights.dtype, device=self.device)
            if u.ndim == 1:
                u = u.expand(num_examples, -1)
            if u.shape != (num_examples, self.n_auxiliary):
                raise ValueError("Auxiliary context must match the requested sample count and width")
            samples = self.gibbs_chain(torch.cat((v, u), -1), num_gibbs=num_steps)
            return self.split_visible_auxiliary(samples)
        u = torch.randint(0, 2, (num_examples, self.n_auxiliary), device=self.device).to(self.weights.dtype)
        if self._gaussian_auxiliary.any():
            bias = torch.zeros_like(self.auxiliary_sigma) if self.auxiliary_bias is None else self.auxiliary_bias
            moments = torch.where(self._gaussian_auxiliary, bias, torch.full_like(bias, .5))
            u = self.sample_u(moments.expand(num_examples, -1))
        for _ in range(num_steps):
            h = self.sample_h(self.vu2h(v, u))
            u = self.sample_u(self.vh2u(v, h))
            v = self.sample_v(self.hu2v(h, u))
        return v, u

    def _init_args(self):
        return {**super()._init_args(), "n_auxiliary": self.n_auxiliary,
                "n_factors": getattr(self, "n_factors", None),
                "auxiliary_bias": self.auxiliary_bias is not None,
                "auxiliary_types": self.auxiliary_types,
                "auxiliary_sigma": self.auxiliary_sigma.detach().cpu().tolist()}

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Checkpoints before mixed-unit support are entirely binary.
        state_dict.setdefault(prefix + "_gaussian_auxiliary", torch.zeros_like(self._gaussian_auxiliary))
        state_dict.setdefault(prefix + "auxiliary_sigma", torch.ones_like(self.auxiliary_sigma))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def __setstate__(self, state):
        super().__setstate__(state)
        width = getattr(self, "n_auxiliary", getattr(self, "n_config", None))
        if "_gaussian_auxiliary" not in self._buffers:
            self.register_buffer("_gaussian_auxiliary", torch.zeros(width, dtype=torch.bool, device=self.device))
        if "auxiliary_sigma" not in self._buffers:
            self.register_buffer("auxiliary_sigma", self.weights.new_ones(width))


def _binary_codebook(state_codes):
    codes = np.asarray(state_codes, dtype=float)
    if (codes.ndim != 2 or not min(codes.shape) or not np.isin(codes, (0, 1)).all()
            or len(np.unique(codes, axis=0)) != len(codes)):
        raise ValueError("Expected unique binary DEBTA state-code rows")
    return codes



class ConditionalTransitionRBM(ThreeWayBinaryBinaryRBM):
    """Binary destination RBM with clamped source/configuration and a codebook.

    Uses the shared factored three-way weights, source interaction, hidden
    sampling, device handling and binary RBM energy kernel. Visible sampling is categorical over valid codes.
    Exact conditional likelihood is fitted by training.fit_transition_rbm;
    unconditional CD/PCD training is not applicable to this explicit-context API.
    Configuration coordinates default to binary; mark real-valued coordinates
    with ``configuration_types=("binary", "gaussian", ...)`` and fixed scales in
    ``configuration_sigma``. Configuration-only Gaussian terms cancel from the
    normalized destination likelihood, so this fit does not learn a prior over
    configurations. NumPy inputs to free_energy/log_probabilities retain the numerical API used
    by path planning; Tensor inputs to free_energy remain differentiable.
    """

    model_type = "binary_source_config_gated_rbm_v1"

    def __init__(self, n_config, state_codes, n_hidden=24, n_factors=12, seed=42,
                 device="cpu", *, configuration_types=None, configuration_sigma=1.):
        codes = _binary_codebook(state_codes)
        for size in (n_config, n_hidden, n_factors):
            if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
                raise ValueError("Model dimensions must be positive integers")
        super().__init__(codes.shape[1], int(n_hidden), int(n_config), device=device,
                         n_factors=int(n_factors), auxiliary_bias=False,
                         auxiliary_types=configuration_types, auxiliary_sigma=configuration_sigma)
        self.double()
        # The transition model is float64; retain the supplied scale precision
        # rather than rounding through the shared base's initial float32 dtype.
        sigma = torch.as_tensor(configuration_sigma, dtype=self.weights.dtype, device=self.device)
        if sigma.ndim == 0:
            sigma = torch.where(self._gaussian_auxiliary, sigma, torch.ones_like(self.auxiliary_sigma))
        self.auxiliary_sigma.copy_(sigma)
        self.n_config, self.n_factors = int(n_config), int(n_factors)
        self.n_state_bits = codes.shape[1]
        self.register_buffer("state_codes", torch.tensor(codes, dtype=self.weights.dtype, device=self.device))
        d, k, h, f = self.n_config, self.n_state_bits, self.n_hidden, self.n_factors
        shapes = {"Ac": (d, k), "As": (k, k), "Bc": (d, h), "Ws": (k, h), "Us": (k, f)}
        self.context_weights = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.zeros(shape, dtype=self.weights.dtype, device=self.device))
            for name, shape in shapes.items()})
        rng = np.random.default_rng(seed)
        with torch.no_grad():
            for name, parameter in self.transition_parameters.items():
                scale = .2 if name in ("G", "Us", "Ud", "V") else .05 if name in ("Ws", "Wd") else 0.
                values = rng.normal(0., scale, tuple(parameter.shape)) if scale else np.zeros(tuple(parameter.shape))
                parameter.copy_(torch.as_tensor(values, device=self.device))
        self.fit_info = None

    @property
    def configuration_types(self):
        return self.auxiliary_types

    @property
    def configuration_sigma(self):
        return self.auxiliary_sigma

    @property
    def transition_parameters(self):
        """Named parameters in the order of the original NPZ checkpoints."""
        p = self.context_weights
        return {"a": self.visible_bias, "Ac": p["Ac"], "As": p["As"],
                "b": self.hidden_bias, "Bc": p["Bc"], "Ws": p["Ws"], "Wd": self.weights,
                "G": self.auxiliary_factors, "Us": p["Us"],
                "Ud": self.visible_factors, "V": self.hidden_factors}

    def parameter_vector(self):
        return torch.cat([p.detach().reshape(-1) for p in self.transition_parameters.values()]).cpu().numpy().copy()

    def conditional_parameters(self, configuration, source):
        c = self.prepare_auxiliary(configuration)
        s = torch.atleast_2d(torch.as_tensor(source, dtype=self.weights.dtype, device=self.device))
        if (s.ndim != 2 or s.shape[1] != self.n_state_bits or len(c) != len(s)
                or not torch.all((s == 0) | (s == 1))):
            raise ValueError("Expected aligned mixed-configuration and binary source-code rows")
        p = self.context_weights
        scaled_c = c / self.configuration_sigma
        a = self.visible_bias + scaled_c @ p["Ac"] + s @ p["As"]
        b = (self.hidden_bias + scaled_c @ p["Bc"] + s @ p["Ws"]
             + self.three_way_hidden(s, c, visible_factors=p["Us"]))
        w = self.effective_weights(c)
        return a, b, w

    def conditional_log_probabilities(self, configuration, source):
        a, b, w = self.conditional_parameters(configuration, source)
        logits = -super().free_energy(self.state_codes[None], weights=w,
                                      visible_bias=a[:, None], hidden_bias=b[:, None])
        return logits.log_softmax(dim=-1)

    def free_energy(self, destinations, *, source, configuration):
        tensor_input = isinstance(destinations, torch.Tensor)
        codes = torch.as_tensor(destinations, dtype=self.weights.dtype, device=self.device)
        codes = torch.atleast_2d(codes)
        if codes.ndim != 2 or codes.shape[1] != self.n_state_bits or not torch.all((codes == 0) | (codes == 1)):
            raise ValueError("Destinations must be binary DEBTA codes")
        a, b, w = self.conditional_parameters(configuration, source)
        if len(a) != 1:
            raise ValueError("Free energy expects one clamped source/configuration")
        energy = super().free_energy(codes[None], weights=w, visible_bias=a[:, None], hidden_bias=b[:, None])[0]
        return energy if tensor_input else energy.detach().cpu().numpy()

    def log_probabilities(self, configurations):
        c = self.prepare_auxiliary(configurations)
        n = len(self.state_codes)
        with torch.no_grad():
            rows = self.conditional_log_probabilities(c.repeat_interleave(n, dim=0), self.state_codes.repeat(len(c), 1))
        return rows.reshape(len(c), n, n).cpu().numpy()

    def v2h(self, visible, pre_sigmoid=False, *, source, configuration):
        a, b, w = self.conditional_parameters(configuration, source)
        visible = torch.as_tensor(visible, dtype=self.weights.dtype, device=self.device)
        visible = torch.atleast_2d(visible)
        if visible.ndim != 2 or visible.shape[1] != self.n_state_bits:
            raise ValueError("Expected destination-code rows")
        return super().v2h(visible[:, None, :], pre_sigmoid=pre_sigmoid,
                           weights=w, hidden_bias=b[:, None]).squeeze(-2)

    def h2v(self, hidden, *, source, configuration):
        """Categorical destination probabilities, rather than independent bits."""
        a, b, w = self.conditional_parameters(configuration, source)
        hidden = torch.as_tensor(hidden, dtype=self.weights.dtype, device=self.device)
        hidden = torch.atleast_2d(hidden)
        if hidden.ndim != 2 or hidden.shape[1] != self.n_hidden:
            raise ValueError("Expected hidden-state rows")
        logits = (a + torch.matmul(hidden[:, None, :], w.transpose(-1, -2)).squeeze(-2)) @ self.state_codes.T
        return logits.softmax(dim=-1)

    def vh2u(self, v, h, *, source):
        """Auxiliary moments for the raw joint energy with source clamped.

        These are not the configuration posterior of normalized transitions or
        paths. Conditional destination training does not identify that marginal.
        """
        v = torch.atleast_2d(torch.as_tensor(v, dtype=self.weights.dtype, device=self.device))
        h = torch.atleast_2d(torch.as_tensor(h, dtype=self.weights.dtype, device=self.device))
        s = torch.atleast_2d(torch.as_tensor(source, dtype=self.weights.dtype, device=self.device))
        if (v.ndim != 2 or h.ndim != 2 or s.ndim != 2 or v.shape[1] != self.n_state_bits
                or h.shape[1] != self.n_hidden or s.shape[1] != self.n_state_bits
                or not torch.all((s == 0) | (s == 1))
                or not torch.isfinite(v).all() or not torch.isfinite(h).all()):
            raise ValueError("Expected destination, hidden and binary source rows")
        p = self.context_weights
        interaction = (v @ p["Ac"].T + h @ p["Bc"].T
                       + (((s @ p["Us"]) + (v @ self.visible_factors))
                          * (h @ self.hidden_factors)) @ self.auxiliary_factors.T)
        return torch.where(self._gaussian_auxiliary,
                           self.configuration_sigma * interaction, interaction.sigmoid())

    def sample_v(self, probabilities):
        return self.state_codes[torch.multinomial(probabilities, 1).squeeze(-1)]

    def recon(self, visible, round_hidden=False, *, source, configuration):
        hidden = self.v2h(visible, source=source, configuration=configuration)
        probability = self.h2v(hidden, source=source, configuration=configuration)
        return probability @ self.state_codes, hidden.round() if round_hidden else hidden

    def forward(self, visible, *, source, configuration):
        return self.recon(visible, source=source, configuration=configuration)[0]

    def gibbs_step(self, visible, sample_visible=True, *, source, configuration):
        hidden_probability = self.v2h(visible, source=source, configuration=configuration)
        hidden = self.sample_h(hidden_probability)
        probability = self.h2v(hidden, source=source, configuration=configuration)
        state = self.sample_v(probability) if sample_visible else probability @ self.state_codes
        return state, probability, hidden, hidden_probability

    def gibbs_chain(self, v_init, num_gibbs=1, use_probability_last_v_update=False, *, source, configuration):
        if isinstance(num_gibbs, bool) or not isinstance(num_gibbs, (int, np.integer)) or num_gibbs < 1:
            raise ValueError("num_gibbs must be a positive integer")
        visible = v_init
        for step in range(num_gibbs):
            visible, _, _, _ = self.gibbs_step(visible, source=source, configuration=configuration,
                sample_visible=not (use_probability_last_v_update and step == num_gibbs - 1))
        return visible

    @torch.no_grad()
    def generate(self, num_examples, num_steps=100, *, source, configuration):
        """Sample valid destination codes with source and configuration clamped."""
        if any(isinstance(n, bool) or not isinstance(n, Integral) or n < 1
               for n in (num_examples, num_steps)):
            raise ValueError("num_examples and num_steps must be positive integers")
        initial = self.state_codes[torch.randint(len(self.state_codes), (num_examples,), device=self.device)]
        return self.gibbs_chain(initial, num_gibbs=num_steps, source=source, configuration=configuration)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        # Native checkpoints previously registered these factors in context_weights.
        for old, new in (("G", "auxiliary_factors"), ("Ud", "visible_factors"), ("V", "hidden_factors")):
            old_key, new_key = prefix + "context_weights." + old, prefix + new
            if old_key in state_dict and new_key not in state_dict:
                state_dict[new_key] = state_dict.pop(old_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict,
                                      missing_keys, unexpected_keys, error_msgs)

    def _init_args(self):
        return dict(n_config=self.n_config, state_codes=self.state_codes.detach().cpu().tolist(),
                    n_hidden=self.n_hidden, n_factors=self.n_factors, device="cpu",
                    configuration_types=self.configuration_types,
                    configuration_sigma=self.configuration_sigma.detach().cpu().tolist())

    def save(self, filename=None, filename_sufix=""):
        if filename is None or Path(filename).suffix != ".npz":
            return super().save(filename, filename_sufix)
        np.savez_compressed(filename, model_type=np.array(self.model_type),
                            state_codes=self.state_codes.detach().cpu().numpy(), n_config=np.array(self.n_config),
                            fit_info=np.array(json.dumps(self.fit_info)),
                            configuration_types=np.array(self.configuration_types),
                            configuration_sigma=self.configuration_sigma.detach().cpu().numpy(),
                            **{name: value.detach().cpu().numpy() for name, value in self.transition_parameters.items()})

    @classmethod
    def load(cls, filename):
        if Path(filename).suffix != ".npz":
            return super().load(filename)
        with np.load(filename, allow_pickle=False) as checkpoint:
            if "model_type" not in checkpoint or str(checkpoint["model_type"]) != cls.model_type:
                raise ValueError("Checkpoint is not a binary source/configuration-conditioned RBM")
            model = cls(int(checkpoint["n_config"]), checkpoint["state_codes"],
                        len(checkpoint["b"]), checkpoint["G"].shape[1],
                        configuration_types=checkpoint["configuration_types"].tolist() if "configuration_types" in checkpoint else None,
                        configuration_sigma=checkpoint["configuration_sigma"] if "configuration_sigma" in checkpoint else 1.)
            with torch.no_grad():
                for name, parameter in model.transition_parameters.items():
                    value = checkpoint[name]
                    if value.shape != tuple(parameter.shape) or not np.isfinite(value).all():
                        raise ValueError("Invalid conditional RBM checkpoint weights")
                    parameter.copy_(torch.as_tensor(value))
            model.fit_info = json.loads(str(checkpoint["fit_info"]))
        return model

    def __setstate__(self, state):
        # Checkpoints predating the nn.Module implementation stored NumPy arrays.
        if "_parameters" not in state and isinstance(state.get("weights"), dict):
            self.__init__(state["n_config"], state["state_codes"], state["n_hidden"], state["n_factors"])
            with torch.no_grad():
                for name, parameter in self.transition_parameters.items():
                    parameter.copy_(torch.as_tensor(state["weights"][name]))
            self.fit_info = state.get("fit_info")
        else:
            super().__setstate__(state)
            # DEBTA extra_state also contains whole pre-refactor nn.Module objects.
            if not hasattr(self, "n_auxiliary"):
                self.n_auxiliary = self.n_config
                self.register_parameter("auxiliary_bias", None)
                for old, new in (("G", "auxiliary_factors"), ("Ud", "visible_factors"), ("V", "hidden_factors")):
                    self.register_parameter(new, self.context_weights.pop(old))




def transition_log_probabilities(model, configurations, n_states=None):
    """Normalized P[c, source, destination] over valid binary DEBTA codes."""
    if not isinstance(model, ConditionalTransitionRBM):
        raise TypeError("Use a ConditionalTransitionRBM with an explicit binary codebook")
    if n_states is not None and n_states != len(model.state_codes):
        raise ValueError("State count differs from the conditional model codebook")
    return model.log_probabilities(configurations)

def transition_energies(model, configuration, state_codes=None):
    """Return -log p(destination | source, configuration) from a transition RBM.

    All rows are normalized over destinations. Apply +inf feasibility masks
    separately: a confident conditional row does not establish source support.
    """
    if not isinstance(model, ConditionalTransitionRBM):
        raise TypeError("Expected a ConditionalTransitionRBM")
    if state_codes is not None and not np.array_equal(state_codes, model.state_codes.detach().cpu().numpy()):
        raise ValueError("State codebook differs from the conditional model")
    return -transition_log_probabilities(model, [configuration])[0]
