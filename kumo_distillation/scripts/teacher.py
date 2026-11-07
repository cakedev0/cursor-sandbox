"""Kumo-Tabular teacher with the settings of the upstream ``kumo_tabular`` model.

Upstream (``electricity_load_forecasting.kumotabular.KumoTabularRegressor``)
calls ``KumoTabular.forward`` once with the full training set as context,
``num_estimators=8``, float16 autocast, and keeps the requested quantiles out
of the 999 predicted ones (sorted to avoid crossing).

``KumoTeacher`` runs the same model, either through the cached ``fit`` /
``predict`` API (the context is encoded once, then many query batches reuse it)
or through chunked ``forward`` calls (the context is re-encoded per chunk).
"""

import functools

import numpy as np
import torch

from folds import QUANTILES


@functools.lru_cache(maxsize=1)
def load_model(size="large", device="cuda"):
    import sdm

    return sdm.models.KumoTabular(task="regression", size=size, device=device)


def _table(values, columns, device):
    import sdm

    tensor = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    return sdm.TableTensor.from_tensor(tensor.to(device), columns=columns)


def _select(out, quantiles):
    available = list(out.numerical_columns)
    names = [f"q{round(q * 999):03d}" for q in quantiles]
    selected = out.numerical[:, [available.index(name) for name in names]]
    return np.sort(selected.float().cpu().numpy(), axis=1)


def _autocast(device):
    return torch.amp.autocast(
        torch.device(device).type,
        dtype=torch.float16,
        enabled=torch.device(device).type == "cuda",
    )


def upstream_forward(X_train, y_train, X_query, quantiles=QUANTILES, device="cuda"):
    """Exactly the upstream ``KumoTabularRegressor(n_estimators=8)`` predict."""
    from electricity_load_forecasting.kumotabular import KumoTabularRegressor

    est = KumoTabularRegressor(quantiles=quantiles, n_estimators=8, device=device)
    return est.fit(X_train, y_train).predict(X_query).to_numpy()


class KumoTeacher:
    def __init__(
        self,
        quantiles=QUANTILES,
        n_estimators=8,
        device="cuda",
        mode="cache",
        batch_size=8192,
        seed=0,
    ):
        self.quantiles = quantiles
        self.n_estimators = n_estimators
        self.device = device
        self.mode = mode
        self.batch_size = batch_size
        self.seed = seed

    def fit(self, X, y):
        self.model_ = load_model(device=self.device)
        self.columns_ = [f"x{i}" for i in range(X.shape[1])]
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y, dtype=np.float32)
        torch.manual_seed(self.seed)
        if self.mode == "cache":
            with torch.inference_mode(), _autocast(self.device):
                self.model_.fit(
                    _table(self.X_, self.columns_, self.device),
                    _table(self.y_[:, None], ["target"], self.device),
                    num_estimators=self.n_estimators,
                )
        return self

    def _predict_batch(self, X):
        x_query = _table(X, self.columns_, self.device)
        with torch.inference_mode(), _autocast(self.device):
            if self.mode == "cache":
                out = self.model_.predict(x_query)
            else:
                out = self.model_(
                    x_context=_table(self.X_, self.columns_, self.device),
                    y_context=_table(self.y_[:, None], ["target"], self.device),
                    x_query=x_query,
                    num_estimators=self.n_estimators,
                )
        return _select(out, self.quantiles)

    def predict(self, X):
        X = np.asarray(X, dtype=np.float32)
        return np.concatenate(
            [
                self._predict_batch(X[i : i + self.batch_size])
                for i in range(0, X.shape[0], self.batch_size)
            ]
        )

    def clear(self):
        if self.mode == "cache":
            self.model_.clear()
        torch.cuda.empty_cache()
