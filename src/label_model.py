"""TabPFN label posterior for the sampler's label-aware terms.

The kNN boundary term (`strategies.label_terms`) predicts a candidate's mode
from its nearest paid run, so its boundary is wherever two paid labels are
equidistant. A tabular foundation model instead returns a posterior
predictive p(mode | x) from the paid runs in context; its normalized entropy

    H~(c) = -sum_k p_k(c) log p_k(c) / log K,  in [0, 1]

is high where the paid runs leave the mode in doubt. It replaces the kNN
boundary term in `vurs-boundary-pfn` and is the whole acquisition in
`pfn-entropy` (see `strategies.PFN_ARMS`).

Caveat carried into the protocol: entropy at a true transition is partly
aleatoric (the label really does flip there), so pure entropy sampling can
keep paying for the same boundary. VURS keeps its coverage and variation
terms beside it for that reason.

Keyless TabPFN v2 weights, CPU, as in Fusion_Microinstability_Classifier
(`fmc.models.tabpfn_benchmark._make_tabpfn`). The acquisition loop uses
ESTIMATORS ensemble members (a ranking needs less than a calibrated score);
scoring uses the library default. Torch threads per process come from
ND_PFN_THREADS so parallel workers do not oversubscribe the CPU.
"""
import os

import numpy as np

ESTIMATORS = 4          # ensemble members inside the acquisition loop
_MODELS = {}


def _model(n_estimators, seed):
    """One reusable classifier per (ensemble size, seed) in this process."""
    key = (n_estimators, seed)
    if key not in _MODELS:
        os.environ.setdefault("TABPFN_ALLOW_CPU_LARGE_DATASET", "true")
        import torch
        from tabpfn import TabPFNClassifier
        from tabpfn.constants import ModelVersion
        threads = os.environ.get("ND_PFN_THREADS")
        if threads:
            torch.set_num_threads(int(threads))
        kwargs = dict(device="cpu", random_state=seed)
        if n_estimators is not None:
            kwargs["n_estimators"] = n_estimators
        _MODELS[key] = TabPFNClassifier.create_default_for_version(ModelVersion.V2, **kwargs)
    return _MODELS[key]


def pfn_posterior(x, labels, query, n_modes, seed=0, n_estimators=ESTIMATORS):
    """p(mode | query) from TabPFN conditioned on the paid runs, one column per mode.

    Modes absent from the paid runs get probability zero here; callers that
    score held runs floor and renormalize. With a single paid mode the
    posterior is a point mass on it.
    """
    labels = np.asarray(labels, int)
    seen = np.unique(labels)
    out = np.zeros((len(query), n_modes))
    if len(seen) == 1:
        out[:, seen[0]] = 1.
        return out
    model = _model(n_estimators, seed).fit(np.asarray(x, float), labels)
    proba = model.predict_proba(np.asarray(query, float))
    out[:, model.classes_.astype(int)] = proba
    return out


def normalized_entropy(posterior):
    """Entropy of each row over log K of the modes present in it: 0 certain, 1 uniform."""
    p = np.clip(posterior, 1e-12, 1.)
    k = max(int((posterior > 0).any(axis=0).sum()), 2)
    return -np.sum(posterior*np.log(p), axis=1)/np.log(k)
