"""Dimension-independent variation estimation and fit-free resolution scores.

One implementation shared by the 2D, 3D and high-dimensional benchmarks so the
resolution spine means the same thing at every input dimension.
"""
from ND_scan_samplers.src.resolution.variation import LocalVariation, knn_variation, metric_fill, region_shapes
from ND_scan_samplers.src.resolution.scores import fit_free_scores, spine_targets

__all__ = ["LocalVariation", "knn_variation", "metric_fill", "region_shapes",
           "fit_free_scores", "spine_targets"]
