"""Contracts of the batched-pool analysis helpers (scripts/analysis/pool_batch_board)."""
from ND_scan_samplers.scripts.analysis.pool_batch_board import (CAMPAIGN, START, batch_size, completed,
                                                                completed_points)
from ND_scan_samplers.scripts.analysis.surprise_board import CHECKPOINTS
from ND_scan_samplers.src.strategies import BATCH_ARMS


def test_batch_size_reads_the_arm_name():
    assert [batch_size(a) for a in ("vurs-b5", "vurs-b20r", "vwrs-b15", "vurs", "space-filling")] == [5, 20, 15, 1, 1]


def test_completed_is_the_last_full_batch_at_or_below_n():
    assert [completed(n, 5) for n in CAMPAIGN] == [19, 34, 59, 94]
    assert [completed(n, 20) for n in CAMPAIGN] == [9, 29, 49, 89]
    assert [completed(n, 1) for n in CAMPAIGN] == list(CAMPAIGN)
    for k in (5, 10, 15, 20):
        for n in CAMPAIGN:
            c = completed(n, k)
            assert c <= n < c+k and (c-START) % k == 0


def test_completed_points_cover_every_batch_size_for_the_sequential_reference():
    need = set(completed_points("vurs")) | set(CHECKPOINTS)
    for arm in BATCH_ARMS:
        assert set(completed_points(arm)) <= need
        assert {completed(n, batch_size(arm)) for n in CAMPAIGN} <= set(completed_points(arm)) | set(CHECKPOINTS)
