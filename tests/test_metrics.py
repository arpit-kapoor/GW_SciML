"""compute_metrics must exclude and report nodes with non-finite predictions instead of failing."""
import numpy as np

from src.inference.metrics import compute_metrics


def _results(n_steps=50, n_nodes=40, seed=0):
    rng = np.random.default_rng(seed)
    targets = rng.normal(size=(n_steps, n_nodes, 1, 2))
    predictions = targets + 0.1 * rng.normal(size=targets.shape)
    return predictions, targets


def test_all_finite_unchanged():
    predictions, targets = _results()
    m = compute_metrics({'val': {'predictions': predictions, 'targets': targets}}, ['mass_concentration', 'head'], [0, 1])
    head = m['val']['head']
    assert head['n_nodes_nonfinite'] == 0 and head['n_nodes_evaluated'] == 40
    diff = (predictions - targets)[..., 1].ravel()
    assert np.isclose(head['rel_l2_error'], np.linalg.norm(diff) / np.linalg.norm(targets[..., 1].ravel()))


def test_blown_up_nodes_excluded_and_counted(capsys):
    predictions, targets = _results()
    predictions[30:, 3, 0, 1] = np.nan          # node 3 blows up at step 31
    predictions[10:, 7, 0, 0] = np.inf          # node 7 blows up at step 11
    m = compute_metrics({'val': {'predictions': predictions, 'targets': targets}}, ['mass_concentration', 'head'], [0, 1])
    log = capsys.readouterr().out
    assert 'BLOWN-UP NODES: 2 of 40 nodes (5.00%)' in log and 'first non-finite at step 11' in log
    keep = np.ones(40, dtype=bool)
    keep[[3, 7]] = False
    expected = compute_metrics({'val': {'predictions': predictions[:, keep], 'targets': targets[:, keep]}},
                               ['mass_concentration', 'head'], [0, 1])
    for col in ('mass_concentration', 'head'):
        assert m['val'][col]['n_nodes_nonfinite'] == 2 and m['val'][col]['n_nodes_evaluated'] == 38
        assert np.isfinite(m['val'][col]['r2'])
        for key in ('nmae', 'rel_l2_error', 'r2', 'kge'):
            assert np.isclose(m['val'][col][key], expected['val'][col][key])
