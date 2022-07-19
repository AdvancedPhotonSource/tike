import numpy as np
import tike.opt


class AlgorithmOptionsStub:

    def __init__(self, costs, window=5) -> None:
        self.costs = costs
        self.convergence_window = window


def test_is_converged():
    assert tike.opt.is_converged(
        AlgorithmOptionsStub((np.arange(11) / 1234).tolist(), 5))
    assert tike.opt.is_converged(
        AlgorithmOptionsStub((np.zeros(11) / 1234).tolist(), 5))
    assert not tike.opt.is_converged(
        AlgorithmOptionsStub((-np.arange(11) / 1234).tolist(), 5))
