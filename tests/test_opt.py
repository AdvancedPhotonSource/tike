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


def test_fit_line():
    result = np.around(tike.opt.fit_line_least_squares(
        y=np.asarray([0, np.log(0.9573), np.log(0.8386)]),
        x=np.asarray([0, 1, 2]),
    ), 4)
    np.testing.assert_array_equal((-0.0880, 0.0148), result)
