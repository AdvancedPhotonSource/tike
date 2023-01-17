import numpy as np
# from tike.communicator import MPICommunicator
import unittest


@unittest.skip(reason="The communicator module is broken/disabled.")
class TestMPICommunicator(unittest.TestCase):
    """Test the functions of the MPICommunicator class."""

    def test_slicing(self):
        """Check correctness of pytcho and tomo data slicing."""
        comm = MPICommunicator()
        # Data comes from forward project with dimensions (theta, v, h)
        shape = [11, 3, 7]
        tomo_data = np.arange(0, np.prod(shape), dtype=np.int32)
        tomo_data = tomo_data.reshape(shape)
        # Pytcho data should be periodic along v
        ptycho_data = comm.get_ptycho_slice(tomo_data)
        # print("{} ptycho_data:\n{}".format(comm.rank, ptycho_data))
        lo = comm.rank * 3
        np.testing.assert_array_equal(ptycho_data[:, 0:3, :],
                                      ptycho_data[:, lo:lo + 3, :])
        # Assert the reverse transform works
        tomo_data1 = comm.get_tomo_slice(ptycho_data)
        np.testing.assert_array_equal(tomo_data, tomo_data1)

    def test_gather(self):
        """Check correctness of data gathering."""
        comm = MPICommunicator()
        data = np.ones([3, 1], dtype=np.int32) * comm.rank
        data = comm.gather(data, root=0, axis=1)
        if comm.rank == 0:
            # print(data)
            truth = np.tile(np.arange(comm.size), [3, 1])
            np.testing.assert_array_equal(data, truth)
        else:
            assert data is None


if __name__ == '__main__':
    unittest.main()
