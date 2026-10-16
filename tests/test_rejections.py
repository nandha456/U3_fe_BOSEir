"""
Things the frontend must REFUSE — and refuse with an error a human can act on
(names the node, says why, says what to do), not a raw Python traceback.
"""
import numpy as np
import pytest
from onnx import helper as H

from oracle import F, vi, const, build, run_frontend


def test_conv1d_is_rejected_cleanly(tmp_path):
    W = np.random.default_rng(0).standard_normal((4, 3, 3)).astype(np.float32)
    m = build([H.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3])],
              [vi("x", F, [1, 3, 16])], [vi("y", F, [None] * 3)], [const(W, "w")])
    with pytest.raises(ValueError, match=r"rank-4|4-D|NCHW"):
        run_frontend(m, tmp_path)


def test_sigmoid_still_rejected_with_reason(tmp_path):
    m = build([H.make_node("Sigmoid", ["x"], ["y"])], [vi("x", F, [1, 4])], [vi("y", F, [1, 4])])
    with pytest.raises(ValueError, match="Sigmoid.*nonlinearity"):
        run_frontend(m, tmp_path)


def test_batchnorm_with_runtime_parameters_is_rejected(tmp_path):
    # gamma/beta/mean/var are graph INPUTS, not constants -> cannot be baked into Mul/Add
    m = build([H.make_node("BatchNormalization", ["x", "g", "b", "m", "v"], ["y"])],
              [vi("x", F, [1, 3, 4, 4])] + [vi(n, F, [3]) for n in "gbmv"], [vi("y", F, [1, 3, 4, 4])])
    with pytest.raises(ValueError, match="BatchNormalization.*compile-time constant"):
        run_frontend(m, tmp_path)


def test_reshape_with_runtime_shape_is_rejected(tmp_path):
    from onnx import TensorProto as TP
    m = build([H.make_node("Reshape", ["x", "shp"], ["y"])],
              [vi("x", F, [1, 6]), vi("shp", TP.INT64, [2])], [vi("y", F, [2, 3])])
    with pytest.raises(ValueError, match="Reshape.*compile-time constant"):
        run_frontend(m, tmp_path)
