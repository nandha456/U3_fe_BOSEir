"""
Ops that are NOT in the 18-op core but can be rewritten EXACTLY using only core ops.
Each test proves the rewrite is numerically identical to onnxruntime running
the original op.
"""
import numpy as np
import pytest
from onnx import helper as H

from oracle import F, vi, const, build, check_against_ort


def _bn_params(c, seed=0):
    rng = np.random.default_rng(seed)
    return [const((rng.random(c) + 0.5).astype(np.float32), "g"),
            const((rng.random(c) - 0.5).astype(np.float32), "b"),
            const((rng.random(c) * 0.2 - 0.1).astype(np.float32), "m"),
            const((rng.random(c) + 0.5).astype(np.float32), "v")]


BN_IN = ["x", "g", "b", "m", "v"]


@pytest.mark.parametrize("shape", [(2, 3, 5, 5), (2, 3, 7), (4, 3)])   # NCHW, NCL, NC
def test_standalone_batchnorm_becomes_mul_add(tmp_path, shape):
    m = build([H.make_node("BatchNormalization", BN_IN, ["y"], epsilon=1e-3)],
              [vi("x", F, list(shape))], [vi("y", F, list(shape))], _bn_params(3))
    graph = check_against_ort(m, tmp_path)
    ops = [n.op_type for n in graph.nodes if n.op_type != "Constant"]
    assert ops == ["Mul", "Add"]


def test_preactivation_batchnorm_before_conv(tmp_path):
    # ResNet-v2 style:  BN -> ReLU -> Conv   (BN can NOT fold into a preceding Conv)
    W = (np.random.default_rng(3).standard_normal((4, 3, 3, 3)) * 0.1).astype(np.float32)
    m = build([H.make_node("BatchNormalization", BN_IN, ["a"]),
               H.make_node("Relu", ["a"], ["act"]),
               H.make_node("Conv", ["act", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1])],
              [vi("x", F, [1, 3, 8, 8])], [vi("y", F, [1, 4, 8, 8])], _bn_params(3) + [const(W, "w")])
    check_against_ort(m, tmp_path)


def test_batchnorm_after_relu_after_conv_rank_is_inferred(tmp_path):
    # BN's input shape is unknown at legalize time -> rank must be derived from producers
    W = (np.random.default_rng(4).standard_normal((3, 3, 3, 3)) * 0.1).astype(np.float32)
    m = build([H.make_node("Conv", ["x", "w"], ["c"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
               H.make_node("Relu", ["c"], ["r"]),
               H.make_node("BatchNormalization", ["r", "g", "b", "m", "v"], ["y"])],
              [vi("x", F, [1, 3, 8, 8])], [vi("y", F, [1, 3, 8, 8])], _bn_params(3) + [const(W, "w")])
    check_against_ort(m, tmp_path)


# ---- Split -----------------------------------------------------------------

def test_split_equal_opset11(tmp_path):
    m = build([H.make_node("Split", ["x"], ["a", "b", "c"], axis=1)],
              [vi("x", F, [2, 6])], [vi("a", F, [2, 2]), vi("b", F, [2, 2]), vi("c", F, [2, 2])], opset=11)
    check_against_ort(m, tmp_path)


def test_split_explicit_sizes_attribute_opset11(tmp_path):
    m = build([H.make_node("Split", ["x"], ["a", "b"], axis=-1, split=[1, 5])],
              [vi("x", F, [2, 6])], [vi("a", F, [2, 1]), vi("b", F, [2, 5])], opset=11)
    check_against_ort(m, tmp_path)


def test_split_explicit_sizes_input_opset13(tmp_path):
    m = build([H.make_node("Split", ["x", "sz"], ["a", "b", "c"], axis=1)],
              [vi("x", F, [2, 6])], [vi("a", F, [2, 1]), vi("b", F, [2, 2]), vi("c", F, [2, 3])],
              [const(np.array([1, 2, 3], dtype=np.int64), "sz")], opset=13)
    check_against_ort(m, tmp_path)


def test_split_num_outputs_uneven_opset18(tmp_path):
    m = build([H.make_node("Split", ["x"], ["a", "b", "c"], axis=1, num_outputs=3)],
              [vi("x", F, [2, 7])], [vi("a", F, [2, 3]), vi("b", F, [2, 3]), vi("c", F, [2, 1])], opset=18)
    check_against_ort(m, tmp_path)


def test_split_feeds_downstream_ops(tmp_path):
    m = build([H.make_node("Split", ["x"], ["a", "b"], axis=1),
               H.make_node("Relu", ["a"], ["ra"]),
               H.make_node("Add", ["ra", "b"], ["y"])],
              [vi("x", F, [2, 6])], [vi("y", F, [2, 3])], opset=13)
    check_against_ort(m, tmp_path)
