"""
Bugs found by probing the frontend against onnxruntime. Each test is named after
the symptom. Before the fixes, every one of these either produced a WRONG
answer silently or crashed with an unhelpful error.
"""
import numpy as np
import pytest
from onnx import helper as H, TensorProto as TP

from oracle import F, vi, const, build, check_against_ort, run_frontend

INT_MAX = 2**31 - 1


def i64(name, values):
    return const(np.array(values, dtype=np.int64), name)


def f32(name, values):
    return const(np.array(values, dtype=np.float32), name)


# ---- Slice -----------------------------------------------------------------

@pytest.mark.parametrize("starts,ends,steps,out_w", [
    ([-2], [INT_MAX], [1], 2),          # last two columns        (negative start)
    ([1], [-1], [1], 6),                # drop first and last     (negative end)
    ([-1], [-INT_MAX], [-1], 8),        # reverse                 (negative step)
    ([0], [8], [3], 3),                 # strided
])
def test_slice_negative_indices_and_steps(tmp_path, starts, ends, steps, out_w):
    m = build([H.make_node("Slice", ["x", "s", "e", "a", "st"], ["y"])],
              [vi("x", F, [1, 8])], [vi("y", F, [1, out_w])],
              [i64("s", starts), i64("e", ends), i64("a", [1]), i64("st", steps)])
    check_against_ort(m, tmp_path)


def test_slice_with_omitted_axes_input(tmp_path):
    # Slice(x, starts, ends, "", steps): empty-string optional input in the MIDDLE
    m = build([H.make_node("Slice", ["x", "s", "e", "", "st"], ["y"])],
              [vi("x", F, [1, 8])], [vi("y", F, [1, 4])],
              [i64("s", [0]), i64("e", [8]), i64("st", [2])])
    check_against_ort(m, tmp_path)


# ---- Conv / pool padding modes --------------------------------------------

@pytest.mark.parametrize("mode", ["SAME_UPPER", "SAME_LOWER", "VALID"])
@pytest.mark.parametrize("stride", [1, 2])
def test_conv_auto_pad(tmp_path, mode, stride):
    W = (np.random.default_rng(0).standard_normal((4, 3, 3, 3)) * 0.1).astype(np.float32)
    m = build([H.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], strides=[stride, stride], auto_pad=mode)],
              [vi("x", F, [1, 3, 8, 8])], [vi("y", F, [None] * 4)], [const(W, "w")])
    check_against_ort(m, tmp_path)


@pytest.mark.parametrize("op", ["MaxPool", "AveragePool"])
@pytest.mark.parametrize("hw,k,s,pads", [
    (5, 2, 2, [0, 0, 0, 0]),            # plain ceil_mode: 5 -> 3 (floor gives 2)
    (5, 2, 2, [1, 1, 1, 1]),            # ceil + padding, last window would start in padding
    (6, 3, 2, [1, 1, 1, 1]),
])
def test_pool_ceil_mode(tmp_path, op, hw, k, s, pads):
    m = build([H.make_node(op, ["x"], ["y"], kernel_shape=[k, k], strides=[s, s], pads=pads, ceil_mode=1)],
              [vi("x", F, [1, 2, hw, hw])], [vi("y", F, [None] * 4)])
    check_against_ort(m, tmp_path)


@pytest.mark.parametrize("include_pad", [0, 1])
def test_avgpool_count_include_pad(tmp_path, include_pad):
    m = build([H.make_node("AveragePool", ["x"], ["y"], kernel_shape=[3, 3], strides=[2, 2],
                           pads=[1, 1, 1, 1], count_include_pad=include_pad)],
              [vi("x", F, [1, 2, 7, 7])], [vi("y", F, [None] * 4)])
    check_against_ort(m, tmp_path)


def test_grouped_depthwise_conv(tmp_path):
    W = (np.random.default_rng(1).standard_normal((6, 1, 3, 3)) * 0.2).astype(np.float32)
    m = build([H.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1], group=6)],
              [vi("x", F, [1, 6, 8, 8])], [vi("y", F, [1, 6, 8, 8])], [const(W, "w")])
    check_against_ort(m, tmp_path)


# ---- Clip with an omitted optional input ----------------------------------

def test_clip_missing_min_only_max(tmp_path):
    # Clip(x, "", max): the frontend used to treat `max` as `min`
    m = build([H.make_node("Clip", ["x", "", "mx"], ["y"])],
              [vi("x", F, [1, 16])], [vi("y", F, [1, 16])], [f32("mx", 0.5)])
    check_against_ort(m, tmp_path)


def test_clip_missing_max_only_min(tmp_path):
    m = build([H.make_node("Clip", ["x", "mn"], ["y"])],
              [vi("x", F, [1, 16])], [vi("y", F, [1, 16])], [f32("mn", -0.5)])
    check_against_ort(m, tmp_path)


# ---- BatchNorm fold must not corrupt a shared weight ----------------------

def test_bn_fold_does_not_corrupt_shared_weight(tmp_path):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((3, 3, 3, 3)) * 0.1).astype(np.float32)
    bn = [const((rng.random(3) + 0.5).astype(np.float32), n) for n in ("g", "b", "m", "v")]
    m = build([H.make_node("Conv", ["x", "w"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
               H.make_node("BatchNormalization", ["c1", "g", "b", "m", "v"], ["c1bn"]),
               H.make_node("Conv", ["x", "w"], ["c2"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
               H.make_node("Add", ["c1bn", "c2"], ["y"])],
              [vi("x", F, [1, 3, 8, 8])], [vi("y", F, [1, 3, 8, 8])], [const(W, "w")] + bn)
    graph = check_against_ort(m, tmp_path)
    assert np.allclose(graph.initializers["w"], W), "the shared original weight was mutated"


def test_bn_not_folded_when_conv_output_is_also_a_graph_output(tmp_path):
    rng = np.random.default_rng(0)
    W = (rng.standard_normal((3, 3, 3, 3)) * 0.1).astype(np.float32)
    bn = [const((rng.random(3) + 0.5).astype(np.float32), n) for n in ("g", "b", "m", "v")]
    m = build([H.make_node("Conv", ["x", "w"], ["c1"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
               H.make_node("BatchNormalization", ["c1", "g", "b", "m", "v"], ["y"])],
              [vi("x", F, [1, 3, 8, 8])],
              [vi("c1", F, [1, 3, 8, 8]), vi("y", F, [1, 3, 8, 8])], [const(W, "w")] + bn)
    check_against_ort(m, tmp_path)


# ---- Quantize / Dequantize dtypes -----------------------------------------

def test_quantize_output_dtype_is_int8(tmp_path):
    m = build([H.make_node("QuantizeLinear", ["x", "s", "z"], ["q"])],
              [vi("x", F, [1, 16])], [vi("q", TP.INT8, [1, 16])],
              [f32("s", 0.05), const(np.array(3, dtype=np.int8), "z")])
    check_against_ort(m, tmp_path)


def test_quantize_default_zero_point_is_uint8(tmp_path):
    m = build([H.make_node("QuantizeLinear", ["x", "s"], ["q"])],
              [vi("x", F, [1, 16])], [vi("q", TP.UINT8, [1, 16])], [f32("s", 0.05)])
    check_against_ort(m, tmp_path)


def test_quantize_dequantize_roundtrip(tmp_path):
    m = build([H.make_node("QuantizeLinear", ["x", "s", "z"], ["q"]),
               H.make_node("DequantizeLinear", ["q", "s", "z"], ["y"])],
              [vi("x", F, [1, 16])], [vi("y", F, [1, 16])],
              [f32("s", 0.05), const(np.array(0, dtype=np.int8), "z")])
    check_against_ort(m, tmp_path)


# ---- ONNX `Constant` NODES (what PyTorch exports) --------------------------

def test_reshape_target_from_constant_node(tmp_path):
    m = build([H.make_node("Constant", [], ["shp"], value=const(np.array([1, -1], dtype=np.int64), "v")),
               H.make_node("Reshape", ["x", "shp"], ["y"])],
              [vi("x", F, [1, 2, 3])], [vi("y", F, [1, 6])])
    check_against_ort(m, tmp_path)


def test_constant_node_value_ints_and_value_float(tmp_path):
    m = build([H.make_node("Constant", [], ["shp"], value_ints=[2, 3]),
               H.make_node("Constant", [], ["hi"], value_float=0.25),
               H.make_node("Reshape", ["x", "shp"], ["r"]),
               H.make_node("Clip", ["r", "", "hi"], ["y"])],
              [vi("x", F, [1, 6])], [vi("y", F, [2, 3])])
    check_against_ort(m, tmp_path)


# ---- Dynamic dimensions ----------------------------------------------------

def _dyn_model():
    W = (np.random.default_rng(0).standard_normal((4, 3, 3, 3)) * 0.1).astype(np.float32)
    return build([H.make_node("Conv", ["x", "w"], ["y"], kernel_shape=[3, 3], pads=[1, 1, 1, 1]),
                  H.make_node("Flatten", ["y"], ["z"])],
                 [vi("x", F, ["N", 3, 8, 8])], [vi("z", F, ["N", 256])], [const(W, "w")])


def test_dynamic_batch_without_binding_gives_clear_error(tmp_path):
    with pytest.raises(ValueError, match=r"dynamic.*--dim|--dim.*dynamic"):
        run_frontend(_dyn_model(), tmp_path)


def test_dynamic_batch_bound_by_dim_name(tmp_path):
    check_against_ort(_dyn_model(), tmp_path, dim_values={"N": 2})


def test_dynamic_batch_bound_by_full_input_shape(tmp_path):
    check_against_ort(_dyn_model(), tmp_path, input_shapes={"x": (3, 3, 8, 8)})


@pytest.mark.parametrize("op", ["MaxPool", "AveragePool"])
def test_pool_default_stride_is_one_not_kernel_size(tmp_path, op):
    # ONNX: omitted `strides` means 1 along every axis. The frontend used to default to the kernel size.
    m = build([H.make_node(op, ["x"], ["y"], kernel_shape=[2, 2])],
              [vi("x", F, [1, 2, 6, 6])], [vi("y", F, [None] * 4)])
    graph = check_against_ort(m, tmp_path)
    assert graph.outputs[0].shape == (1, 2, 5, 5)
