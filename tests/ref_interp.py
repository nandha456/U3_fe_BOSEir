"""
Reference interpreter for BoseIR.

Purpose: EXECUTE a post-pipeline BoseIR graph with plain numpy so we can
compare its numbers against onnxruntime. If the frontend rewrites a model
wrongly (bad BN fold, wrong Clip bound, wrong Split...), the two outputs
disagree and a test fails.

Important design rule: this file is deliberately written WITHOUT importing
anything from frontend/shape_infer.py or legalize.py. If the oracle shared
code with the thing it checks, one bug could hide on both sides. Every op
here is a from-scratch numpy implementation of the ONNX spec.

Usage:
    outputs = run_graph(graph, {"x": np_array})     # -> {output_name: np_array}
"""

import math

import numpy as np


# ---------------------------------------------------------------------------
# Spatial helpers (conv / pool)
# ---------------------------------------------------------------------------

def _out_dim(in_dim, span, pad_total, stride, ceil_mode, pad_begin):
    """Spec formula for conv/pool output size (span = dilation*(k-1)+1)."""
    num = in_dim + pad_total - span
    if not ceil_mode:
        return num // stride + 1
    out = -(-num // stride) + 1
    # PyTorch/ORT rule: the last (ceil) window must start inside input or left pad.
    if (out - 1) * stride >= in_dim + pad_begin:
        out -= 1
    return out


def _conv2d(x, w, b, pads, strides, dils, group):
    N, C, H, W = x.shape
    O, Cg, kh, kw = w.shape
    pt, pl, pb, pr = pads
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb), (pl, pr)))
    Ho = (H + pt + pb - dils[0] * (kh - 1) - 1) // strides[0] + 1
    Wo = (W + pl + pr - dils[1] * (kw - 1) - 1) // strides[1] + 1
    out = np.zeros((N, O, Ho, Wo), dtype=np.float32)
    opg = O // group
    for g in range(group):
        xs = xp[:, g * Cg:(g + 1) * Cg]
        ws = w[g * opg:(g + 1) * opg]
        for i in range(kh):
            for j in range(kw):
                h0, w0 = i * dils[0], j * dils[1]
                patch = xs[:, :, h0:h0 + strides[0] * (Ho - 1) + 1:strides[0],
                           w0:w0 + strides[1] * (Wo - 1) + 1:strides[1]]
                out[:, g * opg:(g + 1) * opg] += np.einsum("nchw,oc->nohw", patch, ws[:, :, i, j])
    if b is not None:
        out += b.reshape(1, -1, 1, 1)
    return out


def _pool2d(x, kind, k, strides, pads, dils, ceil_mode, count_include_pad):
    N, C, H, W = x.shape
    pt, pl, pb, pr = pads
    spans = [dils[0] * (k[0] - 1) + 1, dils[1] * (k[1] - 1) + 1]
    Ho = _out_dim(H, spans[0], pt + pb, strides[0], ceil_mode, pt)
    Wo = _out_dim(W, spans[1], pl + pr, strides[1], ceil_mode, pl)
    extra_h = max(0, (Ho - 1) * strides[0] + spans[0] - (H + pt + pb))
    extra_w = max(0, (Wo - 1) * strides[1] + spans[1] - (W + pl + pr))
    fill = -np.inf if kind == "max" else 0.0
    xp = np.pad(x, ((0, 0), (0, 0), (pt, pb + extra_h), (pl, pr + extra_w)), constant_values=fill)

    # "how many real (countable) elements are in each window" mask, for averaging
    ones = np.ones((1, 1, H, W), dtype=np.float32)
    pad_fill = 1.0 if count_include_pad else 0.0
    mask = np.pad(ones, ((0, 0), (0, 0), (pt, pb), (pl, pr)), constant_values=pad_fill)
    mask = np.pad(mask, ((0, 0), (0, 0), (0, extra_h), (0, extra_w)), constant_values=0.0)

    acc = None
    cnt = np.zeros((1, 1, Ho, Wo), dtype=np.float32)
    for i in range(k[0]):
        for j in range(k[1]):
            h0, w0 = i * dils[0], j * dils[1]
            sl = (slice(None), slice(None),
                  slice(h0, h0 + strides[0] * (Ho - 1) + 1, strides[0]),
                  slice(w0, w0 + strides[1] * (Wo - 1) + 1, strides[1]))
            patch = xp[sl]
            if kind == "max":
                acc = patch if acc is None else np.maximum(acc, patch)
            else:
                acc = patch.copy() if acc is None else acc + patch
                cnt = cnt + mask[sl]
    return acc if kind == "max" else (acc / cnt).astype(np.float32)


# ---------------------------------------------------------------------------
# Quantization helpers
# ---------------------------------------------------------------------------

def _per_axis(v, x, axis):
    v = np.asarray(v)
    if v.ndim == 1 and v.size > 1 and x.ndim > 1:
        shape = [1] * x.ndim
        shape[axis % x.ndim] = v.size
        return v.reshape(shape)
    return v


def _quantize_linear(x, scale, zp, axis):
    qdtype = zp.dtype if zp is not None else np.dtype("uint8")
    info = np.iinfo(qdtype)
    s = _per_axis(scale, x, axis)
    z = _per_axis(zp, x, axis).astype(np.int32) if zp is not None else 0
    q = np.rint(x / s) + z                      # np.rint = round-half-to-even, as the spec says
    return np.clip(q, info.min, info.max).astype(qdtype)


def _dequantize_linear(x, scale, zp, axis):
    s = _per_axis(scale, x, axis)
    z = _per_axis(zp, x, axis).astype(np.float32) if zp is not None else 0.0
    return ((x.astype(np.float32) - z) * s).astype(np.float32)


# ---------------------------------------------------------------------------
# Graph execution
# ---------------------------------------------------------------------------

def _reshape(x, target):
    target = [int(d) for d in target]
    out = [x.shape[i] if d == 0 else d for i, d in enumerate(target)]
    return x.reshape(out)


def _eval_node(node, vals):
    ins = [vals[t.name] if t is not None else None for t in node.inputs]
    a = node.attrs
    op = node.op_type

    if op == "Constant":
        return [a["value"]]
    if op == "Identity":
        return [ins[0]]
    if op == "ReLU":
        return [np.maximum(ins[0], 0)]
    if op == "Add":
        return [ins[0] + ins[1]]
    if op == "Mul":
        return [ins[0] * ins[1]]
    if op in ("Max", "Min"):
        f = np.maximum if op == "Max" else np.minimum
        out = ins[0]
        for v in ins[1:]:
            out = f(out, v)
        return [out]
    if op == "Clip":
        return [np.clip(ins[0], a["min"], a["max"]).astype(ins[0].dtype)]
    if op == "MatMul":
        return [np.matmul(ins[0], ins[1])]
    if op == "Conv2D":
        return [_conv2d(ins[0], ins[1], ins[2] if len(ins) > 2 else None,
                        a.get("pads", [0, 0, 0, 0]), a.get("strides", [1, 1]),
                        a.get("dilations", [1, 1]), a.get("group", 1))]
    if op in ("MaxPool", "AveragePool"):
        k = a["kernel_shape"]
        return [_pool2d(ins[0], "max" if op == "MaxPool" else "avg", k,
                        a.get("strides", [1, 1]), a.get("pads", [0, 0, 0, 0]),
                        a.get("dilations", [1, 1]), bool(a.get("ceil_mode", 0)),
                        bool(a.get("count_include_pad", 0)))]
    if op == "GlobalAveragePool":
        return [ins[0].mean(axis=(2, 3), keepdims=True).astype(np.float32)]
    if op == "Reshape":
        if "flatten_axis" in a:
            ax = a["flatten_axis"]
            d0 = math.prod(ins[0].shape[:ax]) if ax > 0 else 1
            return [ins[0].reshape(d0, -1)]
        return [_reshape(ins[0], ins[1])]
    if op == "Concat":
        return [np.concatenate(ins, axis=a["axis"])]
    if op == "Transpose":
        return [np.transpose(ins[0], a.get("perm"))]
    if op == "Slice":
        x = ins[0]
        starts, ends = ins[1], ins[2]
        axes = ins[3] if len(ins) > 3 and ins[3] is not None else np.arange(len(starts))
        steps = ins[4] if len(ins) > 4 and ins[4] is not None else np.ones(len(starts), dtype=np.int64)
        idx = [slice(None)] * x.ndim
        for s, e, ax, st in zip(starts, ends, axes, steps):
            idx[int(ax)] = slice(int(s), int(e), int(st))     # python slicing == ONNX semantics
        return [x[tuple(idx)]]
    if op == "Split":
        x, axis = ins[0], a["axis"]
        n = len(node.outputs)
        sizes = a.get("split")
        if sizes is None:
            chunk = -(-x.shape[axis] // n)
            sizes = [chunk] * (n - 1) + [x.shape[axis] - chunk * (n - 1)]
        cuts = np.cumsum(sizes)[:-1]
        return list(np.split(x, cuts, axis=axis))
    if op == "QuantizeLinear":
        return [_quantize_linear(ins[0], ins[1], ins[2] if len(ins) > 2 else None, a.get("axis", 1))]
    if op == "DequantizeLinear":
        return [_dequantize_linear(ins[0], ins[1], ins[2] if len(ins) > 2 else None, a.get("axis", 1))]
    if op == "Requantize":
        raise NotImplementedError(
            "Requantize semantics (float 'fake-quant' vs integer output) are an open design "
            "question — see docs/02-fixes-explained.md, 'Open question: Requantize'.")
    raise NotImplementedError(f"ref_interp has no implementation for op '{op}'")


def run_graph(graph, feeds):
    """Execute graph.nodes in order. feeds: {input_name: ndarray}. Returns {output_name: ndarray}."""
    vals = dict(feeds)
    for node in graph.nodes:
        outs = _eval_node(node, vals)
        for t, v in zip(node.outputs, outs):
            vals[t.name] = v
    return {t.name: vals[t.name] for t in graph.outputs}


def run_graph_all(graph, feeds):
    """Like run_graph but returns EVERY tensor value (for intermediate-shape checks)."""
    vals = dict(feeds)
    for node in graph.nodes:
        for t, v in zip(node.outputs, _eval_node(node, vals)):
            vals[t.name] = v
    return vals
