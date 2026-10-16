"""
Stage 4: Shape & Layout Inference

Walks the (already-legalized) graph in order and computes the shape AND dtype of
every tensor, using a small shape function per core op — we only need ~20 of
these, not one per possible ONNX op, because this runs after legalize().

We do NOT trust whatever shape ONNX's value_info said (see import_stage) —
everything here is recomputed from scratch.

Two things this stage also does besides computing shapes:
  * it RESOLVES padding: `auto_pad=SAME_UPPER/SAME_LOWER/VALID` is rewritten into
    explicit `pads` on the node (and `auto_pad` is dropped). After this stage every
    Conv/Pool carries concrete pads, so later stages never need to know about auto_pad.
  * it REFUSES to guess: a shape that isn't fully static, a runtime-computed
    Reshape/Slice parameter, or a non-4-D Conv raises a ValueError that names the node.
"""

import math


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _is_static(shape):
    return shape is not None and all(isinstance(d, int) for d in shape)


def _require_static(node):
    for t in node.inputs:
        if t is not None and not _is_static(t.shape):
            raise ValueError(
                f"Shape inference: input '{t.name}' of node '{node.name}' ({node.op_type}) has "
                f"shape {t.shape}, which is not fully static. BoseIR needs concrete shapes — "
                f"pin dynamic dims with --dim NAME=SIZE or --input-shape.")


def _const(graph, tensor, node, what):
    """Fetch a compile-time-constant tensor's value, or explain why we can't."""
    arr = graph.initializers.get(tensor.name) if tensor is not None else None
    if arr is None:
        raise ValueError(
            f"{node.op_type} '{node.name}': {what} must be a compile-time constant "
            f"(initializer or Constant node), but it is computed at runtime"
            f"{f' (tensor {tensor.name!r})' if tensor is not None else ''}. "
            f"BoseIR is a static-dataflow IR and can't compile that.")
    return arr


def _require_rank4(node, x, what="input"):
    if len(x.shape) != 4:
        raise ValueError(
            f"{node.op_type} '{node.name}': expects a rank-4 (NCHW) {what} but got rank "
            f"{len(x.shape)} {tuple(x.shape)}. 1-D and 3-D convolutions/pools are outside the "
            f"18-op ISA, which only has the 2-D versions.")


def _resolve_pads(node, in_spatial, kernel, strides, dilations):
    """
    Return explicit pads [top, left, bottom, right] for this node, and rewrite the node so
    it carries them (dropping auto_pad). Handles NOTSET / VALID / SAME_UPPER / SAME_LOWER.
    """
    auto = node.attrs.get("auto_pad", "NOTSET")
    if isinstance(auto, bytes):
        auto = auto.decode()
    if auto in ("NOTSET", ""):
        pads = list(node.attrs.get("pads", [0, 0, 0, 0]))
    elif auto == "VALID":
        pads = [0, 0, 0, 0]
    elif auto in ("SAME_UPPER", "SAME_LOWER"):
        begins, ends = [], []
        for size, k, s, d in zip(in_spatial, kernel, strides, dilations):
            out = -(-size // s)                                   # ceil(size / stride)
            total = max(0, (out - 1) * s + d * (k - 1) + 1 - size)
            small, big = total // 2, total - total // 2
            # SAME_UPPER puts the extra padding element at the END, SAME_LOWER at the START
            b, e = (small, big) if auto == "SAME_UPPER" else (big, small)
            begins.append(b)
            ends.append(e)
        pads = begins + ends
    else:
        raise ValueError(f"{node.op_type} '{node.name}': unknown auto_pad value {auto!r}")
    node.attrs["pads"] = pads
    node.attrs.pop("auto_pad", None)
    return pads


def _out_dim(in_dim, k, pad_before, pad_after, stride, dilation, ceil_mode=False):
    span = dilation * (k - 1) + 1
    num = in_dim + pad_before + pad_after - span
    if not ceil_mode:
        return num // stride + 1
    out = -(-num // stride) + 1
    # With ceil_mode the last window may hang off the end — but it must still START inside
    # the input or the left padding, otherwise it would cover only padding. (PyTorch/ORT rule.)
    if (out - 1) * stride >= in_dim + pad_before:
        out -= 1
    return out


# ---------------------------------------------------------------------------
# Shape functions — one per core op
# ---------------------------------------------------------------------------

def _shape_conv2d(node, graph):
    x, w = node.inputs[0], node.inputs[1]
    _require_rank4(node, x)
    _require_rank4(node, w, "weight")
    N, Cin, H, W = x.shape
    Cout, Cg, kh, kw = w.shape
    group = node.attrs.get("group", 1)
    if Cin != Cg * group:
        raise ValueError(f"Conv2D '{node.name}': input has {Cin} channels but weight expects "
                         f"{Cg} x group({group}) = {Cg * group}.")
    strides = node.attrs.get("strides", [1, 1])
    dilations = node.attrs.get("dilations", [1, 1])
    pads = _resolve_pads(node, (H, W), (kh, kw), strides, dilations)   # [top, left, bottom, right]
    Hout = _out_dim(H, kh, pads[0], pads[2], strides[0], dilations[0])
    Wout = _out_dim(W, kw, pads[1], pads[3], strides[1], dilations[1])
    node.outputs[0].shape = (N, Cout, Hout, Wout)
    node.outputs[0].dtype = x.dtype


def _shape_matmul(node, graph):
    a, b = node.inputs[0].shape, node.inputs[1].shape
    # Simplified: 2D matmul, the common case after Gemm decomposition.
    if len(a) == 2 and len(b) == 2:
        if a[1] != b[0]:
            raise ValueError(f"MatMul '{node.name}': shape mismatch {a} x {b}")
        node.outputs[0].shape = (a[0], b[1])
    else:
        raise NotImplementedError(f"MatMul '{node.name}': only 2D x 2D matmul supported right now (got {a} x {b}).")
    node.outputs[0].dtype = node.inputs[0].dtype


def _broadcast_shape(a, b):
    if tuple(a) == tuple(b):
        return tuple(a)
    la, lb = len(a), len(b)
    n = max(la, lb)
    a = (1,) * (n - la) + tuple(a)
    b = (1,) * (n - lb) + tuple(b)
    out = []
    for da, db in zip(a, b):
        if da == db or da == 1 or db == 1:
            out.append(max(da, db))
        else:
            raise ValueError(f"Shapes {a} and {b} aren't broadcast-compatible")
    return tuple(out)


def _shape_elementwise(node, graph):
    """Add / Mul / Max / Min. Max and Min are n-ary in ONNX, so fold over ALL inputs."""
    shape = tuple(node.inputs[0].shape)
    for t in node.inputs[1:]:
        shape = _broadcast_shape(shape, t.shape)
    node.outputs[0].shape = shape
    node.outputs[0].dtype = node.inputs[0].dtype


def _shape_same_as_input(node, graph):
    node.outputs[0].shape = node.inputs[0].shape
    node.outputs[0].dtype = node.inputs[0].dtype


def _shape_pool(node, graph):
    x = node.inputs[0]
    _require_rank4(node, x)
    N, C, H, W = x.shape
    k = node.attrs["kernel_shape"]
    strides = node.attrs.get("strides", [1, 1])        # ONNX default is 1 — NOT the kernel size
    dilations = node.attrs.get("dilations", [1, 1])
    ceil_mode = bool(node.attrs.get("ceil_mode", 0))
    pads = _resolve_pads(node, (H, W), k, strides, dilations)
    Hout = _out_dim(H, k[0], pads[0], pads[2], strides[0], dilations[0], ceil_mode)
    Wout = _out_dim(W, k[1], pads[1], pads[3], strides[1], dilations[1], ceil_mode)
    if Hout < 1 or Wout < 1:
        raise ValueError(f"{node.op_type} '{node.name}': kernel {k} is larger than the padded input "
                         f"{(H, W)} — output would be empty.")
    node.outputs[0].shape = (N, C, Hout, Wout)
    node.outputs[0].dtype = x.dtype


def _shape_global_avg_pool(node, graph):
    x = node.inputs[0]
    _require_rank4(node, x)
    N, C, H, W = x.shape
    node.outputs[0].shape = (N, C, 1, 1)
    node.outputs[0].dtype = x.dtype


def _shape_reshape(node, graph):
    x = node.inputs[0]
    if "flatten_axis" in node.attrs:
        axis = node.attrs["flatten_axis"]
        if axis < 0:
            axis += len(x.shape)
        dim0 = math.prod(x.shape[:axis]) if axis > 0 else 1
        dim1 = math.prod(x.shape[axis:])
        node.outputs[0].shape = (dim0, dim1)
    else:
        if node.attrs.get("allowzero", 0):
            raise ValueError(f"Reshape '{node.name}': allowzero=1 isn't supported.")
        target = _const(graph, node.inputs[1], node, "the target shape").tolist()
        resolved = []
        for i, d in enumerate(target):
            if d == 0:
                resolved.append(x.shape[i])       # 0 means "same as input dim", per ONNX spec
            else:
                resolved.append(int(d))           # -1 stays -1, fixed below
        total = math.prod(x.shape)
        if -1 in resolved:
            known = math.prod(d for d in resolved if d != -1)
            resolved[resolved.index(-1)] = total // known
        if math.prod(resolved) != total:
            raise ValueError(f"Reshape '{node.name}': cannot reshape {tuple(x.shape)} "
                             f"({total} elements) into {tuple(resolved)}.")
        node.outputs[0].shape = tuple(resolved)
    node.outputs[0].dtype = x.dtype


def _shape_concat(node, graph):
    axis = node.attrs["axis"]
    shapes = [t.shape for t in node.inputs]
    out = list(shapes[0])
    out[axis] = sum(s[axis] for s in shapes)
    node.outputs[0].shape = tuple(out)
    node.outputs[0].dtype = node.inputs[0].dtype


def _shape_transpose(node, graph):
    x = node.inputs[0]
    perm = node.attrs.get("perm", list(reversed(range(len(x.shape)))))
    node.outputs[0].shape = tuple(x.shape[p] for p in perm)
    node.outputs[0].dtype = x.dtype


def _shape_slice(node, graph):
    """
    Slice with starts/ends (and optional axes/steps) as constant inputs (opset >= 10).
    Follows the ONNX spec: negative starts/ends/axes count from the end, indices are then
    clamped into range, and a negative step walks backwards. Optional inputs may be
    omitted (None placeholder) — then the defaults apply.
    """
    x = node.inputs[0]
    ins = node.inputs

    def opt(i):
        return ins[i] if len(ins) > i and ins[i] is not None else None

    starts = _const(graph, ins[1], node, "starts").tolist()
    ends = _const(graph, ins[2], node, "ends").tolist()
    axes = _const(graph, opt(3), node, "axes").tolist() if opt(3) is not None else list(range(len(starts)))
    steps = _const(graph, opt(4), node, "steps").tolist() if opt(4) is not None else [1] * len(starts)

    shape = list(x.shape)
    for s, e, ax, st in zip(starts, ends, axes, steps):
        if ax < 0:
            ax += len(shape)
        if st == 0:
            raise ValueError(f"Slice '{node.name}': step must not be 0.")
        dim = shape[ax]
        if s < 0:
            s += dim
        if e < 0:
            e += dim
        if st > 0:
            s, e = max(0, min(s, dim)), max(0, min(e, dim))
        else:
            s, e = max(0, min(s, dim - 1)), max(-1, min(e, dim - 1))
        shape[ax] = len(range(s, e, st))
    node.outputs[0].shape = tuple(shape)
    node.outputs[0].dtype = x.dtype


def _shape_split(node, graph):
    x = node.inputs[0]
    axis = node.attrs["axis"]
    if axis < 0:
        axis += len(x.shape)
    dim = x.shape[axis]
    n = len(node.outputs)
    sizes = node.attrs.get("split")
    if sizes is None:                       # equal split; if uneven, the last piece is smaller
        chunk = -(-dim // n)
        sizes = [chunk] * (n - 1) + [dim - chunk * (n - 1)]
    if len(sizes) != n or sum(sizes) != dim:
        raise ValueError(f"Split '{node.name}': sizes {list(sizes)} don't match {n} outputs / "
                         f"axis length {dim}.")
    for out, size in zip(node.outputs, sizes):
        shape = list(x.shape)
        shape[axis] = int(size)
        out.shape = tuple(shape)
        out.dtype = x.dtype


def _shape_quantize_linear(node, graph):
    # Output type = the zero-point's type (int8/uint8/...); with no zero-point the spec says uint8.
    x = node.inputs[0]
    zp = node.inputs[2] if len(node.inputs) > 2 else None
    node.outputs[0].shape = x.shape
    node.outputs[0].dtype = zp.dtype if zp is not None and zp.dtype else "uint8"


def _shape_dequantize_linear(node, graph):
    # Output type = the scale's type (float32 normally) — NOT the integer input's type.
    x, scale = node.inputs[0], node.inputs[1]
    node.outputs[0].shape = x.shape
    node.outputs[0].dtype = scale.dtype or "float32"


def _shape_requantize(node, graph):
    x = node.inputs[0]
    node.outputs[0].shape = x.shape
    bitwidth = node.attrs.get("bitwidth")
    signed = node.attrs.get("signed", 1)
    if bitwidth is not None:
        bw = int(bitwidth) if not hasattr(bitwidth, "item") else int(bitwidth.item())
        node.outputs[0].dtype = f"{'int' if signed else 'uint'}{bw}"
    else:
        node.outputs[0].dtype = x.dtype


SHAPE_FNS = {
    "Conv2D": _shape_conv2d,
    "MatMul": _shape_matmul,
    "Add": _shape_elementwise,
    "Mul": _shape_elementwise,
    "Max": _shape_elementwise,
    "Min": _shape_elementwise,
    "ReLU": _shape_same_as_input,
    "Clip": _shape_same_as_input,
    "MaxPool": _shape_pool,
    "AveragePool": _shape_pool,
    "GlobalAveragePool": _shape_global_avg_pool,
    "Reshape": _shape_reshape,
    "Concat": _shape_concat,
    "Transpose": _shape_transpose,
    "Slice": _shape_slice,
    "Split": _shape_split,
    "QuantizeLinear": _shape_quantize_linear,
    "DequantizeLinear": _shape_dequantize_linear,
    "Requantize": _shape_requantize,
    "Identity": _shape_same_as_input,
    "Constant": None,  # already resolved during import — nothing to do
}


def infer_shapes(graph):
    for node in graph.nodes:
        if node.op_type == "Constant":
            continue
        fn = SHAPE_FNS.get(node.op_type)
        if fn is None:
            raise ValueError(f"No shape function registered for core op '{node.op_type}' (node '{node.name}') — "
                              f"this means CORE_OPS and SHAPE_FNS have drifted out of sync.")
        _require_static(node)
        fn(node, graph)
    return graph
