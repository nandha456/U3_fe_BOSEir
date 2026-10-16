"""
Stage 1 + 2: Validate and Import

This is the very first thing that touches the .onnx file. Import is a pure,
dumb, 1:1 transcription — every ONNX node becomes one IRNode with the exact
same op_type it had in ONNX. No semantic conversion happens here on purpose;
that's the legalize stage's job. Keeping this stage "dumb" means bugs are
easy to localize.
"""

import numpy as np
import onnx
from onnx import numpy_helper

from frontend.bose_ir import BoseIR, Tensor, IRNode

# ONNX's numeric dtype codes -> simple string names we use internally.
_DTYPE_MAP = {
    1: "float32", 2: "uint8", 3: "int8", 4: "uint16", 5: "int16", 6: "int32", 7: "int64",
    9: "bool", 10: "float16", 11: "float64", 12: "uint32", 13: "uint64",
    16: "bfloat16", 21: "uint4", 22: "int4",     # 4-bit types arrive with opset 21 quantization
}


def validate(onnx_path: str) -> onnx.ModelProto:
    """
    Load the .onnx file and run ONNX's own built-in checker on it.
    Catches a broken/malformed file BEFORE we try to do anything smart
    with it — if this fails, the file itself is the problem, not our code.
    """
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    return model


def _get_opset_version(model: onnx.ModelProto) -> int:
    """
    A .onnx file can technically import several opsets (one per domain).
    We want the version of the MAIN ONNX opset specifically (domain "" or
    "ai.onnx") — that's the one that governs standard op semantics like
    Clip's min/max representation.
    """
    for opset in model.opset_import:
        if opset.domain in ("", "ai.onnx"):
            return opset.version
    return None  # unusual — no main-domain opset declared


def _onnx_attrs_to_dict(node: onnx.NodeProto) -> dict:
    """Convert ONNX's attribute list into a plain python dict."""
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def _constant_node_value(onnx_node: onnx.NodeProto) -> np.ndarray:
    """
    An ONNX `Constant` NODE (not an initializer) stores its payload in one of
    several attributes. PyTorch emits these for Reshape shapes, Slice bounds,
    Clip limits... We turn every flavour into one numpy array so the rest of the
    pipeline only ever has to look in graph.initializers.
    """
    attrs = {a.name: a for a in onnx_node.attribute}
    if "value" in attrs:
        return numpy_helper.to_array(attrs["value"].t)
    if "value_float" in attrs:
        return np.array(attrs["value_float"].f, dtype=np.float32)
    if "value_floats" in attrs:
        return np.array(list(attrs["value_floats"].floats), dtype=np.float32)
    if "value_int" in attrs:
        return np.array(attrs["value_int"].i, dtype=np.int64)
    if "value_ints" in attrs:
        return np.array(list(attrs["value_ints"].ints), dtype=np.int64)
    raise ValueError(
        f"Constant node '{onnx_node.name or onnx_node.output[0]}': unsupported payload "
        f"({sorted(attrs) or 'empty'}) — only value/value_float(s)/value_int(s) are handled "
        f"(sparse_value and string constants are not).")


def import_model(model: onnx.ModelProto) -> BoseIR:
    """
    Walk model.graph.node / initializer / input / output and build our own
    BoseIR/IRNode/Tensor objects — a direct 1:1 transcription of the ONNX
    graph. No op-conversion logic here; that's legalize.legalize().
    """
    onnx_graph = model.graph
    graph = BoseIR(name=onnx_graph.name or "model")
    graph.opset = _get_opset_version(model)

    # 1. Constants (weights, biases, scale factors, ...)
    for init in onnx_graph.initializer:
        arr = numpy_helper.to_array(init)
        graph.initializers[init.name] = arr
        t = graph.get_tensor(init.name)
        t.shape = tuple(arr.shape)
        t.dtype = str(arr.dtype)
        # Constants get their own IRNode so they're visible in the graph,
        # not just floating data — this matches the "Constant" core op.
        const_node = IRNode("Constant", name=f"{init.name}_const", inputs=[], outputs=[t],
                             attrs={"value": arr})
        graph.add_node(const_node)

    # 2. Graph inputs (skip any that are actually initializers, ONNX allows both)
    initializer_names = {i.name for i in onnx_graph.initializer}
    for inp in onnx_graph.input:
        if inp.name in initializer_names:
            continue
        t = graph.get_tensor(inp.name)
        t.shape, t.dtype = _shape_dtype_from_value_info(inp)
        graph.inputs.append(t)

    # 3. Nodes, in the order ONNX gives them (ONNX requires topological order)
    for onnx_node in onnx_graph.node:
        # 3a. An ONNX `Constant` node is just an initializer in disguise — register its
        #     value so later stages can find it in graph.initializers like any weight.
        if onnx_node.op_type == "Constant" and onnx_node.domain in ("", "ai.onnx"):
            arr = _constant_node_value(onnx_node)
            t = graph.get_tensor(onnx_node.output[0])
            t.shape, t.dtype = tuple(arr.shape), str(arr.dtype)
            graph.initializers[t.name] = arr
            graph.add_node(IRNode("Constant", name=onnx_node.name or f"{t.name}_const",
                                  inputs=[], outputs=[t], attrs={"value": arr}))
            continue

        # 3b. "" means "optional input omitted". Keep a None placeholder so the inputs
        #     that follow keep their POSITION (Clip(x, "", max): max must stay 3rd, not become 2nd).
        in_tensors = [graph.get_tensor(n) if n != "" else None for n in onnx_node.input]
        while in_tensors and in_tensors[-1] is None:
            in_tensors.pop()                      # trailing omissions carry no information
        out_tensors = [graph.get_tensor(n) for n in onnx_node.output if n != ""]
        attrs = _onnx_attrs_to_dict(onnx_node)
        node = IRNode(
            op_type=onnx_node.op_type,
            name=onnx_node.name or f"{onnx_node.op_type}_{len(graph.nodes)}",
            inputs=in_tensors,
            outputs=out_tensors,
            attrs=attrs,
            domain=onnx_node.domain,  # "" for standard ONNX, non-empty for QONNX custom ops
        )
        graph.add_node(node)

    # 4. Graph outputs
    for out in onnx_graph.output:
        t = graph.get_tensor(out.name)
        shape, dtype = _shape_dtype_from_value_info(out)
        if t.shape is None:
            t.shape = shape
        if t.dtype is None:
            t.dtype = dtype
        graph.outputs.append(t)

    return graph


def _shape_dtype_from_value_info(value_info):
    """Pull whatever shape/dtype ONNX declared for a graph input/output.

    NOTE: this is a starting point only — per our architecture doc, we don't
    trust this blindly for internal tensors (it's often stale/incomplete).
    Real shape resolution happens in the shape_infer stage.
    """
    shape = None
    dtype = None
    ttype = value_info.type.tensor_type
    if ttype.HasField("elem_type"):
        dtype = _DTYPE_MAP.get(ttype.elem_type, f"dtype_{ttype.elem_type}")
    if ttype.HasField("shape"):
        dims = []
        for d in ttype.shape.dim:
            if d.HasField("dim_value"):
                dims.append(d.dim_value)
            else:
                dims.append(d.dim_param or None)   # "N" for a named symbolic dim, None if unnamed
        shape = tuple(dims)
    return shape, dtype


def bind_input_shapes(graph: BoseIR, input_shapes=None, dim_values=None) -> BoseIR:
    """
    Dynamic dimensions (a batch size written as "N" instead of 1) can't be compiled for a
    static-dataflow chip, and guessing a value would silently change the program. So:
    the caller must pin them, or we stop with a message that says how.

        dim_values   = {"N": 1}                    pin a NAMED dynamic dim everywhere it appears
        input_shapes = {"x": (1, 3, 224, 224)}     pin a whole input's shape explicitly
    """
    input_shapes = input_shapes or {}
    dim_values = dim_values or {}
    for t in graph.inputs:
        if t.name in input_shapes:
            given = tuple(int(d) for d in input_shapes[t.name])
            if t.shape is not None and len(given) != len(t.shape):
                raise ValueError(f"--input-shape for '{t.name}' has rank {len(given)} "
                                 f"but the model declares rank {len(t.shape)}.")
            t.shape = given
            continue
        if t.shape is None:
            raise ValueError(f"Graph input '{t.name}' has no declared shape — pass its shape "
                             f"explicitly, e.g. --input-shape {t.name}=1,3,224,224")
        t.shape = tuple(dim_values.get(d, d) if not isinstance(d, int) else d for d in t.shape)
        bad = [(i, d) for i, d in enumerate(t.shape) if not isinstance(d, int)]
        if bad:
            axis, name = bad[0]
            hint = f"--dim {name}=1" if isinstance(name, str) else f"--input-shape {t.name}=<full shape>"
            raise ValueError(
                f"Graph input '{t.name}' has a dynamic dimension at axis {axis} ({name!r}). "
                f"BoseIR needs fully static shapes — pin it with {hint}.")
    return graph
