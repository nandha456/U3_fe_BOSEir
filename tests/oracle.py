"""
Test helpers: build a tiny ONNX model, run it through BOTH
    (a) onnxruntime            -> the trusted reference ("oracle")
    (b) our frontend + ref_interp -> what BoseIR would compute
and assert they agree on values, shapes AND dtypes.

This is "differential testing": we never hard-code expected numbers, we let a
mature implementation (ORT) tell us the right answer.
"""

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto as TP, helper as H, numpy_helper as nh

from frontend import pipeline
import ref_interp

F = TP.FLOAT
vi = H.make_tensor_value_info
const = nh.from_array          # const(np_array, "name") -> initializer


def build(nodes, inputs, outputs, inits=(), opset=13, name="t"):
    g = H.make_graph(nodes, name, inputs, outputs, initializer=list(inits))
    m = H.make_model(g, opset_imports=[H.make_opsetid("", opset)])
    m.ir_version = 10                       # keep it loadable by the installed onnxruntime
    onnx.checker.check_model(m)
    return m


def run_ort(model, feeds):
    sess = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    names = [o.name for o in sess.get_outputs()]
    return dict(zip(names, sess.run(None, feeds)))


def run_frontend(model, tmp_path, **kw):
    p = tmp_path / "m.onnx"
    onnx.save(model, str(p))
    return pipeline.run(str(p), verbose=False, **kw)


def _random_feeds(graph, seed=0):
    rng = np.random.default_rng(seed)
    feeds = {}
    for t in graph.inputs:
        assert all(isinstance(d, int) for d in t.shape), f"input {t.name} still has dynamic dims: {t.shape}"
        feeds[t.name] = rng.standard_normal(t.shape).astype(np.float32)
    return feeds


def check_against_ort(model, tmp_path, rtol=1e-4, atol=1e-5, feeds=None, **frontend_kw):
    """Run the frontend, execute the IR, compare to ORT. Returns the BoseIR graph."""
    graph = run_frontend(model, tmp_path, **frontend_kw)
    feeds = feeds or _random_feeds(graph)
    expected = run_ort(model, feeds)
    got = ref_interp.run_graph(graph, feeds)

    for t in graph.outputs:
        exp = expected[t.name]
        out = got[t.name]
        assert tuple(t.shape) == exp.shape, f"IR says shape {t.shape} for '{t.name}', ORT produced {exp.shape}"
        assert t.dtype == str(exp.dtype), f"IR says dtype {t.dtype} for '{t.name}', ORT produced {exp.dtype}"
        assert out.shape == exp.shape, f"interpreter shape {out.shape} != ORT {exp.shape} for '{t.name}'"
        np.testing.assert_allclose(out, exp, rtol=rtol, atol=atol, err_msg=f"values differ for '{t.name}'")

    _check_intermediate_shapes(model, graph)
    return graph


def _check_intermediate_shapes(model, graph):
    """Every tensor shape the frontend inferred must match onnx's own shape inference
    (wherever onnx managed to infer a fully static shape)."""
    inferred = onnx.shape_inference.infer_shapes(model)
    for v in inferred.graph.value_info:
        dims = [d.dim_value if d.HasField("dim_value") else None for d in v.type.tensor_type.shape.dim]
        if v.name in graph.tensors and None not in dims:
            assert tuple(graph.tensors[v.name].shape) == tuple(dims), (
                f"tensor '{v.name}': frontend inferred {graph.tensors[v.name].shape}, onnx says {tuple(dims)}")
