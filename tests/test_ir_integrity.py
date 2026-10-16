"""
verify() must catch a *corrupted IR*, not just a wrong op name. Two of the bugs
in the project's history (stale consumer lists, shared weights) were IR
corruption that nothing noticed. These tests pin the new invariants.
"""
import os
import onnx
import pytest

from frontend import pipeline
from frontend.bose_ir import IRNode, Tensor
from frontend.verify import verify

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")


def _good_graph():
    return pipeline.run(os.path.join(MODELS, "residual_block.onnx"), verbose=False)


def test_clean_graph_passes():
    assert verify(_good_graph())


def test_stale_consumer_is_detected():
    g = _good_graph()
    victim = next(iter(g.inputs))
    IRNode("Identity", "ghost", [victim], [Tensor("ghost_out")])     # registers itself as a consumer, but isn't in the graph
    with pytest.raises(ValueError, match="use-def|consumer"):
        verify(g)


def test_wrong_producer_is_detected():
    g = _good_graph()
    node = next(n for n in g.nodes if n.op_type == "Conv2D")
    node.outputs[0].producer = None
    with pytest.raises(ValueError, match="use-def|producer"):
        verify(g)


def test_unresolved_dynamic_dim_is_detected():
    g = _good_graph()
    node = next(n for n in g.nodes if n.op_type == "Conv2D")
    node.outputs[0].shape = ("N",) + tuple(node.outputs[0].shape[1:])
    with pytest.raises(ValueError, match="dynamic|unresolved"):
        verify(g)


def test_tensor_missing_from_registry_is_detected():
    g = _good_graph()
    node = next(n for n in g.nodes if n.op_type == "Conv2D")
    g.tensors.pop(node.outputs[0].name)
    with pytest.raises(ValueError, match="registry|use-def"):
        verify(g)
