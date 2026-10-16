"""
The five models that were already checked into models/ — but now with REAL
assertions instead of "it ran without crashing".
"""
import os
import onnx
import pytest

from oracle import check_against_ort

MODELS = os.path.join(os.path.dirname(__file__), "..", "models")


@pytest.mark.parametrize("name", ["tiny_conv", "convbn", "residual_block", "clip_old_opset", "clip_new_opset"])
def test_model_matches_onnxruntime(name, tmp_path):
    model = onnx.load(os.path.join(MODELS, f"{name}.onnx"))
    check_against_ort(model, tmp_path)


@pytest.mark.parametrize("name", ["convbn", "residual_block"])
def test_batchnorm_is_fully_folded_away(name, tmp_path):
    model = onnx.load(os.path.join(MODELS, f"{name}.onnx"))
    graph = check_against_ort(model, tmp_path)
    ops = [n.op_type for n in graph.nodes]
    assert "BatchNormalization" not in ops
    # BN folded into Conv => no extra Mul/Add nodes needed except the residual Add
    assert "Mul" not in ops
