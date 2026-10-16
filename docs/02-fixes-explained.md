# 02 — The fixes, explained

Each entry follows the same shape: **what you would have seen → why it happened → what
changed → which test proves it.** Entries are grouped by how dangerous they were.

> **Reading tip:** group A is the important one. Those bugs let the compiler *succeed*
> while producing a wrong program.

---

## A. Silent-wrong bugs (compiled "fine", wrong result)

### A1. `Slice` ignored negative indices and steps
* **Seen:** `x[:, -2:]` on a width-8 tensor was reported as shape `(1, 8)`. Correct: `(1, 2)`.
* **Why:** In ONNX, `-2` means "2nd from the end". The code clamped negatives to 0, so it
  read them as "from the start". Negative steps (reversing) were not handled at all.
* **Fix:** `shape_infer._shape_slice` now follows the spec: add the dimension size to
  negative values, then clamp, then count the elements the (possibly negative) step selects.
* **Test:** `test_slice_negative_indices_and_steps` (4 cases).

### A2. Skipped optional inputs shifted everything left — `Clip(x, "", max)`
* **Seen:** a clip with *no minimum but a maximum of 6.0* was compiled as "minimum = 6.0".
  For input `[-3, 2, 7, 9]` ONNX gives `[-3, 2, 6, 6]`; the frontend's meaning gave `[6, 6, 7, 9]`.
* **Why:** ONNX marks a skipped optional input with an empty name `""`. The importer
  threw those away (`if n != ""`), so the remaining inputs slid forward one position and
  `max` was read as the 2nd input — which Clip treats as `min`.
* **Fix:** `import_stage.import_model` now keeps a `None` **placeholder** so every input
  keeps its position. `IRNode` tolerates `None`; `_convert_clip_v11` reads each bound
  with `_opt_const_scalar`, which returns the default (±infinity) when the slot is `None`.
  Trailing omissions are still dropped (they carry no information).
* **Tests:** `test_clip_missing_min_only_max`, `test_clip_missing_max_only_min`,
  `test_slice_with_omitted_axes_input`.

### A3. `auto_pad` was ignored (TensorFlow-style "SAME" padding)
* **Seen:** a 3×3 convolution with `auto_pad=SAME_UPPER` on an 8×8 input was reported as
  6×6 output. It should stay 8×8.
* **Why:** shape inference only looked at the explicit `pads` attribute, which is absent
  when `auto_pad` is used, so it assumed zero padding.
* **Fix:** `shape_infer._resolve_pads` computes the explicit padding for
  `NOTSET`/`VALID`/`SAME_UPPER`/`SAME_LOWER`, writes it into the node's `pads`, and removes
  `auto_pad`. After stage 4 **every Conv/Pool carries explicit pads**, so later stages
  never need to know `auto_pad` existed. (When the total padding is odd, SAME_UPPER puts
  the extra element at the end and SAME_LOWER at the start.)
* **Test:** `test_conv_auto_pad` (3 modes × 2 strides).

### A4. Pooling ignored `ceil_mode`
* **Seen:** MaxPool of a 5×5 input, kernel 2, stride 2, `ceil_mode=1`: reported 2×2, real 3×3.
* **Why:** the output-size formula always rounded down.
* **Fix:** `_out_dim` rounds up in `ceil_mode`, then applies the PyTorch/ORT rule that the
  last window must *start* inside the input or its left padding (otherwise it would cover
  only padding).
* **Test:** `test_pool_ceil_mode`; `test_avgpool_count_include_pad` guards the averaging side.

### A5. Pooling's default stride was the kernel size (it should be 1)
* **Seen:** a `MaxPool` with `kernel_shape=[2,2]` and no `strides` on a 6×6 input produced
  3×3; ONNX says 5×5.
* **Why:** the code defaulted `strides` to `kernel_shape`. That is PyTorch's *Python-API*
  default, not ONNX's. (Exported models usually write `strides` out, which hid this.)
* **Fix:** default `[1, 1]` in `_shape_pool`.
* **Test:** `test_pool_default_stride_is_one_not_kernel_size`.

### A6. BatchNorm folding silently corrupted weights shared with another layer
* **Seen:** when two Convs share one weight tensor (weight tying) and only the first is
  followed by BatchNorm, the second Conv's output changed.
* **Why:** the fold overwrote the shared weight **in place**, so the other reader saw the
  modified values too.
* **Fix:** `legalize._set_private_constant`. If the Conv is the tensor's only reader, it is
  updated in place (cheap). If the tensor is shared, the Conv gets its **own fresh copy**
  (`<node>_weight_folded`) and the original is left untouched.
* **Test:** `test_bn_fold_does_not_corrupt_shared_weight` (also asserts the original array is unchanged).

### A7. BatchNorm folded away a Conv output that the caller could still observe
* **Seen:** if the Conv's raw output was *also* a declared graph output, folding changed
  what that output contained.
* **Fix:** `_fold_batchnorm` skips the fold when the Conv output is in `graph.outputs`.
  The BN is then lowered to Mul + Add instead (page 03), so nothing is lost.
* **Test:** `test_bn_not_folded_when_conv_output_is_also_a_graph_output`.

### A8. `QuantizeLinear` / `DequantizeLinear` reported the wrong dtype
* **Seen:** the output of `QuantizeLinear` was labelled `float32` (it is `int8`/`uint8`).
* **Why:** both were treated as "output type = input type".
* **Fix:** `_shape_quantize_linear` uses the **zero-point's** dtype (default `uint8`, per
  spec); `_shape_dequantize_linear` uses the **scale's** dtype (normally `float32`).
* **Tests:** `test_quantize_output_dtype_is_int8`, `test_quantize_default_zero_point_is_uint8`,
  `test_quantize_dequantize_roundtrip`.

### A9. `Max` / `Min` with more than two inputs only looked at two
* **Fix:** `_shape_elementwise` now folds the broadcast over *all* inputs (ONNX `Max`/`Min` are n-ary).

---

## B. Crashes with unhelpful errors → now work, or fail with a useful message

### B1. ONNX `Constant` nodes were not understood
* **Seen:** `KeyError: 'shp'` on any model where a Reshape's target shape (or a Slice bound,
  or a Clip limit) comes from a `Constant` **node**. This is how PyTorch usually exports them.
* **Why:** the importer registered *initializers* as constants but copied `Constant` nodes
  verbatim, leaving the raw protobuf object in the attributes. Later stages looked in
  `graph.initializers` and found nothing.
* **Fix:** `import_stage._constant_node_value` converts every payload flavour
  (`value`, `value_float(s)`, `value_int(s)`) to a numpy array and registers it as an
  initializer, exactly like a weight. Unsupported payloads (sparse, string) raise a clear error.
* **Tests:** `test_reshape_target_from_constant_node`, `test_constant_node_value_ints_and_value_float`.

### B2. Dynamic dimensions (`"N"` instead of `1`) crashed with `TypeError`
* **Why:** unknown dims became `None`, and `None * 3` explodes deep inside shape inference.
* **Design decision:** a static-dataflow chip cannot run an unknown batch size, and silently
  guessing `1` would change the program. So the user must pin it.
* **Fix:** symbolic names are now kept at import; `import_stage.bind_input_shapes` applies
  `--dim N=1` (pin a named dim everywhere) or `--input-shape x=1,3,224,224` (pin a whole
  input). If anything is left unpinned the error says exactly what to type:
  `Graph input 'x' has a dynamic dimension at axis 0 ('N') ... pin it with --dim N=1.`
  `verify()` additionally rejects any tensor that still has a non-integer dim.
* **Tests:** `test_dynamic_batch_without_binding_gives_clear_error`,
  `test_dynamic_batch_bound_by_dim_name`, `test_dynamic_batch_bound_by_full_input_shape`.

### B3. 1-D / 3-D convolutions died with `not enough values to unpack`
* **Fix:** shape functions call `_require_rank4`, which raises
  `Conv2D 'x': expects a rank-4 (NCHW) input but got rank 3 ...`. `Conv2D` also checks
  `input channels == weight channels × group`.
* **Test:** `test_conv1d_is_rejected_cleanly`.

### B4. A runtime-computed `Reshape` shape / `Slice` bound gave a bare `KeyError`
* **Fix:** `shape_infer._const` raises
  `Reshape 'n': the target shape must be a compile-time constant ... computed at runtime`.
* **Test:** `test_reshape_with_runtime_shape_is_rejected`.

---

## C. IR integrity (so the *next* rewrite can't silently corrupt the graph)

* **Each tensor remembers who produces and who reads it.** If a rewrite forgets to update
  those links, the IR is corrupt in a way that is hard to notice. The original code had one
  such bug (documented in the old README: BN removal left stale `consumers` entries).
* **`verify()` now has a use-def check** (`_check_use_def_consistency`): every node's
  inputs/outputs must agree with the tensors' `.consumers` / `.producer`, every tensor must
  be in the registry, and no tensor may reference a node that isn't in the graph.
  **Tests:** `tests/test_ir_integrity.py` corrupts a good graph four different ways and
  asserts `verify()` catches each.
* **`BoseIR.replace_node()` was dead and wrong** (never called; it detached the wrong side).
  It is fixed, `remove_node()` now detaches first, and both use the new `IRNode.detach()`.
* **`BoseIR.dump()`** (and `--dump`) prints the final graph so you can actually *look* at
  what the pipeline produced — before this the pipeline printed nothing about its result.
* More ONNX dtypes are mapped (int16/uint16/uint32/uint64/bfloat16 and the 4-bit types
  that opset 21 quantization uses) instead of showing up as `dtype_22`.

---

## Open question: what does `Requantize` mean?  (needs a decision — NOT changed)

The QONNX `Quant` op **quantizes and then dequantizes**: it takes a float tensor and
returns a float tensor whose values are snapped to the quantization grid ("fake quant").
The frontend converts it to a `Requantize` node, and `shape_infer._shape_requantize`
labels the output an **integer** type (`int8`, `int4`, ...).

* If your architecture defines `Requantize` as *"produces integers"*, then the conversion drops
  the dequantize half, and the dtype label is a deliberate choice.
* If it should preserve QONNX's float semantics, the dtype is wrong, and the cleanest
  lowering is `QuantizeLinear` + `DequantizeLinear` (for ≥ 8-bit) or a documented
  fake-quant meaning for `Requantize`.

This is a **design decision, not a bug I can fix unilaterally**, and it needs a real
Brevitas-exported model to settle. Until then `ref_interp.py` raises `NotImplementedError`
for `Requantize` instead of guessing, so no test can accidentally "prove" a wrong meaning.
