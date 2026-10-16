# 03 — New operations: BatchNorm (anywhere) and Split

Before: a `BatchNormalization` that was not directly after a Conv was **rejected**
("no legalization rule exists"), and `Split` was listed among the 18 core ops in the README
but had no implementation at all.

---

## BatchNormalization that can't fold → exact `Mul` + `Add`

### Why it matters
The frontend already *folds* `Conv → BatchNorm` by baking the BN into the Conv weights.
But many real networks put BN somewhere a fold can't reach:

* **Pre-activation ResNets** (ResNet-v2): `BN → ReLU → Conv` — the BN comes *before* the Conv.
* BatchNorm after a fully-connected layer (`BatchNorm1d`).
* BN whose preceding Conv output is used by something else too.

All of these used to be rejected, even though the chip has everything needed to run them.

### The math (why it is exact, not an approximation)

Inference-mode BatchNorm is:

```
y = (x − mean) / sqrt(var + ε) · gamma + beta
```

`mean`, `var`, `gamma`, `beta` are constants, so rearrange into **one multiply and one add
per channel**:

```
scale = gamma / sqrt(var + ε)
shift = beta − mean · scale
y     = x · scale + shift
```

`Mul` and `Add` are both core ops. No approximation: the only difference from running
BatchNorm directly is ordinary floating-point rounding.

### The one subtle part: shaping the constants
`scale` and `shift` have one value per channel, `[C]`, but they must line up with the
**channel axis (dim 1)** of the input:

| Input | Constants reshaped to |
|---|---|
| `(N, C)` after a fully-connected layer | `[C]` |
| `(N, C, L)` | `[C, 1]` |
| `(N, C, H, W)` | `[C, 1, 1]` |

That requires the input's **rank**, but shapes of intermediate tensors aren't known until
stage 4 (which runs *after* legalize). `legalize._static_rank` works the rank out from
whatever produced the tensor (a Conv or pool ⇒ 4, `Gemm`/`Flatten` ⇒ 2, element-wise ops ⇒
the max of their inputs, ReLU/Clip/etc. ⇒ same as their input, ...). If it genuinely can't
tell, the frontend **refuses with a clear error** instead of guessing.

### Where in the code
`legalize._convert_batchnorm`, registered as `CONVERT_MAP["BatchNormalization"]`. The
Conv→BN **fold still runs first** (Phase A) and is still preferred, because folding removes the
operation entirely while the lowering keeps two cheap ops.

### What is still rejected (deliberately)
* **Training-mode** BatchNorm (`training_mode=1`, multiple outputs) — export in eval mode.
* BN whose gamma/beta/mean/var are **not constants** — they can't be baked in.

### Tests (`tests/test_decompositions.py`)
`test_standalone_batchnorm_becomes_mul_add` (ranks 4, 3 and 2, and asserts the result is
exactly `[Mul, Add]`), `test_preactivation_batchnorm_before_conv`,
`test_batchnorm_after_relu_after_conv_rank_is_inferred`, plus the rejection tests in
`test_rejections.py`.

---

## Split

`Split` cuts one tensor into several along an axis. It is now a core op (`CORE_OPS`,
`SHAPE_FNS`, converter, interpreter).

### The opset wrinkle (why there are two converters)
Where the piece sizes live changed between ONNX versions — the same situation the
project already handles for `Clip`, using the `VERSIONED_CONVERT_MAP` pattern:

| Opset | Piece sizes are... | Converter |
|---|---|---|
| 1 – 12 | an attribute `split` (optional) | `_convert_split_v1` |
| 13+ | an optional constant **input** (opset 18 also adds `num_outputs`) | `_convert_split_v13` |

Both normalise to the **same** internal form, `attrs = {"axis": a, "split": [sizes] or None}`,
so shape inference, verification and the interpreter see only one representation.
`split=None` means "equal pieces"; if the length doesn't divide evenly the **last** piece is
smaller (the opset-18 rule).

### Tests
`test_split_equal_opset11`, `test_split_explicit_sizes_attribute_opset11`,
`test_split_explicit_sizes_input_opset13`, `test_split_num_outputs_uneven_opset18`,
`test_split_feeds_downstream_ops` — each compared with onnxruntime at the matching opset.
