# 01 — The testing harness

## The problem it solves

Before this work, the `tests/` folder contained scripts that **built** small `.onnx`
files — but nothing that **checked** anything. "It didn't crash" was the only test.
The README said BatchNorm folding was "numerically verified (max diff ~2e-7)", but
that check was not saved anywhere, so nobody could repeat it.

That matters because a compiler's worst failure isn't a crash — it's finishing
successfully and producing a model that computes the **wrong numbers**.

## The idea: let a trusted program grade our homework

We never write the "right answer" by hand. Instead, for every test model we run it
two ways and compare:

```
                       ┌───────────────► onnxruntime ─────────────────────┐
 tiny model (.onnx) ───┤                  (trusted reference = "oracle")   ├──► same? ✔ / ✘
                       └─► our frontend ─► BoseIR ─► ref_interp.py ───────┘
                                                      (executes the IR with numpy)
```

This is called **differential testing**. If our legalization rewrites something
incorrectly, the two answers disagree and the test fails with a message saying which
tensor, and how.

Three things are compared, not just numbers:

1. **Values** — `numpy.allclose` (tolerance 1e-4 relative / 1e-5 absolute, i.e. float32 noise).
2. **Shapes** — the shape our frontend *claims* for each output must equal what ORT really produced.
3. **Dtypes** — likewise (this is what caught `QuantizeLinear` being labelled `float32`).

On top of that, every intermediate tensor's inferred shape is compared with
`onnx.shape_inference`, wherever ONNX itself could work it out.

## The files

| File | Role |
|---|---|
| `tests/ref_interp.py` | A small numpy interpreter for BoseIR. Implements every core op from scratch. |
| `tests/oracle.py` | Helpers: `build()` makes a tiny model, `check_against_ort()` does the comparison above. |
| `tests/test_existing_models.py` | The five models already in `models/`, now with real assertions. |
| `tests/test_regressions.py` | One test per bug that was found (see page 02). |
| `tests/test_decompositions.py` | BatchNorm → Mul+Add and Split, each proven exact (page 03). |
| `tests/test_rejections.py` | Things that must be *refused*, with a helpful message. |
| `tests/test_ir_integrity.py` | `verify()` must detect a deliberately corrupted IR. |
| `pytest.ini` | Tells pytest where to find the code (`pythonpath = . tests`). |
| `tests/make_*.py` | Unchanged: the original model generators. |

### Why `ref_interp.py` imports nothing from `frontend/`

If the checker shared code with the thing it checks, a single mistake could appear on
both sides and cancel out. So the interpreter re-implements each op from the ONNX spec.
It even re-derives the pooling output-size formula separately — and the real
`onnxruntime` is the tie-breaker for both.

## Anatomy of one test

```python
def test_clip_missing_min_only_max(tmp_path):
    # Clip(x, "", max): "no min, but a max" — the frontend used to read `max` as `min`
    m = build([H.make_node("Clip", ["x", "", "mx"], ["y"])],      # 1. build a 1-node model
              [vi("x", F, [1, 16])], [vi("y", F, [1, 16])],
              [f32("mx", 0.5)])
    check_against_ort(m, tmp_path)                                # 2. run both ways, compare
```

`check_against_ort` does: save the model → run the frontend → generate random input →
run ORT → run `ref_interp` on the BoseIR → compare values, shapes, dtypes.

## Adding a new test (recipe)

1. Pick a failing case. Build the smallest model that shows it (one or two nodes).
2. Declare outputs with `vi("y", F, [None] * rank)` if you don't want to hand-compute the shape.
3. Call `check_against_ort(model, tmp_path)`.
4. Run `python -m pytest -k your_test_name`. It should **fail first** (proving it detects the bug), then pass after the fix.

## How to read a failure

```
AssertionError: IR says shape (1, 8) for 'y', ORT produced (1, 2)
```
→ shape inference is wrong (the frontend and the oracle disagree about size).

```
AssertionError: IR says dtype float32 for 'q', ORT produced int8
```
→ a dtype rule is wrong.

```
AssertionError: Not equal to tolerance ... Mismatched elements: 8 / 16
```
→ the *rewrite* is numerically wrong (e.g. a bad BN fold).

## Evidence that the harness works

* **Red before green.** The tests were written *before* the fixes. Run against the
  original frontend, **39 of the first 58 tests failed** — each failure is a real bug.
* **Interpreter validated.** Cases the original frontend already handled correctly
  (grouped/depthwise conv, `count_include_pad` pooling, plain Conv→BN→ReLU) passed
  immediately, which shows the interpreter agrees with ORT where it should.
* **Mutation check.** After the fixes (60 tests, all green) two of the original bugs
  were deliberately reintroduced — negative-Slice handling and in-place mutation of a
  shared weight. The suite went red (4 failures). Restoring the fixes made it green again.

## Known limits (be honest about what is *not* covered)

* All test models are tiny and hand-built. **No real exported network** (e.g. a
  torchvision MobileNet) has been run yet — that is the top item on page 04.
* Only `float32` compute paths are exercised.
* `Requantize` is **not** interpreted: its meaning is an open design question
  (page 02, last section). The interpreter raises `NotImplementedError` on purpose
  rather than guess.
* `MatMul` is only exercised in the 2-D case, because that is all the frontend supports.
