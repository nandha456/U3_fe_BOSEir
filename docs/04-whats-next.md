# 04 — What's next (priority order)

What is **done**: a differential-test harness (60 tests), ~15 correctness fixes, IR integrity
checks, dynamic-dim handling, BatchNorm lowering, `Split`.

What is **not** done — and why it is in this order:

1. **Run real exported models.** Everything so far uses tiny hand-built graphs. Export
   MobileNetV2, ResNet-18 and SqueezeNet from torchvision at a few opsets, add them as a
   test corpus, and track "how many pass". This tells you which gaps *actually* matter
   instead of guessing. (Needs `torch`; was not installed where this work was done.)
2. **Decide what `Requantize` means**, then validate QONNX against a real Brevitas export
   (details: page 02, last section). Until then quantized-QONNX models are the least-trusted path.
3. **Close the remaining README gaps** — each is small, and now has a harness to check it:
   * Batched `MatMul` shape inference (NumPy semantics: broadcast batch dims, 1-D promotion).
   * `Slice` attribute form for opset < 10 (use the `VERSIONED_CONVERT_MAP` pattern).
   * `Gemm` with `alpha`/`beta` ≠ 1 (extra scalar `Mul` on the product / on `C`).
4. **Constant-fold the weight `Transpose` that `Gemm(transB=1)` leaves at runtime**
   (PyTorch's `Linear` always exports `transB=1`, so every FC layer currently carries a
   pointless runtime Transpose of a constant).
5. **More exact decompositions** (each testable the same way as BatchNorm): `Sub`,
   `Div`-by-constant, `LeakyRelu` → `Max(x, αx)`, `HardSigmoid` → `Clip`,
   `Squeeze`/`Unsqueeze` → `Reshape`, `Pad` folded into Conv `pads`.
6. **Graph optimisation passes** on top of the now-trustworthy IR: constant folding,
   identity/dead-node elimination, folding a BatchNorm that directly *precedes* a Conv into
   that Conv's weights, DQ→op→Q fusion.
7. **A handoff format:** serialise BoseIR (JSON + weights) with a version number so the
   middle-end has a stable contract.

> Rule of thumb for every item above: **write the failing comparison-with-ORT test first**,
> then make it pass. That habit is the whole point of page 01.
