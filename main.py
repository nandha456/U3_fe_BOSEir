"""
Entry point. Run like:

    python main.py models/tiny_conv.onnx
    python main.py models/tiny_conv.onnx --dump                  # also print the final BoseIR
    python main.py my_model.onnx --dim N=1                       # pin a dynamic dimension called "N"
    python main.py my_model.onnx --input-shape x=1,3,224,224     # pin a whole input shape
"""

import argparse
import sys

from frontend import pipeline


def _parse_kv(items, parse_value):
    out = {}
    for item in items or []:
        if "=" not in item:
            sys.exit(f"Expected NAME=VALUE, got '{item}'")
        k, v = item.split("=", 1)
        out[k] = parse_value(v)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="ONNX -> BoseIR frontend")
    ap.add_argument("onnx_file")
    ap.add_argument("--dim", action="append", metavar="NAME=SIZE",
                    help="pin a named dynamic dimension (repeatable)")
    ap.add_argument("--input-shape", action="append", metavar="INPUT=D0,D1,...",
                    help="pin the full shape of one graph input (repeatable)")
    ap.add_argument("--dump", action="store_true", help="print the resulting BoseIR")
    args = ap.parse_args()

    graph = pipeline.run(
        args.onnx_file,
        dim_values=_parse_kv(args.dim, int),
        input_shapes=_parse_kv(args.input_shape, lambda v: tuple(int(d) for d in v.split(","))),
    )
    if args.dump:
        print(graph.dump())
