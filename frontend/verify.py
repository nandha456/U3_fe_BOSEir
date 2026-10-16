"""
Stage 6: IR Verification

The last gate before handing the graph to the middle-end. Checks:
    1. every node's op is one of our 18 core ops (nothing slipped through
       legalize with the wrong op type — defense in depth)
    2. the graph is in a valid topological order (every node's inputs are
       produced earlier, or are graph inputs/constants)
    3. every tensor has a resolved shape and dtype
    4. no orphaned nodes (every output either feeds something else, or is
       a declared graph output)
    5. use-def consistency: every "who reads / who produces" link recorded on the
       tensors agrees with the actual node list (catches stale references left
       behind by a rewrite — the class of bug that once hid a dead node)
    6. no dynamic dimensions survive: every dim of every tensor is a plain int
"""

CORE_OPS = {
    "Conv2D", "MatMul", "Add", "Mul", "Max", "Min", "ReLU", "Clip",
    "MaxPool", "AveragePool", "GlobalAveragePool", "Reshape", "Concat",
    "Slice", "Split", "Transpose", "QuantizeLinear", "DequantizeLinear",
    "Requantize", "Constant", "Identity",
}


def verify(graph):
    _check_only_core_ops(graph)
    _check_use_def_consistency(graph)
    _check_topological_order(graph)
    _check_shapes_and_dtypes_resolved(graph)
    _check_no_orphans(graph)
    return True


def _check_only_core_ops(graph):
    for node in graph.nodes:
        if node.op_type not in CORE_OPS:
            raise ValueError(
                f"IR VERIFICATION FAILED: node '{node.name}' has op_type '{node.op_type}', "
                f"which is not one of our 18 core ops. This means legalize() has a bug — "
                f"it should never have let this through."
            )


def _check_use_def_consistency(graph):
    """The links stored on Tensors (.producer / .consumers) must match the real node list."""
    in_graph = {id(n) for n in graph.nodes}

    for node in graph.nodes:
        for t in node.inputs:
            if t is None:
                continue
            if graph.tensors.get(t.name) is not t:
                raise ValueError(f"IR VERIFICATION FAILED (use-def): node '{node.name}' reads tensor "
                                 f"'{t.name}' that is missing from (or different from) the graph's tensor registry.")
            if sum(1 for c in t.consumers if c is node) != sum(1 for i in node.inputs if i is t):
                raise ValueError(f"IR VERIFICATION FAILED (use-def): tensor '{t.name}' doesn't list "
                                 f"node '{node.name}' as a consumer the right number of times.")
        for t in node.outputs:
            if t.producer is not node:
                raise ValueError(f"IR VERIFICATION FAILED (use-def): tensor '{t.name}' is an output of "
                                 f"node '{node.name}' but its .producer points at {t.producer!r}.")
            if graph.tensors.get(t.name) is not t:
                raise ValueError(f"IR VERIFICATION FAILED (use-def): output tensor '{t.name}' of node "
                                 f"'{node.name}' is missing from the graph's tensor registry.")

    for t in graph.tensors.values():
        for c in t.consumers:
            if id(c) not in in_graph:
                raise ValueError(f"IR VERIFICATION FAILED (use-def): tensor '{t.name}' lists consumer "
                                 f"'{c.name}' which is not in the graph — a stale reference left by a rewrite.")
        if t.producer is not None and id(t.producer) not in in_graph:
            raise ValueError(f"IR VERIFICATION FAILED (use-def): tensor '{t.name}' has producer "
                             f"'{t.producer.name}' which is not in the graph.")


def _check_topological_order(graph):
    produced = set()
    for t in graph.inputs:
        produced.add(t.name)
    for t in graph.tensors.values():
        if t.producer is None:
            produced.add(t.name)  # graph input or constant with no explicit producer node

    for node in graph.nodes:
        for t in node.inputs:
            if t is None:
                continue                      # omitted optional input
            if t.name not in produced:
                raise ValueError(
                    f"IR VERIFICATION FAILED: node '{node.name}' reads tensor '{t.name}' "
                    f"before it's produced. BoseIR is not in valid topological order."
                )
        for t in node.outputs:
            produced.add(t.name)


def _check_shapes_and_dtypes_resolved(graph):
    for node in graph.nodes:
        for t in node.outputs:
            if t.shape is None:
                raise ValueError(f"IR VERIFICATION FAILED: tensor '{t.name}' (from node '{node.name}') "
                                  f"has no resolved shape — shape inference missed it.")
            if not all(isinstance(d, int) for d in t.shape):
                raise ValueError(f"IR VERIFICATION FAILED: tensor '{t.name}' (from node '{node.name}') "
                                  f"still has an unresolved dynamic dimension in shape {t.shape}.")
            if t.dtype is None:
                raise ValueError(f"IR VERIFICATION FAILED: tensor '{t.name}' (from node '{node.name}') "
                                  f"has no resolved dtype.")


def _check_no_orphans(graph):
    graph_output_names = {t.name for t in graph.outputs}
    for node in graph.nodes:
        for t in node.outputs:
            if len(t.consumers) == 0 and t.name not in graph_output_names:
                raise ValueError(
                    f"IR VERIFICATION FAILED: tensor '{t.name}' (from node '{node.name}') "
                    f"has no consumers and isn't a declared graph output — dead/orphaned node."
                )
