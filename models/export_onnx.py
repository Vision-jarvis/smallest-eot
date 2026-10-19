"""Export a binary sklearn HistGradientBoostingClassifier to a plain ONNX graph.

skl2onnx's HGB converter breaks with current onnx (bool attributes), and the head is tiny, so
we emit the graph directly: one `ai.onnx.ml.TreeEnsembleRegressor` that sums every tree's leaf
values (rule: go left iff x <= threshold, the HGB rule for non-missing inputs; the runtime
replaces missing features with a finite sentinel, so the NaN branch is never taken), plus the
baseline log-odds, then a Sigmoid. Output `p_eot` has shape (N, 1).

Note on "int8": the head is a few hundred trees of float thresholds (~100 KB); quantising it
buys nothing. The int8 budget in the statement matters for the GPU audio encoder (PENDING).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper


def hgb_to_onnx(model, n_features: int) -> onnx.ModelProto:
    if model.n_trees_per_iteration_ != 1:
        raise ValueError("binary classifiers only")
    attrs = {k: [] for k in ("nodes_treeids", "nodes_nodeids", "nodes_featureids", "nodes_values", "nodes_modes", "nodes_truenodeids", "nodes_falsenodeids", "target_treeids", "target_nodeids", "target_ids", "target_weights")}
    for tid, (pred,) in enumerate(model._predictors):
        nodes = pred.nodes
        for nid, nd in enumerate(nodes):
            attrs["nodes_treeids"].append(tid)
            attrs["nodes_nodeids"].append(nid)
            if nd["is_leaf"]:
                attrs["nodes_featureids"].append(0)
                attrs["nodes_values"].append(0.0)
                attrs["nodes_modes"].append("LEAF")
                attrs["nodes_truenodeids"].append(0)
                attrs["nodes_falsenodeids"].append(0)
                attrs["target_treeids"].append(tid)
                attrs["target_nodeids"].append(nid)
                attrs["target_ids"].append(0)
                attrs["target_weights"].append(float(nd["value"]))
            else:
                attrs["nodes_featureids"].append(int(nd["feature_idx"]))
                attrs["nodes_values"].append(float(nd["num_threshold"]))
                attrs["nodes_modes"].append("BRANCH_LEQ")
                attrs["nodes_truenodeids"].append(int(nd["left"]))
                attrs["nodes_falsenodeids"].append(int(nd["right"]))
    base = float(np.ravel(model._baseline_prediction)[0])
    tree = helper.make_node("TreeEnsembleRegressor", ["x"], ["raw"], domain="ai.onnx.ml", n_targets=1, aggregate_function="SUM", base_values=[base], post_transform="NONE", **attrs)
    sig = helper.make_node("Sigmoid", ["raw"], ["p_eot"])
    g = helper.make_graph([tree, sig], "pulse_eot_hi_head", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, n_features])], [helper.make_tensor_value_info("p_eot", TensorProto.FLOAT, [None, 1])])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17), helper.make_opsetid("ai.onnx.ml", 3)])
    m.ir_version = 9
    onnx.checker.check_model(m)
    return m


def export(model, n_features: int, path: Path, X_check: np.ndarray, tol: float = 1e-4) -> float:
    """Write the graph and verify it reproduces sklearn's probabilities; returns max |dp|."""
    import onnxruntime as ort

    path.write_bytes(hgb_to_onnx(model, n_features).SerializeToString())
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    p_onnx = sess.run(None, {"x": X_check.astype(np.float32)})[0][:, 0]
    err = float(np.max(np.abs(p_onnx - model.predict_proba(X_check)[:, 1])))
    if err > tol:
        raise RuntimeError(f"ONNX parity failed: max |dp| = {err}")
    return err
