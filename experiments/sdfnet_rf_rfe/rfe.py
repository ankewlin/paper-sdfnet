"""Random-forest RFE grouped by statistical feature type (16 elements each).

RF is only the feature-ranking estimator. The final localizer is SDF-Net.
All selector settings are fixed before observing downstream test results.
"""
import numpy as np
from sklearn.ensemble import RandomForestRegressor


def select_features(statistics, target, names, keep=7):
    """Fit only training positions; recursively remove the weakest group."""
    statistics = np.asarray(statistics, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if statistics.ndim != 3 or statistics.shape[2] != len(names):
        raise ValueError("Expected [training positions, elements, feature types].")
    if target.shape != (len(statistics), 3) or len(statistics) < 6:
        raise ValueError("Expected at least six training positions with xyz labels.")
    if not 1 <= keep < len(names) or len(set(names)) != len(names):
        raise ValueError("Invalid requested feature count or duplicate names.")
    if not np.isfinite(statistics).all() or not np.isfinite(target).all():
        raise ValueError("Training data must be finite.")
    groups = statistics.transpose(0, 2, 1)
    settings = dict(n_estimators=300, criterion="squared_error", max_depth=None,
                    min_samples_split=2, min_samples_leaf=1, max_features=1.0,
                    bootstrap=True, random_state=42, n_jobs=-1)
    active, trace = list(range(len(names))), []
    while len(active) > keep:
        current = groups[:, active, :].reshape(len(groups), -1)
        print("RF-RFE fitting {} feature types ({} trees)...".format(
            len(active), settings["n_estimators"]), flush=True)
        # Native multi-output RF fits xyz jointly in the original common units.
        model = RandomForestRegressor(**settings).fit(current, target)
        scores = model.feature_importances_.reshape(len(active), groups.shape[2]).sum(axis=1)
        if not np.isfinite(scores).all() or scores.sum() <= 0:
            raise ValueError("RF produced no usable feature importance scores.")
        remove = int(np.argmin(scores))
        trace.append({"scores": {names[i]: float(s) for i, s in zip(active, scores)},
                      "removed": names[active[remove]]})
        print("RFE removes: " + names[active[remove]], flush=True)
        active.pop(remove)
    return {"method": "Grouped RF-RFE", "estimator": "RandomForestRegressor",
            "selected_features": [names[i] for i in active],
            "selected_source_indices": active, "candidate_features": list(names),
            "training_positions": len(groups), "elements_per_group": groups.shape[2],
            "keep": keep, "rf_parameters": settings, "elimination_trace": trace,
            "group_score": "sum of impurity-based feature importances over array elements",
            "selection_data": "training split only; no validation or test-guided selection",
            "hyperparameters": "fixed before downstream evaluation; no tuning on localization results",
            "normalization": "existing prepared training-only normalization; xyz in common metre units",
            "rfe_reference_doi": "10.1023/A:1012487302797",
            "rf_reference_doi": "10.1023/A:1010933404324"}

"""
运行rfe代码
python -c "import sys; sys.path.insert(0, 'experiments/sdfnet_rf_rfe'); from rfe import select_features; from run import read_prepared, SOURCE_STAT_COLUMNS, save_json; splits, _ = read_prepared('data/prepared.npz', ('train',)); train = splits['train']; result = select_features(train['statistical'], train['target'], SOURCE_STAT_COLUMNS, keep=7); save_json('rfe_selection.json', result); print('Selected:', result['selected_features'])"
"""
