"""Classic ML baseline for fault detection experiments."""

from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

COLUMN_ALIASES = {
    "fault_label": "etiqueta_fallo",
    "exercise": "ejercicio",
    "source": "origen",
    "athlete_id": "voluntario_id",
}


def _resolve_column(df: pd.DataFrame, english: str) -> str:
    """Return the column name present in ``df`` for the English alias ``english``."""
    if english in df.columns:
        return english
    fallback = COLUMN_ALIASES.get(english)
    if fallback and fallback in df.columns:
        return fallback
    raise KeyError(f"Column '{english}' not found in DataFrame (nor its Spanish alias '{fallback}').")


def train_classic_model(final_df_path: str, output_model_path: str) -> dict[str, float]:
    """Train a RandomForest baseline and return cross-validated metrics."""
    df = pd.read_csv(final_df_path)
    fault_col = _resolve_column(df, "fault_label")
    athlete_col = _resolve_column(df, "athlete_id")

    drop_columns = {
        athlete_col,
        fault_col,
        _resolve_column(df, "exercise"),
        _resolve_column(df, "source"),
        "video_id",
    }
    X = df.drop(columns=drop_columns)
    y = df[fault_col]
    groups = df[athlete_col]

    gkf = GroupKFold(n_splits=5)
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(random_state=42)),
        ]
    )
    param_grid = {
        "clf__n_estimators": [50, 100],
        "clf__max_depth": [None, 10, 20],
    }
    grid = GridSearchCV(pipe, param_grid, cv=gkf.split(X, y, groups), scoring="f1", n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_

    precisions: list[float] = []
    recalls: list[float] = []
    f1s: list[float] = []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        params = grid.best_params_["clf"]
        model = RandomForestClassifier(**params, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))

    metrics = {
        "precision_mean": sum(precisions) / len(precisions),
        "recall_mean": sum(recalls) / len(recalls),
        "f1_mean": sum(f1s) / len(f1s),
    }
    joblib.dump(best_model, output_model_path)
    return metrics
