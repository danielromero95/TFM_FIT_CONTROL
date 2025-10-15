import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score

def train_classic_model(final_df_path, output_model_path):
    df = pd.read_csv(final_df_path)
    X = df.drop(columns=['video_id','etiqueta_fallo','ejercicio','origen','voluntario_id'])
    y = df['etiqueta_fallo']
    groups = df['voluntario_id']  # Para asegurarnos de que no repita en folds
    # Definir GroupKFold
    gkf = GroupKFold(n_splits=5)
    # Pipeline (opcional con StandardScaler)
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', RandomForestClassifier(random_state=42))
    ])
    # GridSearchCV para hiperparámetros básicos
    param_grid = {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [None, 10, 20]
    }
    grid = GridSearchCV(pipe, param_grid, cv=gkf.split(X, y, groups), scoring='f1', n_jobs=-1)
    grid.fit(X, y)
    best_model = grid.best_estimator_
    # Evaluación final en folds
    precisions, recalls, f1s = [], [], []
    for train_idx, test_idx in gkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = RandomForestClassifier(**grid.best_params_['clf'], random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        precisions.append(precision_score(y_test, y_pred))
        recalls.append(recall_score(y_test, y_pred))
        f1s.append(f1_score(y_test, y_pred))
    metrics = {
        'precision_mean': sum(precisions)/len(precisions),
        'recall_mean': sum(recalls)/len(recalls),
        'f1_mean': sum(f1s)/len(f1s)
    }
    # Guardar modelo con joblib
    import joblib
    joblib.dump(best_model, output_model_path)
    return metrics
