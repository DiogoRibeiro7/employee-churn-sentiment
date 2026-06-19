"""Tests for training, scoring, and evaluation helper scripts."""

import pandas as pd
from sklearn.datasets import make_classification

from scripts.evaluate_model import evaluate_model_bundle
from scripts.predict_risk import load_model_bundle, score_dataset
from scripts.train_model import save_model_bundle, train_model_bundle


def test_train_and_evaluate_baseline_bundle(tmp_path) -> None:
    X, y = make_classification(
        n_samples=40,
        n_features=4,
        n_informative=3,
        n_redundant=0,
        random_state=7,
    )
    df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
    df["target"] = y

    bundle, metrics = train_model_bundle(df, target_column="target")
    output_model = tmp_path / "model.pkl"
    save_model_bundle(bundle, output_model)

    loaded_bundle = load_model_bundle(output_model)
    evaluated = evaluate_model_bundle(loaded_bundle, df, top_k=5)

    assert output_model.exists()
    assert loaded_bundle["model_name"] == "log_reg"
    assert metrics.keys() == {"roc_auc", "f1", "precision_at_k"}
    assert evaluated.keys() == {"roc_auc", "f1", "precision_at_k"}


def test_train_score_and_export_combined_bundle(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "employee_id": [1, 2, 3, 4, 5, 6],
            "date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "age": [25, 40, 35, 28, 31, 45],
            "tenure": [1, 5, 3, 2, 4, 6],
            "feedback": [
                "I love my job",
                "I hate this place",
                "Happy and delighted",
                "Angry and mad",
                "I feel confident and happy",
                "I am annoyed and afraid",
            ],
            "target": [0, 1, 0, 1, 0, 1],
        }
    )

    bundle, _ = train_model_bundle(
        df,
        target_column="target",
        text_column="feedback",
        test_size=0.5,
        random_state=0,
    )
    model_path = tmp_path / "combined.pkl"
    output_csv = tmp_path / "scores.csv"
    save_model_bundle(bundle, model_path)

    scores = score_dataset(
        load_model_bundle(model_path),
        df.drop(columns=["target"]),
        id_column="employee_id",
        date_column="date",
        output_path=output_csv,
    )

    assert output_csv.exists()
    assert set(scores.columns) == {"employee_id", "week_start", "churn_risk"}
    assert scores["churn_risk"].between(0, 1).all()
