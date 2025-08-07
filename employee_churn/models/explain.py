"""Model explanation utilities using SHAP."""

import pandas as pd
import shap
from sklearn.base import BaseEstimator


def explain_with_shap(model: BaseEstimator, data: pd.DataFrame) -> pd.DataFrame:
    """Compute SHAP values for a trained model and dataset.

    Args:
        model: Fitted estimator supporting ``predict_proba`` or ``decision_function``.
        data: Feature matrix used for generating explanations.

    Returns:
        DataFrame of SHAP values aligned with ``data`` index and columns.
    """
    explainer = shap.Explainer(model, data)
    shap_values = explainer(data)
    return pd.DataFrame(shap_values.values, columns=data.columns, index=data.index)
