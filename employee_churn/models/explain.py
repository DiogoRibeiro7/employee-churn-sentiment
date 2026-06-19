"""Model explanation utilities using SHAP."""

import numpy as np
import pandas as pd
import shap
from sklearn.base import BaseEstimator


def explain_with_shap(
    model: BaseEstimator, data: pd.DataFrame, class_index: int = 1
) -> pd.DataFrame:
    """Compute SHAP values for a trained model and dataset.

    Tree ensembles return one set of SHAP values per class, producing a
    ``(n_samples, n_features, n_classes)`` array. In that case the values for
    ``class_index`` (the positive class by default) are returned so the result
    is always a 2-D, feature-aligned DataFrame.

    Args:
        model: Fitted estimator supporting ``predict_proba`` or
            ``decision_function``.
        data: Feature matrix used for generating explanations.
        class_index: Class whose contributions to return for multiclass output.

    Returns:
        DataFrame of SHAP values aligned with ``data`` index and columns.
    """
    explainer = shap.Explainer(model, data)
    values = np.asarray(explainer(data).values)
    if values.ndim == 3:
        # Shape (n_samples, n_features, n_classes) -> pick one class.
        index = min(class_index, values.shape[2] - 1)
        values = values[:, :, index]
    return pd.DataFrame(values, columns=data.columns, index=data.index)
