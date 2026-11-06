"""Reference baseline that predicts counts without looking at the cells (model B0, session mean).

Both functions take per-image frames with ``live_true, dead_true`` (and ``source`` for B0) and return the
test frame with ``live_pred, dead_pred`` added.
"""

from __future__ import annotations

import pandas as pd


def predict_mean(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Every test image gets the training-set mean Live and Dead counts (B0's fallback; not a named baseline)."""
    return test.assign(live_pred=train["live_true"].mean(), dead_pred=train["dead_true"].mean())


def predict_session_mean(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """B0: every test image gets its session's training mean (the global mean for unseen sessions).

    On the random split this measures the "recognize the session" shortcut; under
    leave-one-session-out it falls back to the training-set mean.
    """
    means = train.groupby("source")[["live_true", "dead_true"]].mean()
    live = test["source"].map(means["live_true"]).fillna(train["live_true"].mean())
    dead = test["source"].map(means["dead_true"]).fillna(train["dead_true"].mean())
    return test.assign(live_pred=live, dead_pred=dead)
