"""
Data cleaning for selected_data/.

Detects and forward-fills missing business days in a price series.
Uses pd.bdate_range (Mon-Fri) as the reference calendar.
US market holidays fall within business days and are also forward-filled,
which is the standard convention (price unchanged on holiday).
"""

import pandas as pd


def fill_missing_business_days(series: pd.Series) -> pd.Series:
    """
    Reindex series to cover every business day in its date range,
    forward-filling any gaps.

    Returns the cleaned series (same name preserved).
    """
    full_range = pd.bdate_range(start=series.index.min(), end=series.index.max())
    n_missing  = len(full_range.difference(series.index))

    if n_missing > 0:
        # Roughly 9 US market holidays per year end up in bdate_range.
        # Anything beyond that is a genuine data gap.
        year_span      = (series.index.max() - series.index.min()).days / 365.25
        expected_hols  = round(year_span * 9)
        true_gaps      = max(0, n_missing - expected_hols)

        if true_gaps > 0:
            print(f"    [clean] {series.name}: {true_gaps} true gap(s) detected "
                  f"(+{expected_hols} expected holidays), forward-filling all {n_missing}")
        else:
            print(f"    [clean] {series.name}: {n_missing} holiday gap(s) filled (no true gaps)")

        series = series.reindex(full_range).ffill()
        series.name = series.name  # ffill drops name on some pandas versions

    return series


def clean_series(series: pd.Series) -> pd.Series:
    """Entry point: apply all cleaning steps."""
    series = fill_missing_business_days(series)
    return series
