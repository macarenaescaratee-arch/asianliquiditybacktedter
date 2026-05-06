"""
Streamlit dashboard entry (run with: streamlit run dashboard/app.py).

Future use: charts via Plotly, filters for symbols and date ranges, sidebar controls
for strategy parameters, and links to generated HTML/PDF reports under ``reports/``.
"""

from __future__ import annotations

# Streamlit is imported inside main() so ``python -c "import dashboard.app"`` works in CI
# without requiring a Streamlit runtime during lightweight imports.


def main() -> None:
    import streamlit as st

    st.set_page_config(page_title="AsianLiquidityBacktester", layout="wide")
    st.title("AsianLiquidityBacktester")
    st.write(
        "Dashboard skeleton — connect to backtest outputs and analytics in later phases."
    )


if __name__ == "__main__":
    main()
