# sonnet-colab-macro-loaders

Colab-ready script (Sonnet) that pulls fundamental and macro data with safe, non-fatal execution:
- MOEX ISS (description, dividends, splits, last price)
- CBR USD/RUB daily series and key rate table
- Yahoo Finance fundamentals via `yfinance`
- Optional: Alpha Vantage, Financial Modeling Prep, EODHD (API keys optional)
- e-disclosure.ru search template

Every block is wrapped in `safe_run`: errors are printed but execution continues.

## File
- `colab_macro_loaders_sonnet.py`

## Run
In Colab or locally (Python 3.9+):
```bash
pip install yfinance beautifulsoup4 lxml pandas numpy requests
python colab_macro_loaders_sonnet.py
```

Set optional API keys via env vars: `ALPHAVANTAGE_API_KEY`, `FMP_API_KEY`, `EODHD_API_KEY`.
