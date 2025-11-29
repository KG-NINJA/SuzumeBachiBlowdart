# System State Report (Flat Model Structure)

## models/ inventory
- Structure is flat (no per-ticker subdirectories); files named `<TICKER>_info.json`, `<TICKER>_model.json`, `<TICKER>_scaler.pkl`.
- File counts: 20 model artifacts (json), 10 info files, 10 scaler pickles; no `.lgb` or `.txt` LightGBM files.
- Unique tickers detected: AAPL, AMD, AMZN, GOOGL, META, MSFT, NFLX, NVDA, QQQ, TSLA, plus `lgb` prefix from shared models.

## Sample per-ticker file sets
- AAPL: `AAPL_info.json`, `AAPL_model.json`, `AAPL_scaler.pkl`
- AMD: `AMD_info.json`, `AMD_model.json`, `AMD_scaler.pkl`
- AMZN: `AMZN_info.json`, `AMZN_model.json`, `AMZN_scaler.pkl`
- GOOGL: `GOOGL_info.json`, `GOOGL_model.json`, `GOOGL_scaler.pkl`
- META: `META_info.json`, `META_model.json`, `META_scaler.pkl`

## Info file schema
- Keys: `ticker`, `accuracy`, `previous_accuracy`, `accuracy_improvement`, `train_samples`, `test_samples`, `total_train_samples`, `features`, `learning_type`, `model_path`, `scaler_path`, `timestamp`.
- Missing keys across files: `regime`, `model_type`, `version`.

## Accuracy snapshot (from *_info.json)
- AAPL: 0.375
- AMD: 0.5625
- AMZN: 0.5625
- GOOGL: 0.75
- META: 0.6875
- MSFT: 0.625
- NFLX: 0.5
- NVDA: 0.5625
- QQQ: 0.4375
- TSLA: 0.6875

## Identification
- Core training/prediction code (e.g., `blowdart_ml_engine.py`) references flat structure via `_info.json` artifacts.
- GitHub Actions workflows call `python simple_daily_prediction.py --mode train`; no nested model paths referenced.

## Conclusion
- System currently operates with a flat models directory and relies on `_info.json` metadata storing accuracy; regime/version metadata absent.
