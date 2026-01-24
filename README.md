# momentum

## Daily workflow

1. Refresh data + signals:
   ```bash
   python updater/update_parquet.py
   ```
2. Run backtests (separate process after the update completes):
   ```bash
   python updater/run_backtest_systems.py
   ```

Alternatively, run both steps in sequence:
```bash
python updater/run_daily_pipeline.py
```
