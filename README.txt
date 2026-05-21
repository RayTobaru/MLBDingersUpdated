IMPORT-STABILITY FIX PACK

Use these replacements:

Project root:
- fetch.py <= fetch.py
- precompute_everything.py <= precompute_everything.py

legacy folder:
- legacy/fetch.py <= legacy_fetch_import_safe.py
- legacy/precompute_everything.py <= legacy_precompute_everything_import_safe.py

mlb_model folder:
- mlb_model/predict_pitcher_ko.py <= predict_pitcher_ko_import_safe.py
- mlb_model/predict_batter_outcomes.py <= predict_batter_outcomes_import_safe.py
- mlb_model/shared_game_utils.py <= shared_game_utils_import_safe.py
- mlb_model/season_backtest.py <= season_backtest_import_safe.py

Concrete fixes in this pass:
1. Added missing _row_eligibility_flag to predict_batter_outcomes.py
2. Fixed broken BATTER_CAREER / pc_bat references in predict_pitcher_ko.py
3. Added root fetch.py and precompute_everything.py shims so import fetch / import precompute_everything resolve consistently
4. Included import-safe legacy fetch / precompute replacements
5. Cleaned season_backtest.py duplicate imports / __all__

After replacing files, clear pycache folders and rerun:
Get-ChildItem -Recurse -Directory -Filter __pycache__ | Remove-Item -Recurse -Force
python cli.py
