# MLB Dingers Prediction Model

A Python-based MLB prediction system for pitcher strikeouts, batter outcomes, home run probability, and sportsbook value analysis. The project combines player/team data, matchup context, Monte Carlo-style probability outputs, custom pitch-zone adjustments, probability calibration, and FanDuel odds comparison to create daily MLB prediction reports.

> This project is a personal sports analytics and machine learning portfolio project. It is not financial advice, betting advice, or guaranteed betting guidance.

---

## Project Overview

The model generates daily MLB outputs for:

- Pitcher strikeout probabilities
- Batter hit and home run probabilities
- Full matchup and full-slate reports
- FanDuel home run value boards
- Zone-based 3x3 matchup adjustments
- Probability calibration using saved predictions and actual results
- Walk-forward calibration and actuals tracking

The system is mainly operated through the command-line interface in `cli.py`.

---

## Screenshot Gallery

The screenshots below are stored in the repository's `data/` folder.

### Main CLI Menu

![Main CLI Menu](data/Cliimage.png)

### Top HR Angles Output

![Top HR Angles Output](data/HRboard.png)

### Pitcher KO Output

![Pitcher KO Output](data/KOoutput%20table.png)

### FanDuel HR Value Board

---

## Key Features

### 1. Pitcher Strikeout Prediction

The pitcher KO module produces projected strikeout probabilities across multiple thresholds:

- `P(K≥4)`
- `P(K≥5)`
- `P(K≥6)`
- `P(K≥7)`
- `P(K≥8)`

It also includes calibrated strikeout probabilities and recommendation fields such as:

- `P_cal(K≥4)` through `P_cal(K≥8)`
- `anchor_k6`
- `k5_k6_blend`
- `tail_k7_k8_blend`
- `recommended_focus`

These outputs help separate safer strikeout anchors from higher-upside tail outcomes.

---

### 2. Pitcher KO 3x3 Zone Layer

The pitcher KO model includes a 3x3 pitch-zone context layer designed to improve matchup-specific strikeout projections.

The evidence ladder is:

```text
exact pitcher rows
→ pitcher-family rows
→ pitcher archetype rows
→ hand matchup rows
→ league rows
```

This allows the model to still produce context-aware outputs when exact pitcher data is limited.

Example diagnostic columns include:

- `zone_pitch_k_mult`
- `zone_pitch_confidence`
- `zone_pitch_pitcher_arch`
- `zone_pitch_candidate_source`
- `zone_pitch_sample_regime`
- `zone_pitch_exact_pitcher_rows`
- `zone_pitch_pitcher_archetype_rows`
- `zone_pitch_hand_rows`
- `zone_pitch_league_rows`

---

### 3. Batter Outcome Prediction

The batter module projects several offensive outcomes, including:

- Hit probability
- Home run probability
- Expected hits
- Expected home runs
- Singles, doubles, triples, and total hit context
- Batting order and lineup role effects

Important output columns include:

- `P(H>=1)`
- `P_cal(H>=1)`
- `P(HR>=1)`
- `P_cal(HR>=1)`
- `exp_hits`
- `exp_hr`
- `hit_pa`
- `hr_pa`

The raw probabilities are preserved while calibrated probabilities are added for better comparison against actual outcomes.

---

### 4. Batter HR 3x3 Zone Layer

The HR model includes a batter-side 3x3 context system that estimates contact and damage quality against the opposing pitcher profile.

The HR evidence ladder is:

```text
exact batter vs pitcher-family/zone
→ batter-family profile
→ batter archetype
→ pitcher archetype
→ hand matchup
→ league rows
```

Example HR context columns include:

- `zone_hr_overall_mult`
- `zone_hr_contact_mult`
- `zone_hr_damage_mult`
- `zone_hr_barrel_mult`
- `zone_hr_air_mult`
- `zone_hr_pull_air_mult`
- `zone_hr_confidence`
- `zone_hr_candidate_source`
- `zone_hr_sample_regime`
- `zone_hr_batter_archetype`
- `zone_hr_pitcher_arch`

---

### 5. Probability Calibration

The model includes a probability calibration layer that compares saved model predictions against actual outcomes.

The calibration workflow can build 10-day and 90-day actuals files and create probability calibration maps for:

- Hits
- Home runs

The calibrated outputs are saved alongside raw probabilities:

```text
P(H>=1)          raw hit probability
P_cal(H>=1)      calibrated hit probability

P(HR>=1)         raw HR probability
P_cal(HR>=1)     calibrated HR probability
```

The system also supports blended calibration so the output does not collapse all players into the same bucket. This helps preserve model ranking while correcting probability bias.

---

### 6. FanDuel HR Value Board

The value board reads a manually maintained `FDodds.csv` file and compares model HR probability against market-implied probability.

It can output:

- Raw HR probability
- Calibrated HR probability
- FanDuel odds
- FanDuel implied probability
- Fair American odds
- Edge percentage points
- Expected value per $100
- Value tag

Important columns include:

- `P(HR>=1)`
- `P_cal(HR>=1)`
- `model_prob_hr`
- `model_prob_source`
- `fd_odds`
- `fd_implied_prob`
- `fair_odds_american`
- `edge_pct_pts`
- `ev_per_100`
- `value_tag`

When calibrated probability is available, the value board uses `P_cal(HR>=1)` as the preferred probability source.

---

## CLI Menu

The main workflow is run through:

```bash
python cli.py
```

Current menu options include:

```text
1) Build / refresh precompute caches
2) Refresh auto defense proxies only
3) Pitcher KO output for one matchup
4) Batter outcomes for one matchup
5) Full daily pipeline for one matchup
6) Full daily pipeline for full slate
7) Prediction report for one matchup
8) Prediction report for full slate
9) Full-season replay / backtest
10) Walk-forward calibration summary from saved files
11) Closing-line benchmark from saved files
12) Zone HR 3x3 output audit
13) FanDuel HR value board
14) Build/update probability calibration maps
0) Exit
```

---

## Typical Daily Workflow

A standard daily run usually follows this order:

```text
1. Activate the virtual environment
2. Update FDodds.csv if using FanDuel HR value comparison
3. Run option 1 if caches need to be refreshed
4. Run option 6 for the full slate
5. Review batter, KO, and HR value outputs in the outputs/ folder
```

Activate the virtual environment in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Then run:

```powershell
python cli.py
```

---

## Calibration Workflow

To update actuals and probability calibration maps:

```text
14) Build/update probability calibration maps
```

Recommended windows:

```text
10,90
```

This generates calibration maps under:

```text
outputs/calibration/
```

The latest maps are used automatically by future batter outputs.

---

## Output Files

Common outputs are written to the `outputs/` folder.

Examples:

```text
outputs/ko_YYYY-MM-DD_TEAM_TEAM.csv
outputs/ko_slate_YYYY-MM-DD.csv
outputs/batters_YYYY-MM-DD_TEAM_TEAM.csv
outputs/batters_slate_YYYY-MM-DD.csv
outputs/batters_slate_YYYY-MM-DD_top_hr.csv
outputs/batters_slate_YYYY-MM-DD_top_hits.csv
outputs/batters_slate_YYYY-MM-DD_hr_values.csv
outputs/calibration/
```

---

## Project Structure

```text
Dingers_hotfix/
├── cli.py
├── FDodds.csv
├── data/
│   ├── Cliimage.png
│   ├── HRboard.png
│   ├── KOoutput table.png
│   ├── hrvalue.png
│   └── team_defense_proxies.csv
├── mlb_model/
│   ├── predict_pitcher_ko.py
│   ├── predict_batter_outcomes.py
│   ├── prob_calibration.py
│   ├── fd_value.py
│   ├── reporting.py
│   ├── zone_pitch_features.py
│   └── zone_hr_features.py
├── outputs/
├── cache/
└── README.md
```

---

## Requirements

This project uses Python and common data science libraries such as:

- pandas
- numpy
- scipy
- scikit-learn
- joblib
- pybaseball or related MLB data utilities, depending on local setup

Install dependencies from your project environment as needed.

Example:

```powershell
python -m pip install -r requirements.txt
```

---

## Notes and Limitations

- The model depends on the quality of lineup, roster, pitcher, and market data.
- Early-season samples can be noisy, so calibration and shrinkage are important.
- FanDuel odds are manually maintained through `FDodds.csv`.
- Outputs are analytical estimates and should not be treated as guaranteed results.
- The model is built for portfolio, research, and educational sports analytics purposes.

---

## Disclaimer

This project is for educational and portfolio use only. It does not guarantee betting results and should not be interpreted as financial advice or a recommendation to gamble.
