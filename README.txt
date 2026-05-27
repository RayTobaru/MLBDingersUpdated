# MLB Dingers Prediction Model

A Python-based MLB analytics system for pitcher strikeouts, batter outcomes, home run probability, and sportsbook value analysis. The project combines daily MLB matchup data, player-level context, custom pitch-zone adjustments, probability calibration, and FanDuel odds comparison to generate daily command-line reports.

> This project is a personal sports analytics and machine learning portfolio project. It is not financial advice or guaranteed betting guidance.

---

## Screenshot Gallery

Screenshots are stored in `docs/screenshots/`. If an image does not display on GitHub, confirm that the file exists in the repository, the path is spelled exactly the same, and the image has been committed and pushed.

### Main CLI Menu

![Main CLI Menu](docs/screenshots/cli_menu.png)

### Top HR Angles Output

![Top HR Angles Output](docs/screenshots/top_hr_angles.png)

### Pitcher KO Output

![Pitcher KO Output](docs/screenshots/pitcher_ko_output.png)

### FanDuel HR Value Board

![FanDuel HR Value Board](docs/screenshots/fanduel_hr_value_board.png)

---

## Project Overview

The model produces daily MLB prediction outputs for:

- Pitcher strikeout probability thresholds
- Batter hit and home run probabilities
- One-game, selected-range, and full-slate reports
- FanDuel home run value boards
- Zone-based matchup context adjustments
- Probability calibration using saved predictions and actual results
- Walk-forward style calibration and actuals tracking

The main workflow is run through the command-line interface in `cli.py`.

---

## Key Features

### 1. Pitcher Strikeout Prediction

The pitcher KO module estimates strikeout outcomes across multiple thresholds:

- `P(K≥4)`
- `P(K≥5)`
- `P(K≥6)`
- `P(K≥7)`
- `P(K≥8)`

It also outputs calibrated probabilities and ranking fields:

- `P_cal(K≥x)`
- `anchor_k6`
- `k5_k6_blend`
- `tail_k7_k8_blend`
- `recommended_focus`

These fields help separate safer strikeout anchors from higher-tail strikeout candidates.

---

### 2. Pitcher KO 3x3 Zone Layer

The pitcher KO model includes a 3x3 pitch-zone context layer designed to account for pitcher attack zones and pitch-family tendencies.

The evidence ladder is:

```text
exact pitcher rows
→ pitcher-family rows
→ pitcher archetype rows
→ hand matchup rows
→ league rows
```

Example output fields include:

- `zone_pitch_k_mult`
- `zone_pitch_confidence`
- `zone_pitch_pitcher_arch`
- `zone_pitch_candidate_source`
- `zone_pitch_sample_regime`
- `zone_pitch_exact_pitcher_rows`
- `zone_pitch_pitcher_archetype_rows`
- `zone_pitch_hand_rows`
- `zone_pitch_league_rows`

This makes the strikeout model more transparent by showing whether a matchup adjustment came from exact pitcher evidence, pitcher archetype fallback, hand matchup fallback, or league-level fallback.
![Alt text] (data/KOoutput table.png)

---

### 3. Batter Outcomes and Home Run Probability

The batter model generates player-level outcome estimates, including:

- `exp_hits`
- `P(H>=1)`
- `P_cal(H>=1)`
- `exp_hr`
- `P(HR>=1)`
- `P_cal(HR>=1)`
- `hit_pa`
- `hr_pa`

Raw probabilities are preserved, while calibrated probabilities are added as separate columns. This makes it possible to compare the original model output against the historically calibrated probability.

---

### 4. Batter/HR 3x3 Zone Layer

The home run model includes a batter-side 3x3 zone and contact-damage layer.

The evidence ladder is:

```text
exact batter vs pitcher-family/zone
→ batter-family profile
→ batter archetype
→ pitcher archetype
→ hand matchup
→ league
```

Important fields include:

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

This layer gives additional context around whether a batter’s home run probability is being adjusted by contact quality, air-ball tendency, pull-air profile, or matchup archetype.

---

### 5. Probability Calibration

The model supports a calibration workflow that compares saved predictions against actual results over rolling windows such as 10 days and 90 days.

The calibration system keeps three versions of key probabilities:

```text
P(HR>=1)           raw model HR probability
P_mapcal(HR>=1)    pure calibration-map probability
P_cal(HR>=1)       blended calibrated HR probability

P(H>=1)            raw model hit probability
P_mapcal(H>=1)     pure calibration-map probability
P_cal(H>=1)        blended calibrated hit probability
```

The blended `P_cal` columns are designed to correct broad probability bias while preserving model ranking better than a pure bucket-based calibration map.

Calibration outputs are saved under:

```text
outputs/calibration/
```

---

### 6. FanDuel HR Value Board

The FanDuel value board compares model home run probability against market odds from `FDodds.csv`.

The value layer uses calibrated probability when available:

```text
model_prob_source = P_cal(HR>=1)
```

Common value-board fields include:

- `fd_odds`
- `fd_implied_prob`
- `fair_odds_american`
- `edge_pct_pts`
- `ev_per_100`
- `value_tag`
- `model_prob_hr`
- `model_prob_source`

This allows the project to estimate whether a home run price is fair, overvalued, or potentially positive expected value based on the model.

---

## CLI Menu

The project is controlled from `cli.py`:

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

```text
1. Activate the virtual environment
2. Update FDodds.csv if using FanDuel value boards
3. Run option 1 if caches need to be refreshed
4. Run option 6 for a full daily slate
5. Review KO output, top HR angles, top hit angles, and HR value board
6. Run option 14 periodically to refresh calibration maps
```

Example:

```powershell
.\.venv\Scripts\Activate.ps1
python cli.py
```

---

## Project Structure

```text
Dingers_hotfix/
├── cli.py
├── FDodds.csv
├── build_last10_actuals.py
├── run_calibration_suite.py
├── mlb_model/
│   ├── predict_pitcher_ko.py
│   ├── predict_batter_outcomes.py
│   ├── zone_pitch_features.py
│   ├── zone_hr_features.py
│   ├── prob_calibration.py
│   ├── fd_value.py
│   └── reporting.py
├── outputs/
│   ├── calibration/
│   ├── *_compact.csv
│   ├── *_top_hr.csv
│   ├── *_top_hits.csv
│   └── *_hr_values.csv
└── docs/
    └── screenshots/
        ├── cli_menu.png
        ├── top_hr_angles.png
        ├── pitcher_ko_output.png
        └── fanduel_hr_value_board.png
```

---

## FanDuel Odds Input Format

The project supports a manually updated `FDodds.csv` file. A simple format is:

```csv
player_name,fd_odds
Kyle Schwarber,540
Aaron Judge,520
Shohei Ohtani,490
```

The parser also supports one-column pasted odds formats from sportsbook pages, as long as player names and odds can be identified.

---

## Example Outputs

Typical generated files include:

```text
outputs/ko_YYYY-MM-DD_TEAM_TEAM.csv
outputs/ko_slate_YYYY-MM-DD.csv
outputs/batters_YYYY-MM-DD_TEAM_TEAM.csv
outputs/batters_slate_YYYY-MM-DD.csv
outputs/batters_slate_YYYY-MM-DD_top_hr.csv
outputs/batters_slate_YYYY-MM-DD_top_hits.csv
outputs/batters_slate_YYYY-MM-DD_hr_values.csv
outputs/calibration/prob_calibration_hr_latest_map.csv
outputs/calibration/prob_calibration_hit_latest_map.csv
```

---

## Installation Notes

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the CLI:

```powershell
python cli.py
```

---

## Current Development Focus

Recent improvements include:

- Pitcher KO 3x3 zone context layer
- Pitcher archetype fallback for low-sample pitchers
- Batter/HR 3x3 contact-damage layer
- FanDuel HR value board integration
- 10-day and 90-day actuals/calibration workflows
- Blended calibrated probabilities for hits and HRs
- Full-slate and selected-range CLI outputs

Future improvements may include:

- More robust backtesting by player subgroup
- Closing-line benchmarking
- Additional market support beyond HR props
- Cleaner dashboard-style visualization
- More formal model evaluation metrics

---

## Disclaimer

This repository is for educational and portfolio purposes only. Sports predictions are uncertain, and model outputs should not be interpreted as guaranteed outcomes or financial advice.
