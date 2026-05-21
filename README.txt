@'
# MLB Dingers Prediction Model

A Python-based MLB prediction system for pitcher strikeouts, batter outcomes, home run probability, and sportsbook value analysis. The project combines player/team data, matchup context, Monte Carlo-style probability outputs, custom 3x3 pitch-zone adjustments, and FanDuel odds comparison to create daily betting-style reports.

> This project is designed as a personal sports analytics and machine learning portfolio project. It is not financial advice or guaranteed betting guidance.

---

## Project Overview

This model generates daily MLB prediction outputs for:

- Pitcher strikeout probabilities
- Batter hit and home run probabilities
- Full matchup and full-slate batter reports
- FanDuel home run value boards
- Zone-based 3x3 matchup adjustments
- Walk-forward calibration and actuals tracking

The system is run through a command-line interface in `cli.py`.

---

## Screenshot Gallery

Add screenshots to the `docs/screenshots/` folder and update the image paths below.

### Main CLI Menu

![Main CLI Menu](docs/screenshots/main_cli_menu.png)

### Batter HR Value Board

![HR Value Board](docs/screenshots/hr_value_board.png)

### Top HR Angles Output

![Top HR Angles](docs/screenshots/top_hr_angles.png)

### Pitcher KO Output

![Pitcher KO Output](docs/screenshots/pitcher_ko_output.png)

---

## Key Features

### 1. Pitcher Strikeout Prediction

The pitcher KO module produces projected strikeout probabilities across multiple thresholds, such as:

- `P(K≥4)`
- `P(K≥5)`
- `P(K≥6)`
- `P(K≥7)`
- `P(K≥8)`

It also includes calibrated probabilities and recommendation fields such as:

- `anchor_k6`
- `k5_k6_blend`
- `tail_k7_k8_blend`
- `recommended_focus`

---

### 2. Pitcher KO 3x3 Zone Layer

The pitcher KO model includes a 3x3 pitch-zone context layer.

The evidence ladder is:

```text
exact pitcher rows
→ pitcher-family rows
→ pitcher archetype rows
→ hand matchup rows
→ league rows
