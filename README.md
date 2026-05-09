# Perceived Tabooness of English Swear Words

A data analysis and visualization project based on a survey conducted in a university linguistics course. Participants rated the perceived tabooness of four English swear words used in three different sentence contexts on a 1–7 scale.

---

## Overview

This project investigates how **sentence context** affects the perceived tabooness of swear words. The same word can feel very different depending on whether it is used as a directed insult, an emotional expression, or in a neutral, informative statement.

**Swear words analyzed:** *Fuck, Bitch, Hell, Shit*

**Sentence types:**
- **Invective**: directed insult (e.g., *"Go fuck yourself."*)
- **Emotive**: expressing the speaker's feeling (e.g., *"I can't wait to fucking graduate."*)
- **Informative**: neutral or literal context (e.g., *"…my roommate fucking someone."*)

**Scale:** 1 = not taboo at all → 7 = extremely taboo  
**Sample size:** n = 50 respondents

---

## Visualizations

The script produces five plots:

1. **Grouped bar chart**: mean tabooness by word and sentence type, with error bars (± 1 SEM)
2. **Horizontal bar chart**: overall mean by sentence type across all words
3. **Heatmap**: word × sentence type mean ratings at a glance
4. **Radar chart**: sentence type profiles compared across all four words
5. **Strip plot**: individual respondent ratings with word-level mean overlay

---

## Project Structure

```
├── tabooness_analysis.py         # Standalone Python script
├── tabooness_analysis.ipynb      # Jupyter Notebook version (cell-by-cell)
├── tabooness_analysis.png        # Combined output figure
├── Tabooness_Survey_Responses.csv  # Raw survey data
├── requirements.txt              # Python dependencies
└── README.md
```

---

## Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Set up a virtual environment (recommended)
```bash
python -m venv venv
source venv/bin/activate        # Mac / Linux
venv\Scripts\activate           # Windows
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the analysis
**Python script:**
```bash
python tabooness_analysis.py
```

**Jupyter Notebook:**
```bash
jupyter notebook tabooness_analysis.ipynb
```

The script will print a summary table to the console and save the combined figure as `tabooness_analysis.png`.

---

## Key Findings

| Sentence Type | Overall Mean (1–7) |
|---|---|
| Invective | 4.10 |
| Word itself | 3.10 |
| Informative | 2.45 |
| Emotive | 2.25 |

- **Invective** use consistently produced the highest tabooness ratings across all words.
- **Emotive** use was rated as the least taboo, suggesting that expressive language is perceived more leniently.
- **Hell** received notably lower ratings than the other three words across all sentence types.
- **Religion** had no notable effect on perceived tabooness of hell.

---

## Dependencies

- Python 3.10+
- pandas
- matplotlib
- numpy

---

## License

This project is for academic and portfolio purposes.
