"""
Perceived Tabooness of English Swear Words
===========================================
Analyzes survey data collected in a university linguistics course.
Participants rated the perceived tabooness of swear words used in
three sentence types (invective, emotive, informative) on a 1–7 scale.

Swear words covered: Fuck, Bitch, Hell, Shit
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.gridspec import GridSpec

# ── 0. Configuration ──────────────────────────────────────────────────────────

CSV_PATH = "Tabooness_Survey_Responses.csv"

# Map raw column names → structured metadata
SENTENCE_META = {
    # FUCK
    "Go fuck yourself.": {
        "word": "Fuck", "type": "Invective",
        "label": "Go fuck yourself.",
    },
    "I can't wait to fucking graduate.": {
        "word": "Fuck", "type": "Emotive",
        "label": "I can't wait to fucking graduate.",
    },
    "I left the apartment as soon as I could after seeing my roommate fucking someone.": {
        "word": "Fuck", "type": "Informative",
        "label": "…my roommate fucking someone.",        # shortened for display
    },
    "The word Fuck itself": {
        "word": "Fuck", "type": "Word itself",
        "label": "The word 'Fuck'",
    },
    # BITCH
    "You are such a bitch.": {
        "word": "Bitch", "type": "Invective",
        "label": "You are such a bitch.",
    },
    "That was a bitch of a test.": {
        "word": "Bitch", "type": "Emotive",
        "label": "That was a bitch of a test.",
    },
    "The dog we thought that was a male turned out to be a bitch.": {
        "word": "Bitch", "type": "Informative",
        "label": "…turned out to be a bitch.",
    },
    "The word Bitch": {
        "word": "Bitch", "type": "Word itself",
        "label": "The word 'Bitch'",
    },
    # HELL
    "Go to hell!": {
        "word": "Hell", "type": "Invective",
        "label": "Go to hell!",
    },
    "That was one hell of a ride!": {
        "word": "Hell", "type": "Emotive",
        "label": "That was one hell of a ride!",
    },
    "In the bible, those who have sinned go to hell after their death. ": {
        "word": "Hell", "type": "Informative",
        "label": "…go to hell after their death.",
    },
    "The word Hell": {
        "word": "Hell", "type": "Word itself",
        "label": "The word 'Hell'",
    },
    # SHIT
    "You piece of shit!": {
        "word": "Shit", "type": "Invective",
        "label": "You piece of shit!",
    },
    "My shitty car stopped working yesterday.": {
        "word": "Shit", "type": "Emotive",
        "label": "My shitty car stopped working yesterday.",
    },
    "I spent a whole hour trying to pick up my dog's shit.": {
        "word": "Shit", "type": "Informative",
        "label": "…pick up my dog's shit.",
    },
}

WORDS = ["Fuck", "Bitch", "Hell", "Shit"]
TYPES = ["Invective", "Emotive", "Informative", "Word itself"]

TYPE_COLORS = {
    "Invective":   "#C0392B",   # red
    "Emotive":     "#E67E22",   # orange
    "Informative": "#2980B9",   # blue
    "Word itself": "#8E44AD",   # purple
}

WORD_COLORS = {
    "Fuck":  "#C0392B",
    "Bitch": "#E67E22",
    "Hell":  "#2980B9",
    "Shit":  "#27AE60",
}


# ── 1. Load & reshape ─────────────────────────────────────────────────────────

def load_data(path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (respondents_df, long_df) where long_df has columns
    [word, type, label, rating].

    Rows whose Timestamp is not a real date string (e.g. 'Average',
    'Word averages', 'Maximum', 'Minimum') are excluded so they do not
    inflate the sample size or skew the means.
    """
    raw = pd.read_csv(path)

    # Keep only genuine respondent rows (real date-like timestamps)
    NON_RESPONDENT_LABELS = {"Average", "Word averages", "Maximum", "Minimum"}
    respondents = raw[
        raw["Timestamp"].notna()
        & ~raw["Timestamp"].isin(NON_RESPONDENT_LABELS)
    ].copy()

    records = []
    for col, meta in SENTENCE_META.items():
        if col in respondents.columns:
            numeric = pd.to_numeric(respondents[col], errors="coerce").dropna()
            for val in numeric:
                records.append({
                    "word":   meta["word"],
                    "type":   meta["type"],
                    "label":  meta["label"],
                    "rating": float(val),
                })

    long = pd.DataFrame(records)
    return respondents, long


# ── 2. Compute means ──────────────────────────────────────────────────────────

def compute_means(long: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (by_word_type, by_type) mean DataFrames."""
    by_word_type = (
        long.groupby(["word", "type"])["rating"]
        .agg(mean="mean", sem="sem", n="count")
        .reset_index()
    )
    by_type = (
        long.groupby("type")["rating"]
        .agg(mean="mean", sem="sem", n="count")
        .reset_index()
    )
    return by_word_type, by_type


# ── 3. Pretty-print table ─────────────────────────────────────────────────────

def print_summary_table(by_word_type: pd.DataFrame) -> None:
    print("\n" + "=" * 62)
    print("  Perceived Tabooness — Mean Ratings by Word & Sentence Type")
    print("  (Scale: 1 = not taboo at all, 7 = extremely taboo)")
    print("=" * 62)
    pivot = by_word_type.pivot(index="word", columns="type", values="mean")
    pivot = pivot.reindex(index=WORDS, columns=TYPES)
    pivot["Row Mean"] = pivot.mean(axis=1)
    col_means = pivot.mean(axis=0)
    col_means.name = "Col Mean"
    pivot = pd.concat([pivot, col_means.to_frame().T])
    print(pivot.round(2).to_string())
    print("=" * 62 + "\n")


# ── 4. Plots ──────────────────────────────────────────────────────────────────

def style_axes(ax, title: str, ylabel: str = "Mean Tabooness Rating (1–7)") -> None:
    ax.set_title(title, fontsize=13, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_ylim(1, 7.4)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def plot_grouped_bar(ax, by_word_type: pd.DataFrame) -> None:
    """Bar chart: x=words, groups=sentence types."""
    x = np.arange(len(WORDS))
    n_types = len(TYPES)
    w = 0.18
    offsets = np.linspace(-(n_types - 1) / 2 * w, (n_types - 1) / 2 * w, n_types)

    for i, stype in enumerate(TYPES):
        subset = by_word_type[by_word_type["type"] == stype].set_index("word")
        means = [subset.loc[word, "mean"] if word in subset.index else 0 for word in WORDS]
        sems  = [subset.loc[word, "sem"]  if word in subset.index else 0 for word in WORDS]
        bars = ax.bar(
            x + offsets[i], means, w,
            color=TYPE_COLORS[stype], label=stype,
            zorder=3, edgecolor="white", linewidth=0.5,
        )
        ax.errorbar(
            x + offsets[i], means, yerr=sems,
            fmt="none", color="black", capsize=3, linewidth=1, zorder=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(WORDS, fontsize=11)
    ax.legend(title="Sentence Type", fontsize=9, title_fontsize=9,
              loc="upper right", framealpha=0.9)
    style_axes(ax, "Mean Tabooness by Word & Sentence Type")


def plot_type_comparison(ax, by_type: pd.DataFrame) -> None:
    """Horizontal bar chart comparing sentence types overall."""
    order = by_type.set_index("type").reindex(TYPES).reset_index()
    colors = [TYPE_COLORS[t] for t in order["type"]]
    bars = ax.barh(
        order["type"], order["mean"],
        color=colors, edgecolor="white", height=0.55, zorder=3,
    )
    ax.errorbar(
        order["mean"], order["type"],
        xerr=order["sem"],
        fmt="none", color="black", capsize=4, linewidth=1.2, zorder=4,
    )
    for bar, mean in zip(bars, order["mean"]):
        ax.text(
            bar.get_width() + 0.20, bar.get_y() + bar.get_height() / 2,
            f"{mean:.2f}", va="center", fontsize=10, fontweight="bold",
        )
    ax.set_xlim(1, 8)
    ax.set_xlabel("Mean Tabooness Rating (1–7)", fontsize=10)
    ax.xaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("Overall Mean by Sentence Type", fontsize=13, fontweight="bold", pad=10)


def plot_heatmap(ax, by_word_type: pd.DataFrame) -> None:
    """Heatmap: words × sentence types."""
    pivot = by_word_type.pivot(index="word", columns="type", values="mean")
    pivot = pivot.reindex(index=WORDS, columns=TYPES)
    data = pivot.values

    im = ax.imshow(data, cmap="RdYlGn_r", vmin=1, vmax=7, aspect="auto")
    ax.set_xticks(range(len(TYPES)))
    ax.set_xticklabels(TYPES, fontsize=10)
    ax.set_yticks(range(len(WORDS)))
    ax.set_yticklabels(WORDS, fontsize=11, fontweight="bold")
    for i in range(len(WORDS)):
        for j in range(len(TYPES)):
            ax.text(
                j, i, f"{data[i, j]:.2f}",
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="white" if data[i, j] > 4.5 or data[i, j] < 2.5 else "black",
            )
    plt.colorbar(im, ax=ax, shrink=0.85, label="Mean Rating (1–7)")
    ax.set_title("Tabooness Heatmap (Word × Sentence Type)",
                 fontsize=13, fontweight="bold", pad=10)


def plot_word_radar(ax, by_word_type: pd.DataFrame) -> None:
    """Radar / spider chart per sentence type across words."""
    categories = WORDS
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(1, 7)
    ax.set_yticks([2, 3, 4, 5, 6, 7])
    ax.set_yticklabels(["2", "3", "4", "5", "6", "7"], fontsize=7, color="grey")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.xaxis.grid(True, linestyle="-", alpha=0.3)
    ax.spines["polar"].set_visible(False)

    for stype in TYPES:
        subset = by_word_type[by_word_type["type"] == stype].set_index("word")
        vals = [subset.loc[w, "mean"] if w in subset.index else 1 for w in categories]
        vals += vals[:1]
        ax.plot(angles, vals, linewidth=2, color=TYPE_COLORS[stype], label=stype)
        ax.fill(angles, vals, alpha=0.08, color=TYPE_COLORS[stype])

    ax.legend(
        loc="upper right", bbox_to_anchor=(1.35, 1.15),
        title="Sentence Type", fontsize=9, title_fontsize=9,
    )
    ax.set_title("Radar: Sentence Type Profiles Across Words",
                 fontsize=13, fontweight="bold", pad=20)


def plot_strip_with_mean(ax, long: pd.DataFrame) -> None:
    """Individual ratings as jittered dots, word-level mean overlay."""
    rng = np.random.default_rng(42)
    for xi, word in enumerate(WORDS):
        subset = long[long["word"] == word]["rating"]
        jitter = rng.uniform(-0.25, 0.25, size=len(subset))
        ax.scatter(
            np.full(len(subset), xi) + jitter,
            subset,
            alpha=0.25, s=20,
            color=WORD_COLORS[word], zorder=2,
        )
        mean_val = subset.mean()
        ax.hlines(mean_val, xi - 0.35, xi + 0.35,
                  colors=WORD_COLORS[word], linewidth=3, zorder=3)
        ax.text(xi + 0.38, mean_val, f"{mean_val:.2f}",
                va="center", fontsize=9, fontweight="bold", color=WORD_COLORS[word])

    ax.set_xticks(range(len(WORDS)))
    ax.set_xticklabels(WORDS, fontsize=11)
    style_axes(ax, "Individual Ratings & Word-Level Mean")
    handles = [
        mpatches.Patch(color=WORD_COLORS[w], label=w) for w in WORDS
    ]
    ax.legend(handles=handles, fontsize=9, title="Word", title_fontsize=9,
              loc="upper right", framealpha=0.9)


# ── 5. Main ───────────────────────────────────────────────────────────────────

def main() -> None:
    raw, long = load_data(CSV_PATH)
    n_respondents = len(raw)
    by_word_type, by_type = compute_means(long)

    print_summary_table(by_word_type)

    # ── Figure layout ──────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(18, 16))
    fig.suptitle(
        "Perceived Tabooness of English Swear Words\n"
        f"Survey data · n = {n_respondents} respondents · 1–7 scale",
        fontsize=16, fontweight="bold", y=0.98,
    )

    gs = GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])          # grouped bar (wide)
    ax2 = fig.add_subplot(gs[0, 2])           # overall type comparison
    ax3 = fig.add_subplot(gs[1, :2])          # heatmap (wide)
    ax4 = fig.add_subplot(gs[1, 2], polar=True)  # radar
    ax5 = fig.add_subplot(gs[2, :])           # strip plot (full width)

    plot_grouped_bar(ax1, by_word_type)
    plot_type_comparison(ax2, by_type)
    plot_heatmap(ax3, by_word_type)
    plot_word_radar(ax4, by_word_type)
    plot_strip_with_mean(ax5, long)

    # Footnote
    fig.text(
        0.5, 0.01,
        "Invective = directed insult · Emotive = speaker's feeling · "
        "Informative = neutral / literal context\n"
        "Error bars show ± 1 standard error of the mean.",
        ha="center", fontsize=9, color="gray",
    )

    out_path = "tabooness_analysis.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {out_path}")
    plt.show()


if __name__ == "__main__":
    main()
