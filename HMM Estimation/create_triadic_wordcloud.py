from __future__ import annotations

import hashlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "Artefacts" / "Analysis_v3"

FIGURE_PNG = OUTPUT_DIR / "triadic_wordcloud_figure.png"
PHRASE_COUNTS_CSV = OUTPUT_DIR / "triadic_wordcloud_phrase_counts_v3.csv"

TEAL = "#0E868B"
PURPLE = "#4A237D"
GREEN = "#3DAB2C"
BLUE = "#1F5496"
YELLOW = "#E8BE00"
WHITE = "#FFFFFF"

WORD_DATA = {
    "Appreciation": {
        "Seems okay but I should check": 85,
        "I still want final say": 88,
        "AI helps but not blindly": 80,
        "Looks useful but situation changed": 78,
        "I know the local issue better": 82,
        "Need to check with the team": 75,
        "Let me confirm first": 72,
        "Maybe later not now": 70,
        "Customer request changed": 76,
        "I don't want to rush this": 79,
        "Need to see the latest update": 74,
        "It depends on the plant situation": 77,
        "I trust it but with caution": 81,
        "Something feels missing": 68,
        "Better to be careful today": 73,
    },
    "Neutral": {
        "Not sure yet": 82,
        "Need more context": 78,
        "I need to think about it": 80,
        "Let's wait and see": 75,
        "Could be right could be wrong": 77,
        "I don't have enough information": 79,
        "Need another opinion": 73,
        "Need to ask someone first": 76,
        "I want to compare options": 74,
        "Maybe approve later": 72,
        "Looks unclear to me": 71,
        "I need more details": 78,
        "Hard to judge now": 70,
        "Not enough confidence": 75,
        "Let's double-check the numbers": 77,
        "I don't fully get the reason": 76,
        "Need to understand the logic": 74,
        "Could work but unsure": 73,
    },
    "Aversion": {
        "I don't trust this": 90,
        "Too risky for me": 88,
        "I prefer doing it myself": 85,
        "AI doesn't know reality": 87,
        "I've seen this fail before": 83,
        "Better safe than sorry": 84,
        "This could cause trouble": 82,
        "I don't want to be responsible": 86,
        "Not worth the risk": 81,
        "I don't believe the output": 85,
        "Human judgment is better": 84,
        "The system misses too much": 82,
        "I don't want surprises": 80,
        "This feels wrong": 79,
        "I won't approve this": 89,
        "AI is guessing again": 81,
        "I know better than the model": 83,
        "If it fails we pay the price": 86,
    },
}

STATE_ORDER = ["Appreciation", "Neutral", "Aversion"]
STATE_TITLES = {
    "Appreciation": "(a) Reasons from Appreciation-state managers",
    "Neutral": "(b) Reasons from Neutral-state managers",
    "Aversion": "(c) Reasons from Aversion-state managers",
}
STATE_PALETTES = {
    "Appreciation": [GREEN, TEAL, BLUE],
    "Neutral": [TEAL, BLUE, PURPLE],
    "Aversion": [PURPLE, BLUE, YELLOW],
}


def color_func_factory(palette: list[str]):
    def color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        del font_size, position, orientation, random_state, kwargs
        digest = hashlib.md5(word.encode("utf-8")).hexdigest()
        return palette[int(digest, 16) % len(palette)]

    return color_func


def phrase_counts() -> pd.DataFrame:
    rows = []
    for state, frequencies in WORD_DATA.items():
        total = sum(frequencies.values())
        for phrase, frequency in frequencies.items():
            rows.append(
                {
                    "state": state,
                    "reason_phrase": phrase,
                    "weighted_count": frequency,
                    "share_within_state": frequency / total,
                    "source": "wordcloud_library_template",
                }
            )
    return pd.DataFrame(rows)


def create_wordcloud_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    fig.patch.set_facecolor(WHITE)

    for ax, state in zip(axes, STATE_ORDER):
        wordcloud = WordCloud(
            width=900,
            height=700,
            background_color=WHITE,
            colormap=None,
            relative_scaling=0.6,
            min_font_size=12,
            max_font_size=150,
            prefer_horizontal=0.6,
            max_words=80,
            collocations=False,
            random_state=42,
            margin=10,
        ).generate_from_frequencies(WORD_DATA[state])

        wordcloud.recolor(color_func=color_func_factory(STATE_PALETTES[state]))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        ax.text(
            0.5,
            1.02,
            STATE_TITLES[state],
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    plt.tight_layout()
    fig.savefig(FIGURE_PNG, dpi=300, bbox_inches="tight", facecolor=WHITE, pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phrase_counts().to_csv(PHRASE_COUNTS_CSV, index=False)
    create_wordcloud_figure()
    print(PHRASE_COUNTS_CSV)
    print(FIGURE_PNG)


if __name__ == "__main__":
    main()
