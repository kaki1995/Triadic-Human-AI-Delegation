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

FIGURE_PNG = OUTPUT_DIR / "triadic_wordcloud_contextual_figure.png"
FIGURE_SVG = OUTPUT_DIR / "triadic_wordcloud_contextual_figure.svg"
PHRASE_COUNTS_CSV = OUTPUT_DIR / "triadic_wordcloud_contextual_phrase_counts_v3.csv"

TEAL = "#0E868B"
PURPLE = "#4A237D"
GREEN = "#3DAB2C"
BLUE = "#1F5496"
YELLOW = "#E8BE00"
WHITE = "#FFFFFF"

REASON_DATA = {
    "Appreciation": [
        ("I still want final say", 92, "Performance pressure"),
        ("I trust it but with caution", 89, "Forecast accuracy"),
        ("Seems okay but I should check", 86, "Forecast accuracy"),
        ("Looks useful but situation changed", 83, "Demand volatility"),
        ("I know the local issue better", 80, "Task complexity"),
        ("Need to check with the team", 77, "Task complexity"),
        ("AI helps but not blindly", 74, "Task complexity"),
        ("Need to see the latest update", 73, "Decision latency"),
        ("Germany shift changed", 70, "Task complexity"),
        ("China supplier is late", 68, "Supply disruptions"),
        ("Better to be careful today", 66, "Recent negative shock"),
        ("I don't want to rush this", 64, "Decision latency"),
    ],
    "Neutral": [
        ("Not sure yet", 90, "Task complexity"),
        ("I need more details", 86, "Forecast accuracy"),
        ("I need to think about it", 84, "Task complexity"),
        ("Need to understand the logic", 82, "Forecast accuracy"),
        ("I don't have enough information", 80, "Task complexity"),
        ("Need more context", 78, "Task complexity"),
        ("Need to ask someone first", 76, "Performance pressure"),
        ("Let's double-check the numbers", 74, "Forecast accuracy"),
        ("Could be right could be wrong", 72, "Forecast accuracy"),
        ("Mexico border delay unclear", 70, "Decision latency"),
        ("Need another opinion", 68, "Performance pressure"),
        ("Maybe approve later", 66, "Decision latency"),
    ],
    "Aversion": [
        ("I don't trust this", 92, "Forecast accuracy"),
        ("AI doesn't know reality", 90, "Demand volatility"),
        ("Too risky for me", 88, "Supply disruptions"),
        ("I won't approve this", 86, "Performance pressure"),
        ("I prefer doing it myself", 84, "Performance pressure"),
        ("I don't believe the output", 82, "Forecast accuracy"),
        ("Human judgment is better", 80, "Task complexity"),
        ("The system misses too much", 78, "Task complexity"),
        ("If it fails we pay the price", 76, "Target difficulty"),
        ("Brazil port backlog is risky", 74, "Supply disruptions"),
        ("Better safe than sorry", 72, "Recent negative shock"),
        ("I've seen this fail before", 70, "Recent negative shock"),
    ],
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
    for state, reasons in REASON_DATA.items():
        total = sum(frequency for _, frequency, _ in reasons)
        for phrase, frequency, contextual_factor in reasons:
            rows.append(
                {
                    "state": state,
                    "reason_phrase": phrase,
                    "contextual_factor": contextual_factor,
                    "weighted_count": frequency,
                    "share_within_state": frequency / total,
                    "source": "wordcloud_library_contextual_template",
                }
            )
    return pd.DataFrame(rows)


def frequencies_for_state(state: str) -> dict[str, int]:
    return {phrase: frequency for phrase, frequency, _ in REASON_DATA[state]}


def create_wordcloud_figure() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(22, 9))
    fig.patch.set_facecolor(WHITE)

    for ax, state in zip(axes, STATE_ORDER):
        frequencies = frequencies_for_state(state)
        wordcloud = WordCloud(
            width=900,
            height=700,
            background_color=WHITE,
            colormap=None,
            relative_scaling=0.45,
            min_font_size=12,
            max_font_size=115,
            prefer_horizontal=1.0,
            max_words=80,
            collocations=False,
            random_state=42,
            margin=10,
        ).generate_from_frequencies(frequencies)

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
    fig.savefig(FIGURE_SVG, bbox_inches="tight", facecolor=WHITE, pad_inches=0.3)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    phrase_counts().to_csv(PHRASE_COUNTS_CSV, index=False)
    create_wordcloud_figure()
    print(PHRASE_COUNTS_CSV)
    print(FIGURE_PNG)
    print(FIGURE_SVG)


if __name__ == "__main__":
    main()
