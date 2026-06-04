from __future__ import annotations

from create_ai_authority_three_panel_joint_effects_v3 import (
    FigureConfig,
    StateCoefficients,
    create_figure,
)


# Placeholder values designed to create readable thesis-style surfaces.
# Replace these with fitted HMM/state-specific escalation estimates before
# final reporting.
ESCALATION_COEFFICIENTS: dict[str, StateCoefficients] = {
    "Aversion": StateCoefficients(
        alpha=0.62,
        beta1_x=-0.06,
        beta2_x2=-0.20,
        beta3_y=-0.12,
        beta4_y2=-0.16,
        beta5_xy=-0.18,
        beta6_c=-0.07,
        beta7_xc=-0.08,
        beta8_yc=-0.12,
    ),
    "Neutral": StateCoefficients(
        alpha=0.50,
        beta1_x=-0.04,
        beta2_x2=-0.18,
        beta3_y=-0.08,
        beta4_y2=-0.14,
        beta5_xy=-0.12,
        beta6_c=-0.05,
        beta7_xc=-0.05,
        beta8_yc=-0.08,
    ),
    "Appreciation": StateCoefficients(
        alpha=0.36,
        beta1_x=-0.03,
        beta2_x2=-0.14,
        beta3_y=-0.10,
        beta4_y2=-0.12,
        beta5_xy=-0.10,
        beta6_c=-0.04,
        beta7_xc=-0.04,
        beta8_yc=-0.06,
    ),
}

ESCALATION_P_VALUES: dict[str, dict[str, float]] = {
    "Aversion": {
        "alpha": 0.002,
        "beta1_x": 0.004,
        "beta2_x2": 0.006,
        "beta3_y": 0.002,
        "beta4_y2": 0.008,
        "beta5_xy": 0.003,
        "beta6_c": 0.004,
        "beta7_xc": 0.007,
        "beta8_yc": 0.006,
    },
    "Neutral": {
        "alpha": 0.003,
        "beta1_x": 0.009,
        "beta2_x2": 0.007,
        "beta3_y": 0.005,
        "beta4_y2": 0.008,
        "beta5_xy": 0.006,
        "beta6_c": 0.006,
        "beta7_xc": 0.018,
        "beta8_yc": 0.009,
    },
    "Appreciation": {
        "alpha": 0.004,
        "beta1_x": 0.012,
        "beta2_x2": 0.009,
        "beta3_y": 0.004,
        "beta4_y2": 0.007,
        "beta5_xy": 0.008,
        "beta6_c": 0.015,
        "beta7_xc": 0.026,
        "beta8_yc": 0.011,
    },
}

ESCALATION_CONFIG = FigureConfig(
    outcome_var="escalation_share",
    outcome_label="Escalation Rate",
    output_stem="figure_escalation_three_panel_joint_effects_v3",
    surface_alpha=0.72,
    gradient_norm_mode="global",
    gradient_padding=0.24,
)


def main() -> None:
    paths = create_figure(
        config=ESCALATION_CONFIG,
        fallback_coefficients=ESCALATION_COEFFICIENTS,
        fallback_p_values=ESCALATION_P_VALUES,
    )
    print("Saved Escalation Rate three-panel joint-effects outputs:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
