"""Generate the common 3D test grid for all implementations."""

import json
import os

NU_VALUES = [
    0,
    0.25,
    0.5,
    1,
    1.5,
    2,
    5,
    10,
    25,
    50,
    75,
    85,
    90,
    100,
    150,
    200,
    500,
    1000,
    -0.5,
    -1.5,
    -2,
]

UNSCALED_Z = [
    -50,
    -40,
    -30,
    -25,
    -20,
    -15,
    -12,
    -10,
    -8,
    -6,
    -4,
    -3,
    -2,
    -1,
    -0.5,
    -0.1,
    -0.001,
    -1e-6,
    0,
    1e-6,
    0.001,
    0.1,
    0.5,
    1,
    2,
    3,
    4,
    6,
    8,
    10,
    12,
    15,
    20,
    25,
    30,
    40,
    50,
]

SCALED_Z = [
    -1000,
    -500,
    -300,
    -200,
    -100,
    -50,
    -30,
    -10,
    -5,
    -1,
    -0.1,
    -0.01,
    0.001,
    0.01,
    0.1,
    1,
    5,
    10,
    30,
    50,
    100,
    200,
    300,
    500,
    1000,
]

UNSCALED_FUNCTIONS = [
    "besselj",
    "bessely",
    "besseli",
    "besselk",
    "hankel1",
    "hankel2",
    "airy",
    "airyprime",
    "biry",
    "biryprime",
]

SCALED_FUNCTIONS = [
    "besselj_scaled",
    "bessely_scaled",
    "besseli_scaled",
    "besselk_scaled",
    "hankel1_scaled",
    "hankel2_scaled",
    "airy_scaled",
    "airyprime_scaled",
    "biry_scaled",
    "biryprime_scaled",
]


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "..", "results")
    os.makedirs(results_dir, exist_ok=True)

    grid = {
        "unscaled_grid": {
            "nu_values": NU_VALUES,
            "re_values": UNSCALED_Z,
            "im_values": UNSCALED_Z,
            "functions": UNSCALED_FUNCTIONS,
        },
        "scaled_grid": {
            "nu_values": NU_VALUES,
            "re_values": SCALED_Z,
            "im_values": SCALED_Z,
            "functions": SCALED_FUNCTIONS,
        },
    }

    out_path = os.path.join(results_dir, "test_grid.json")
    with open(out_path, "w") as f:
        json.dump(grid, f, indent=2)

    n_unscaled = len(NU_VALUES) * len(UNSCALED_Z) * len(UNSCALED_Z)
    n_scaled = len(NU_VALUES) * len(SCALED_Z) * len(SCALED_Z)
    print(f"Grid generated: {out_path}")
    print(
        f"  Unscaled: {len(NU_VALUES)} nu x {len(UNSCALED_Z)} re x {len(UNSCALED_Z)} im = {n_unscaled} points"
    )
    print(
        f"  Scaled:   {len(NU_VALUES)} nu x {len(SCALED_Z)} re x {len(SCALED_Z)} im = {n_scaled} points"
    )


if __name__ == "__main__":
    main()
