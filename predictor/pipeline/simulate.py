"""Simulation stage: generate a seeded, synthetic primer dataset.

No real wet-lab data exists in this repo (see CLAUDE.md's data rules), so the
pipeline is bootstrapped on a fully reproducible simulated dataset. The label
is a **proxy** — a thermodynamic desirability score pushed through a sigmoid,
plus a latent per-template random intercept and noise. It is emphatically not
an experimental measurement, and the provenance sidecar flags it as a proxy
so no downstream metric is ever mistaken for validation on real data.

The latent per-template effect induces within-group correlation, which is the
whole reason grouped CV (grouping by ``template_id``) is mandatory: it is
deliberately NOT exposed as a feature.

CLI-able: invoked directly or via ``predictor/workflows/Snakefile``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

from primer_core.features.composition import gc_clamp, gc_content
from primer_core.features.thermo import (
    calc_hairpin,
    calc_homodimer,
    calc_tm,
    three_prime_end_dg,
)

_BASES = "ACGT"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _desirability_z(seq: str, thermo_settings: dict[str, Any]) -> tuple[float, dict[str, float]]:
    """Compute the proxy desirability score ``z`` and the features behind it.

    The +coefficient * delta-G terms penalize strong 3'-end / dimer / hairpin
    structure (delta-G values are negative), i.e. they encode mispriming and
    robustness risk. Returns ``z`` plus the raw features (for optional reuse).
    """
    gc = gc_content(seq)
    clamp = gc_clamp(seq, 5)
    tm = calc_tm(seq, **thermo_settings)
    hairpin = calc_hairpin(seq, **thermo_settings)
    homodimer = calc_homodimer(seq, **thermo_settings)
    tpe = three_prime_end_dg(seq, **thermo_settings)

    z = (
        -0.5 * ((tm - 60.0) / 5.0) ** 2
        - 0.5 * ((gc - 0.5) / 0.15) ** 2
        + 0.27 * min(clamp, 3)
        + 0.4 * tpe
        + 0.3 * homodimer
        + 0.2 * hairpin
    )
    features = {
        "gc_content": gc,
        "gc_clamp": float(clamp),
        "tm": tm,
        "hairpin_dg": hairpin,
        "homodimer_dg": homodimer,
        "three_prime_end_dg": tpe,
    }
    return z, features


_PROVENANCE = {
    "source": "simulated (seeded)",
    "license": "CC0 / synthetic",
    "label_description": (
        "sigmoid of a thermodynamic desirability score + latent per-template " "effect + noise"
    ),
    "is_proxy_label": True,
    "grouping_key": "template_id",
    "notes": (
        "PROXY LABEL for plumbing only. Generated deterministically from a "
        "seed; NOT experimental data. The label is a thermodynamic desirability "
        "score (Tm/GC/clamp/3'-end/dimer/hairpin) pushed through a sigmoid, plus "
        "a latent per-template random intercept and Gaussian noise. The latent "
        "template effect is not a feature and induces within-group correlation, "
        "which is why grouped CV by template_id is required. Do not report any "
        "metric from this dataset as validated on real experimental efficiency."
    ),
}


def simulate_dataset(
    output_path: str | Path,
    seed: int = 1234,
    n_templates: int = 60,
    primers_per_template: int = 8,
    template_length: int = 400,
    thermo_settings: dict[str, Any] | None = None,
) -> Path:
    """Generate a seeded synthetic primer dataset and its provenance sidecar.

    Args:
        output_path: Destination CSV path; the provenance sidecar is written
            alongside it as ``<stem>.provenance.yaml``.
        seed: RNG seed for full reproducibility.
        n_templates: Number of synthetic template "genes".
        primers_per_template: Primers sampled per template.
        template_length: Length of each synthetic template.
        thermo_settings: primer3 settings passed to the thermo feature calls.

    Returns:
        Path to the written CSV.
    """
    import numpy as np
    import pandas as pd
    import yaml

    thermo_settings = thermo_settings or {}
    rng = np.random.RandomState(seed)

    rows: list[dict[str, Any]] = []
    for t in range(n_templates):
        template_id = f"t{t}"
        template_seq = "".join(rng.choice(list(_BASES), size=template_length))
        # Latent per-template random intercept — NOT exposed as a feature.
        template_effect = float(rng.normal(0.0, 0.10))

        for p in range(primers_per_template):
            primer_len = int(rng.randint(18, 26))  # 18..25 inclusive
            site_start = int(rng.randint(0, template_length - primer_len + 1))
            seq = template_seq[site_start : site_start + primer_len]

            z, _feats = _desirability_z(seq, thermo_settings)
            noise = float(rng.normal(0.0, 0.08))
            efficiency = _sigmoid(z) + template_effect + noise
            efficiency = float(min(1.0, max(0.0, efficiency)))

            rows.append(
                {
                    "primer_id": f"{template_id}_p{p}",
                    "sequence": seq,
                    "template_id": template_id,
                    "label": efficiency,
                    "template_seq": template_seq,
                    "site_start": site_start,
                }
            )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_path, index=False)

    sidecar = output_path.with_suffix(".provenance.yaml")
    with sidecar.open("w") as handle:
        yaml.safe_dump(_PROVENANCE, handle, sort_keys=False)

    return output_path


def simulate_from_config(
    output_path: str | Path,
    config_path: str | Path = "predictor/workflows/configs/config.yaml",
) -> Path:
    """Run ``simulate_dataset`` using parameters from the pipeline config."""
    import yaml

    with open(config_path) as handle:
        config = yaml.safe_load(handle)
    data_cfg = config.get("data", {})
    thermo_settings = config.get("thermo", {}).get("primer3_settings", {})
    return simulate_dataset(
        output_path,
        seed=data_cfg.get("seed", 1234),
        n_templates=data_cfg.get("n_templates", 60),
        primers_per_template=data_cfg.get("primers_per_template", 8),
        template_length=data_cfg.get("template_length", 400),
        thermo_settings=thermo_settings,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Simulate a seeded primer dataset.")
    parser.add_argument("--output", required=True, help="Path to write dataset CSV")
    parser.add_argument(
        "--config",
        default="predictor/workflows/configs/config.yaml",
        help="Path to pipeline config YAML",
    )
    return parser


def main() -> None:
    """CLI entrypoint: ``python -m predictor.pipeline.simulate``."""
    args = _build_arg_parser().parse_args()
    simulate_from_config(args.output, args.config)


if __name__ == "__main__":
    main()
