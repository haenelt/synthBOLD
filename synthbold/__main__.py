"""CLI entry point for synthBOLD data generation."""

import argparse
import logging
from pathlib import Path

from synthbold.config import Config
from synthbold.io import save_batch
from synthbold.pipeline import SynthPipeline, SynthSample
from synthbold.transforms import GaussianNoise, RicianNoise
from synthbold.transforms.functional import Pipeline, zscore


def setup_logging(level: int = logging.INFO) -> None:
    """Configures global logging for CLI entrypoint. In production, set this to
    logging.WARNING or logging.ERROR."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)-20.20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="synthbold", description="Generate synthetic BOLD data and save to disk."
    )
    add = parser.add_argument
    add(
        "--output",
        required=True,
        type=Path,
        help="Output directory. Created if not exists.",
    )
    add(
        "--n-sample",
        type=int,
        default=10,
        help="Number of generated label maps (default: 10).",
    )
    add(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for data generation (default: 4).",
    )
    add(
        "--config",
        type=Path,
        default=None,
        help="Path to a YAML config file (default: None).",
    )
    add(
        "--include-spheres",
        action="store_true",
        help="Include sphere label maps (default: False).",
    )
    add(
        "--include-tetrahedra",
        action="store_true",
        help="Include tetrahedron label maps (default: False).",
    )
    add(
        "--include-cubes",
        action="store_true",
        help="Include cube label maps (default: False).",
    )
    add(
        "--include-toroids",
        action="store_true",
        help="Include toroid label maps (default: False).",
    )
    add(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility (default: None).",
    )
    add(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity (default: INFO). Use WARNING or ERROR in production.",
    )
    return parser


def standardize(s: SynthSample) -> SynthSample:
    """Z-score normalize the field map statistics."""
    return s._replace(
        dbz_mean=zscore(s.dbz_mean),
        dbz_variance=zscore(s.dbz_variance, center=False),
        dbz_skewness=zscore(s.dbz_skewness),
        dbz_kurtosis=zscore(s.dbz_kurtosis),
    )


def main() -> None:
    """Parse CLI arguments, build the data generator, and run synthesis.

    Raises:
        ValueError: If `args.label` does not match a known label type (not reachable
            in practice, since argparse's `choices` already restrict it).
    """
    parser = get_parser()
    args = parser.parse_args()
    setup_logging(level=getattr(logging, args.log_level))
    logger = logging.getLogger(__name__)

    # Config is frozen; seed override requires a copy.
    config = Config.from_yaml(args.config) if args.config is not None else Config()
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    # instantiate pipeline
    logger.info("Instantiating synthesis pipeline.")
    indices = None
    synth_pipeline = SynthPipeline(
        dirname=args.output,
        n_samples=args.n_sample,
        config=config,
        indices=indices,
        include_spheres=args.include_spheres,
        include_tetrahedra=args.include_tetrahedra,
        include_cubes=args.include_cubes,
        include_toroids=args.include_toroids,
    )

    # Generate batch
    logger.info("Generating synthetic data.")
    sample, sample_distractor, _ = synth_pipeline(batch_size=args.batch_size)

    # Apply random noise to synthesized data (RicianNoise to magnitude data and
    # GaussianNoise to phase data)
    logger.info("Applying noise.")
    magnitude_noise = Pipeline(
        [RicianNoise.from_config(config)],
        prob=config.transform_prob,
        device=config.device,
        seed=config.seed,
    )
    phase_noise = Pipeline(
        [GaussianNoise.from_config(config)],
        prob=config.transform_prob,
        device=config.device,
        seed=config.seed,
    )

    sample = sample._replace(
        magnitude=magnitude_noise(sample.magnitude),
        phase=phase_noise(sample.phase),
    )
    if sample_distractor is not None:
        sample_distractor = sample_distractor._replace(
            magnitude=magnitude_noise(sample_distractor.magnitude),
            phase=phase_noise(sample_distractor.phase),
        )

    # Standardize
    logger.info("Standardize field map statistics.")
    sample = standardize(sample)
    if sample_distractor is not None:
        sample_distractor = standardize(sample_distractor)

    logger.info("Save to disk.")
    save_batch(sample._asdict(), args.output, "nii", "", None)
    if sample_distractor is not None:
        save_batch(sample_distractor._asdict(), args.output, "nii", "_distractor", None)


if __name__ == "__main__":
    main()
