"""CLI entry point for synthBOLD data generation."""

import argparse
import logging
from pathlib import Path

from synthbold.base import BaseGeometry
from synthbold.config import Config
from synthbold.geometries import Cubes, Cylinders, Shapes, Spheres, Tetrahedra, Toroids


def setup_logging(level: int = logging.INFO) -> None:
    """Configures global logging for CLI entrypoint."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)-20.20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def get_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="synthbold", description="Generate synthetic data and save to disk."
    )
    add = parser.add_argument
    add(
        "-o",
        "--output",
        required=True,
        type=Path,
        help="Output file path. Use .zarr or .nii extension to select format.",
    )
    add(
        "-l",
        "--label",
        required=True,
        choices=["shapes", "cylinders", "spheres", "tetrahedra", "cubes", "toroids"],
        help="Type of label map to generate.",
    )
    add("--n-sample", type=int, default=1, help="Number of samples (default: 1).")
    add("--config", type=Path, default=None, help="Path to a YAML config file.")
    add("--seed", type=int, default=None, help="Random seed for reproducibility.")
    return parser


def main() -> None:
    """Parse CLI arguments, build the data generator, and run synthesis."""
    setup_logging()
    parser = get_parser()
    args = parser.parse_args()
    logger = logging.getLogger(__name__)

    if args.output.suffix not in {".zarr", ".nii"}:
        parser.error(f"Output must have a .zarr or .nii extension, got: {args.output}")

    # Config is frozen; seed override requires a copy.
    config = Config.from_yaml(args.config) if args.config is not None else Config()
    if args.seed is not None:
        config = config.model_copy(update={"seed": args.seed})

    generator: BaseGeometry
    match args.label:
        case "shapes":
            generator = Shapes.from_config(config)
        case "cylinders":
            generator = Cylinders.from_config(config)
        case "tetrahedra":
            generator = Tetrahedra.from_config(config)
        case "spheres":
            generator = Spheres.from_config(config)
        case "cubes":
            generator = Cubes.from_config(config)
        case "toroids":
            generator = Toroids.from_config(config)
        case _:
            raise ValueError(f"Unknown label: {args.label}")

    logger.info("Generating %d maps (%s)", args.n_sample, args.label)
    generator(n_sample=args.n_sample, fname=args.output)


if __name__ == "__main__":
    main()
