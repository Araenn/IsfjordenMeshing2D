"""Generate data/Isfjorden.txt from an external 2D model pickle."""

import argparse
from pathlib import Path

from convertDatasFunctions import export_bathy_from_pkl


ROOT = Path(__file__).resolve().parent
VS_MODEL_PICKLE_PATH = Path(
    "/home/lea/Desktop/code/resonance_model/resonance_model/picking/vs_model_2d.pkl"
)
OUTPUT_PATH = ROOT / "data" / "Isfjorden.txt"
DX = 4.08  # Horizontal geometry resampling step in metres; set for your dataset.


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vs-model", type=Path, default=VS_MODEL_PICKLE_PATH,
                        help="External pickle containing x_km and z_m sample columns.")
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH,
                        help="Destination geometry TXT file (default: data/Isfjorden.txt).")
    parser.add_argument("--dx", type=float, default=DX,
                        help="Horizontal resampling step in metres (default: 4.08).")
    args = parser.parse_args(argv)
    model_path = args.vs_model.expanduser()
    output_path = args.output.expanduser()
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    if not output_path.is_absolute():
        output_path = ROOT / output_path
    try:
        export_bathy_from_pkl(model_path, output_path, dx=args.dx)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
