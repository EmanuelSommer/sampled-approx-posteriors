"""Pool results from multiple experiments."""

import argparse
import json
import os
from pathlib import Path

import yaml


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Pool results from multiple experiments."
    )
    parser.add_argument(
        "--dir", type=str, required=True, help="Directory containing experiment results"
    )
    parser.add_argument(
        "--filter", type=str, default="", help="Filter for directories via a substring"
    )
    parser.add_argument(
        "--what",
        type=str,
        nargs="+",
        default=["PredPerf", "Lppd", "BasicConfig"],
        help="What to pool (list)",
    )
    parser.add_argument(
        "--warmstart", action="store_true", help="Pool warmstart as well"
    )
    parser.add_argument("--verbose", action="store_true", help="Verbose")
    parser.add_argument(
        "--output", type=str, default="pooled_results.json", help="Output file"
    )
    # add example usage for help
    parser.epilog = "Example usage: python pool_results.py --dir /path/to/dir --filter 'exp1' --what 'PredPerf' 'Lppd' --warmstart --verbose --output pooled_results.json"

    return parser.parse_args()


def main():
    """Summarizes the results of the multiple experiments."""
    args = parse_args()
    DIR = Path(args.dir)
    FILTER = args.filter

    results_json = {}
    all_dirs = os.listdir(DIR)
    # only directories
    filtered_dirs = [d for d in all_dirs if os.path.isdir(DIR / d)]
    if FILTER != "":
        filtered_dirs = [d for d in filtered_dirs if FILTER in d]

    for d in filtered_dirs:
        if args.verbose:
            print("#" * 40)  # noqa: T201
            print(f"# Processing {d}")  # noqa: T201
        subdirs = os.listdir(DIR / d / "eval")
        # only npy and json files
        subdirs = [s for s in subdirs if s.endswith(".npy") or s.endswith(".json")]
        # only files that contain the what
        subdirs = [s for s in subdirs if any([w in s for w in args.what])]
        # if warmstart is not requested, remove warmstart files containing 'warm'
        if not args.warmstart:
            subdirs = [s for s in subdirs if "warm" not in s]
        if subdirs == []:
            continue
        results_json[d] = {}
        for subdir in subdirs:
            if "BasicConfig" in args.what:
                with open(DIR / d / "config.yaml", "r") as f:
                    config_dict = yaml.load(f, Loader=yaml.FullLoader)
                    results_json[d]["config"] = config_dict

            subdir_str = subdir.split(".")[0]
            if subdir.endswith(".npy"):
                print(".npy processing not implemented yet")  # noqa: T201
            elif subdir.endswith(".json"):
                with open(DIR / d / "eval" / subdir, "r") as f:
                    res_dict = json.load(f)

                    if args.verbose:
                        print(f"{subdir_str}")  # noqa: T201
                    # if res dict has only one key and a single value, unpack it
                    if len(res_dict.keys()) == 1 and isinstance(
                        res_dict[list(res_dict.keys())[0]], float
                    ):
                        res_dict = res_dict[list(res_dict.keys())[0]]
                        if args.verbose:
                            print(f"    {round(res_dict, 5)}")  # noqa: T201
                    else:
                        if args.verbose:
                            print(res_dict)  # noqa: T201
                    results_json[d][subdir_str] = res_dict

    # save the results
    with open(DIR / args.output, "wt") as f:
        json.dump(results_json, f)

    if args.verbose:
        print("#" * 40)  # noqa: T201
        print(f"Pooled results saved to {args.output}")  # noqa: T201


if __name__ == "__main__":
    main()
