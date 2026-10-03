"""Export one coherent GENE scan family from the fmc dataset as a pool CSV.

Run with an environment that has pandas (the Fusion_Microinstability_Classifier
venv does; this repo's does not need it):

    python scripts/export_mgkdb_pool.py \
        --data C:/Users/joesc/git/Fusion_Microinstability_Classifier/data \
        --family global_cfs_cdirs_m2116_hatch_pscans \
        --output data/pools/hatch_pscans_global.csv

Coordinates are design variables, not raw local inputs. Each scan directory is
one profile variant; its temperature and density gradient factors are the
directory means of omt_e and omn_e divided by the median over directories at
the same radius. Within a directory the factor drifts slightly with radius
because the variants were not pure multipliers, so the mean is a label for the
variant, and the drift is recorded in the export.
"""
import argparse
import glob
import pathlib

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=pathlib.Path, required=True)
    parser.add_argument("--family", required=True,
                        help="campaign parent, e.g. global_cfs_cdirs_m2116_hatch_pscans")
    parser.add_argument("--output", type=pathlib.Path, required=True)
    args = parser.parse_args()

    df = pd.concat([pd.read_pickle(f) for f in sorted(glob.glob(
        str(args.data/"mode_ID_GENE_data_chunk_*.pkl")))], ignore_index=True)
    df = df[df.directory.str.contains(args.family + "_scanfiles", regex=False)].copy()
    if df.empty:
        raise SystemExit(f"no rows for family {args.family}")
    df["x0"] = df.x0.round(4)

    per = df.groupby(["directory", "x0"])[["omt_e", "omn_e"]].first().reset_index()
    median = per.groupby("x0")[["omt_e", "omn_e"]].transform("median")
    per["factor_T"] = per.omt_e/median.omt_e
    per["factor_n"] = per.omn_e/median.omn_e
    factors = per.groupby("directory").agg(
        factor_T=("factor_T", "mean"), factor_n=("factor_n", "mean"),
        drift_T=("factor_T", lambda s: float(s.max()-s.min())),
        drift_n=("factor_n", lambda s: float(s.max()-s.min())))

    out = df.join(factors, on="directory")
    out = out[["directory", "factor_T", "factor_n", "drift_T", "drift_n", "x0", "kymin",
               "gamma", "omega", "mode_ID"]].rename(columns={"kymin": "ky"})
    out["mode_ID"] = out.mode_ID.fillna("")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False, float_format="%.8g")
    print(f"{len(out)} rows, {out.directory.nunique()} directories -> {args.output}")
    print(f"gamma == -1 sentinels: {int((out.gamma == -1).sum())}; "
          f"unlabelled: {int((out.mode_ID == '').sum())}; "
          f"max within-directory factor drift T/n: "
          f"{factors.drift_T.max():.3f}/{factors.drift_n.max():.3f}")


if __name__ == "__main__":
    main()
