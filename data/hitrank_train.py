"""Train a HitRanker model (data/hitrank_*.txt) with LightGBM LambdaRank.

Input is a TSV with the columns query, target (database keys) and one column per feature
(names as in HitRanker.cpp). It is not made by Foldseek itself. A hit is correct if the query and
the target are in the same SCOP fold. Training is deterministic; the output keeps only what
HitRanker reads.
Usage: hitrank_train.py native|target|seq features.tsv query.lookup target.lookup scop_lookup.tsv out.txt
"""

import sys

import lightgbm as lgb
import pandas as pd

# no logE (depends on the database size) and no counts that grow with the alignment length
SPEC = [
    "bits", "fident", "gapres", "bits_per_col", "s3_mean", "sa_mean", "id3", "pos3", "posrun",
    "span_ratio", "cont_both", "cont_span", "rg_ratio",
    "a_col_same", "a_col_wrong", "a_col_qsse_gap", "a_col_tsse_gap",
    "a_match_q", "a_match_t", "a_unm_qE", "a_unm_qH", "a_unm_tE", "a_unm_tH",
]  # fmt: skip
CONTACTS = ["cont_both", "cont_span", "rg_ratio"]
MODELS = {
    # the other features overfit to SCOP domains when the native factor is available
    "native": (["bits", "native", "lddt", "tm"],
               "HitRanker model for query and target C-alpha: LambdaRank 100 trees x 7 leaves on SCOP40, foldseek2 12-state"),
    "target": (SPEC,
               "HitRanker model for target C-alpha only: LambdaRank 100 trees x 7 leaves on SCOP40 with ModernProst queries, foldseek2 12-state"),
    "seq": ([f for f in SPEC if f not in CONTACTS],
            "HitRanker model without C-alpha: LambdaRank 100 trees x 7 leaves on SCOP40 with ModernProst queries and targets, foldseek2 12-state"),
}  # fmt: skip
KEEP = ("feature_names", "Tree", "num_leaves", "split_feature", "threshold", "decision_type",
        "left_child", "right_child", "leaf_value")  # fmt: skip


def read_names(path):
    with open(path) as fh:
        return {int(line.split()[0]): line.split()[1] for line in fh}


def main():
    kind, features, query_lookup, target_lookup, scop_lookup, out = sys.argv[1:7]
    columns, description = MODELS[kind]

    hits = pd.read_csv(features, sep="\t")
    hits["q"] = hits["query"].map(read_names(query_lookup))
    hits["t"] = hits["target"].map(read_names(target_lookup))
    hits = hits[hits.q != hits.t]
    with open(scop_lookup) as fh:
        fold = {name: ".".join(scop.split(".")[:2]) for name, scop in (line.split() for line in fh)}
    qfold = hits.q.map(fold)
    hits["tp"] = ((qfold == hits.t.map(fold)) & qfold.notna()).astype(int)
    # LambdaRank needs the hits of one query to be contiguous
    hits = hits.sort_values(["q", "bits"], ascending=[True, False]).reset_index(drop=True)

    model = lgb.LGBMRanker(
        objective="lambdarank",
        lambdarank_truncation_level=200,
        n_estimators=100,
        num_leaves=7,
        learning_rate=0.15,
        subsample=0.8,
        subsample_freq=1,
        colsample_bytree=0.8,
        deterministic=True,
        force_row_wise=True,
        verbose=-1,
    )
    model.fit(hits[columns], hits.tp.values, group=hits.groupby("q", sort=False).size().values)

    lines = [f"# {description}"]
    for line in model.booster_.model_to_string().splitlines():
        if line == "end of trees":
            break
        if "=" in line and line.split("=", 1)[0] in KEEP:
            lines.append(line)
    lines.append("end of trees")
    with open(out, "w") as fh:
        fh.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
