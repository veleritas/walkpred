import pandas as pd
import numpy as np

import subprocess

from itertools import product
from itertools import chain

from tqdm import tqdm
from collections import defaultdict

from glob import glob

def all_pairs(df):
    chem = df["chemical_id"].unique()
    dise = df["disease_id"].unique()

    return set(product(chem, dise))

def pair_to_df(pairs):
    return pd.DataFrame(list(pairs), columns = ["chemical_id", "disease_id"])

def df_to_pairs(df):
    return set(zip(df["chemical_id"], df["disease_id"]))


np.random.seed(72345612)

nodes = pd.read_csv("tmp/hetionet_nodes.tsv", sep='\t')

edges = None

gold = pd.read_csv("tmp/hetionet_gold.tsv", sep='\t')

all_gpairs = df_to_pairs(gold)
holdout_ratio = 0.2


def split_data():
    holdout = gold.sample(frac = holdout_ratio)

    tpairs = all_gpairs - df_to_pairs(holdout)

    train = (pair_to_df(tpairs)
        .merge(gold, how="left", on=["chemical_id", "disease_id"])
    )

    holdout_assumed_false = all_pairs(holdout) - df_to_pairs(holdout) - df_to_pairs(train)
    holdout_final = holdout.append(pair_to_df(holdout_assumed_false))

    #---------------------

    train_assumed_false = (all_pairs(train) - df_to_pairs(train)
                           - df_to_pairs(holdout_final)
    )
    train_final = train.append(pair_to_df(train_assumed_false))

    assert df_to_pairs(train_final).isdisjoint(df_to_pairs(holdout_final))
    return (holdout_final, train_final)

def clean_df(df):
    """Remove empty cells.
    Set numeric labels for edge type.
    """

    return (df
        [["chemical_id", "disease_id", "etype"]]
        .fillna(0)
        .replace({"TREATS_CDtDO": 1})
    )

def subsample(df, M=4):
    """Subsample the training data to remove the vast majority of
    negative training examples."""

    positives = df.query("etype == 1")

    return (positives
        .append(
            (df
                .query("etype == 0")
                .sample(len(positives) * M)
            )
        )
        .reset_index(drop=True)
    )

def add_uids(df):
    return (df
        .merge(
            nodes[["node_uid", "node_id"]],
            how="inner", left_on="chemical_id", right_on="node_id"
        )
        .merge(
            nodes[["node_uid", "node_id"]],
            how="inner", left_on="disease_id", right_on="node_id"
        )
        .drop(["node_id_x", "node_id_y"], axis=1)
        .rename(columns={
            "node_uid_x": "chemical_uid",
            "node_uid_y": "disease_uid"
        })
    )

def build_adjlist(train, edges, idx, fbase):
    pos = train.query("etype == 1")

    adjlist = defaultdict(set)

#    for suid, tuid in tqdm(
#        zip(
#            chain(pos["chemical_uid"], edges["source_uid"]),
#            chain(pos["disease_uid"], edges["target_uid"])
#        ),
#        desc="Building",
#        total=len(pos) + len(edges)
#    ):

    for suid, tuid in tqdm(
        zip(edges["source_uid"], edges["target_uid"]),
        desc="Building",
        total=len(edges)
    ):

        adjlist[suid].add(tuid)
        adjlist[tuid].add(suid)

    # write to file
    fname = "tmp/data/adjlist/{}_adj_{}.txt".format(fbase, idx)
    with open(fname, "w") as fout:
        for key, vals in tqdm(adjlist.items(), desc="Saving"):
            vals = sorted(list(vals))
            vals = list(map(str, vals))

            fout.write("{} {}\n".format(key, " ".join(vals)))

def work(fbase):
    K = 4
    for idx in range(K):
        print("Splitting data for fold {}".format(idx))
        # sample gold standard and split
        holdout, train = split_data()

        holdout = (holdout
            .pipe(clean_df)
            .pipe(add_uids)
        )

        train = (train
            .pipe(clean_df)
            .pipe(subsample)
            .pipe(add_uids)
        )


#        # reduce the number of negatives in the holdout
#        hold_pos = len(holdout.query("etype == 1"))
#        if len(holdout.query("etype == 0")) > hold_pos:
#            temp = (holdout
#                .query("etype == 0")
#                .sample(n=hold_pos)
#                .append(holdout.query("etype == 1"))
#            )
#
#            holdout = temp



        holdout.to_csv(
            "tmp/data/holdout/{}_holdout_{}.tsv".format(fbase, idx),
            sep='\t', index=False
        )

        train.to_csv(
            "tmp/data/train/{}_train_{}.tsv".format(fbase, idx),
            sep='\t', index=False
        )

        # append gold training edges to network
        build_adjlist(train, edges, idx, fbase)

        subprocess.run(["bash", "make_embedding.sh", fbase, str(idx)])

def main():
    global edges

    for edge_fname in glob("tmp/edges/fake*.tsv"):
#    for edge_fname in chain(["tmp/edges/empty.tsv"], glob("tmp/edges/drug*.tsv")):
        edges = pd.read_csv(edge_fname, sep='\t', low_memory=False)
        fbase = edge_fname.split("/")[2][:-4]

        print(fbase)
        work(fbase)

if __name__ == "__main__":
    main()
