from typing import Dict, Sequence, Tuple

import ir_measures
import numpy as np
import pandas as pd
import torch


def create_run_from_scores(
    query_ids: Sequence[str], doc_ids: Sequence[Tuple[str, ...]], scores: torch.Tensor
) -> pd.DataFrame:
    num_docs = [len(ids) for ids in doc_ids]
    scores = scores.float().cpu().detach().numpy().reshape(-1)
    df = pd.DataFrame(
        {
            "query_id": np.array(query_ids).repeat(num_docs),
            "q0": 0,
            "doc_id": np.array(sum(map(lambda x: list(x), doc_ids), [])),
            "score": scores,
            "system": "lightning_ir",
        }
    )
    df["rank"] = df.groupby("query_id")["score"].rank(ascending=False, method="first")

    def key(series: pd.Series) -> pd.Series:
        if series.name == "query_id":
            return series.map({query_id: i for i, query_id in enumerate(query_ids)})
        return series

    df = df.sort_values(["query_id", "rank"], ascending=[True, True], key=key)
    return df


def create_qrels_from_dicts(qrels: Sequence[Dict[str, int]]) -> pd.DataFrame:
    return pd.DataFrame.from_records(qrels)


def evaluate_run(
    run: pd.DataFrame, qrels: pd.DataFrame, measures: Sequence[str]
) -> Dict[str, float]:
    parsed_measures = [ir_measures.parse_measure(measure) for measure in measures]
    metrics = {
        str(key): val
        for key, val in ir_measures.calc_aggregate(parsed_measures, qrels, run).items()
    }
    return metrics
