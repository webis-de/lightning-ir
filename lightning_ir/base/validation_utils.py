"""Validation utilities module for Lightning IR.

This module contains utility functions for validation and evaluation of Lightning IR models."""

from typing import Dict, Sequence

import ir_measures
import numpy as np
import pandas as pd
import torch


def create_run_from_scores(
    query_ids: Sequence[str], doc_ids: Sequence[Sequence[str]], scores: torch.Tensor
) -> pd.DataFrame:
    """Convenience function to create a run DataFrame from query and document ids and scores.

    Args:
        query_ids (Sequence[str]): List of query IDs.
        doc_ids (Sequence[Sequence[str]]): List of lists containing document IDs for each query.
        scores (torch.Tensor): Tensor containing scores for each query-document pair.
    Returns:
        pd.DataFrame: DataFrame containing the run information with columns:
            query_id, q0, doc_id, score, rank, and system.
    """
    num_docs = [len(ids) for ids in doc_ids]
    df = pd.DataFrame(
        {
            "query_id": np.array(query_ids).repeat(num_docs),
            "q0": 0,
            "doc_id": sum(map(lambda x: list(x), doc_ids), []),
            "score": scores.float().numpy(force=True).reshape(-1),
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
    """Convenience function to create a qrels DataFrame from a list of dictionaries.

    Args:
        qrels (Sequence[Dict[str, int]]): Mappings of doc_id -> relevance for each query. Defaults to None.
    Returns:
        pd.DataFrame: DataFrame with columns: query_id, q0, doc_id, and relevance.
    """
    return pd.DataFrame.from_records(qrels)


def evaluate_run(run: pd.DataFrame, qrels: pd.DataFrame, measures: Sequence[str]) -> Dict[str, float]:
    """Convenience function to evaluate a run against qrels using a set of measures.

    .. _ir-measures: https://ir-measur.es/en/latest/index.html

    Args:
        run (pd.DataFrame): Parse TREC run.
        qrels (pd.DataFrame): Parse TREC qrels.
        measures (Sequence[str]): Metrics corresponding to ir-measures_ measure strings.
    Returns:
        Dict[str, float]: Calculated metrics.
    """
    parsed_measures = [ir_measures.parse_measure(measure) for measure in measures]
    metrics = {str(measure): measure.calc_aggregate(qrels, run) for measure in parsed_measures}
    return metrics
