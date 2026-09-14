import importlib.util
import json
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import ir_datasets
import pandas as pd
import pytest
from _pytest.fixtures import SubRequest

from lightning_ir import BiEncoderModule, LightningIRDataModule, LightningIRModule, LightningIRTrainer, RunDataset
from lightning_ir.callbacks import (
    HuggingFaceExportCallback,
    IndexCallback,
    MeanValidationMetricCallback,
    RegisterLocalDatasetCallback,
    ReRankCallback,
    SearchCallback,
)
from lightning_ir.models import DprConfig
from lightning_ir.retrieve import (
    FaissFlatIndexConfig,
    FaissIVFIndexConfig,
    FaissSearchConfig,
    IndexConfig,
    PlaidIndexConfig,
    PlaidSearchConfig,
    SearchConfig,
    SeismicIndexConfig,
    SeismicSearchConfig,
    TorchDenseIndexConfig,
    TorchDenseSearchConfig,
    TorchSparseIndexConfig,
    TorchSparseSearchConfig,
)

from .conftest import CORPUS_DIR, DATA_DIR


@pytest.fixture(
    params=[
        FaissFlatIndexConfig(),
        FaissIVFIndexConfig(num_centroids=16),
        TorchSparseIndexConfig(),
        TorchDenseIndexConfig(),
        PlaidIndexConfig(num_centroids=8),
        SeismicIndexConfig(num_postings=32),
    ],
    ids=["Faiss", "FaissIVF", "Sparse", "Dense", "Plaid", "Seismic"],
)
def index_config(request: SubRequest) -> IndexConfig:
    return request.param


def run_datamodule(module: LightningIRModule, inference_datasets: Sequence[RunDataset]) -> LightningIRDataModule:
    datamodule = LightningIRDataModule(num_workers=0, inference_batch_size=2, inference_datasets=inference_datasets)
    datamodule.setup(stage="test")
    return datamodule


# @pytest.mark.parametrize("devices", (1, 2))
def test_index_callback(
    tmp_path: Path,
    bi_encoder_module: BiEncoderModule,
    doc_datamodule: LightningIRDataModule,
    index_config: IndexConfig,
    # devices: int,
):
    if bi_encoder_module.config.model_type not in index_config.SUPPORTED_MODELS:
        pytest.skip(
            f"Indexing not supported for {bi_encoder_module.config.__class__.__name__} model "
            f"and {index_config.__class__.__name__} indexer"
        )

    if "Seismic" in index_config.__class__.__name__:
        if importlib.util.find_spec("seismic") is None:
            pytest.skip("seismic package is not available")

    index_dir = tmp_path / "index"
    index_callback = IndexCallback(index_config=index_config, index_dir=index_dir)

    trainer = LightningIRTrainer(
        # devices=devices,
        logger=False,
        enable_checkpointing=False,
        callbacks=[index_callback],
    )
    trainer.index(bi_encoder_module, datamodule=doc_datamodule)

    assert doc_datamodule.inference_datasets is not None
    assert index_callback.indexer.num_embeddings and index_callback.indexer.num_docs
    assert index_callback.indexer.num_embeddings >= index_callback.indexer.num_docs

    dataset_id = doc_datamodule.inference_datasets[0].dataset_id
    index_dir = index_dir / dataset_id
    assert (
        (index_dir / "index.faiss").exists()  # faiss
        or (index_dir / "index.pt").exists()  # sparse
        or (index_dir / "centroids.npy").exists()  # plaid
        or (index_dir / ".index.seismic").exists()  # seismic
    )
    assert (index_dir / "doc_ids.txt").exists()
    doc_ids_path = index_dir / "doc_ids.txt"
    doc_ids = doc_ids_path.read_text().split()
    for idx, doc_id in enumerate(doc_ids):
        assert doc_id == f"doc_id_{idx + 1}"
    assert (index_dir / "config.json").exists()


def get_index(
    bi_encoder_module: BiEncoderModule,
    doc_datamodule: LightningIRDataModule,
    search_config: SearchConfig,
) -> Path:
    index_config: IndexConfig
    if isinstance(search_config, FaissSearchConfig):
        index_type = "faiss"
        index_config = FaissFlatIndexConfig()
    elif isinstance(search_config, TorchSparseSearchConfig):
        index_type = "sparse"
        index_config = TorchSparseIndexConfig()
    elif isinstance(search_config, TorchDenseSearchConfig):
        index_type = "dense"
        index_config = TorchDenseIndexConfig()
    elif isinstance(search_config, PlaidSearchConfig):
        index_type = "plaid"
        index_config = PlaidIndexConfig(num_centroids=8)
    elif isinstance(search_config, SeismicSearchConfig):
        index_type = "seismic"
        index_config = SeismicIndexConfig(num_postings=32)
    else:
        raise ValueError("Unknown search_config type")
    index_dir = (
        DATA_DIR
        / "indexes"
        / f"{index_type}-{bi_encoder_module.config.model_type}-{bi_encoder_module.config.similarity_function}"
    )
    if index_dir.exists():
        return index_dir

    index_callback = IndexCallback(index_config=index_config, index_dir=index_dir)

    trainer = LightningIRTrainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[index_callback],
    )
    trainer.test(bi_encoder_module, datamodule=doc_datamodule)
    return index_dir


@pytest.mark.parametrize(
    "search_config",
    (
        FaissSearchConfig(k=3, imputation_strategy="min", candidate_k=3),
        FaissSearchConfig(k=3, imputation_strategy="gather", candidate_k=3),
        PlaidSearchConfig(k=3, centroid_score_threshold=0),
        TorchSparseSearchConfig(k=3),
        TorchDenseSearchConfig(k=3),
        SeismicSearchConfig(k=3),
    ),
    ids=["FaissMin", "FaissGather", "Plaid", "Sparse", "Dense", "Seismic"],
)
def test_search_callback(
    tmp_path: Path,
    bi_encoder_module: BiEncoderModule,
    query_datamodule: LightningIRDataModule,
    doc_datamodule: LightningIRDataModule,
    search_config: SearchConfig,
):
    if bi_encoder_module.config.model_type not in search_config.SUPPORTED_MODELS:
        pytest.skip(
            f"Searching not supported for {bi_encoder_module.config.__class__.__name__} model and "
            f"{search_config.__class__.__name__} searcher"
        )

    if "Seismic" in search_config.__class__.__name__:
        if importlib.util.find_spec("seismic") is None:
            pytest.skip("seismic package is not available")

    index_dir = get_index(bi_encoder_module, doc_datamodule, search_config)
    save_dir = tmp_path / "runs"
    search_callback = SearchCallback(search_config=search_config, index_dir=index_dir, save_dir=save_dir, use_gpu=False)

    trainer = LightningIRTrainer(
        logger=False,
        enable_checkpointing=False,
        callbacks=[search_callback],
        inference_mode=False,
    )
    trainer.test(bi_encoder_module, datamodule=query_datamodule)

    for dataloader in trainer.test_dataloaders:
        dataset = dataloader.dataset
        dataset_id = dataset.dataset_id.replace("/", "-")
        assert (save_dir / f"{dataset_id}.run").exists()
        run_df = pd.read_csv(
            save_dir / f"{dataset_id}.run",
            sep="\t",
            header=None,
            names=["query_id", "Q0", "doc_id", "rank", "score", "system"],
        )
        assert run_df["query_id"].nunique() == len(dataset)


def test_rerank_callback(tmp_path: Path, module: LightningIRModule, inference_datasets: Sequence[RunDataset]):
    datamodule = run_datamodule(module, inference_datasets)
    save_dir = tmp_path / "runs"
    rerank_callback = ReRankCallback(save_dir)
    trainer = LightningIRTrainer(logger=False, enable_checkpointing=False, callbacks=[rerank_callback])
    trainer.re_rank(module, datamodule)

    for dataloader in trainer.test_dataloaders:
        dataset = dataloader.dataset
        dataset_id = dataset.dataset_id.replace("/", "-")
        assert (save_dir / f"{dataset_id}.run").exists()
        run_df = pd.read_csv(
            save_dir / f"{dataset_id}.run",
            sep="\t",
            header=None,
            names=["query_id", "Q0", "doc_id", "rank", "score", "system"],
        )
        assert run_df["query_id"].nunique() == len(dataset)


def test_register_local_dataset_callback(model_name_or_path: str):
    callback = RegisterLocalDatasetCallback(
        dataset_id="test",
        docs=str(CORPUS_DIR / "docs.tsv"),
        queries=str(CORPUS_DIR / "queries.tsv"),
        qrels=str(CORPUS_DIR / "qrels.tsv"),
        docpairs=str(CORPUS_DIR / "docpairs.tsv"),
    )
    module = LightningIRModule(model_name_or_path=model_name_or_path, config=DprConfig(embedding_dim=4))
    datamodule = LightningIRDataModule(train_dataset=RunDataset("test"), train_batch_size=2)

    trainer = LightningIRTrainer(logger=False, enable_checkpointing=False, callbacks=[callback])

    trainer.test(module, datamodule)

    assert ir_datasets.registry._registered.get("test") is not None


class _StubTrainer:
    """Minimal stand-in for the parts of ``Trainer`` the export callback touches.

    A real ``fit`` would need a downloaded backbone; the callback only reads ``global_step``,
    ``callback_metrics``, ``log_dir`` and the two flags, so a stub keeps this test offline.
    """

    def __init__(self, log_dir: Path):
        self.log_dir = str(log_dir)
        self.global_step = 0
        self.sanity_checking = False
        self.is_global_zero = True
        self.callback_metrics: dict[str, float] = {}
        self.callbacks: list = []


class _StubModule:
    def __init__(self):
        self.config = SimpleNamespace(save_step=None)
        self.saved_steps: list[int] = []

    def save_pretrained(self, save_path: str | Path) -> None:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "config.json").write_text(json.dumps({"save_step": self.config.save_step}))
        self.saved_steps.append(self.config.save_step)


def _validate(callback: HuggingFaceExportCallback, trainer: _StubTrainer, module: _StubModule, step: int, **metrics):
    trainer.global_step = step
    trainer.callback_metrics.update(metrics)
    callback.on_validation_end(trainer, module)


def test_hugging_face_export_callback_selects_best(tmp_path: Path):
    trainer = _StubTrainer(tmp_path)
    module = _StubModule()
    callback = HuggingFaceExportCallback(dirpath="hf_best", monitor="val_mean_nDCG@10", mode="max")

    _validate(callback, trainer, module, 1000, **{"val_mean_nDCG@10": 0.30})
    _validate(callback, trainer, module, 2000, **{"val_mean_nDCG@10": 0.50})
    _validate(callback, trainer, module, 3000, **{"val_mean_nDCG@10": 0.40})

    # relative dirpath resolves against log_dir, and only improvements are exported
    export_dir = tmp_path / "hf_best"
    assert callback.best_step == 2000
    assert callback.best_model_score == 0.50
    assert callback.num_exports == 2
    assert module.saved_steps == [1000, 2000]

    info = json.loads((export_dir / "export_info.json").read_text())
    assert info == {
        "monitor": "val_mean_nDCG@10",
        "mode": "max",
        "score": 0.50,
        "global_step": 2000,
        "selection": "best",
    }
    # the exported config carries the step it was exported at, as the built-in export does
    assert json.loads((export_dir / "config.json").read_text())["save_step"] == 2000
    # but the in-memory value is left alone: it reports the step of the shared
    # huggingface_checkpoint/ export, which only a .ckpt write updates
    assert module.config.save_step is None

    # a further, non-improving validation must not overwrite the export
    trainer.global_step = 4000
    callback.on_fit_end(trainer, module)
    assert callback.best_step == 2000
    assert module.saved_steps == [1000, 2000]


def test_hugging_face_export_callback_min_mode_and_sanity_check(tmp_path: Path):
    trainer = _StubTrainer(tmp_path)
    module = _StubModule()
    callback = HuggingFaceExportCallback(dirpath=tmp_path / "hf_min", monitor="loss", mode="min")

    trainer.sanity_checking = True
    _validate(callback, trainer, module, 0, loss=99.0)
    assert callback.num_exports == 0

    trainer.sanity_checking = False
    _validate(callback, trainer, module, 500, loss=2.0)
    _validate(callback, trainer, module, 1000, loss=3.0)
    _validate(callback, trainer, module, 1500, loss=1.0)
    assert callback.best_step == 1500
    assert callback.best_model_score == 1.0
    assert module.saved_steps == [500, 1500]


def test_hugging_face_export_callback_final_export(tmp_path: Path):
    trainer = _StubTrainer(tmp_path)
    module = _StubModule()
    callback = HuggingFaceExportCallback(dirpath="hf_final")

    # no monitor: validations export nothing, on_fit_end exports the final weights
    _validate(callback, trainer, module, 1000, **{"val_mean_nDCG@10": 0.9})
    assert callback.num_exports == 0

    trainer.global_step = 50100
    callback.on_fit_end(trainer, module)
    assert callback.num_exports == 1
    assert callback.best_step == 50100
    info = json.loads((tmp_path / "hf_final" / "export_info.json").read_text())
    assert info["selection"] == "final" and info["monitor"] is None and info["global_step"] == 50100


def test_hugging_face_export_callback_missing_monitor(tmp_path: Path):
    trainer = _StubTrainer(tmp_path)
    module = _StubModule()
    callback = HuggingFaceExportCallback(dirpath="hf_best", monitor="val_mean_nDCG@10")

    # failing loudly on the first validation beats discovering after 50 hours that the run
    # exported nothing at all
    with pytest.raises(ValueError, match="val_mean_nDCG@10"):
        _validate(callback, trainer, module, 1000, other_metric=0.5)


def test_hugging_face_export_callback_bad_mode():
    with pytest.raises(ValueError, match="mode must be"):
        HuggingFaceExportCallback(dirpath="hf", monitor="m", mode="maximum")


def test_mean_validation_metric_callback_finds_export_callback(tmp_path: Path):
    trainer = _StubTrainer(tmp_path)
    mean_callback = MeanValidationMetricCallback(name="val_mean_robust_nDCG@10")
    export_callback = HuggingFaceExportCallback(dirpath="hf_best_robust", monitor="val_mean_robust_nDCG@10")
    other_export = HuggingFaceExportCallback(dirpath="hf_final")
    trainer.callbacks = [mean_callback, export_callback, other_export]

    # the end-of-run summary must not claim "nothing was selected" when an export callback is
    # the thing selecting on the metric
    assert mean_callback._monitoring_export_callbacks(trainer) == [export_callback]
    assert mean_callback._monitoring_checkpoint_callback(trainer) is None
