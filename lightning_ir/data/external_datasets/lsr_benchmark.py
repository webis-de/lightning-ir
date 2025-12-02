def register_lsr_dataset_to_ir_datasets(dataset):
    try:
        from lsr_benchmark import register_to_ir_datasets
    except ImportError:
        msg = (
            "I could not import the lsr_benchmark package. Please install the lsr-benchmark via "
            + f"'pip3 install lsr-benchmark' to load ir_datasets from there. I got the dataset id '{dataset}'."
        )
        print(msg)
        raise ValueError(msg)

    register_to_ir_datasets(dataset.replace("lsr-benchmark/", ""))
