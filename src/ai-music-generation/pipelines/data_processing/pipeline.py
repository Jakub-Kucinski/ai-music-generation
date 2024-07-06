from kedro.pipeline import Pipeline, node, pipeline


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=lambda x: x,
                inputs="test_kedro_run_data",
                outputs="processed_data",
                name="data_processing",
            )
        ],
        inputs=None,
        outputs=None,
        parameters=None,
        namespace=None,
        tags=None,
    )
