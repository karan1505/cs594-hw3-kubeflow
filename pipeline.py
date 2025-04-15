from kfp import dsl
from kfp.components import load_component_from_file

load_op = load_component_from_file("pipeline_components/load_and_clean.yaml")
feature_op = load_component_from_file("pipeline_components/feature_engineer.yaml")
train_op = load_component_from_file("pipeline_components/train_model.yaml")
eval_op = load_component_from_file("pipeline_components/evaluate_model.yaml")

@dsl.pipeline(name="IMDB ML Pipeline", description="Pipeline using YAML component defs")
def imdb_pipeline():
    load_task = load_op(
        input_csv_path="data/imdb_top_1000.csv",
        output_csv_path="data/imdb_cleaned.csv"
    )

    feat_task = feature_op(
        input_csv_path="data/imdb_cleaned.csv",
        output_features_path="data/features.csv",
        output_labels_path="data/labels.csv"
    )

    train_task = train_op(
        features_csv="data/features.csv",
        labels_csv="data/labels.csv",
        model_output_path="model/rf_model.pkl"
    )

    eval_task = eval_op(
        features_csv="data/features.csv",
        labels_csv="data/labels.csv",
        model_path="model/rf_model.pkl"
    )
