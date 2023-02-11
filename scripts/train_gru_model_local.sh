echo "Using mlflow tracking url $MLFLOW_TRACKING_URI"
echo "Using python path $PYTHONPATH"

python bin/run_train_gru.py config/train_local.yaml model_artifacts/saved_gru_model_local datasets/generated/full/annotations/full_annotations.csv
