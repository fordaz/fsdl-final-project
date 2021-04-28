echo "Using mlflow tracking url $MLFLOW_TRACKING_URI"
echo "Using python path $PYTHONPATH"

python bin/run_train.py config/train_local.yaml model_artifacts/saved_model_local datasets/generated/training/annotations/all_clean_annotations.json