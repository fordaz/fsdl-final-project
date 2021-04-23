echo "Using mlflow tracking url $MLFLOW_TRACKING_URI"
echo "Using python path $PYTHONPATH"

# setting up conda
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh

# activating the aws instance conda environment ready for DL
conda activate pytorch_latest_p37

# installing the application dependencies
pip install -r requirements.txt

# pulling the training datasets
dvc pull

# launching the training
python bin/run_train.py