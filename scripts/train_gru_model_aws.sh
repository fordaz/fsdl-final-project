echo "Using mlflow tracking url $MLFLOW_TRACKING_URI"
echo "Using python path $PYTHONPATH"

# setting up conda
source /home/ec2-user/anaconda3/etc/profile.d/conda.sh

# activating the aws instance conda environment ready for DL
conda activate pytorch_latest_p37

# installing the application dependencies
pip install -r requirements-aws.txt

# to avoid conflict trying to remove ruamel
pip install dvc --ignore-installed ruamel.yaml

# pulling the training datasets
dvc pull

# launching the training
python bin/run_train_gru.py $1 $2 $3