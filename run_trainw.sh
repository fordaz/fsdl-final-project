echo $MLFLOW_TRACKING_URI

export PYTHONPATH=/home/ec2-user/workspace/fsdl-final-project

echo `id`

echo $PYTHONPATH

source /home/ec2-user/anaconda3/etc/profile.d/conda.sh

conda activate pytorch_p36

python bin/run_train.py