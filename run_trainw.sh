export MLFLOW_TRACKING_URI="postgresql://mlflowDbUser:demoable.secret.001@terraform-20210415031713058400000001.c7anop1qkfor.us-west-2.rds.amazonaws.com/mlflowDbStore"

export PYTHONPATH=/home/ec2-user/workspace/fsdl-final-project

conda activate pytorch_p36

python bin/run_train.py