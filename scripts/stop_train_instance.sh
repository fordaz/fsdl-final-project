aws ec2 stop-instances --instance-ids $(cat ./infra_state/training_instance_id.txt)
