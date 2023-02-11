host_name=$(cat ./infra_state/training_instance_host.txt)

ssh -i ~/.ssh/deep-learning-kp.pem ec2-user@$host_name
