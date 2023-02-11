
ts=$(date +%d-%m-%Y_%H-%M-%S)

echo "Using timestamp $ts"

cp /etc/ansible/hosts /etc/ansible/hosts.$ts

host_name=$(cat ./infra_state/training_instance_host.txt)

echo "$host_name       ansible_ssh_private_key_file=/Users/fordaz/.ssh/deep-learning-kp.pem    ansible_user=ec2-user" > /etc/ansible/hosts
