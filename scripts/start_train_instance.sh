aws ec2 start-instances --instance-ids $(cat ./infra_state/training_instance_id.txt)

aws ec2 describe-instances --query 'Reservations[*].Instances[*].{Instance:PublicDnsName}' --output text > ./infra_state/training_instance_host.txt

./scripts/update_ansible_inventory.sh
