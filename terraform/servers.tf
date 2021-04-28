resource "aws_instance" "dl_experiment" {
  ami             = var.ami_id
  instance_type   = var.instance_size
  key_name        = var.ami_key_pair_name
  security_groups = [aws_security_group.dl_experments_ingress_all.id]
  cpu_core_count = var.cpu_count
  cpu_threads_per_core = var.threads_per_core
  tags = {
    Name = var.ami_name
  }

  subnet_id = aws_subnet.subnet_dl_a.id
}
