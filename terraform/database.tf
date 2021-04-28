resource "aws_db_instance" "default" {
  allocated_storage      = 20
  engine                 = "postgres"
  instance_class         = "db.t3.micro"
  name                   = "mlflowDbStore"
  username               = "mlflowDbUser"
  password               = trimspace(file("${path.module}/secrets/mydb1-password.txt"))
  skip_final_snapshot    = true
  publicly_accessible    = true
  db_subnet_group_name   = aws_db_subnet_group.db_subnet_group_dl_a.name
  vpc_security_group_ids = [aws_security_group.dl_experiment_db_sg.id]
}