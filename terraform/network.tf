resource "aws_vpc" "dl_experiments_vpc" {
  cidr_block           = "10.0.0.0/16"
  enable_dns_hostnames = true
  enable_dns_support   = true
  tags = {
    Name = "dl_experiments_vpc"
  }
}

resource "aws_subnet" "subnet_dl_a" {
  cidr_block              = cidrsubnet(aws_vpc.dl_experiments_vpc.cidr_block, 3, 1)
  vpc_id                  = aws_vpc.dl_experiments_vpc.id
  map_public_ip_on_launch = true
  availability_zone       = "us-west-2a"
}

resource "aws_subnet" "subnet_dl_b" {
  cidr_block              = cidrsubnet(aws_vpc.dl_experiments_vpc.cidr_block, 5, 1)
  vpc_id                  = aws_vpc.dl_experiments_vpc.id
  map_public_ip_on_launch = true
  availability_zone       = "us-west-2b"
}

resource "aws_db_subnet_group" "db_subnet_group_dl_a" {
  name       = "main_db_subnet"
  subnet_ids = [aws_subnet.subnet_dl_a.id, aws_subnet.subnet_dl_b.id]

  tags = {
    Name = "The DB subnet group"
  }
}

resource "aws_internet_gateway" "dl_experiment_igw" {
  vpc_id = aws_vpc.dl_experiments_vpc.id
  tags = {
    Name = "dl_experiment_igw"
  }
}

resource "aws_route_table" "dl_experiment_rtable" {
  vpc_id = aws_vpc.dl_experiments_vpc.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.dl_experiment_igw.id
  }

  tags = {
    Name = "dl_experiment_rtable"
  }
}

resource "aws_route_table_association" "dl_subnet_assoc" {
  subnet_id      = aws_subnet.subnet_dl_a.id
  route_table_id = aws_route_table.dl_experiment_rtable.id
}

resource "aws_security_group" "dl_experments_ingress_all" {
  name = "allow_all_sg"

  vpc_id = aws_vpc.dl_experiments_vpc.id

  ingress {
    cidr_blocks = [
      "0.0.0.0/0"
    ]
    from_port = 22
    to_port   = 22
    protocol  = "tcp"
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "dl_experiment_db_sg" {
  name = "allow_db_access_sg"

  description = "RDS postgres servers (terraform-managed)"

  vpc_id = aws_vpc.dl_experiments_vpc.id

  # Only postgres in
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Allow all outbound traffic.
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
