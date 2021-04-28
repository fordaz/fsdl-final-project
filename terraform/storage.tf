resource "aws_s3_bucket" "dl_experiments_bucket" {
  bucket = "fordaz-fsdl-2021-final-project"
  acl    = "private"

  tags = {
    Name        = "fsdl-2021"
    Environment = "dev"
  }
}