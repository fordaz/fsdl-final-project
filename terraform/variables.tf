variable "ami_name" {
  default = "dl_ami"
}

variable "instance_size" {
  default = "p2.xlarge"
}

variable "ami_id" {
  default = "ami-01fe82fb0c26ae580"
}

variable "ami_key_pair_name" {
  default = "deep-learning-kp"
}

variable "cpu_count" {
  default = 1
}

variable "threads_per_core" {
  default = 1
}