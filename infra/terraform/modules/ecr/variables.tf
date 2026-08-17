variable "name_prefix" {
  type        = string
  description = "Prefixo de naming dos recursos."
}

variable "repo_suffix" {
  type        = string
  description = "Sufixo do nome do repositorio ECR (ex.: api, grafana)."
  default     = "api"
}
