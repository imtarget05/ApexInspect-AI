variable "resource_group_name" {
  type        = string
  default     = "rg-portfolio-prod"
  description = "Name of the Azure Resource Group"
}

variable "location" {
  type        = string
  default     = "southeastasia"
  description = "Azure region for deployment"
}

variable "environment" {
  type        = string
  default     = "production"
  description = "Environment tier (production, staging, dev)"
}

variable "prefix" {
  type        = string
  default     = "portfolio"
  description = "Resource prefix for naming"
}

variable "acr_name" {
  type        = string
  default     = ""
  description = "Optional custom ACR name (auto-generated if empty)"
}

variable "key_vault_name" {
  type        = string
  default     = ""
  description = "Optional custom Key Vault name (auto-generated if empty)"
}

variable "storage_account_name" {
  type        = string
  default     = ""
  description = "Optional custom Storage Account name (auto-generated if empty)"
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "Docker image tag to deploy"
}

variable "groq_api_key" {
  type        = string
  default     = ""
  sensitive   = true
  description = "Groq API Key for the LangGraph Agent"
}
