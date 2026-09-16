terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.100"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }
}

provider "azurerm" {
  features {
    resource_group {
      prevent_deletion_if_contains_resources = false
    }
    key_vault {
      purge_soft_delete_on_destroy    = true
      recover_soft_deleted_key_vaults = true
    }
  }
}

data "azurerm_client_config" "current" {}

resource "random_string" "suffix" {
  length  = 6
  special = false
  upper   = false
}

# 1. Resource Group
resource "azurerm_resource_group" "rg" {
  name     = var.resource_group_name
  location = var.location
  tags = {
    Environment = var.environment
    ManagedBy   = "Terraform"
    Project     = "ApexInspect-AI"
  }
}

# 2. Azure Container Registry (ACR)
resource "azurerm_container_registry" "acr" {
  name                = coalesce(var.acr_name, "crportfolio${random_string.suffix.result}")
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "Basic"
  admin_enabled       = true
  tags                = azurerm_resource_group.rg.tags
}

# 3. Log Analytics Workspace (Required for Container Apps Monitoring)
resource "azurerm_log_analytics_workspace" "logs" {
  name                = "${var.prefix}-law"
  resource_group_name = azurerm_resource_group.rg.name
  location            = azurerm_resource_group.rg.location
  sku                 = "PerGB2018"
  retention_in_days   = 30
  tags                = azurerm_resource_group.rg.tags
}

# 4. Azure Container Apps Managed Environment
resource "azurerm_container_app_environment" "env" {
  name                       = "${var.prefix}-cae-env"
  resource_group_name        = azurerm_resource_group.rg.name
  location                   = azurerm_resource_group.rg.location
  log_analytics_workspace_id = azurerm_log_analytics_workspace.logs.id
  tags                       = azurerm_resource_group.rg.tags
}

# 5. Azure Key Vault (DevSecOps - Centralized Secrets Management)
resource "azurerm_key_vault" "kv" {
  name                        = coalesce(var.key_vault_name, "kv-${var.prefix}-${random_string.suffix.result}")
  resource_group_name         = azurerm_resource_group.rg.name
  location                    = azurerm_resource_group.rg.location
  tenant_id                   = data.azurerm_client_config.current.tenant_id
  sku_name                    = "standard"
  soft_delete_retention_days  = 7
  purge_protection_enabled    = false
  enable_rbac_authorization   = true
  tags                        = azurerm_resource_group.rg.tags
}

# 6. Storage Account (Blob storage for real PCB captures and inspection logs)
resource "azurerm_storage_account" "storage" {
  name                     = coalesce(var.storage_account_name, "st${var.prefix}${random_string.suffix.result}")
  resource_group_name      = azurerm_resource_group.rg.name
  location                 = azurerm_resource_group.rg.location
  account_tier             = "Standard"
  account_replication_type = "LRS"
  tags                     = azurerm_resource_group.rg.tags
}

resource "azurerm_storage_container" "defect_images" {
  name                  = "defect-images"
  storage_account_name  = azurerm_storage_account.storage.name
  container_access_type = "private"
}

# 7. Azure Container App (ApexInspect AI - Single Container Serverless Deployment)
resource "azurerm_container_app" "app" {
  name                         = "apexinspect-ai"
  container_app_environment_id = azurerm_container_app_environment.env.id
  resource_group_name          = azurerm_resource_group.rg.name
  revision_mode                = "Single"
  tags                         = azurerm_resource_group.rg.tags

  registry {
    server               = azurerm_container_registry.acr.login_server
    username             = azurerm_container_registry.acr.admin_username
    password_secret_name = "acr-password"
  }

  secret {
    name  = "acr-password"
    value = azurerm_container_registry.acr.admin_password
  }

  secret {
    name  = "groq-api-key"
    value = var.groq_api_key != "" ? var.groq_api_key : "placeholder_key"
  }

  template {
    # Scale-to-zero: 0 instances when idle = $0 compute cost
    min_replicas = 0
    max_replicas = 1

    container {
      name   = "apexinspect-ai"
      image  = "${azurerm_container_registry.acr.login_server}/apexinspect-ai:${var.image_tag}"
      cpu    = 0.5
      memory = "1.0Gi"

      env {
        name  = "STREAMLIT_SERVER_PORT"
        value = "7860"
      }
      env {
        name  = "STREAMLIT_SERVER_ADDRESS"
        value = "0.0.0.0"
      }
      env {
        name        = "GROQ_API_KEY"
        secret_name = "groq-api-key"
      }
    }
  }

  ingress {
    external_enabled = true
    target_port      = 7860
    traffic_weight {
      percentage      = 100
      latest_revision = true
    }
  }
}
