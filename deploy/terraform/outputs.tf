output "resource_group_name" {
  value       = azurerm_resource_group.rg.name
  description = "Created Resource Group"
}

output "acr_login_server" {
  value       = azurerm_container_registry.acr.login_server
  description = "Azure Container Registry Login Server"
}

output "key_vault_uri" {
  value       = azurerm_key_vault.kv.vault_uri
  description = "Azure Key Vault URI"
}

output "storage_account_name" {
  value       = azurerm_storage_account.storage.name
  description = "Storage Account Name"
}

output "container_app_fqdn" {
  value       = "https://${azurerm_container_app.app.ingress[0].fqdn}"
  description = "Public URL of ApexInspect AI"
}
