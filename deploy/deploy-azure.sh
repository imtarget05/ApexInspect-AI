#!/usr/bin/env bash
# ==============================================================================
# ApexInspect AI — Automated Deployment Script for Azure Container Apps (ACA)
# Cost Optimization: Configured with min-replicas=0 (Scale-to-Zero = 0 USD idle)
# ==============================================================================

set -euo pipefail

RESOURCE_GROUP="${RESOURCE_GROUP:-rg-portfolio-prod}"
LOCATION="${LOCATION:-southeastasia}"
ACR_NAME="${ACR_NAME:-}"
CONTAINERAPPS_ENVIRONMENT="${CONTAINERAPPS_ENVIRONMENT:-cae-portfolio-env}"
APP_NAME="apexinspect-ai"
TARGET_PORT=7860
IMAGE_TAG="latest"

echo "=========================================================="
echo "🚀 Deploying ApexInspect AI to Azure Container Apps"
echo "Resource Group: $RESOURCE_GROUP | Location: $LOCATION"
echo "=========================================================="

if ! command -v az &> /dev/null; then
    echo "❌ Error: Azure CLI (az) is not installed."
    echo "💡 Install via Homebrew: brew install azure-cli"
    exit 1
fi

echo "🔍 Verifying Azure login status..."
az account show --output none || {
    echo "⚠️ Not logged in. Prompting for Azure login..."
    az login
}

echo "📦 Ensuring Resource Group [$RESOURCE_GROUP] exists..."
az group create --name "$RESOURCE_GROUP" --location "$LOCATION" --output table

if [ -z "$ACR_NAME" ]; then
    SUBSCRIPTION_ID=$(az account show --query id -o tsv | tr -d '-' | cut -c1-6)
    ACR_NAME="crportfolio${SUBSCRIPTION_ID}"
fi

echo "🏭 Ensuring Azure Container Registry [$ACR_NAME] exists..."
if ! az acr show --name "$ACR_NAME" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    az acr create \
        --name "$ACR_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --sku Basic \
        --admin-enabled true \
        --location "$LOCATION" \
        --output table
fi

echo "🔨 Building and pushing image [$APP_NAME:$IMAGE_TAG] using ACR cloud build..."
az acr build \
    --registry "$ACR_NAME" \
    --image "${APP_NAME}:${IMAGE_TAG}" \
    . \
    --output table

ACR_LOGIN_SERVER=$(az acr show --name "$ACR_NAME" --query loginServer -o tsv)
ACR_ADMIN_PASSWORD=$(az acr credential show --name "$ACR_NAME" --query "passwords[0].value" -o tsv)
ACR_ADMIN_USERNAME=$(az acr credential show --name "$ACR_NAME" --query username -o tsv)

echo "🌐 Ensuring Container Apps Environment [$CONTAINERAPPS_ENVIRONMENT] exists..."
if ! az containerapp env show --name "$CONTAINERAPPS_ENVIRONMENT" --resource-group "$RESOURCE_GROUP" &> /dev/null; then
    az containerapp env create \
        --name "$CONTAINERAPPS_ENVIRONMENT" \
        --resource-group "$RESOURCE_GROUP" \
        --location "$LOCATION" \
        --output table
fi

echo "🚀 Deploying Container App [$APP_NAME] with Scale-to-Zero (min-replicas=0)..."

az containerapp create \
    --name "$APP_NAME" \
    --resource-group "$RESOURCE_GROUP" \
    --environment "$CONTAINERAPPS_ENVIRONMENT" \
    --image "${ACR_LOGIN_SERVER}/${APP_NAME}:${IMAGE_TAG}" \
    --registry-server "$ACR_LOGIN_SERVER" \
    --registry-username "$ACR_ADMIN_USERNAME" \
    --registry-password "$ACR_ADMIN_PASSWORD" \
    --target-port "$TARGET_PORT" \
    --ingress external \
    --min-replicas 0 \
    --max-replicas 1 \
    --cpu 0.5 \
    --memory 1.0Gi \
    --env-vars \
        "GROQ_API_KEY=${GROQ_API_KEY:-}" \
        "STREAMLIT_SERVER_PORT=7860" \
        "STREAMLIT_SERVER_ADDRESS=0.0.0.0" \
    --output table || {
    echo "⚠️ Updating existing container app..."
    az containerapp update \
        --name "$APP_NAME" \
        --resource-group "$RESOURCE_GROUP" \
        --image "${ACR_LOGIN_SERVER}/${APP_NAME}:${IMAGE_TAG}" \
        --output table
}

APP_URL=$(az containerapp show --name "$APP_NAME" --resource-group "$RESOURCE_GROUP" --query "properties.configuration.ingress.fqdn" -o tsv)

echo "=========================================================="
echo "✅ DEPLOYMENT SUCCESSFUL!"
echo "🔗 Public Live URL: https://${APP_URL}"
echo "=========================================================="
