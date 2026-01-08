#!/usr/bin/env bash
set -euo pipefail

#################################
# VARIABLES
#################################
RESOURCE_GROUP="rg-nlp-deployment"
LOCATION="swedencentral"

ACR_NAME="acrnubhav1767816776"
IMAGE_NAME="nlp-api"
IMAGE_TAG="latest"

CONTAINER_ENV="env-nlp"
APP_NAME="nlp-service"

#################################
# LOGIN ACR
#################################
echo "🔐 Connexion à Azure Container Registry..."

ACR_SERVER=$(az acr show -n "$ACR_NAME" --query loginServer -o tsv | tr -d '\r')
echo "ACR_SERVER : $ACR_SERVER"

az acr login -n "$ACR_NAME"

#################################
# BUILD + TAG + PUSH IMAGE
#################################
echo "🐳 Build de l'image Docker..."
docker build -t "$IMAGE_NAME" .

echo "🏷️ Tag de l'image..."
docker tag "$IMAGE_NAME" "$ACR_SERVER/$IMAGE_NAME:$IMAGE_TAG"

echo "📤 Push vers ACR..."
docker push "$ACR_SERVER/$IMAGE_NAME:$IMAGE_TAG"

echo "✅ Image Docker envoyée avec succès"

#################################
# CONTAINER APP
#################################
echo "🚀 Déploiement sur Azure Container Apps..."

# Credentials ACR
ACR_USER=$(az acr credential show -n "$ACR_NAME" --query username -o tsv | tr -d '\r')
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv | tr -d '\r')

# Création de l'environnement (si déjà existant → OK)
az containerapp env create \
  --name "$CONTAINER_ENV" \
  --resource-group "$RESOURCE_GROUP" \
  --location "$LOCATION" || true

# Création de l'application (CPU/RAM VALIDES)
az containerapp create \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --environment "$CONTAINER_ENV" \
  --image "$ACR_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
  --target-port 8000 \
  --ingress external \
  --registry-server "$ACR_SERVER" \
  --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS" \
  --cpu 2.0 \
  --memory 4.0Gi

#################################
# URL PUBLIQUE
#################################
APP_URL=$(az containerapp show \
  --name "$APP_NAME" \
  --resource-group "$RESOURCE_GROUP" \
  --query properties.configuration.ingress.fqdn \
  -o tsv | tr -d '\r')

echo "=========================================="
echo "✅ DÉPLOIEMENT NLP RÉUSSI"
echo "🌍 URL : https://$APP_URL"
echo "=========================================="
