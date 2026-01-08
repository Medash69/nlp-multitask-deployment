#!/usr/bin/env bash
set -euo pipefail

RESOURCE_GROUP="rg-nlp-deployment"
LOCATION="swedencentral"
ACR_NAME="acrnubhav1767816776"
IMAGE_NAME="nlp-api"
IMAGE_TAG="latest"

echo "🔍 Vérification du registre..."
ACR_SERVER=$(az acr show -n "$ACR_NAME" --query loginServer -o tsv | tr -d '\r')

echo "🔑 Connexion au registre $ACR_SERVER..."
az acr login -n "$ACR_NAME"

echo "🏗️  Build de l'image..."
docker build -t "$IMAGE_NAME" .

echo "🏷️  Tag de l'image..."
docker tag "$IMAGE_NAME" "$ACR_SERVER/$IMAGE_NAME:$IMAGE_TAG"

echo "📤 Push vers ACR (Patientez...)"
docker push "$ACR_SERVER/$IMAGE_NAME:$IMAGE_TAG"

echo "🚀 Déploiement sur Azure Container Apps (2.0 CPU / 4.0Gi)..."
ACR_USER=$(az acr credential show -n "$ACR_NAME" --query username -o tsv | tr -d '\r')
ACR_PASS=$(az acr credential show -n "$ACR_NAME" --query "passwords[0].value" -o tsv | tr -d '\r')

az containerapp create \
  --name nlp-service \
  --resource-group "$RESOURCE_GROUP" \
  --environment env-nlp \
  --image "$ACR_SERVER/$IMAGE_NAME:$IMAGE_TAG" \
  --target-port 8000 \
  --ingress external \
  --registry-server "$ACR_SERVER" \
  --registry-username "$ACR_USER" \
  --registry-password "$ACR_PASS" \
  --cpu 2.0 --memory 4.0Gi

APP_URL=$(az containerapp show -n nlp-service -g "$RESOURCE_GROUP" --query properties.configuration.ingress.fqdn -o tsv | tr -d '\r')

echo "=========================================="
echo "✅ DÉPLOIEMENT NLP RÉUSSI"
echo "URL : https://$APP_URL"
echo "=========================================="