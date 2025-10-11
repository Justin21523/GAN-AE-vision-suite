#!/bin/bash

# GAN-AE-VISION-SUITE Kubernetes Deployment Script

set -e

echo "🚀 Deploying GAN-AE-VISION-SUITE to Kubernetes"

# Configuration
NAMESPACE="gan-ae-vision"
APP_NAME="gan-ae-vision"
IMAGE_TAG="${1:-latest}"
KUBE_CONTEXT="${2:-$(kubectl config current-context)}"

echo "📋 Deployment Configuration:"
echo "  Namespace: $NAMESPACE"
echo "  App Name: $APP_NAME"
echo "  Image Tag: $IMAGE_TAG"
echo "  K8s Context: $KUBE_CONTEXT"

# Check if kubectl is available
if ! command -v kubectl &> /dev/null; then
    echo "❌ kubectl is not installed"
    exit 1
fi

# Set kubectl context
kubectl config use-context "$KUBE_CONTEXT"

# Create namespace if it doesn't exist
if ! kubectl get namespace "$NAMESPACE" &> /dev/null; then
    echo "📁 Creating namespace: $NAMESPACE"
    kubectl create namespace "$NAMESPACE"
fi

# Create AI Warehouse PVC
echo "💾 Creating AI Warehouse PVC..."
kubectl apply -f deploy/k8s/ai-warehouse-pvc.yaml -n "$NAMESPACE"

# Wait for PVC to be bound
echo "⏳ Waiting for PVC to be bound..."
kubectl wait --for=jsonpath='{.status.phase}'=Bound pvc/ai-warehouse-pvc -n "$NAMESPACE" --timeout=300s

# Build and push Docker image (if not using existing)
if [ "$IMAGE_TAG" = "latest" ]; then
    echo "🐳 Building Docker image..."
    docker build -t gan-ae-vision:latest -f deploy/Dockerfile .

    # If using a registry, push the image
    # docker tag gan-ae-vision:latest your-registry/gan-ae-vision:latest
    # docker push your-registry/gan-ae-vision:latest
fi

# Deploy application
echo "📦 Deploying application..."
kubectl apply -f deploy/k8s/gan-ae-vision.yaml -n "$NAMESPACE"

# Wait for deployments to be ready
echo "⏳ Waiting for deployments to be ready..."
kubectl wait --for=condition=available deployment/gan-ae-vision-api -n "$NAMESPACE" --timeout=600s
kubectl wait --for=condition=available deployment/gan-ae-vision-ui -n "$NAMESPACE" --timeout=600s

# Check pod status
echo "🔍 Checking pod status..."
kubectl get pods -n "$NAMESPACE" -l app=gan-ae-vision-api
kubectl get pods -n "$NAMESPACE" -l app=gan-ae-vision-ui

# Get service URLs
echo "🌐 Service URLs:"
kubectl get svc -n "$NAMESPACE" | grep gan-ae-vision

# If ingress is enabled, show ingress information
if kubectl get ingress gan-ae-vision-ingress -n "$NAMESPACE" &> /dev/null; then
    echo "🔗 Ingress Information:"
    kubectl get ingress gan-ae-vision-ingress -n "$NAMESPACE"
fi

echo "✅ Deployment completed successfully!"
echo ""
echo "📋 Next steps:"
echo "  1. Check application logs: kubectl logs -n $NAMESPACE deployment/gan-ae-vision-api"
echo "  2. Access the UI through the service or ingress"
echo "  3. Monitor resource usage: kubectl top pods -n $NAMESPACE"