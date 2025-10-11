#!/bin/bash

# GAN-AE-VISION-SUITE Monitoring Setup Script

set -e

echo "📊 Setting up monitoring for GAN-AE-VISION-SUITE"

# Create monitoring directory
MONITORING_DIR="monitoring"
mkdir -p "$MONITORING_DIR"

# Generate Prometheus configuration
cat > "$MONITORING_DIR/prometheus.yml" << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  # - "first_rules.yml"
  # - "second_rules.yml"

scrape_configs:
  - job_name: 'gan-ae-vision'
    static_configs:
      - targets: ['localhost:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
EOF

# Generate Grafana dashboard configuration
cat > "$MONITORING_DIR/dashboard.json" << EOF
{
  "dashboard": {
    "id": null,
    "title": "GAN-AE-VISION-SUITE Metrics",
    "tags": ["gan", "ai", "monitoring"],
    "timezone": "browser",
    "panels": [],
    "time": {
      "from": "now-6h",
      "to": "now"
    },
    "refresh": "30s"
  }
}
EOF

# Create Docker Compose for monitoring stack
cat > "$MONITORING_DIR/docker-compose.monitoring.yml" << EOF
version: '3.8'

services:
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--web.console.libraries=/etc/prometheus/console_libraries'
      - '--web.console.templates=/etc/prometheus/consoles'
      - '--storage.tsdb.retention.time=200h'
      - '--web.enable-lifecycle'
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    volumes:
      - grafana_data:/var/lib/grafana
      - ./dashboard.json:/etc/grafana/provisioning/dashboards/dashboard.json
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped

  node-exporter:
    image: prom/node-exporter:latest
    ports:
      - "9100:9100"
    restart: unless-stopped

volumes:
  prometheus_data:
  grafana_data:
EOF

echo "✅ Monitoring setup completed!"
echo ""
echo "📋 To start monitoring stack:"
echo "  cd $MONITORING_DIR && docker-compose -f docker-compose.monitoring.yml up -d"
echo ""
echo "🌐 Access URLs:"
echo "  - Prometheus: http://localhost:9090"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Node Exporter: http://localhost:9100"