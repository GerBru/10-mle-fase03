# Imagem do Grafana com datasource e dashboards provisionados.
# Usada no ECS/Fargate (AWS), onde nao ha volume para montar o provisioning.
# No ambiente local o docker-compose continua montando ./monitoring/grafana/provisioning.
FROM grafana/grafana:latest

# Copia datasource (Prometheus) + provider de dashboards + JSON do dashboard.
COPY provisioning/ /etc/grafana/provisioning/
