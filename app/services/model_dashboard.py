from typing import Dict, List, Optional
import asyncio
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from prometheus_client import CollectorRegistry, Counter, Histogram, Gauge
import torch
from app.monitoring.metrics import MODEL_PERFORMANCE_METRICS

logger = logging.getLogger(__name__)

@dataclass
class ModelVersion:
    version_id: str
    model_name: str
    created_at: datetime
    metrics: Dict[str, float]
    size_mb: float
    deployment_status: str
    traffic_percentage: float
    performance_stats: Dict[str, float]

class ModelDashboard:
    def __init__(self):
        self.registry = CollectorRegistry()
        self.metrics = self._setup_metrics()
        self.active_versions: Dict[str, ModelVersion] = {}
        self.deployment_history: List[Dict] = []
        
    def _setup_metrics(self) -> Dict:
        """Initialize dashboard metrics"""
        return {
            "version_performance": Histogram(
                "model_version_performance",
                "Performance metrics per model version",
                ["model_name", "version", "metric"]
            ),
            "deployment_status": Gauge(
                "model_deployment_status",
                "Deployment status of model versions",
                ["model_name", "version"]
            ),
            "traffic_distribution": Gauge(
                "model_traffic_distribution",
                "Traffic distribution across versions",
                ["model_name", "version"]
            ),
            "error_rate": Counter(
                "model_error_rate",
                "Error rate per model version",
                ["model_name", "version", "error_type"]
            )
        }

    async def register_model_version(
        self,
        model_name: str,
        version_id: str,
        model: torch.nn.Module
    ) -> ModelVersion:
        """Register a new model version for monitoring"""
        try:
            # Calculate model size
            size_mb = sum(
                p.nelement() * p.element_size() for p in model.parameters()
            ) / (1024 * 1024)
            
            version = ModelVersion(
                version_id=version_id,
                model_name=model_name,
                created_at=datetime.now(),
                metrics={},
                size_mb=size_mb,
                deployment_status="registered",
                traffic_percentage=0.0,
                performance_stats={}
            )
            
            self.active_versions[version_id] = version
            
            # Initialize metrics
            self.metrics["deployment_status"].labels(
                model_name=model_name,
                version=version_id
            ).set(1)
            
            return version

        except Exception as e:
            logger.error(f"Failed to register model version: {e}")
            raise

    async def update_metrics(
        self,
        version_id: str,
        metrics: Dict[str, float]
    ):
        """Update performance metrics for a model version"""
        try:
            version = self.active_versions[version_id]
            version.metrics.update(metrics)
            
            for metric_name, value in metrics.items():
                self.metrics["version_performance"].labels(
                    model_name=version.model_name,
                    version=version_id,
                    metric=metric_name
                ).observe(value)

        except Exception as e:
            logger.error(f"Failed to update metrics: {e}")
            raise

    async def manage_deployment(
        self,
        version_id: str,
        traffic_percentage: float,
        rollback_threshold: Optional[Dict[str, float]] = None
    ):
        """Manage canary deployment of a model version"""
        try:
            version = self.active_versions[version_id]
            previous_version = self._get_previous_version(version.model_name)
            
            # Update traffic distribution
            version.traffic_percentage = traffic_percentage
            if previous_version:
                previous_version.traffic_percentage = 100 - traffic_percentage
            
            self.metrics["traffic_distribution"].labels(
                model_name=version.model_name,
                version=version_id
            ).set(traffic_percentage)
            
            # Monitor performance for rollback
            if rollback_threshold:
                await self._monitor_rollback_criteria(
                    version,
                    rollback_threshold
                )
            
            # Update deployment history
            self.deployment_history.append({
                "timestamp": datetime.now(),
                "version_id": version_id,
                "traffic_percentage": traffic_percentage,
                "metrics": version.metrics.copy()
            })

        except Exception as e:
            logger.error(f"Deployment management failed: {e}")
            raise

    async def _monitor_rollback_criteria(
        self,
        version: ModelVersion,
        thresholds: Dict[str, float]
    ):
        """Monitor performance metrics for automatic rollback"""
        try:
            for metric_name, threshold in thresholds.items():
                current_value = version.metrics.get(metric_name)
                if current_value and current_value < threshold:
                    logger.warning(
                        f"Model version {version.version_id} performing below "
                        f"threshold for {metric_name}: {current_value} < {threshold}"
                    )
                    await self._initiate_rollback(version)
                    break

        except Exception as e:
            logger.error(f"Rollback monitoring failed: {e}")
            raise

    async def _initiate_rollback(self, version: ModelVersion):
        """Initiate rollback to previous stable version"""
        try:
            previous_version = self._get_previous_version(version.model_name)
            if previous_version:
                # Restore traffic to previous version
                await self.manage_deployment(
                    previous_version.version_id,
                    100.0
                )
                version.deployment_status = "rolled_back"
                logger.info(
                    f"Rolled back from {version.version_id} to "
                    f"{previous_version.version_id}"
                )

        except Exception as e:
            logger.error(f"Rollback failed: {e}")
            raise

    def _get_previous_version(self, model_name: str) -> Optional[ModelVersion]:
        """Get the previous stable version of a model"""
        stable_versions = [
            v for v in self.active_versions.values()
            if v.model_name == model_name and
            v.deployment_status == "stable"
        ]
        return max(
            stable_versions,
            key=lambda x: x.created_at
        ) if stable_versions else None

    async def get_dashboard_data(
        self,
        model_name: Optional[str] = None,
        time_range: Optional[timedelta] = None
    ) -> Dict:
        """Get dashboard data for visualization"""
        try:
            if time_range is None:
                time_range = timedelta(days=7)

            cutoff_time = datetime.now() - time_range
            
            # Filter relevant data
            versions = [
                v for v in self.active_versions.values()
                if (not model_name or v.model_name == model_name) and
                v.created_at >= cutoff_time
            ]
            
            # Compile dashboard data
            return {
                "versions": [
                    {
                        "version_id": v.version_id,
                        "model_name": v.model_name,
                        "created_at": v.created_at.isoformat(),
                        "metrics": v.metrics,
                        "size_mb": v.size_mb,
                        "status": v.deployment_status,
                        "traffic": v.traffic_percentage
                    }
                    for v in versions
                ],
                "deployment_history": [
                    h for h in self.deployment_history
                    if h["timestamp"] >= cutoff_time
                ],
                "performance_summary": await self._generate_performance_summary(
                    versions
                )
            }

        except Exception as e:
            logger.error(f"Failed to generate dashboard data: {e}")
            raise

    async def _generate_performance_summary(
        self,
        versions: List[ModelVersion]
    ) -> Dict:
        """Generate performance summary statistics"""
        try:
            summary = {}
            for version in versions:
                if version.model_name not in summary:
                    summary[version.model_name] = {
                        "total_versions": 0,
                        "active_versions": 0,
                        "avg_performance": {},
                        "best_version": None,
                        "latest_version": None
                    }
                
                model_summary = summary[version.model_name]
                model_summary["total_versions"] += 1
                
                if version.deployment_status != "rolled_back":
                    model_summary["active_versions"] += 1
                
                # Update average performance
                for metric, value in version.metrics.items():
                    if metric not in model_summary["avg_performance"]:
                        model_summary["avg_performance"][metric] = []
                    model_summary["avg_performance"][metric].append(value)
                
                # Track best and latest versions
                if (not model_summary["best_version"] or
                    version.metrics.get("accuracy", 0) >
                    model_summary["best_version"].metrics.get("accuracy", 0)):
                    model_summary["best_version"] = version
                
                if (not model_summary["latest_version"] or
                    version.created_at >
                    model_summary["latest_version"].created_at):
                    model_summary["latest_version"] = version
            
            # Calculate averages
            for model_name in summary:
                avg_perf = summary[model_name]["avg_performance"]
                summary[model_name]["avg_performance"] = {
                    metric: sum(values) / len(values)
                    for metric, values in avg_perf.items()
                }
            
            return summary

        except Exception as e:
            logger.error(f"Failed to generate performance summary: {e}")
            raise 