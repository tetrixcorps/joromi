import psutil
import GPUtil
import docker
from datetime import datetime
import logging
import asyncio

class ServiceMonitor:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.logger = logging.getLogger(__name__)

    def get_service_stats(self):
        """Get stats for all running services"""
        stats = {}
        for container in self.docker_client.containers.list():
            try:
                stats[container.name] = {
                    'cpu': container.stats(stream=False)['cpu_stats'],
                    'memory': container.stats(stream=False)['memory_stats'],
                    'status': container.status
                }
            except Exception as e:
                self.logger.error(f"Error getting stats for {container.name}: {e}")
        return stats

    def get_gpu_stats(self):
        """Get GPU statistics"""
        try:
            gpus = GPUtil.getGPUs()
            return [{
                'id': gpu.id,
                'load': gpu.load,
                'memory_used': gpu.memoryUsed,
                'memory_total': gpu.memoryTotal,
                'temperature': gpu.temperature
            } for gpu in gpus]
        except Exception as e:
            self.logger.error(f"Error getting GPU stats: {e}")
            return []

    def get_system_stats(self):
        """Get system-wide statistics"""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent
        }

    async def monitor_services(self):
        """Monitor all services and resources"""
        while True:
            try:
                timestamp = datetime.now().isoformat()
                stats = {
                    'timestamp': timestamp,
                    'services': self.get_service_stats(),
                    'gpus': self.get_gpu_stats(),
                    'system': self.get_system_stats()
                }
                self.logger.info(f"System Stats: {stats}")
                await asyncio.sleep(60)  # Monitor every minute
            except Exception as e:
                self.logger.error(f"Monitoring error: {e}")
                await asyncio.sleep(5) 