import torch
import psutil
import threading
import time
from typing import Dict, List
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger('gpu_monitor')

@dataclass
class GPUStats:
    memory_used: int
    memory_total: int
    utilization: float
    temperature: float
    power_usage: float

class GPUMonitor:
    def __init__(self, interval: int = 5):
        self.interval = interval
        self.stats_history: List[Dict] = []
        self.running = False
        self.monitor_thread = None

    def start(self):
        """Start GPU monitoring"""
        if not self.running:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()

    def stop(self):
        """Stop GPU monitoring"""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join()

    def _monitor_loop(self):
        """Main monitoring loop"""
        while self.running:
            try:
                stats = self.get_gpu_stats()
                self.stats_history.append(stats)
                
                # Keep last hour of stats (with 5s interval)
                max_history = 720  # 3600/5
                if len(self.stats_history) > max_history:
                    self.stats_history.pop(0)
                
                # Log warnings if necessary
                self._check_warnings(stats)
                
                time.sleep(self.interval)
                
            except Exception as e:
                logger.error(f"Error in GPU monitoring: {str(e)}")
                time.sleep(self.interval)

    def get_gpu_stats(self) -> Dict[str, GPUStats]:
        """Get current GPU statistics"""
        stats = {}
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                handle = torch.cuda.device(i)
                props = torch.cuda.get_device_properties(i)
                
                memory = torch.cuda.memory_stats(i)
                memory_used = memory.get('allocated_bytes.all.current', 0)
                memory_total = props.total_memory
                
                stats[f'gpu_{i}'] = GPUStats(
                    memory_used=memory_used,
                    memory_total=memory_total,
                    utilization=torch.cuda.utilization(i),
                    temperature=torch.cuda.temperature(i),
                    power_usage=torch.cuda.power_usage(i) if hasattr(torch.cuda, 'power_usage') else 0
                )
        
        return stats

    def _check_warnings(self, stats: Dict[str, GPUStats]):
        """Check for warning conditions"""
        for gpu_id, gpu_stats in stats.items():
            memory_usage_pct = (gpu_stats.memory_used / gpu_stats.memory_total) * 100
            
            if memory_usage_pct > 90:
                logger.warning(f"{gpu_id}: High memory usage ({memory_usage_pct:.1f}%)")
            
            if gpu_stats.temperature > 80:
                logger.warning(f"{gpu_id}: High temperature ({gpu_stats.temperature}°C)")
            
            if gpu_stats.utilization > 95:
                logger.warning(f"{gpu_id}: High GPU utilization ({gpu_stats.utilization}%)")

    def get_memory_usage(self) -> Dict[str, float]:
        """Get memory usage percentage for each GPU"""
        stats = self.get_gpu_stats()
        return {
            gpu_id: (gpu_stats.memory_used / gpu_stats.memory_total) * 100
            for gpu_id, gpu_stats in stats.items()
        }

    def get_utilization(self) -> Dict[str, float]:
        """Get GPU utilization percentage"""
        stats = self.get_gpu_stats()
        return {
            gpu_id: gpu_stats.utilization
            for gpu_id, gpu_stats in stats.items()
        } 