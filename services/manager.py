import asyncio
from typing import Dict, Type
from services.base_service import BaseModelService
from services.asr_service import ASRService
from services.translation_service import TranslationService
import uvicorn
import subprocess
import os
from typing import Optional

class ServiceManager:
    def __init__(self):
        self.services: Dict[str, BaseModelService] = {}
        self.initialize_services()

    def initialize_services(self):
        self.services = {
            "asr": ASRService(port=8001),
            "translation": TranslationService(port=8002),
            # Add other services...
        }

    async def start_services(self):
        for service_name, service in self.services.items():
            try:
                await service.initialize()
                uvicorn.run(
                    service.app,
                    host="0.0.0.0",
                    port=service.port,
                    log_level="info"
                )
            except Exception as e:
                print(f"Failed to start {service_name}: {str(e)}")

    async def stop_services(self):
        # Cleanup and shutdown logic
        pass

class TmuxManager:
    def __init__(self):
        self.session_name = "joromigpt"

    def create_session(self) -> None:
        """Create a new tmux session"""
        try:
            subprocess.run(['tmux', 'new-session', '-d', '-s', self.session_name])
        except Exception as e:
            logger.error(f"Failed to create tmux session: {e}")

    def create_monitoring_window(self) -> None:
        """Create monitoring window with service stats"""
        try:
            subprocess.run([
                'tmux', 'new-window', '-t', f'{self.session_name}:',
                '-n', 'monitoring'
            ])
            subprocess.run([
                'tmux', 'send-keys', '-t', f'{self.session_name}:monitoring',
                'python3 -m services.monitor', 'C-m'
            ])
        except Exception as e:
            logger.error(f"Failed to create monitoring window: {e}")

    def create_service_window(self, service_name: str) -> None:
        """Create window for specific service"""
        try:
            subprocess.run([
                'tmux', 'new-window', '-t', f'{self.session_name}:',
                '-n', service_name
            ])
            subprocess.run([
                'tmux', 'send-keys', '-t', f'{self.session_name}:{service_name}',
                f'docker-compose logs -f {service_name}', 'C-m'
            ])
        except Exception as e:
            logger.error(f"Failed to create service window: {e}")

    def attach_session(self) -> None:
        """Attach to the tmux session"""
        try:
            subprocess.run(['tmux', 'attach-session', '-t', self.session_name])
        except Exception as e:
            logger.error(f"Failed to attach to session: {e}")

if __name__ == "__main__":
    manager = ServiceManager()
    asyncio.run(manager.start_services()) 