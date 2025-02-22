#!/usr/bin/env python3
import os
import sys
import argparse
import subprocess
from pathlib import Path

SUPERVISORD_TEMPLATE = """
[program:{service_name}]
command=docker-compose up {service_name}
directory={working_dir}
user={user}
autostart=true
autorestart=true
stderr_logfile=/var/log/joromigpt/{service_name}.err.log
stdout_logfile=/var/log/joromigpt/{service_name}.out.log
environment=
    CUDA_VISIBLE_DEVICES={gpu_id},
    PYTHONPATH={working_dir}
"""

SYSTEMD_TEMPLATE = """
[Unit]
Description=JoromiGPT {service_name} Service
After=docker.service
Requires=docker.service

[Service]
Type=simple
User={user}
Environment=CUDA_VISIBLE_DEVICES={gpu_id}
Environment=PYTHONPATH={working_dir}
WorkingDirectory={working_dir}
ExecStart=/usr/local/bin/docker-compose up {service_name}
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
"""

class ServiceManager:
    def __init__(self):
        self.working_dir = Path(__file__).parent.parent.absolute()
        self.user = os.getenv('USER', 'root')
        self.services = {
            'gateway': {'gpu_id': 'none'},
            'asr': {'gpu_id': '0'},
            'tts': {'gpu_id': '1'},
            'translation': {'gpu_id': '2'},
            'banking': {'gpu_id': '3'},
            'visual_qa': {'gpu_id': '4'}
        }

    def setup_supervisord(self):
        """Setup supervisord configurations for all services"""
        config_dir = Path('/etc/supervisor/conf.d')
        log_dir = Path('/var/log/joromigpt')

        # Create log directory
        subprocess.run(['sudo', 'mkdir', '-p', str(log_dir)])
        subprocess.run(['sudo', 'chown', f'{self.user}:{self.user}', str(log_dir)])

        for service, config in self.services.items():
            conf_content = SUPERVISORD_TEMPLATE.format(
                service_name=service,
                working_dir=self.working_dir,
                user=self.user,
                gpu_id=config['gpu_id']
            )
            
            conf_file = config_dir / f'joromigpt_{service}.conf'
            with open(conf_file, 'w') as f:
                f.write(conf_content)

        # Reload supervisord
        subprocess.run(['sudo', 'supervisorctl', 'reread'])
        subprocess.run(['sudo', 'supervisorctl', 'update'])

    def setup_systemd(self):
        """Setup systemd service files for all services"""
        service_dir = Path('/etc/systemd/system')

        for service, config in self.services.items():
            service_content = SYSTEMD_TEMPLATE.format(
                service_name=service,
                working_dir=self.working_dir,
                user=self.user,
                gpu_id=config['gpu_id']
            )
            
            service_file = service_dir / f'joromigpt_{service}.service'
            with open(service_file, 'w') as f:
                f.write(service_content)

        # Reload systemd
        subprocess.run(['sudo', 'systemctl', 'daemon-reload'])

    def start_services(self, system='supervisord'):
        """Start all services using specified system"""
        for service in self.services:
            if system == 'supervisord':
                subprocess.run(['sudo', 'supervisorctl', 'start', f'joromigpt_{service}'])
            else:
                subprocess.run(['sudo', 'systemctl', 'start', f'joromigpt_{service}'])

    def stop_services(self, system='supervisord'):
        """Stop all services using specified system"""
        for service in self.services:
            if system == 'supervisord':
                subprocess.run(['sudo', 'supervisorctl', 'stop', f'joromigpt_{service}'])
            else:
                subprocess.run(['sudo', 'systemctl', 'stop', f'joromigpt_{service}'])

    def status(self, system='supervisord'):
        """Check status of all services"""
        if system == 'supervisord':
            subprocess.run(['sudo', 'supervisorctl', 'status'])
        else:
            for service in self.services:
                subprocess.run(['sudo', 'systemctl', 'status', f'joromigpt_{service}'])

def main():
    parser = argparse.ArgumentParser(description='JoromiGPT Service Manager')
    parser.add_argument('--system', choices=['supervisord', 'systemd'], default='supervisord',
                      help='Service management system to use')
    parser.add_argument('command', choices=['setup', 'start', 'stop', 'status'],
                      help='Command to execute')

    args = parser.parse_args()
    manager = ServiceManager()

    if args.command == 'setup':
        if args.system == 'supervisord':
            manager.setup_supervisord()
        else:
            manager.setup_systemd()
    elif args.command == 'start':
        manager.start_services(args.system)
    elif args.command == 'stop':
        manager.stop_services(args.system)
    elif args.command == 'status':
        manager.status(args.system)

if __name__ == '__main__':
    if os.geteuid() != 0:
        print("This script must be run as root or with sudo")
        sys.exit(1)
    main() 