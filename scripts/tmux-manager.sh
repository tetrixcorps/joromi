#!/bin/bash

# Create a new tmux session named "joromigpt"
tmux new-session -d -s joromigpt

# Window 1: Service Status and Monitoring
tmux rename-window -t joromigpt:1 'monitor'
tmux send-keys -t joromigpt:monitor 'watch -n 1 "docker-compose ps && echo \"\nGPU Status:\" && nvidia-smi"' C-m

# Window 2: Service Logs
tmux new-window -t joromigpt -n 'logs'
tmux split-window -h
tmux select-pane -t 0
tmux send-keys 'docker-compose logs -f gateway asr translation' C-m
tmux select-pane -t 1
tmux send-keys 'docker-compose logs -f tts banking' C-m

# Window 3: Resource Monitoring
tmux new-window -t joromigpt -n 'resources'
tmux split-window -h
tmux select-pane -t 0
tmux send-keys 'htop' C-m
tmux select-pane -t 1
tmux send-keys 'watch -n 1 nvidia-smi' C-m

# Window 4: Service Control
tmux new-window -t joromigpt -n 'control'
tmux send-keys 'echo "Service Control Panel" && echo "1. Start all services: docker-compose up -d" && echo "2. Stop all services: docker-compose down" && echo "3. Restart service: docker-compose restart [service]"' C-m

# Window 5: Model Management
tmux new-window -t joromigpt -n 'models'
tmux split-window -h
tmux select-pane -t 0
tmux send-keys 'echo "Model Cache Directory:" && ls -l /path/to/model-cache' C-m
tmux select-pane -t 1
tmux send-keys 'docker-compose exec gateway python3 -c "from services.manager import ModelManager; ModelManager.check_models()"' C-m

# Attach to the session
tmux attach-session -t joromigpt 