import React, { useState, useEffect } from 'react';
import {
  LineChart,
  BarChart,
  Gauge,
  MetricsTable,
  DeploymentHistory
} from './charts';
import { ModelSelector, TimeRangeSelector } from './controls';

interface DashboardProps {
  modelName: string;
  refreshInterval?: number;
}

export const ModelDashboard: React.FC<DashboardProps> = ({
  modelName,
  refreshInterval = 30000
}) => {
  const [dashboardData, setDashboardData] = useState(null);
  const [timeRange, setTimeRange] = useState(7);
  
  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await fetch(
          `/api/dashboard/models/${modelName}?days=${timeRange}`
        );
        const data = await response.json();
        setDashboardData(data);
      } catch (error) {
        console.error('Failed to fetch dashboard data:', error);
      }
    };
    
    fetchData();
    const interval = setInterval(fetchData, refreshInterval);
    return () => clearInterval(interval);
  }, [modelName, timeRange, refreshInterval]);
  
  if (!dashboardData) return <div>Loading...</div>;
  
  return (
    <div className="model-dashboard">
      <header className="dashboard-header">
        <h1>{modelName} Performance Dashboard</h1>
        <TimeRangeSelector
          value={timeRange}
          onChange={setTimeRange}
        />
      </header>
      
      <div className="metrics-overview">
        <Gauge
          value={dashboardData.performance_summary.memory_usage}
          title="Memory Usage"
        />
        <Gauge
          value={dashboardData.performance_summary.gpu_utilization}
          title="GPU Utilization"
        />
      </div>
      
      <div className="performance-charts">
        <LineChart
          data={dashboardData.versions}
          metrics={["accuracy", "latency"]}
          title="Performance Trends"
        />
        <BarChart
          data={dashboardData.deployment_history}
          metric="traffic_percentage"
          title="Traffic Distribution"
        />
      </div>
      
      <DeploymentHistory
        history={dashboardData.deployment_history}
      />
      
      <MetricsTable
        versions={dashboardData.versions}
        metrics={[
          "accuracy",
          "latency",
          "throughput",
          "error_rate"
        ]}
      />
    </div>
  );
}; 