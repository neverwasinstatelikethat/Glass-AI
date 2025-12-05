import React from 'react';
import useDashboardData from '../hooks/useDashboardData';

const TestDashboardData: React.FC = () => {
  const { data, loading, error } = useDashboardData();

  if (loading) {
    return <div>Loading...</div>;
  }

  if (error) {
    return <div>Error: {error}</div>;
  }

  return (
    <div>
      <h1>Dashboard Data Test</h1>
      {data && (
        <div>
          <h2>KPI Data</h2>
          <p>Quality Rate: {data.kpiData.qualityRate}%</p>
          <p>Defect Count: {data.kpiData.defectCount}</p>
          <p>Units Produced: {data.kpiData.unitsProduced}</p>
          <p>Uptime: {data.kpiData.uptime}%</p>

          <h2>Real-time Metrics</h2>
          <ul>
            {data.realTimeMetrics.map((metric, index) => (
              <li key={index}>
                {metric.name}: {metric.value} {metric.unit} (Icon: {metric.icon})
              </li>
            ))}
          </ul>

          <h2>AI Recommendations</h2>
          <ul>
            {data.aiRecommendations.map((rec, index) => (
              <li key={index}>
                {rec.text} - Priority: {rec.priority} (Icon: {rec.icon})
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default TestDashboardData;