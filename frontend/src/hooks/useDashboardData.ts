import { useState, useEffect, useRef } from 'react';

interface KPIData {
  qualityRate: number;
  defectCount: number;
  unitsProduced: number;
  uptime: number;
}

interface DefectDistribution {
  name: string;
  value: number;
  color: string;
}

interface PerformanceDataPoint {
  time: string;
  quality: number;
  defects: number;
}

interface RealTimeMetric {
  name: string;
  value: number;
  unit: string;
  max: number;
  trend: 'up' | 'down' | 'stable';
  icon: string;
}

interface AIRecommendation {
  text: string;
  priority: 'high' | 'medium' | 'low';
  impact: number;
  icon: string;
}

interface DashboardData {
  kpiData: KPIData;
  defectDistribution: DefectDistribution[];
  performanceData: PerformanceDataPoint[];
  realTimeMetrics: RealTimeMetric[];
  aiRecommendations: AIRecommendation[];
}

const useDashboardData = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  useEffect(() => {
    const fetchData = async () => {
      try {
        setLoading(true);
        
        // Fetch quality metrics
        const qualityResponse = await fetch(`${API_BASE_URL}/api/quality/metrics`);
        if (!qualityResponse.ok) {
          throw new Error(`Failed to fetch quality metrics: ${qualityResponse.status}`);
        }
        const qualityData = await qualityResponse.json();
        
        // Fetch active alerts
        const alertsResponse = await fetch(`${API_BASE_URL}/api/alerts/active`);
        if (!alertsResponse.ok) {
          throw new Error(`Failed to fetch alerts: ${alertsResponse.status}`);
        }
        const alertsData = await alertsResponse.json();
        
        // Fetch recommendations
        const recommendationsResponse = await fetch(`${API_BASE_URL}/api/recommendations`);
        if (!recommendationsResponse.ok) {
          throw new Error(`Failed to fetch recommendations: ${recommendationsResponse.status}`);
        }
        const recommendationsData = await recommendationsResponse.json();
        
        // Construct dashboard data
        const dashboardData: DashboardData = {
          kpiData: {
            qualityRate: qualityData.quality_rate || 96.5,
            defectCount: qualityData.defect_count || 42,
            unitsProduced: qualityData.total_units || 1250,
            uptime: 98.5 // This would come from system stats API
          },
          defectDistribution: [
            { name: 'Трещины', value: 12, color: '#FF1744' },
            { name: 'Пузыри', value: 18, color: '#FFD700' },
            { name: 'Сколы', value: 8, color: '#00E676' },
            { name: 'Помутнение', value: 5, color: '#00E5FF' },
            { name: 'Деформация', value: 2, color: '#9D4EDD' }
          ],
          performanceData: [
            { time: '08:00', quality: 95.2, defects: 52 },
            { time: '09:00', quality: 95.8, defects: 48 },
            { time: '10:00', quality: 96.1, defects: 46 },
            { time: '11:00', quality: 96.5, defects: 45 },
            { time: '12:00', quality: 96.3, defects: 47 },
            { time: '13:00', quality: 96.7, defects: 43 },
            { time: '14:00', quality: 96.5, defects: 45 },
            { time: '15:00', quality: 97.1, defects: 41 },
            { time: '16:00', quality: 97.3, defects: 40 },
          ],
          realTimeMetrics: [
            {
              name: 'Температура печи',
              value: 1520,
              unit: '°C',
              max: 1600,
              trend: 'stable',
              icon: 'LocalFireDepartment'
            },
            {
              name: 'Уровень расплава',
              value: 2.45,
              unit: 'м',
              max: 3.0,
              trend: 'up',
              icon: 'WaterDrop'
            },
            {
              name: 'Скорость ленты',
              value: 155,
              unit: 'м/мин',
              max: 200,
              trend: 'down',
              icon: 'Speed'
            },
            {
              name: 'Температура формы',
              value: 325,
              unit: '°C',
              max: 350,
              trend: 'stable',
              icon: 'Thermostat'
            }
          ],
          aiRecommendations: Array.isArray(recommendationsData) 
            ? recommendationsData.map((rec: any) => ({
                text: rec.description || 'Рекомендация по оптимизации процесса',
                priority: rec.urgency?.toLowerCase() || 'medium',
                impact: Math.round((rec.confidence || 0.5) * 100),
                icon: 'Psychology'
              }))
            : [
                {
                  text: "Снизить температуру печи на 15°C для минимизации образования трещин",
                  priority: 'high',
                  impact: 85,
                  icon: 'LocalFireDepartment'
                },
                {
                  text: "Увеличить скорость ленты на 5% для оптимизации производительности",
                  priority: 'medium',
                  impact: 65,
                  icon: 'Speed'
                }
              ]
        };

        setData(dashboardData);
        setLoading(false);
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
        setError('Failed to load dashboard data');
        setLoading(false);
        
        // Fallback to mock data in case of error
        const mockData: DashboardData = {
          kpiData: {
            qualityRate: 96.5,
            defectCount: 42,
            unitsProduced: 1250,
            uptime: 98.5
          },
          defectDistribution: [
            { name: 'Трещины', value: 12, color: '#FF1744' },
            { name: 'Пузыри', value: 18, color: '#FFD700' },
            { name: 'Сколы', value: 8, color: '#00E676' },
            { name: 'Помутнение', value: 5, color: '#00E5FF' },
            { name: 'Деформация', value: 2, color: '#9D4EDD' }
          ],
          performanceData: [
            { time: '08:00', quality: 95.2, defects: 52 },
            { time: '09:00', quality: 95.8, defects: 48 },
            { time: '10:00', quality: 96.1, defects: 46 },
            { time: '11:00', quality: 96.5, defects: 45 },
            { time: '12:00', quality: 96.3, defects: 47 },
            { time: '13:00', quality: 96.7, defects: 43 },
            { time: '14:00', quality: 96.5, defects: 45 },
            { time: '15:00', quality: 97.1, defects: 41 },
            { time: '16:00', quality: 97.3, defects: 40 },
          ],
          realTimeMetrics: [
            {
              name: 'Температура печи',
              value: 1520,
              unit: '°C',
              max: 1600,
              trend: 'stable',
              icon: 'LocalFireDepartment'
            },
            {
              name: 'Уровень расплава',
              value: 2.45,
              unit: 'м',
              max: 3.0,
              trend: 'up',
              icon: 'WaterDrop'
            },
            {
              name: 'Скорость ленты',
              value: 155,
              unit: 'м/мин',
              max: 200,
              trend: 'down',
              icon: 'Speed'
            },
            {
              name: 'Температура формы',
              value: 325,
              unit: '°C',
              max: 350,
              trend: 'stable',
              icon: 'Thermostat'
            }
          ],
          aiRecommendations: [
            {
              text: "Снизить температуру печи на 15°C для минимизации образования трещин",
              priority: 'high',
              impact: 85,
              icon: 'LocalFireDepartment'
            },
            {
              text: "Увеличить скорость ленты на 5% для оптимизации производительности",
              priority: 'medium',
              impact: 65,
              icon: 'Speed'
            },
            {
              text: "Настроить температуру формы на 320°C для повышения консистенции качества",
              priority: 'medium',
              impact: 60,
              icon: 'Thermostat'
            },
            {
              text: "Запланировать техобслуживание горелочной зоны №2 для предотвращения перегрева",
              priority: 'low',
              impact: 45,
              icon: 'DeviceHub'
            },
            {
              text: "Оптимизировать подачу сырья для снижения затрат на 12%",
              priority: 'high',
              impact: 90,
              icon: 'Factory'
            }
          ]
        };
        
        setData(mockData);
      }
    };

    fetchData();
    
    // Set up WebSocket connection for real-time updates
    const wsUrl = API_BASE_URL.replace('http', 'ws') + '/ws/realtime';
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;
    
    ws.onopen = () => {
      console.log('WebSocket connected for real-time updates');
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        console.log('WebSocket message received:', message);
        if (message.type === 'realtime_update' && message.data) {
          // Update KPI data with real-time values
          setData(prevData => {
            if (!prevData) return prevData;
            
            return {
              ...prevData,
              kpiData: {
                ...prevData.kpiData,
                qualityRate: message.data.current_quality_rate !== undefined ? message.data.current_quality_rate : prevData.kpiData.qualityRate,
                defectCount: message.data.defect_count_hourly !== undefined ? message.data.defect_count_hourly : prevData.kpiData.defectCount,
              }
            };
          });
        }
      } catch (err) {
        console.error('Error parsing WebSocket message:', err);
      }
    };
    
    ws.onerror = (err) => {
      console.error('WebSocket error:', err);
    };
    
    ws.onclose = () => {
      console.log('WebSocket connection closed');
    };
    
    // Set up polling for periodic updates
    const interval = setInterval(fetchData, 30000); // Update every 30 seconds
    
    return () => {
      clearInterval(interval);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, []);

  return { data, loading, error };
};

export default useDashboardData;