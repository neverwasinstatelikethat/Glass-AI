import { useState, useEffect, useRef, useCallback } from 'react';

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

export interface DashboardData {
  kpiData: KPIData;
  defectDistribution: DefectDistribution[];
  performanceData: PerformanceDataPoint[];
  realTimeMetrics: RealTimeMetric[];
  aiRecommendations: AIRecommendation[];
}

const DEFECT_COLORS: Record<string, string> = {
  crack: '#FF1744',
  bubble: '#FFD700',
  chip: '#00E676',
  stain: '#9D4EDD',
  cloudiness: '#00E5FF',
  deformation: '#FF3366'
};

const DEFECT_NAMES: Record<string, string> = {
  crack: 'Трещины',
  bubble: 'Пузыри',
  chip: 'Сколы',
  stain: 'Пятна',
  cloudiness: 'Помутнение',
  deformation: 'Деформация'
};

const useDashboardData = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const fetchCountRef = useRef(0);
  const performanceHistoryRef = useRef<PerformanceDataPoint[]>([]);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Fetch all dashboard data from backend APIs
  const fetchData = useCallback(async () => {
    try {
      fetchCountRef.current += 1;
      const isFirstFetch = fetchCountRef.current === 1;
      if (isFirstFetch) setLoading(true);
      
      // Parallel fetch all endpoints
      const [qualityRes, alertsRes, trendsRes, efficiencyRes, rlRecsRes, statsRes, dtStateRes] = await Promise.allSettled([
        fetch(`${API_BASE_URL}/api/quality/metrics`),
        fetch(`${API_BASE_URL}/api/alerts/active`),
        fetch(`${API_BASE_URL}/api/analytics/defect-trends?timerange=24h&grouping=hourly`),
        fetch(`${API_BASE_URL}/api/analytics/production-efficiency?timerange=24h`),
        fetch(`${API_BASE_URL}/api/rl/recommendations/detailed`),
        fetch(`${API_BASE_URL}/api/statistics`),
        fetch(`${API_BASE_URL}/api/digital-twin/state`)
      ]);

      // Parse quality metrics
      let qualityData: any = {};
      if (qualityRes.status === 'fulfilled' && qualityRes.value.ok) {
        qualityData = await qualityRes.value.json();
      }

      // Parse statistics for uptime
      let statsData: any = {};
      if (statsRes.status === 'fulfilled' && statsRes.value.ok) {
        statsData = await statsRes.value.json();
      }

      // Parse digital twin state for real-time sensor values
      let dtState: any = {};
      if (dtStateRes.status === 'fulfilled' && dtStateRes.value.ok) {
        const dtResponse = await dtStateRes.value.json();
        dtState = dtResponse.data || dtResponse;
      }

      // Parse defect trends for distribution
      let trendsData: any = { data_points: [] };
      if (trendsRes.status === 'fulfilled' && trendsRes.value.ok) {
        trendsData = await trendsRes.value.json();
      }

      // Parse efficiency data for performance chart
      let efficiencyData: any = { data_points: [] };
      if (efficiencyRes.status === 'fulfilled' && efficiencyRes.value.ok) {
        efficiencyData = await efficiencyRes.value.json();
      }

      // Parse RL recommendations
      let recommendationsData: any[] = [];
      if (rlRecsRes.status === 'fulfilled' && rlRecsRes.value.ok) {
        const rlData = await rlRecsRes.value.json();
        recommendationsData = rlData.recommendations || rlData || [];
      }
      
      // Fallback to regular recommendations
      if (!recommendationsData || recommendationsData.length === 0) {
        try {
          const fallbackRes = await fetch(`${API_BASE_URL}/api/recommendations`);
          if (fallbackRes.ok) {
            recommendationsData = await fallbackRes.json();
          }
        } catch (e) {
          console.warn('Fallback recommendations failed:', e);
        }
      }

      // Build defect distribution from trends data
      const defectDistribution: DefectDistribution[] = [];
      const defectTypes = ['crack', 'bubble', 'chip', 'stain', 'cloudiness', 'deformation'];
      
      if (trendsData.data_points && trendsData.data_points.length > 0) {
        // Aggregate defects from all data points
        const aggregated: Record<string, number> = {};
        defectTypes.forEach(dt => aggregated[dt] = 0);
        
        trendsData.data_points.forEach((dp: any) => {
          defectTypes.forEach(dt => {
            aggregated[dt] += dp[dt] || 0;
          });
        });
        
        defectTypes.forEach(dt => {
          if (aggregated[dt] > 0) {
            defectDistribution.push({
              name: DEFECT_NAMES[dt] || dt,
              value: aggregated[dt],
              color: DEFECT_COLORS[dt] || '#888888'
            });
          }
        });
      }
      
      // Fallback if no data
      if (defectDistribution.length === 0) {
        defectDistribution.push(
          { name: 'Трещины', value: 0, color: '#FF1744' },
          { name: 'Пузыри', value: 0, color: '#FFD700' },
          { name: 'Сколы', value: 0, color: '#00E676' }
        );
      }

      // Build performance data from efficiency or trends
      let performanceData: PerformanceDataPoint[] = [];
      
      if (efficiencyData.data_points && efficiencyData.data_points.length > 0) {
        performanceData = efficiencyData.data_points.map((dp: any) => {
          const ts = new Date(dp.timestamp);
          return {
            time: ts.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
            quality: dp.quality_rate || dp.quality || 95,
            defects: dp.defect_count || Math.round((100 - (dp.quality_rate || 95)) * 10)
          };
        }).slice(-12); // Last 12 points
      } else if (trendsData.data_points && trendsData.data_points.length > 0) {
        performanceData = trendsData.data_points.map((dp: any) => {
          const ts = new Date(dp.timestamp);
          return {
            time: ts.toLocaleTimeString('ru-RU', { hour: '2-digit', minute: '2-digit' }),
            quality: 100 - (dp.total_defects || 0) * 0.5,
            defects: dp.total_defects || 0
          };
        }).slice(-12);
      }
      
      // Store in history and merge
      if (performanceData.length > 0) {
        performanceHistoryRef.current = performanceData;
      } else {
        performanceData = performanceHistoryRef.current;
      }

      // Build real-time metrics from digital twin state
      const sensorParams = dtState.parameters || dtState.sensor_values || {};
      const realTimeMetrics: RealTimeMetric[] = [
        {
          name: 'Температура печи',
          value: Math.round(sensorParams.furnace_temperature || sensorParams.furnace?.temperature || 1520),
          unit: '°C',
          max: 1600,
          trend: 'stable',
          icon: 'LocalFireDepartment'
        },
        {
          name: 'Давление печи',
          value: Math.round((sensorParams.furnace_pressure || sensorParams.furnace?.pressure || 25) * 10) / 10,
          unit: 'кПа',
          max: 50,
          trend: 'stable',
          icon: 'WaterDrop'
        },
        {
          name: 'Скорость ленты',
          value: Math.round(sensorParams.belt_speed || sensorParams.forming?.speed || 155),
          unit: 'м/мин',
          max: 200,
          trend: 'stable',
          icon: 'Speed'
        },
        {
          name: 'Температура формы',
          value: Math.round(sensorParams.mold_temp || sensorParams.forming?.mold_temp || 320),
          unit: '°C',
          max: 400,
          trend: 'stable',
          icon: 'Thermostat'
        }
      ];

      // Build AI recommendations from RL data
      const aiRecommendations: AIRecommendation[] = Array.isArray(recommendationsData) && recommendationsData.length > 0
        ? recommendationsData.map((rec: any) => ({
            text: rec.text || rec.description || rec.action || 'Рекомендация RL агента',
            priority: (rec.priority || rec.urgency || 'medium').toLowerCase() as 'high' | 'medium' | 'low',
            impact: rec.impact || Math.round((rec.confidence || rec.expected_improvement || 0.7) * 100),
            icon: rec.icon || getIconForParameter(rec.parameter)
          }))
        : [];

      // Calculate uptime from stats
      let uptime = 98.5;
      if (statsData.uptime) {
        // Parse uptime string like "0:15:32.123456"
        const parts = statsData.uptime.split(':');
        if (parts.length >= 2) {
          const hours = parseFloat(parts[0]);
          const minutes = parseFloat(parts[1]);
          uptime = Math.min(99.9, 95 + (hours * 60 + minutes) / 60);
        }
      }

      const dashboardData: DashboardData = {
        kpiData: {
          qualityRate: qualityData.quality_rate || dtState.quality_score * 100 || 0,
          defectCount: qualityData.defect_count || trendsData.total_defects || 0,
          unitsProduced: qualityData.total_units || 0,
          uptime: uptime
        },
        defectDistribution,
        performanceData,
        realTimeMetrics,
        aiRecommendations
      };

      setData(dashboardData);
      setLoading(false);
      setError(null);
      
    } catch (err) {
      console.error('Error fetching dashboard data:', err);
      setError('Failed to load dashboard data');
      setLoading(false);
    }
  }, [API_BASE_URL]);

  // Helper function to get icon for parameter
  const getIconForParameter = (param: string): string => {
    const iconMap: Record<string, string> = {
      'furnace_temperature': 'LocalFireDepartment',
      'belt_speed': 'Speed',
      'mold_temp': 'Thermostat',
      'forming_pressure': 'Compress',
      'cooling_rate': 'AcUnit',
      'energy_consumption': 'ElectricBolt'
    };
    return iconMap[param] || 'Psychology';
  };

  useEffect(() => {
    // Initial fetch
    fetchData();
    
    // Set up polling for periodic updates (every 10 minutes = 600000 ms)
    const interval = setInterval(fetchData, 600000);
    
    return () => {
      clearInterval(interval);
    };
  }, [fetchData]);

  return { data, loading, error };
};

export default useDashboardData;