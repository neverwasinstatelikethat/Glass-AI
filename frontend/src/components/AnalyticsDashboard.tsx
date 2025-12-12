import React, { useState, useEffect, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  Grid,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Stack,
  Card,
  CardContent,
  Tabs,
  Tab,
  CircularProgress,
  Alert,
  Chip,
  IconButton,
  Tooltip as MuiTooltip,
  alpha,
  useTheme,
  Divider,
  Button,
  Menu,
  Fade,
  Zoom,
  LinearProgress
} from '@mui/material';
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  AreaChart,
  Area,
  ZAxis,
  Brush,
  PieChart,
  Pie,
  Sector,
  Treemap,
  Sankey,
  Label,
  LabelList,
  ReferenceLine,
  ReferenceArea,
  ComposedChart
} from 'recharts';
import * as MUIIcons from '@mui/icons-material';
import {
  TrendingUp,
  BarChart as BarChartIcon,
  ScatterPlot,
  BubbleChart,
  Download,
  FilterList,
  Refresh,
  InfoOutlined,
  Timeline,
  TableChart,
  PictureAsPdf,
  InsertChart,
  Timeline as TimelineIcon,
  ShowChart,
  Equalizer,
  PieChart as PieChartIcon,
  GridOn,
  MultilineChart,
  Speed,
  AutoGraph,
  TrendingFlat,
  LocalFireDepartment,
  Engineering,
  PrecisionManufacturing,
  QrCodeScanner,
  Checklist,
  Factory,
  DeviceHub,
  Psychology
} from '@mui/icons-material';
import * as XLSX from 'xlsx';

const API_BASE_URL = 'http://localhost:8000';

interface DefectTrendData {
  timestamp: string;
  total_defects: number;
  crack: number;
  bubble: number;
  chip: number;
  stain: number;
  cloudiness: number;
  deformation: number;
}

interface EfficiencyData {
  timestamp: string;
  production_rate: number;
  quality_rate: number;
  efficiency_score: number;
  operator_id: string;
  shift: string;
  downtime: number;
  oee: number;
}

interface CorrelationData {
  [parameter: string]: {
    [defect: string]: number;
  };
}

interface HeatmapData {
  time: string;
  parameter: string;
  value: number;
  defect_rate: number;
}

interface StatisticalControlData {
  timestamp: string;
  value: number;
  ucl: number;
  lcl: number;
  cl: number;
  is_out_of_control: boolean;
}

interface ProductionMetrics {
  target_production: number;
  actual_production: number;
  good_units: number;
  defective_units: number;
  uptime: number;
  downtime: number;
  availability: number;
  performance: number;
  quality: number;
  oee: number;
}

const COLORS = [
  '#0066FF', '#00E5FF', '#00E676', '#FFD700', '#FF3366', '#8B5CF6', 
  '#FF6B6B', '#4ECDC4', '#FFD166', '#06D6A0', '#118AB2', '#EF476F'
];

const CustomTooltip = ({ active, payload, label }: any) => {
  const theme = useTheme();
  
  if (active && payload && payload.length) {
    return (
      <Paper
        elevation={8}
        sx={{
          p: 2,
          backgroundColor: 'rgba(13, 27, 42, 0.95)',
          border: `1px solid ${theme.palette.primary.main}50`,
          borderRadius: 2,
          minWidth: 220,
          backdropFilter: 'blur(10px)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4)'
        }}
      >
        <Typography variant="subtitle2" fontWeight="bold" color="white" gutterBottom>
          {label && typeof label === 'string' && !isNaN(Date.parse(label)) 
            ? new Date(label).toLocaleString('ru-RU', {
                day: '2-digit',
                month: 'short',
                hour: '2-digit',
                minute: '2-digit'
              })
            : label}
        </Typography>
        <Divider sx={{ my: 1, borderColor: theme.palette.primary.main }} />
        {payload.map((entry: any, index: number) => (
          <Box key={index} sx={{ 
            display: 'flex', 
            justifyContent: 'space-between', 
            alignItems: 'center', 
            mb: 0.5,
            p: 1,
            borderRadius: 1,
            bgcolor: 'rgba(255, 255, 255, 0.05)'
          }}>
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
              <Box sx={{ 
                width: 10, 
                height: 10, 
                borderRadius: '50%', 
                bgcolor: entry.color,
                boxShadow: `0 0 8px ${entry.color}`
              }} />
              <Typography variant="body2" color="text.secondary">
                {entry.name}
              </Typography>
            </Box>
            <Typography variant="body2" fontWeight="bold" color="white">
              {typeof entry.value === 'number' ? entry.value.toFixed(2) : entry.value}
            </Typography>
          </Box>
        ))}
      </Paper>
    );
  }
  return null;
};

const AnalyticsDashboard: React.FC = () => {
  const theme = useTheme();
  const [exportAnchorEl, setExportAnchorEl] = useState<null | HTMLElement>(null);
  
  const [activeTab, setActiveTab] = useState(0);
  const [timeRange, setTimeRange] = useState('24h');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedDefects, setSelectedDefects] = useState<string[]>([]);
  const [viewMode, setViewMode] = useState<'chart' | 'table'>('chart');
  const [activePieIndex, setActivePieIndex] = useState(0);
  
  const [defectTrends, setDefectTrends] = useState<DefectTrendData[]>([]);
  const [efficiencyData, setEfficiencyData] = useState<EfficiencyData[]>([]);
  const [correlations, setCorrelations] = useState<CorrelationData>({});
  const [heatmapData, setHeatmapData] = useState<HeatmapData[]>([]);
  const [controlChartData, setControlChartData] = useState<StatisticalControlData[]>([]);
  const [productionMetrics, setProductionMetrics] = useState<ProductionMetrics | null>(null);

  const fetchDefectTrends = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/analytics/defect-trends?timerange=${timeRange}&grouping=hourly`
      );
      
      if (!response.ok) throw new Error('Failed to fetch defect trends');
      
      const data = await response.json();
      setDefectTrends(data.data_points || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const fetchEfficiencyData = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/analytics/production-efficiency?timerange=${timeRange}`
      );
      
      if (!response.ok) throw new Error('Failed to fetch efficiency data');
      
      const data = await response.json();
      setEfficiencyData(data.data_points || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const fetchCorrelations = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/analytics/parameter-correlations`);
      
      if (!response.ok) throw new Error('Failed to fetch correlations');
      
      const data = await response.json();
      setCorrelations(data.correlations || {});
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
    } finally {
      setLoading(false);
    }
  };

  const fetchHeatmapData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/analytics/heatmap?timerange=${timeRange}`);
      if (response.ok) {
        const data = await response.json();
        setHeatmapData(data.data || []);
      }
    } catch (err) {
      console.error('Error fetching heatmap:', err);
    }
  };

  const fetchControlChartData = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/analytics/control-chart?timerange=${timeRange}`);
      if (response.ok) {
        const data = await response.json();
        setControlChartData(data.data || []);
      }
    } catch (err) {
      console.error('Error fetching control chart:', err);
    }
  };

  const fetchProductionMetrics = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/analytics/production-metrics?timerange=${timeRange}`);
      if (response.ok) {
        const data = await response.json();
        setProductionMetrics(data.metrics || null);
      }
    } catch (err) {
      console.error('Error fetching production metrics:', err);
    }
  };

  useEffect(() => {
    fetchDefectTrends();
    fetchEfficiencyData();
    fetchCorrelations();
    fetchHeatmapData();
    fetchControlChartData();
    fetchProductionMetrics();
  }, [timeRange]);

  // Экспорт в Excel
  const exportToExcel = () => {
    const workbook = XLSX.utils.book_new();
    
    // Лист с трендами дефектов
    const defectSheet = XLSX.utils.json_to_sheet(defectTrends.map(item => ({
      Дата: new Date(item.timestamp).toLocaleString('ru-RU'),
      'Всего дефектов': item.total_defects,
      'Трещины': item.crack,
      'Пузыри': item.bubble,
      'Сколы': item.chip,
      'Пятна': item.stain,
      'Помутнения': item.cloudiness,
      'Деформации': item.deformation
    })));
    XLSX.utils.book_append_sheet(workbook, defectSheet, 'Тренды дефектов');
    
    // Лист с эффективностью
    const efficiencySheet = XLSX.utils.json_to_sheet(efficiencyData.map(item => ({
      Дата: new Date(item.timestamp).toLocaleString('ru-RU'),
      'Производительность': item.production_rate,
      'Качество': item.quality_rate,
      'Эффективность': item.efficiency_score,
      'OEE': item.oee,
      'Простой': item.downtime,
      'Оператор': item.operator_id,
      'Смена': item.shift
    })));
    XLSX.utils.book_append_sheet(workbook, efficiencySheet, 'Эффективность');
    
    XLSX.writeFile(workbook, 'analytics-dashboard.xlsx');
    setExportAnchorEl(null);
  };

  // Экспорт в CSV
  const exportToCSV = () => {
    const csvContent = [
      'Дата,Всего дефектов,Трещины,Пузыри,Сколы,Пятна,Помутнения,Деформации',
      ...defectTrends.map(item => 
        `${new Date(item.timestamp).toLocaleString('ru-RU')},${item.total_defects},${item.crack},${item.bubble},${item.chip},${item.stain},${item.cloudiness},${item.deformation}`
      )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = 'analytics-dashboard.csv';
    link.click();
    setExportAnchorEl(null);
  };

  const getDefectDistributionRadar = () => {
    if (defectTrends.length === 0) return [];
    
    const latest = defectTrends[defectTrends.length - 1];
    const total = latest.total_defects || 1;
    
    return [
      { defect: 'Трещины', value: (latest.crack / total) * 100, fullMark: 100 },
      { defect: 'Пузыри', value: (latest.bubble / total) * 100, fullMark: 100 },
      { defect: 'Сколы', value: (latest.chip / total) * 100, fullMark: 100 },
      { defect: 'Пятна', value: (latest.stain / total) * 100, fullMark: 100 },
      { defect: 'Помутнения', value: (latest.cloudiness / total) * 100, fullMark: 100 },
      { defect: 'Деформации', value: (latest.deformation / total) * 100, fullMark: 100 }
    ];
  };

  const getDefectSummary = () => {
    if (defectTrends.length === 0) return null;
    const latest = defectTrends[defectTrends.length - 1];
    return {
      total: latest.total_defects,
      byType: [
        { name: 'Трещины', value: latest.crack, color: theme.palette.error.main },
        { name: 'Пузыри', value: latest.bubble, color: theme.palette.warning.main },
        { name: 'Сколы', value: latest.chip, color: theme.palette.info.main },
        { name: 'Пятна', value: latest.stain, color: theme.palette.success.main },
        { name: 'Помутнения', value: latest.cloudiness, color: theme.palette.primary.main },
        { name: 'Деформации', value: latest.deformation, color: theme.palette.secondary.main }
      ].filter(item => item.value > 0) // Фильтруем нулевые значения
    };
  };

  const getCorrelationColor = (value: number) => {
    if (value > 0.7) return theme.palette.error.main;
    if (value > 0.4) return theme.palette.warning.main;
    if (value > 0.1) return theme.palette.success.main;
    if (value > -0.2) return theme.palette.info.main;
    return theme.palette.primary.main;
  };

  // Подготовка данных для тепловой карты (исправленная версия)
  const getHeatmapChartData = () => {
    if (heatmapData.length === 0) return [];
    
    const times = Array.from(new Set(heatmapData.map(d => d.time))).slice(0, 12);
    const params = Array.from(new Set(heatmapData.map(d => d.parameter))).slice(0, 8);
    
    return times.map(time => {
      const obj: any = { time };
      params.forEach(param => {
        const item = heatmapData.find(d => d.time === time && d.parameter === param);
        obj[param] = item ? item.value : 0;
      });
      return obj;
    });
  };

  const handleRefresh = () => {
    fetchDefectTrends();
    fetchEfficiencyData();
    fetchCorrelations();
    fetchHeatmapData();
    fetchControlChartData();
    fetchProductionMetrics();
  };

  // Подготовка данных для нового дашборда "Эффективность по сменам"
  const getShiftEfficiency = () => {
    const shifts = efficiencyData.reduce((acc, item) => {
      if (!acc[item.shift]) {
        acc[item.shift] = {
          totalEfficiency: 0,
          count: 0,
          productionRate: 0,
          qualityRate: 0,
          oee: 0
        };
      }
      acc[item.shift].totalEfficiency += item.efficiency_score;
      acc[item.shift].productionRate += item.production_rate;
      acc[item.shift].qualityRate += item.quality_rate;
      acc[item.shift].oee += item.oee || 0;
      acc[item.shift].count++;
      return acc;
    }, {} as any);

    return Object.entries(shifts).map(([shift, data]: [string, any]) => ({
      shift,
      efficiency: (data.totalEfficiency / data.count).toFixed(1),
      productionRate: (data.productionRate / data.count).toFixed(1),
      qualityRate: (data.qualityRate / data.count).toFixed(1),
      oee: (data.oee / data.count).toFixed(1)
    }));
  };

  // Расширенные метрики KPI
  const getKPIMetrics = () => {
    if (efficiencyData.length === 0) return null;
    
    const avgEfficiency = efficiencyData.reduce((sum, item) => sum + item.efficiency_score, 0) / efficiencyData.length;
    const avgProduction = efficiencyData.reduce((sum, item) => sum + item.production_rate, 0) / efficiencyData.length;
    const avgQuality = efficiencyData.reduce((sum, item) => sum + item.quality_rate, 0) / efficiencyData.length;
    const avgOEE = efficiencyData.reduce((sum, item) => sum + (item.oee || 0), 0) / efficiencyData.length;
    
    return {
      efficiency: avgEfficiency,
      production: avgProduction,
      quality: avgQuality,
      oee: avgOEE,
      totalProduction: defectTrends.reduce((sum, item) => sum + item.total_defects, 0) * 10, // Предполагаем 10 единиц на дефект
      defectRate: defectTrends.length > 0 ? 
        (defectTrends[defectTrends.length - 1].total_defects / (defectTrends[defectTrends.length - 1].total_defects * 10)) * 100 : 0,
      availability: productionMetrics?.availability || 95,
      performance: productionMetrics?.performance || 90
    };
  };

  const onPieEnter = (_: any, index: number) => {
    setActivePieIndex(index);
  };

  const renderActiveShape = (props: any) => {
    const { cx, cy, midAngle, innerRadius, outerRadius, startAngle, endAngle, fill, payload, percent, value } = props;
    const sin = Math.sin(-midAngle * (Math.PI / 180));
    const cos = Math.cos(-midAngle * (Math.PI / 180));
    const sx = cx + (outerRadius + 10) * cos;
    const sy = cy + (outerRadius + 10) * sin;
    const mx = cx + (outerRadius + 30) * cos;
    const my = cy + (outerRadius + 30) * sin;
    const ex = mx + (cos >= 0 ? 1 : -1) * 22;
    const ey = my;
    const textAnchor = cos >= 0 ? 'start' : 'end';

    return (
      <g>
        <Sector
          cx={cx}
          cy={cy}
          innerRadius={innerRadius}
          outerRadius={outerRadius}
          startAngle={startAngle}
          endAngle={endAngle}
          fill={fill}
        />
        <Sector
          cx={cx}
          cy={cy}
          startAngle={startAngle}
          endAngle={endAngle}
          innerRadius={outerRadius + 6}
          outerRadius={outerRadius + 10}
          fill={fill}
        />
        <path d={`M${sx},${sy}L${mx},${my}L${ex},${ey}`} stroke={fill} fill="none" />
        <circle cx={ex} cy={ey} r={2} fill={fill} stroke="none" />
        <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} textAnchor={textAnchor} fill="white">
          {`${payload.name}`}
        </text>
        <text x={ex + (cos >= 0 ? 1 : -1) * 12} y={ey} dy={18} textAnchor={textAnchor} fill="#999">
          {`${value} (${(percent * 100).toFixed(2)}%)`}
        </text>
      </g>
    );
  };

  return (
    <Box style={{ padding: '24px' }}>
      {/* Header с экспортом */}
      <Paper elevation={0} sx={{ 
        p: 3, 
        mb: 3, 
        background: 'linear-gradient(135deg, rgba(13, 27, 42, 0.8) 0%, rgba(0, 29, 61, 0.6) 100%)',
        border: '1px solid rgba(0, 102, 255, 0.3)',
        borderRadius: 3,
        backdropFilter: 'blur(20px)',
      }}>
        <Stack direction="row" justifyContent="space-between" alignItems="center">
          <Box>
            <Typography variant="h4" fontWeight="bold" color="white">
              Расширенная аналитика производства
            </Typography>
            <Typography variant="body2" color="text.secondary">
              РТУ МИРЭА • Система мониторинга качества стекла
            </Typography>
          </Box>
          
          <Stack direction="row" spacing={2} alignItems="center">
            <FormControl sx={{ minWidth: 180 }}>
              <InputLabel sx={{ color: 'text.secondary' }}>Временной диапазон</InputLabel>
              <Select
                value={timeRange}
                label="Временной диапазон"
                onChange={(e) => setTimeRange(e.target.value)}
                sx={{
                  backgroundColor: 'rgba(13, 27, 42, 0.6)',
                  borderRadius: 2,
                  color: 'white',
                  '& .MuiOutlinedInput-notchedOutline': {
                    borderColor: 'rgba(0, 102, 255, 0.3)',
                  },
                  '&:hover .MuiOutlinedInput-notchedOutline': {
                    borderColor: theme.palette.primary.main,
                  },
                }}
              >
                <MenuItem value="24h">Последние 24 часа</MenuItem>
                <MenuItem value="7d">Последние 7 дней</MenuItem>
                <MenuItem value="30d">Последние 30 дней</MenuItem>
              </Select>
            </FormControl>
            
            <Button
              variant="contained"
              onClick={handleRefresh}
              startIcon={<Refresh />}
              sx={{
                background: 'linear-gradient(135deg, #0066FF 0%, #00E5FF 100%)',
                '&:hover': {
                  background: '#4D8FFF',
                },
              }}
            >
              Обновить
            </Button>
            
            <Button
              variant="contained"
              onClick={(e) => setExportAnchorEl(e.currentTarget)}
              startIcon={<Download />}
              sx={{
                background: 'linear-gradient(135deg, #00E676 0%, #00C853 100%)',
                '&:hover': {
                  background: '#69F0AE',
                },
              }}
            >
              Экспорт
            </Button>
            
            <Menu
              anchorEl={exportAnchorEl}
              open={Boolean(exportAnchorEl)}
              onClose={() => setExportAnchorEl(null)}
              TransitionComponent={Fade}
            >
              <MenuItem onClick={exportToExcel} sx={{ gap: 1 }}>
                <TableChart fontSize="small" />
                Экспорт в Excel
              </MenuItem>
              <MenuItem onClick={exportToCSV} sx={{ gap: 1 }}>
                <GridOn fontSize="small" />
                Экспорт в CSV
              </MenuItem>
            </Menu>
          </Stack>
        </Stack>
      </Paper>

      {/* Error Alert */}
      {error && (
        <Zoom in={Boolean(error)}>
          <Alert 
            severity="error" 
            sx={{ 
              mb: 2,
              borderRadius: 2,
              background: 'linear-gradient(135deg, rgba(255, 23, 68, 0.1) 0%, rgba(255, 23, 68, 0.05) 100%)',
              border: '1px solid rgba(255, 23, 68, 0.3)',
              backdropFilter: 'blur(10px)',
            }} 
            onClose={() => setError(null)}
          >
            {error}
          </Alert>
        </Zoom>
      )}

      {/* Tabs */}
      <Paper elevation={0} sx={{ 
        mb: 3,
        background: 'rgba(13, 27, 42, 0.6)',
        backdropFilter: 'blur(20px)',
        border: '1px solid rgba(0, 102, 255, 0.15)',
        borderRadius: 3,
      }}>
        <Tabs 
          value={activeTab} 
          onChange={(_, newValue) => setActiveTab(newValue)}
          variant="scrollable"
          scrollButtons="auto"
          sx={{
            '& .MuiTab-root': {
              textTransform: 'none',
              fontWeight: 600,
              color: 'text.secondary',
              minHeight: 64,
              '&.Mui-selected': {
                color: 'primary.main',
              },
            },
            '& .MuiTabs-indicator': {
              background: 'linear-gradient(90deg, #0066FF, #00E5FF)',
              height: 3,
              borderRadius: '3px 3px 0 0',
            },
          }}
        >
          <Tab icon={<TrendingUp />} label="Тренды дефектов" />
          <Tab icon={<ShowChart />} label="Производительность" />
          <Tab icon={<BubbleChart />} label="Корреляции" />
          <Tab icon={<PieChartIcon />} label="Распределение" />
          <Tab icon={<Equalizer />} label="Тепловая карта" />
          <Tab icon={<TimelineIcon />} label="Контрольные карты" />
          <Tab icon={<AutoGraph />} label="KPI и метрики" />
        </Tabs>
      </Paper>

      {/* Loading Indicator */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 8 }}>
          <CircularProgress sx={{ color: theme.palette.primary.main }} />
        </Box>
      )}

      {/* Tab Panels */}
      {!loading && (
        <>
          {/* Tab 1: Defect Trends */}
          {activeTab === 0 && (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                }}>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom>
                    Динамика дефектов по времени
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <AreaChart data={defectTrends}>
                      <CartesianGrid 
                        strokeDasharray="3 3" 
                        stroke="rgba(255, 255, 255, 0.1)"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="timestamp"
                        tickFormatter={(value) => new Date(value).toLocaleTimeString('ru-RU', { hour: '2-digit' })}
                        stroke="rgba(255, 255, 255, 0.7)"
                        tickLine={false}
                        axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                      />
                      <YAxis 
                        stroke="rgba(255, 255, 255, 0.7)"
                        tickLine={false}
                        axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                      />
                      <Tooltip 
                        content={<CustomTooltip />}
                        contentStyle={{
                          backgroundColor: 'rgba(13, 27, 42, 0.95)',
                          border: '1px solid rgba(0, 102, 255, 0.5)',
                          borderRadius: 8,
                          backdropFilter: 'blur(10px)'
                        }}
                      />
                      <Legend />
                      
                      <Area
                        type="monotone"
                        dataKey="total_defects"
                        stroke={theme.palette.primary.main}
                        fill={`url(#colorTotal)`}
                        fillOpacity={0.3}
                        strokeWidth={2}
                        name="Всего дефектов"
                      />
                      
                      <defs>
                        <linearGradient id="colorTotal" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={theme.palette.primary.main} stopOpacity={0.8}/>
                          <stop offset="95%" stopColor={theme.palette.primary.main} stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                    </AreaChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              <Grid item xs={12} md={6}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                  height: '100%'
                }}>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom>
                    Суммарное количество дефектов
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <ComposedChart data={defectTrends.slice(-12)}>
                      <CartesianGrid 
                        strokeDasharray="3 3" 
                        stroke="rgba(255, 255, 255, 0.1)"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="timestamp"
                        tickFormatter={(value) => new Date(value).toLocaleTimeString('ru-RU', { hour: '2-digit' })}
                        stroke="rgba(255, 255, 255, 0.7)"
                        tickLine={false}
                        axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                      />
                      <YAxis 
                        stroke="rgba(255, 255, 255, 0.7)"
                        tickLine={false}
                        axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                      />
                      <Tooltip 
                        content={<CustomTooltip />}
                        contentStyle={{
                          backgroundColor: 'rgba(13, 27, 42, 0.95)',
                          border: '1px solid rgba(0, 102, 255, 0.5)',
                          borderRadius: 8,
                          backdropFilter: 'blur(10px)'
                        }}
                      />
                      <Bar
                        dataKey="total_defects"
                        fill="url(#colorBar)"
                        radius={[4, 4, 0, 0]}
                        name="Всего дефектов"
                      />
                      <Line
                        type="monotone"
                        dataKey="total_defects"
                        stroke={theme.palette.warning.main}
                        strokeWidth={2}
                        dot={false}
                      />
                      <defs>
                        <linearGradient id="colorBar" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor={theme.palette.primary.main} stopOpacity={0.8}/>
                          <stop offset="95%" stopColor={theme.palette.secondary.main} stopOpacity={0.3}/>
                        </linearGradient>
                      </defs>
                    </ComposedChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              <Grid item xs={12} md={6}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                  height: '100%'
                }}>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom>
                    Распределение по типам
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={getDefectSummary()?.byType}>
                      <CartesianGrid 
                        strokeDasharray="3 3" 
                        stroke="rgba(255, 255, 255, 0.1)"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="name"
                        stroke="rgba(255, 255, 255, 0.7)"
                        tickLine={false}
                        axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                      />
                      <YAxis 
                        stroke="rgba(255, 255, 255, 0.7)"
                        tickLine={false}
                        axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                      />
                      <Tooltip 
                        content={<CustomTooltip />}
                        contentStyle={{
                          backgroundColor: 'rgba(13, 27, 42, 0.95)',
                          border: '1px solid rgba(0, 102, 255, 0.5)',
                          borderRadius: 8,
                          backdropFilter: 'blur(10px)'
                        }}
                      />
                      <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                        {getDefectSummary()?.byType.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.color}
                            stroke="rgba(255, 255, 255, 0.2)"
                            strokeWidth={1}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>
          )}

          {/* Tab 2: Production Efficiency - Улучшенная версия */}
          {activeTab === 1 && (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                }}>
                  <Stack direction="row" justifyContent="space-between" alignItems="center" mb={3}>
                    <Typography variant="h6" fontWeight="bold" color="white">
                      Производительность vs Качество
                    </Typography>
                    <Stack direction="row" spacing={1}>
                      <Chip
                        label="График"
                        onClick={() => setViewMode('chart')}
                        color={viewMode === 'chart' ? 'primary' : 'default'}
                        variant={viewMode === 'chart' ? 'filled' : 'outlined'}
                        sx={{ backdropFilter: 'blur(10px)' }}
                      />
                      <Chip
                        label="Таблица"
                        onClick={() => setViewMode('table')}
                        color={viewMode === 'table' ? 'primary' : 'default'}
                        variant={viewMode === 'table' ? 'filled' : 'outlined'}
                        sx={{ backdropFilter: 'blur(10px)' }}
                      />
                    </Stack>
                  </Stack>
                  
                  {viewMode === 'chart' ? (
                    <ResponsiveContainer width="100%" height={400}>
                      <LineChart data={efficiencyData}>
                        <CartesianGrid 
                          strokeDasharray="3 3" 
                          stroke="rgba(255, 255, 255, 0.1)"
                          vertical={false}
                        />
                        <XAxis
                          dataKey="timestamp"
                          tickFormatter={(value) => new Date(value).toLocaleTimeString('ru-RU', { hour: '2-digit' })}
                          stroke="rgba(255, 255, 255, 0.7)"
                          tickLine={false}
                          axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                        />
                        <YAxis 
                          yAxisId="left"
                          stroke="rgba(255, 255, 255, 0.7)"
                          tickLine={false}
                          axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                          label={{ 
                            value: 'Производительность (ед/ч)', 
                            angle: -90, 
                            position: 'insideLeft',
                            fill: 'rgba(255, 255, 255, 0.7)'
                          }}
                        />
                        <YAxis 
                          yAxisId="right" 
                          orientation="right"
                          stroke="rgba(255, 255, 255, 0.7)"
                          tickLine={false}
                          axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                          label={{ 
                            value: 'Качество (%)', 
                            angle: 90, 
                            position: 'insideRight',
                            fill: 'rgba(255, 255, 255, 0.7)'
                          }}
                        />
                        <Tooltip 
                          content={<CustomTooltip />}
                          contentStyle={{
                            backgroundColor: 'rgba(13, 27, 42, 0.95)',
                            border: '1px solid rgba(0, 102, 255, 0.5)',
                            borderRadius: 8,
                            backdropFilter: 'blur(10px)'
                          }}
                        />
                        <Legend />
                        
                        <Line
                          yAxisId="left"
                          type="monotone"
                          dataKey="production_rate"
                          stroke={theme.palette.primary.main}
                          strokeWidth={3}
                          dot={false}
                          activeDot={{ r: 8 }}
                          name="Производительность (ед/ч)"
                        />
                        
                        <Line
                          yAxisId="right"
                          type="monotone"
                          dataKey="quality_rate"
                          stroke={theme.palette.success.main}
                          strokeWidth={3}
                          strokeDasharray="5 5"
                          dot={false}
                          activeDot={{ r: 8 }}
                          name="Качество (%)"
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  ) : (
                    <Box sx={{ maxHeight: 400, overflow: 'auto' }}>
                      <table style={{ width: '100%', borderCollapse: 'collapse', color: 'white' }}>
                        <thead>
                          <tr style={{ background: 'rgba(0, 102, 255, 0.2)' }}>
                            <th style={{ padding: '12px', textAlign: 'left' }}>Время</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>Производительность</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>Качество</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>OEE</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>Оператор</th>
                            <th style={{ padding: '12px', textAlign: 'left' }}>Смена</th>
                          </tr>
                        </thead>
                        <tbody>
                          {efficiencyData.slice(-20).map((item, index) => (
                            <tr key={index} style={{ 
                              borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
                              background: index % 2 ? 'rgba(255, 255, 255, 0.05)' : 'transparent'
                            }}>
                              <td style={{ padding: '12px' }}>
                                {new Date(item.timestamp).toLocaleTimeString('ru-RU')}
                              </td>
                              <td style={{ padding: '12px' }}>{item.production_rate.toFixed(1)} ед/ч</td>
                              <td style={{ padding: '12px' }}>{item.quality_rate.toFixed(1)}%</td>
                              <td style={{ padding: '12px' }}>
                                <Box sx={{ 
                                  display: 'inline-block',
                                  padding: '4px 8px',
                                  borderRadius: 1,
                                  background: item.oee > 85 ? 'rgba(0, 230, 118, 0.2)' :
                                            item.oee > 70 ? 'rgba(255, 215, 0, 0.2)' :
                                            'rgba(255, 23, 68, 0.2)',
                                  color: item.oee > 85 ? '#00E676' :
                                        item.oee > 70 ? '#FFD700' :
                                        '#FF1744',
                                  fontWeight: 'bold'
                                }}>
                                  {item.oee ? item.oee.toFixed(1) : '0.0'}%
                                </Box>
                              </td>
                              <td style={{ padding: '12px' }}>{item.operator_id}</td>
                              <td style={{ padding: '12px' }}>{item.shift}</td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </Box>
                  )}
                </Paper>
              </Grid>

              <Grid item xs={12}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                }}>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom>
                    Эффективность по сменам
                  </Typography>
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={getShiftEfficiency()}>
                      <CartesianGrid 
                        strokeDasharray="3 3" 
                        stroke="rgba(255, 255, 255, 0.1)"
                        vertical={false}
                      />
                      <XAxis
                        dataKey="shift"
                        stroke="rgba(255, 255, 255, 0.7)"
                        tickLine={false}
                        axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                      />
                      <YAxis 
                        stroke="rgba(255, 255, 255, 0.7)"
                        tickLine={false}
                        axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                      />
                      <Tooltip 
                        content={<CustomTooltip />}
                        contentStyle={{
                          backgroundColor: 'rgba(13, 27, 42, 0.95)',
                          border: '1px solid rgba(0, 102, 255, 0.5)',
                          borderRadius: 8,
                          backdropFilter: 'blur(10px)'
                        }}
                      />
                      <Legend />
                      <Bar
                        dataKey="efficiency"
                        fill={theme.palette.primary.main}
                        radius={[4, 4, 0, 0]}
                        name="Эффективность (%)"
                      />
                      <Bar
                        dataKey="oee"
                        fill={theme.palette.success.main}
                        radius={[4, 4, 0, 0]}
                        name="OEE (%)"
                      />
                    </BarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>
          )}

          {/* Tab 3: Parameter Correlations */}
          {activeTab === 2 && (
            <Paper elevation={0} sx={{ 
              p: 3,
              background: 'rgba(13, 27, 42, 0.6)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(0, 102, 255, 0.15)',
              borderRadius: 3,
            }}>
              <Stack direction="row" justifyContent="space-between" alignItems="flex-start" mb={3}>
                <Box>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom>
                    Корреляция параметров процесса с типами дефектов
                  </Typography>
                  <Typography variant="body2" color="text.secondary">
                    Анализ влияния параметров производства на возникновение дефектов
                  </Typography>
                </Box>
                <MuiTooltip title="Корреляция показывает взаимосвязь между параметрами и дефектами">
                  <IconButton size="small" sx={{ color: 'text.secondary' }}>
                    <InfoOutlined />
                  </IconButton>
                </MuiTooltip>
              </Stack>
              
              <Grid container spacing={2}>
                {Object.entries(correlations).map(([param, defectCorrs]) => {
                  const maxCorr = Math.max(...Object.values(defectCorrs).map(Math.abs));
                  
                  return (
                    <Grid item xs={12} md={6} lg={4} key={param}>
                      <Card sx={{ 
                        background: 'rgba(13, 27, 42, 0.8)',
                        border: '1px solid rgba(0, 102, 255, 0.2)',
                        borderRadius: 2,
                        transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
                        '&:hover': {
                          transform: 'translateY(-8px)',
                          boxShadow: '0 12px 40px rgba(0, 102, 255, 0.3)',
                          borderColor: theme.palette.primary.main,
                        },
                        height: '100%',
                      }}>
                        <CardContent>
                          <Stack direction="row" justifyContent="space-between" alignItems="center" mb={2}>
                            <Typography variant="subtitle1" fontWeight="bold" color="white">
                              {param.replace(/_/g, ' ').toUpperCase()}
                            </Typography>
                            <Chip
                              label={`${maxCorr.toFixed(2)}`}
                              size="small"
                              sx={{
                                backgroundColor: getCorrelationColor(maxCorr) + '20',
                                color: getCorrelationColor(maxCorr),
                                fontWeight: 'bold',
                                border: `1px solid ${getCorrelationColor(maxCorr)}`,
                              }}
                            />
                          </Stack>
                          
                          <Stack spacing={1}>
                            {Object.entries(defectCorrs).map(([defect, correlation]) => (
                              <Box key={defect}>
                                <Stack direction="row" justifyContent="space-between" alignItems="center" mb={0.5}>
                                  <Typography variant="body2" color="text.secondary">
                                    {defect}
                                  </Typography>
                                  <Typography
                                    variant="body2"
                                    fontWeight="bold"
                                    sx={{ 
                                      color: getCorrelationColor(correlation),
                                      textShadow: `0 0 10px ${getCorrelationColor(correlation)}40`
                                    }}
                                  >
                                    {correlation > 0 ? '+' : ''}{correlation.toFixed(2)}
                                  </Typography>
                                </Stack>
                                <Box sx={{ 
                                  width: '100%', 
                                  height: 6, 
                                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                                  borderRadius: 3,
                                  overflow: 'hidden'
                                }}>
                                  <Box
                                    sx={{
                                      width: `${Math.abs(correlation) * 100}%`,
                                      height: '100%',
                                      background: `linear-gradient(90deg, ${getCorrelationColor(correlation)}80, ${getCorrelationColor(correlation)})`,
                                      borderRadius: 3,
                                      transition: 'width 0.3s',
                                      boxShadow: `0 0 8px ${getCorrelationColor(correlation)}`
                                    }}
                                  />
                                </Box>
                              </Box>
                            ))}
                          </Stack>
                        </CardContent>
                      </Card>
                    </Grid>
                  );
                })}
              </Grid>
            </Paper>
          )}

          {/* Tab 4: Defect Distribution */}
          {activeTab === 3 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={6}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                  height: '100%'
                }}>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom>
                    Радиальное распределение дефектов
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <RadarChart data={getDefectDistributionRadar()}>
                      <PolarGrid stroke="rgba(255, 255, 255, 0.2)" />
                      <PolarAngleAxis
                        dataKey="defect"
                        stroke="rgba(255, 255, 255, 0.7)"
                      />
                      <PolarRadiusAxis stroke="rgba(255, 255, 255, 0.7)" />
                      <Radar
                        name="Процент"
                        dataKey="value"
                        stroke={theme.palette.primary.main}
                        fill={theme.palette.primary.main}
                        fillOpacity={0.4}
                        strokeWidth={2}
                      />
                      <Tooltip 
                        contentStyle={{
                          background: 'rgba(13, 27, 42, 0.95)',
                          border: `1px solid ${theme.palette.primary.main}50`,
                          borderRadius: 8,
                          backdropFilter: 'blur(10px)',
                        }}
                        formatter={(value) => [`${Number(value).toFixed(1)}%`, 'Доля']}
                      />
                    </RadarChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>

              <Grid item xs={12} md={6}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                  height: '100%'
                }}>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom>
                    Распределение дефектов по типам
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <PieChart>
                      <Pie
                        activeIndex={activePieIndex}
                        activeShape={renderActiveShape}
                        data={getDefectSummary()?.byType}
                        cx="50%"
                        cy="50%"
                        innerRadius={60}
                        outerRadius={100}
                        fill="#8884d8"
                        dataKey="value"
                        onMouseEnter={onPieEnter}
                      >
                        {getDefectSummary()?.byType.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.color}
                          />
                        ))}
                      </Pie>
                      <Tooltip 
                        formatter={(value) => [value, 'Количество']}
                        contentStyle={{
                          background: 'rgba(13, 27, 42, 0.95)',
                          border: `1px solid ${theme.palette.primary.main}50`,
                          borderRadius: 8,
                          backdropFilter: 'blur(10px)',
                        }}
                      />
                    </PieChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>
          )}

          {/* Tab 5: Heatmap */}
          {activeTab === 4 && (
            <Paper elevation={0} sx={{ 
              p: 3,
              background: 'rgba(13, 27, 42, 0.6)',
              backdropFilter: 'blur(20px)',
              border: '1px solid rgba(0, 102, 255, 0.15)',
              borderRadius: 3,
            }}>
              <Typography variant="h6" fontWeight="bold" color="white" gutterBottom>
                Тепловая карта параметров производства
              </Typography>
              {heatmapData.length > 0 ? (
                <ResponsiveContainer width="100%" height={500}>
                  <BarChart
                    data={getHeatmapChartData()}
                    margin={{ top: 20, right: 30, left: 20, bottom: 50 }}
                  >
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                    <XAxis 
                      dataKey="time" 
                      stroke="rgba(255, 255, 255, 0.7)"
                      tickLine={false}
                      axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                    />
                    <YAxis 
                      stroke="rgba(255, 255, 255, 0.7)"
                      tickLine={false}
                      axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                    />
                    <Tooltip 
                      contentStyle={{
                        background: 'rgba(13, 27, 42, 0.95)',
                        border: `1px solid ${theme.palette.primary.main}50`,
                        borderRadius: 8,
                        backdropFilter: 'blur(10px)',
                      }}
                    />
                    <Legend />
                    {heatmapData.length > 0 && Object.keys(getHeatmapChartData()[0] || {})
                      .filter(key => key !== 'time')
                      .map((param, index) => (
                        <Bar
                          key={param}
                          dataKey={param}
                          stackId="a"
                          fill={COLORS[index % COLORS.length]}
                          name={param}
                          radius={[2, 2, 0, 0]}
                        />
                      ))
                    }
                  </BarChart>
                </ResponsiveContainer>
              ) : (
                <Box sx={{ height: 500, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Typography color="text.secondary">Данные тепловой карты отсутствуют</Typography>
                </Box>
              )}
            </Paper>
          )}

          {/* Tab 6: Control Charts */}
          {activeTab === 5 && (
            <Grid container spacing={3}>
              <Grid item xs={12}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                }}>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom>
                    Контрольная карта процесса (X-bar)
                  </Typography>
                  <ResponsiveContainer width="100%" height={400}>
                    <LineChart data={controlChartData}>
                      <CartesianGrid 
                        strokeDasharray="3 3" 
                        stroke="rgba(255, 255, 255, 0.1)"
                      />
                      <XAxis
                        dataKey="timestamp"
                        tickFormatter={(value) => new Date(value).toLocaleTimeString('ru-RU', { hour: '2-digit' })}
                        stroke="rgba(255, 255, 255, 0.7)"
                        tickLine={false}
                        axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                      />
                      <YAxis 
                        stroke="rgba(255, 255, 255, 0.7)"
                        tickLine={false}
                        axisLine={{ stroke: 'rgba(0, 102, 255, 0.3)' }}
                      />
                      <Tooltip 
                        content={<CustomTooltip />}
                        contentStyle={{
                          backgroundColor: 'rgba(13, 27, 42, 0.95)',
                          border: '1px solid rgba(0, 102, 255, 0.5)',
                          borderRadius: 8,
                          backdropFilter: 'blur(10px)'
                        }}
                      />
                      <Legend />
                      
                      <ReferenceLine y={controlChartData[0]?.cl} stroke={theme.palette.success.main} strokeDasharray="5 5" />
                      <ReferenceLine y={controlChartData[0]?.ucl} stroke={theme.palette.error.main} strokeDasharray="3 3" />
                      <ReferenceLine y={controlChartData[0]?.lcl} stroke={theme.palette.warning.main} strokeDasharray="3 3" />
                      
                      <Line
                        type="monotone"
                        dataKey="value"
                        stroke={theme.palette.primary.main}
                        strokeWidth={2}
                        dot={{ r: 4 }}
                        activeDot={{ r: 8 }}
                        name="Измеренное значение"
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </Paper>
              </Grid>
            </Grid>
          )}

          {/* Tab 7: KPI и метрики */}
          {activeTab === 6 && (
            <Grid container spacing={3}>
              <Grid item xs={12} md={8}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                }}>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom mb={3}>
                    Ключевые показатели эффективности (KPI)
                  </Typography>
                  
                  {getKPIMetrics() && (
                    <Grid container spacing={3}>
                      <Grid item xs={12} md={6}>
                        <Card sx={{ 
                          p: 2,
                          background: 'linear-gradient(135deg, rgba(0, 102, 255, 0.1) 0%, rgba(0, 229, 255, 0.05) 100%)',
                          border: '1px solid rgba(0, 102, 255, 0.3)',
                          height: '100%'
                        }}>
                          <Stack direction="row" alignItems="center" spacing={2} mb={2}>
                            <Speed sx={{ color: theme.palette.primary.main, fontSize: 40 }} />
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Общая эффективность оборудования (OEE)
                              </Typography>
                              <Typography variant="h3" fontWeight="bold" color="white">
                                {getKPIMetrics()?.oee.toFixed(1)}%
                              </Typography>
                            </Box>
                          </Stack>
                          <LinearProgress 
                            variant="determinate" 
                            value={getKPIMetrics()?.oee || 0}
                            sx={{ 
                              height: 8, 
                              borderRadius: 4,
                              bgcolor: 'rgba(255, 255, 255, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                background: 'linear-gradient(90deg, #0066FF, #00E5FF)'
                              }
                            }}
                          />
                        </Card>
                      </Grid>

                      <Grid item xs={12} md={6}>
                        <Card sx={{ 
                          p: 2,
                          background: 'linear-gradient(135deg, rgba(0, 230, 118, 0.1) 0%, rgba(0, 200, 83, 0.05) 100%)',
                          border: '1px solid rgba(0, 230, 118, 0.3)',
                          height: '100%'
                        }}>
                          <Stack direction="row" alignItems="center" spacing={2} mb={2}>
                            <AutoGraph sx={{ color: theme.palette.success.main, fontSize: 40 }} />
                            <Box>
                              <Typography variant="body2" color="text.secondary">
                                Качество продукции
                              </Typography>
                              <Typography variant="h3" fontWeight="bold" color="white">
                                {getKPIMetrics()?.quality.toFixed(1)}%
                              </Typography>
                            </Box>
                          </Stack>
                          <LinearProgress 
                            variant="determinate" 
                            value={getKPIMetrics()?.quality || 0}
                            sx={{ 
                              height: 8, 
                              borderRadius: 4,
                              bgcolor: 'rgba(255, 255, 255, 0.1)',
                              '& .MuiLinearProgress-bar': {
                                background: 'linear-gradient(90deg, #00E676, #00C853)'
                              }
                            }}
                          />
                        </Card>
                      </Grid>

                      <Grid item xs={12} md={4}>
                        <Card sx={{ 
                          p: 2,
                          background: 'rgba(13, 27, 42, 0.8)',
                          border: '1px solid rgba(255, 215, 0, 0.3)',
                          height: '100%'
                        }}>
                          <Stack alignItems="center" spacing={1}>
                            <Engineering sx={{ color: theme.palette.warning.main, fontSize: 30 }} />
                            <Typography variant="body2" color="text.secondary" align="center">
                              Доступность
                            </Typography>
                            <Typography variant="h4" fontWeight="bold" color="white">
                              {getKPIMetrics()?.availability.toFixed(1)}%
                            </Typography>
                          </Stack>
                        </Card>
                      </Grid>

                      <Grid item xs={12} md={4}>
                        <Card sx={{ 
                          p: 2,
                          background: 'rgba(13, 27, 42, 0.8)',
                          border: '1px solid rgba(255, 51, 102, 0.3)',
                          height: '100%'
                        }}>
                          <Stack alignItems="center" spacing={1}>
                            <PrecisionManufacturing sx={{ color: theme.palette.secondary.main, fontSize: 30 }} />
                            <Typography variant="body2" color="text.secondary" align="center">
                              Производительность
                            </Typography>
                            <Typography variant="h4" fontWeight="bold" color="white">
                              {getKPIMetrics()?.performance.toFixed(1)}%
                            </Typography>
                          </Stack>
                        </Card>
                      </Grid>

                      <Grid item xs={12} md={4}>
                        <Card sx={{ 
                          p: 2,
                          background: 'rgba(13, 27, 42, 0.8)',
                          border: '1px solid rgba(139, 92, 246, 0.3)',
                          height: '100%'
                        }}>
                          <Stack alignItems="center" spacing={1}>
                            <LocalFireDepartment sx={{ color: theme.palette.info.main, fontSize: 30 }} />
                            <Typography variant="body2" color="text.secondary" align="center">
                              Уровень дефектов
                            </Typography>
                            <Typography variant="h4" fontWeight="bold" color="white">
                              {getKPIMetrics()?.defectRate.toFixed(2)}%
                            </Typography>
                          </Stack>
                        </Card>
                      </Grid>

                      <Grid item xs={12}>
                        <Card sx={{ 
                          p: 3,
                          background: 'linear-gradient(135deg, rgba(13, 27, 42, 0.8) 0%, rgba(0, 29, 61, 0.6) 100%)',
                          border: '1px solid rgba(0, 102, 255, 0.3)',
                        }}>
                          <Typography variant="h6" color="white" gutterBottom>
                            Производственные метрики
                          </Typography>
                          <Grid container spacing={2}>
                            <Grid item xs={6} md={3}>
                              <Stack alignItems="center">
                                <Typography variant="body2" color="text.secondary">
                                  Целевое производство
                                </Typography>
                                <Typography variant="h5" fontWeight="bold" color="white">
                                  {productionMetrics?.target_production || 0} ед.
                                </Typography>
                              </Stack>
                            </Grid>
                            <Grid item xs={6} md={3}>
                              <Stack alignItems="center">
                                <Typography variant="body2" color="text.secondary">
                                  Фактическое производство
                                </Typography>
                                <Typography variant="h5" fontWeight="bold" color={theme.palette.success.main}>
                                  {productionMetrics?.actual_production || 0} ед.
                                </Typography>
                              </Stack>
                            </Grid>
                            <Grid item xs={6} md={3}>
                              <Stack alignItems="center">
                                <Typography variant="body2" color="text.secondary">
                                  Качественные единицы
                                </Typography>
                                <Typography variant="h5" fontWeight="bold" color="white">
                                  {productionMetrics?.good_units || 0} ед.
                                </Typography>
                              </Stack>
                            </Grid>
                            <Grid item xs={6} md={3}>
                              <Stack alignItems="center">
                                <Typography variant="body2" color="text.secondary">
                                  Простой
                                </Typography>
                                <Typography variant="h5" fontWeight="bold" color={theme.palette.error.main}>
                                  {productionMetrics?.downtime || 0} мин.
                                </Typography>
                              </Stack>
                            </Grid>
                          </Grid>
                        </Card>
                      </Grid>
                    </Grid>
                  )}
                </Paper>
              </Grid>

              {/* Real-time metrics panel */}
              <Grid item xs={12}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                  height: '100%'
                }}>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom mb={3}>
                    Метрики в реальном времени
                  </Typography>
                  
                  <Grid container spacing={3}>
                    {/* Example real-time metrics */}
                    <Grid item xs={12} sm={6} md={3}>
                      <Card sx={{ p: 2, background: 'rgba(13, 27, 42, 0.8)', border: '1px solid rgba(0, 102, 255, 0.3)' }}>
                        <Stack direction="row" alignItems="center" spacing={2} mb={2}>
                          <Box sx={{ width: 48, height: 48, borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(135deg, #0066FF, #00E5FF)' }}>
                            <LocalFireDepartment sx={{ color: '#FFFFFF', fontSize: 24 }} />
                          </Box>
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              Температура печи
                            </Typography>
                            <Typography variant="h4" fontWeight="bold" color="white">
                              1520°C
                            </Typography>
                          </Box>
                        </Stack>
                        <LinearProgress variant="determinate" value={76} />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                          Норма: 1400-1600°C
                        </Typography>
                      </Card>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={3}>
                      <Card sx={{ p: 2, background: 'rgba(13, 27, 42, 0.8)', border: '1px solid rgba(255, 51, 102, 0.3)' }}>
                        <Stack direction="row" alignItems="center" spacing={2} mb={2}>
                          <Box sx={{ width: 48, height: 48, borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(135deg, #FF3366, #FF9E6D)' }}>
                            <Speed sx={{ color: '#FFFFFF', fontSize: 24 }} />
                          </Box>
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              Скорость ленты
                            </Typography>
                            <Typography variant="h4" fontWeight="bold" color="white">
                              150 м/мин
                            </Typography>
                          </Box>
                        </Stack>
                        <LinearProgress variant="determinate" value={75} />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                          Норма: 120-200 м/мин
                        </Typography>
                      </Card>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={3}>
                      <Card sx={{ p: 2, background: 'rgba(13, 27, 42, 0.8)', border: '1px solid rgba(0, 230, 118, 0.3)' }}>
                        <Stack direction="row" alignItems="center" spacing={2} mb={2}>
                          <Box sx={{ width: 48, height: 48, borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(135deg, #00E676, #6EFFB2)' }}>
                            <DeviceHub sx={{ color: '#FFFFFF', fontSize: 24 }} />
                          </Box>
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              Давление
                            </Typography>
                            <Typography variant="h4" fontWeight="bold" color="white">
                              15.2 кПа
                            </Typography>
                          </Box>
                        </Stack>
                        <LinearProgress variant="determinate" value={30} />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                          Норма: 0-50 кПа
                        </Typography>
                      </Card>
                    </Grid>
                    
                    <Grid item xs={12} sm={6} md={3}>
                      <Card sx={{ p: 2, background: 'rgba(13, 27, 42, 0.8)', border: '1px solid rgba(157, 78, 221, 0.3)' }}>
                        <Stack direction="row" alignItems="center" spacing={2} mb={2}>
                          <Box sx={{ width: 48, height: 48, borderRadius: '12px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'linear-gradient(135deg, #9D4EDD, #E0AAFF)' }}>
                            <Psychology sx={{ color: '#FFFFFF', fontSize: 24 }} />
                          </Box>
                          <Box>
                            <Typography variant="body2" color="text.secondary">
                              Вероятность дефектов
                            </Typography>
                            <Typography variant="h4" fontWeight="bold" color="white">
                              12%
                            </Typography>
                          </Box>
                        </Stack>
                        <LinearProgress variant="determinate" value={12} />
                        <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                          Прогноз ИИ
                        </Typography>
                      </Card>
                    </Grid>
                  </Grid>
                </Paper>
              </Grid>

              <Grid item xs={12} md={4}>
                <Paper elevation={0} sx={{ 
                  p: 3,
                  background: 'rgba(13, 27, 42, 0.6)',
                  backdropFilter: 'blur(20px)',
                  border: '1px solid rgba(0, 102, 255, 0.15)',
                  borderRadius: 3,
                  height: '100%'
                }}>
                  <Typography variant="h6" fontWeight="bold" color="white" gutterBottom>
                    Сводка по дефектам
                  </Typography>
                  <Grid container spacing={2}>
                    {getDefectSummary()?.byType.map((defect, index) => (
                      <Grid item xs={12} sm={6} md={4} key={defect.name}>
                        <Card sx={{ 
                          p: 2,
                          background: `linear-gradient(135deg, ${defect.color}20 0%, ${defect.color}10 100%)`,
                          border: `1px solid ${defect.color}40`,
                          transition: 'transform 0.3s',
                          '&:hover': {
                            transform: 'translateY(-4px)',
                          }
                        }}>
                          <Stack direction="row" justifyContent="space-between" alignItems="center">
                            <Stack direction="row" alignItems="center" spacing={2}>
                              <Box sx={{ 
                                width: 40, 
                                height: 40, 
                                borderRadius: '50%', 
                                bgcolor: defect.color,
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                              }}>
                                <Checklist sx={{ color: 'white', fontSize: 20 }} />
                              </Box>
                              <Box>
                                <Typography variant="body1" fontWeight="bold" color="white">
                                  {defect.name}
                                </Typography>
                                <Typography variant="caption" color="text.secondary">
                                  {((defect.value / (getDefectSummary()?.total || 1)) * 100).toFixed(1)}% от общего числа
                                </Typography>
                              </Box>
                            </Stack>
                            <Typography variant="h4" fontWeight="bold" color={defect.color}>
                              {defect.value}
                            </Typography>
                          </Stack>
                        </Card>
                      </Grid>
                    ))}
                  </Grid>
                </Paper>
              </Grid>
            </Grid>
          )}
        </>
      )}
    </Box>
  );
};

export default AnalyticsDashboard;