import React, { useState, useEffect } from 'react';
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
  CircularProgress,
  Alert,
  Chip
} from '@mui/material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from 'recharts';
import { useTheme } from '@mui/material/styles';
import { Psychology, TrendingUp, TrendingDown } from '@mui/icons-material';

const API_BASE_URL = 'http://localhost:8000';

interface FeatureImportance {
  feature: string;
  importance: number;
  category: 'thermal' | 'mechanical' | 'chemical' | 'visual' | 'network' | 'temporal';
}

interface ShapValue {
  feature: string;
  value: number;
  shap_value: number;
}

interface ShapData {
  prediction_id: string;
  timestamp: string;
  base_value: number;
  predicted_value: number;
  shap_values: ShapValue[];
  force_plot_data: {
    pushing_higher: string[];
    pushing_lower: string[];
  };
}

const ModelExplainability: React.FC = () => {
  const theme = useTheme();
  
  const [modelType, setModelType] = useState<'lstm' | 'gnn' | 'vit'>('lstm');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  const [featureImportance, setFeatureImportance] = useState<FeatureImportance[]>([]);
  const [shapData, setShapData] = useState<ShapData | null>(null);

  // Fetch feature importance
  const fetchFeatureImportance = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(
        `${API_BASE_URL}/api/explainability/feature-importance?model_type=${modelType}`
      );
      
      if (!response.ok) {
        throw new Error('Failed to fetch feature importance');
      }
      
      const data = await response.json();
      setFeatureImportance(data.features || []);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error fetching feature importance:', err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch SHAP values
  const fetchShapValues = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`${API_BASE_URL}/api/explainability/shap-values`);
      
      if (!response.ok) {
        throw new Error('Failed to fetch SHAP values');
      }
      
      const data = await response.json();
      setShapData(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error');
      console.error('Error fetching SHAP values:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchFeatureImportance();
    fetchShapValues();
  }, [modelType]);

  const getCategoryColor = (category: string) => {
    const colors: Record<string, string> = {
      thermal: theme.palette.error.main,
      mechanical: theme.palette.info.main,
      chemical: theme.palette.warning.main,
      visual: theme.palette.success.main,
      network: theme.palette.primary.main,
      temporal: theme.palette.secondary.main
    };
    return colors[category] || theme.palette.grey[500];
  };

  const getShapColor = (value: number) => {
    return value > 0 ? theme.palette.error.main : theme.palette.success.main;
  };

  return (
    <Box>
      {/* Header */}
      <Stack direction="row" justifyContent="space-between" alignItems="center" mb={3}>
        <Stack direction="row" alignItems="center" spacing={2}>
          <Psychology sx={{ fontSize: 40, color: theme.palette.primary.main }} />
          <Typography variant="h5" fontWeight="bold">
            Объяснимость моделей
          </Typography>
        </Stack>
        
        <FormControl sx={{ minWidth: 180 }}>
          <InputLabel>Модель</InputLabel>
          <Select
            value={modelType}
            label="Модель"
            onChange={(e) => setModelType(e.target.value as 'lstm' | 'gnn' | 'vit')}
          >
            <MenuItem value="lstm">LSTM (Временные ряды)</MenuItem>
            <MenuItem value="gnn">GNN (Сенсорная сеть)</MenuItem>
            <MenuItem value="vit">ViT (Визуальная классификация)</MenuItem>
          </Select>
        </FormControl>
      </Stack>

      {/* Error Alert */}
      {error && (
        <Alert severity="error" sx={{ mb: 2 }} onClose={() => setError(null)}>
          {error}
        </Alert>
      )}

      {/* Loading Indicator */}
      {loading && (
        <Box sx={{ display: 'flex', justifyContent: 'center', p: 4 }}>
          <CircularProgress />
        </Box>
      )}

      {/* Content */}
      {!loading && (
        <Grid container spacing={3}>
          {/* Feature Importance */}
          <Grid item xs={12} lg={7}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Важность признаков (SHAP Analysis)
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Чем выше значение, тем больше влияние признака на предсказание модели
              </Typography>
              
              <ResponsiveContainer width="100%" height={400}>
                <BarChart
                  data={featureImportance}
                  layout="vertical"
                  margin={{ top: 5, right: 30, left: 120, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" stroke={theme.palette.divider} />
                  <XAxis
                    type="number"
                    stroke={theme.palette.text.secondary}
                    domain={[0, 'auto']}
                  />
                  <YAxis
                    type="category"
                    dataKey="feature"
                    stroke={theme.palette.text.secondary}
                    width={110}
                    tick={{ fontSize: 12 }}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: theme.palette.background.paper,
                      border: `1px solid ${theme.palette.divider}`
                    }}
                  />
                  <Bar dataKey="importance" name="Важность">
                    {featureImportance.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={getCategoryColor(entry.category)} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>

              <Box sx={{ mt: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  Категории признаков:
                </Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {Array.from(new Set(featureImportance.map(f => f.category))).map(category => (
                    <Chip
                      key={category}
                      label={category}
                      size="small"
                      sx={{
                        backgroundColor: `${getCategoryColor(category)}30`,
                        color: getCategoryColor(category),
                        fontWeight: 'bold'
                      }}
                    />
                  ))}
                </Stack>
              </Box>
            </Paper>
          </Grid>

          {/* SHAP Force Plot */}
          <Grid item xs={12} lg={5}>
            <Paper sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                SHAP Force Plot (Последний прогноз)
              </Typography>
              <Typography variant="body2" color="text.secondary" gutterBottom>
                Как признаки влияют на конкретное предсказание
              </Typography>

              {shapData && (
                <Box>
                  {/* Prediction Summary */}
                  <Card sx={{ mb: 2, backgroundColor: theme.palette.background.default }}>
                    <CardContent>
                      <Stack spacing={2}>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Базовая вероятность дефекта
                          </Typography>
                          <Typography variant="h5" fontWeight="bold">
                            {(shapData.base_value * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Прогнозируемая вероятность
                          </Typography>
                          <Typography
                            variant="h5"
                            fontWeight="bold"
                            sx={{ color: shapData.predicted_value > 0.5 ? theme.palette.error.main : theme.palette.success.main }}
                          >
                            {(shapData.predicted_value * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                        <Box>
                          <Typography variant="caption" color="text.secondary">
                            Вклад признаков
                          </Typography>
                          <Typography
                            variant="h5"
                            fontWeight="bold"
                            sx={{ color: (shapData.predicted_value - shapData.base_value) > 0 ? theme.palette.error.main : theme.palette.success.main }}
                          >
                            {((shapData.predicted_value - shapData.base_value) * 100 > 0 ? '+' : '')}
                            {((shapData.predicted_value - shapData.base_value) * 100).toFixed(1)}%
                          </Typography>
                        </Box>
                      </Stack>
                    </CardContent>
                  </Card>

                  {/* SHAP Values List */}
                  <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                    Топ признаков по влиянию:
                  </Typography>
                  <Stack spacing={1}>
                    {shapData.shap_values.slice(0, 7).map((sv, index) => (
                      <Box
                        key={index}
                        sx={{
                          display: 'flex',
                          justifyContent: 'space-between',
                          alignItems: 'center',
                          p: 1.5,
                          borderRadius: 1,
                          backgroundColor: `${getShapColor(sv.shap_value)}15`,
                          borderLeft: `4px solid ${getShapColor(sv.shap_value)}`
                        }}
                      >
                        <Box>
                          <Typography variant="body2" fontWeight="bold">
                            {sv.feature.replace(/_/g, ' ')}
                          </Typography>
                          <Typography variant="caption" color="text.secondary">
                            Значение: {sv.value}
                          </Typography>
                        </Box>
                        <Stack direction="row" alignItems="center" spacing={0.5}>
                          {sv.shap_value > 0 ? (
                            <TrendingUp sx={{ color: getShapColor(sv.shap_value), fontSize: 20 }} />
                          ) : (
                            <TrendingDown sx={{ color: getShapColor(sv.shap_value), fontSize: 20 }} />
                          )}
                          <Typography
                            variant="body2"
                            fontWeight="bold"
                            sx={{ color: getShapColor(sv.shap_value) }}
                          >
                            {sv.shap_value > 0 ? '+' : ''}{sv.shap_value.toFixed(3)}
                          </Typography>
                        </Stack>
                      </Box>
                    ))}
                  </Stack>

                  <Typography variant="caption" color="text.secondary" sx={{ mt: 2, display: 'block' }}>
                    ID прогноза: {shapData.prediction_id}
                    <br />
                    Время: {new Date(shapData.timestamp).toLocaleString('ru-RU')}
                  </Typography>
                </Box>
              )}
            </Paper>
          </Grid>
        </Grid>
      )}
    </Box>
  );
};

export default ModelExplainability;
