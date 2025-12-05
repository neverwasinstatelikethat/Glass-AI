/**
 * ExplainabilityPanel Component - Phase 6
 * Displays SHAP feature attributions and model explanations
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  LinearProgress,
  Tooltip,
  Skeleton,
  Alert,
  Grid,
  Paper,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Psychology,
  TrendingUp,
  TrendingDown,
  Insights,
  AutoAwesome,
} from '@mui/icons-material';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip as RechartsTooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts';
import { explainabilityApi, FeatureAttribution } from '../services/pipelineApi';

const GlassCard = styled(Card)(({ theme }) => ({
  background: 'rgba(13, 27, 42, 0.4)',
  backdropFilter: 'blur(20px) saturate(180%)',
  border: '1px solid rgba(0, 102, 255, 0.2)',
  borderRadius: '24px',
  padding: theme.spacing(3),
  transition: 'all 0.3s ease',
  '&:hover': {
    borderColor: 'rgba(0, 229, 255, 0.5)',
    boxShadow: '0 8px 32px rgba(0, 102, 255, 0.2)',
  },
}));

const FeatureBar = styled(Box)(({ theme }) => ({
  display: 'flex',
  alignItems: 'center',
  marginBottom: theme.spacing(2),
  padding: theme.spacing(1.5),
  background: 'rgba(255, 255, 255, 0.03)',
  borderRadius: '12px',
  transition: 'all 0.2s ease',
  '&:hover': {
    background: 'rgba(255, 255, 255, 0.06)',
  },
}));

interface ExplainabilityPanelProps {
  modelName?: string;
  refreshInterval?: number; // milliseconds
}

const ExplainabilityPanel: React.FC<ExplainabilityPanelProps> = ({ 
  modelName = 'lstm',
  refreshInterval = 10000 
}) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [topFeatures, setTopFeatures] = useState<FeatureAttribution[]>([]);
  const [timestamp, setTimestamp] = useState<string>('');

  const fetchExplanation = async () => {
    try {
      setLoading(true);
      const data = await explainabilityApi.getPredictionExplanation(modelName);
      
      if (data.explanations && data.explanations.top_features) {
        setTopFeatures(data.explanations.top_features);
      } else if (data.explanations && data.explanations.shap_values) {
        // Sort SHAP values by absolute importance
        const sorted = [...data.explanations.shap_values]
          .sort((a, b) => Math.abs(b.importance) - Math.abs(a.importance))
          .slice(0, 10);
        setTopFeatures(sorted);
      }
      
      setTimestamp(data.timestamp);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch explanations:', err);
      setError('Failed to load explainability data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchExplanation();
    const interval = setInterval(fetchExplanation, refreshInterval);
    return () => clearInterval(interval);
  }, [modelName, refreshInterval]);

  const getImportanceColor = (importance: number) => {
    if (isNaN(importance)) return '#00E676'; // Default color for NaN values
    const absImportance = Math.abs(importance);
    if (absImportance > 0.7) return '#FF3366';
    if (absImportance > 0.4) return '#FFD700';
    if (absImportance > 0.2) return '#00E5FF';
    return '#00E676';
  };

  const formatFeatureName = (name: string) => {
    if (!name) return 'Unknown Feature';
    return name
      .split('_')
      .map(word => word.charAt(0).toUpperCase() + word.slice(1))
      .join(' ');
  };

  if (loading && topFeatures.length === 0) {
    return (
      <GlassCard>
        <CardContent>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '16px' }}>
            <Psychology sx={{ mr: 1, color: '#00E5FF' }} />
            <Typography variant="h6" color="white">
              Model Explainability
            </Typography>
          </div>
          <Skeleton variant="rectangular" height={300} sx={{ borderRadius: 2 }} />
        </CardContent>
      </GlassCard>
    );
  }
  if (error) {
    return (
      <GlassCard>
        <CardContent>
          <Alert severity="error">{error}</Alert>
        </CardContent>
      </GlassCard>
    );
  }

  const chartData = topFeatures
    .filter(feature => feature.importance !== undefined && !isNaN(feature.importance))
    .map(feature => ({
      name: formatFeatureName(feature.feature_name),
      importance: feature.importance,
      absImportance: Math.abs(feature.importance),
    }));
  return (
    <GlassCard>
      <CardContent>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '24px' }}>
          <div style={{ display: 'flex', alignItems: 'center' }}>
            <Psychology sx={{ mr: 1, color: '#00E5FF', fontSize: 32 }} />
            <div>
              <Typography variant="h5" color="white" fontWeight="bold">
                Model Explainability
              </Typography>
              <Typography variant="caption" color="rgba(255, 255, 255, 0.6)">
                SHAP Feature Importance Analysis
              </Typography>
            </div>
          </div>
          <Chip
            icon={<Insights />}
            label={`Model: ${modelName.toUpperCase()}`}
            sx={{
              background: 'linear-gradient(135deg, #0066FF 0%, #00E5FF 100%)',
              color: 'white',
              fontWeight: 'bold',
            }}
          />
        </div>
        {/* Top Features Bar Chart */}
        <div style={{ marginBottom: '32px' }}>
          {chartData && chartData.length > 0 ? (
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={chartData} layout="horizontal">
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255, 255, 255, 0.1)" />
                <XAxis 
                  type="number" 
                  stroke="rgba(255, 255, 255, 0.6)"
                  domain={[-1, 1]}
                  tickFormatter={(value) => {
                    if (isNaN(value)) return '0';
                    return value.toString();
                  }}
                />
                <YAxis 
                  type="category" 
                  dataKey="name" 
                  stroke="rgba(255, 255, 255, 0.6)"
                  width={150}
                  tickFormatter={(value) => {
                    if (!value) return 'Unknown';
                    return value.toString();
                  }}
                />
                <RechartsTooltip
                  contentStyle={{
                    background: 'rgba(13, 27, 42, 0.95)',
                    border: '1px solid rgba(0, 229, 255, 0.3)',
                    borderRadius: '8px',
                    color: 'white',
                  }}
                />
                <Bar dataKey="importance" radius={[0, 8, 8, 0]}>
                  {chartData.map((entry, index) => (
                    <Cell 
                      key={`cell-${index}`} 
                      fill={getImportanceColor(isNaN(entry.importance) ? 0 : entry.importance)} 
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          ) : (
            <div style={{ height: '300px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'rgba(255, 255, 255, 0.6)' }}>
              No feature importance data available
            </div>
          )}
        </div>
        {/* Feature Details List */}
        <div>
          <Typography variant="subtitle2" color="rgba(255, 255, 255, 0.7)" style={{ marginBottom: '16px' }}>
            Top Contributing Features
          </Typography>
          <Grid container spacing={2}>
            {topFeatures.slice(0, 6).map((feature, index) => (
              <Grid item xs={12} sm={6} key={index}>
                <FeatureBar>
                  <div style={{ flex: 1 }}>
                    <div style={{ display: 'flex', alignItems: 'center', marginBottom: '4px' }}>
                      {feature.importance > 0 ? (
                        <TrendingUp sx={{ fontSize: 16, color: '#00E676', mr: 0.5 }} />
                      ) : (
                        <TrendingDown sx={{ fontSize: 16, color: '#FF3366', mr: 0.5 }} />
                      )}
                      <Typography variant="body2" color="white" fontWeight="500">
                        {formatFeatureName(feature.feature_name)}
                      </Typography>
                    </div>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                      <LinearProgress
                        variant="determinate"
                        value={isNaN(Math.abs(feature.importance) * 100) ? 0 : Math.abs(feature.importance) * 100}
                        sx={{
                          flex: 1,
                          height: 6,
                          borderRadius: 3,
                          backgroundColor: 'rgba(255, 255, 255, 0.1)',
                          '& .MuiLinearProgress-bar': {
                            background: getImportanceColor(feature.importance),
                            borderRadius: 3,
                          },
                        }}
                      />
                      <Typography
                        variant="caption"
                        fontWeight="bold"
                        sx={{
                          color: getImportanceColor(feature.importance),
                          minWidth: 45,
                          textAlign: 'right',
                        }}
                      >
                        {isNaN(Math.abs(feature.importance) * 100) ? '0.0%' : (Math.abs(feature.importance) * 100).toFixed(1) + '%'}
                      </Typography>
                    </div>
                  </div>
                </FeatureBar>
              </Grid>
            ))}
          </Grid>
        </div>
        {/* Timestamp */}
        {timestamp && (
          <div style={{ marginTop: '24px', textAlign: 'center' }}>
            <Typography variant="caption" color="rgba(255, 255, 255, 0.5)">
              Last updated: {new Date(timestamp).toLocaleTimeString()}
            </Typography>
          </div>
        )}
      </CardContent>
    </GlassCard>
  );
};

export default ExplainabilityPanel;
