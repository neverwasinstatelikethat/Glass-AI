/**
 * MetricsMonitor Component - Phase 7
 * Displays pipeline performance metrics and system health
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Grid,
  Box,
  Chip,
  LinearProgress,
  Alert,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  Speed,
  CheckCircle,
  Error,
  Timeline,
  AccessTime,
} from '@mui/icons-material';
import { metricsApi, PipelineMetrics } from '../services/pipelineApi';

const GlassCard = styled(Card)(() => ({
  background: 'rgba(13, 27, 42, 0.4)',
  backdropFilter: 'blur(20px) saturate(180%)',
  border: '1px solid rgba(0, 102, 255, 0.2)',
  borderRadius: '24px',
  padding: '24px',
  transition: 'all 0.3s ease',
  '&:hover': {
    borderColor: 'rgba(0, 229, 255, 0.5)',
    boxShadow: '0 8px 32px rgba(0, 102, 255, 0.2)',
  },
}));

const MetricCard = styled(Box)(() => ({
  padding: '16px',
  background: 'rgba(255, 255, 255, 0.03)',
  borderRadius: '12px',
  border: '1px solid rgba(255, 255, 255, 0.1)',
}));

interface MetricsMonitorProps {
  refreshInterval?: number;
}

const MetricsMonitor: React.FC<MetricsMonitorProps> = ({ refreshInterval = 5000 }) => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [metrics, setMetrics] = useState<PipelineMetrics | null>(null);

  const fetchMetrics = async () => {
    try {
      const data = await metricsApi.getPipelineMetrics();
      setMetrics(data.pipeline_metrics);
      setError(null);
    } catch (err) {
      console.error('Failed to fetch metrics:', err);
      setError('Failed to load metrics');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchMetrics();
    const interval = setInterval(fetchMetrics, refreshInterval);
    return () => clearInterval(interval);
  }, [refreshInterval]);

  if (error) {
    return (
      <GlassCard>
        <Alert severity="error">{error}</Alert>
      </GlassCard>
    );
  }

  const successRate = metrics 
    ? ((metrics.successful_predictions / (metrics.successful_predictions + metrics.failed_predictions || 1)) * 100) || 0
    : 0;

  return (
    <GlassCard>
      <CardContent>
        <div style={{ display: 'flex', alignItems: 'center', marginBottom: '24px' }}>
          <Speed sx={{ mr: 1, color: '#00E5FF', fontSize: 32 }} />
          <div>
            <Typography variant="h5" sx={{ color: 'white', fontWeight: 'bold' }}>
              Pipeline Metrics
            </Typography>
            <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
              Real-time Performance Monitoring
            </Typography>
          </div>
        </div>
        <Grid container spacing={2}>
          <Grid item xs={12} sm={6} md={3}>
            <MetricCard>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                <Timeline sx={{ color: '#00E676', mr: 1 }} />
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Executions
                </Typography>
              </div>
              <Typography variant="h4" sx={{ color: 'white', fontWeight: 'bold' }}>
                {metrics?.pipeline_executions || 0}
              </Typography>
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <MetricCard>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                <CheckCircle sx={{ color: '#00E676', mr: 1 }} />
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Success Rate
                </Typography>
              </div>
              <Typography variant="h4" sx={{ color: '#00E676', fontWeight: 'bold' }}>
                {isNaN(successRate) ? '0.0%' : successRate.toFixed(1) + '%'}
              </Typography>
              <LinearProgress
                variant="determinate"
                value={isNaN(successRate) ? 0 : successRate}
                sx={{
                  mt: 1,
                  height: 4,
                  borderRadius: 2,
                  backgroundColor: 'rgba(255, 255, 255, 0.1)',
                  '& .MuiLinearProgress-bar': {
                    background: 'linear-gradient(90deg, #00E676 0%, #00E5FF 100%)',
                  },
                }}
              />
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <MetricCard>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                <AccessTime sx={{ color: '#FFD700', mr: 1 }} />
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Avg Latency
                </Typography>
              </div>
              <Typography variant="h4" sx={{ color: 'white', fontWeight: 'bold' }}>
                {metrics?.avg_latency_ms ? (isNaN(metrics.avg_latency_ms) ? '0' : metrics.avg_latency_ms.toFixed(0)) : '0'}
                <Typography component="span" variant="caption" sx={{ ml: 1, color: 'rgba(255, 255, 255, 0.6)' }}>
                  ms
                </Typography>
              </Typography>
            </MetricCard>
          </Grid>

          <Grid item xs={12} sm={6} md={3}>
            <MetricCard>
              <div style={{ display: 'flex', alignItems: 'center', marginBottom: '8px' }}>
                <Error sx={{ color: '#FF3366', mr: 1 }} />
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.7)' }}>
                  Failed
                </Typography>
              </div>
              <Typography variant="h4" sx={{ color: '#FF3366', fontWeight: 'bold' }}>
                {metrics?.failed_predictions || 0}
              </Typography>
            </MetricCard>
          </Grid>
        </Grid>

        <div style={{ marginTop: '24px' }}>
          <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 2 }}>
            Stage Performance
          </Typography>
          <Grid container spacing={2}>
            <Grid item xs={12} md={4}>
              <div style={{ padding: '16px', background: 'rgba(0, 102, 255, 0.1)', borderRadius: '8px' }}>
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                  Feature Extraction
                </Typography>
                <Typography variant="h6" sx={{ color: '#00E5FF', fontWeight: 'bold' }}>
                  {metrics?.feature_extraction_time_ms ? (isNaN(metrics.feature_extraction_time_ms) ? '0.0' : metrics.feature_extraction_time_ms.toFixed(1)) : '0.0'} ms
                </Typography>
              </div>
            </Grid>
            <Grid item xs={12} md={4}>
              <div style={{ padding: '16px', background: 'rgba(0, 230, 118, 0.1)', borderRadius: '8px' }}>
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                  Prediction
                </Typography>
                <Typography variant="h6" sx={{ color: '#00E676', fontWeight: 'bold' }}>
                  {metrics?.prediction_time_ms ? (isNaN(metrics.prediction_time_ms) ? '0.0' : metrics.prediction_time_ms.toFixed(1)) : '0.0'} ms
                </Typography>
              </div>
            </Grid>
            <Grid item xs={12} md={4}>
              <div style={{ padding: '16px', background: 'rgba(255, 215, 0, 0.1)', borderRadius: '8px' }}>
                <Typography variant="caption" sx={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                  Explanation
                </Typography>
                <Typography variant="h6" sx={{ color: '#FFD700', fontWeight: 'bold' }}>
                  {metrics?.explanation_time_ms ? (isNaN(metrics.explanation_time_ms) ? '0.0' : metrics.explanation_time_ms.toFixed(1)) : '0.0'} ms
                </Typography>
              </div>
            </Grid>
          </Grid>
        </div>
      </CardContent>
    </GlassCard>
  );
};

export default MetricsMonitor;
