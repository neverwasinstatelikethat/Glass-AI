/**
 * AutonomyStatus Component - Phase 5
 * Displays autonomous action status and recommendations
 */

import React, { useState, useEffect } from 'react';
import {
  Card,
  CardContent,
  Typography,
  Box,
  Chip,
  Button,
  Alert,
  Grid,
  Divider,
} from '@mui/material';
import { styled } from '@mui/material/styles';
import {
  SmartToy,
  CheckCircle,
  Warning,
  Security,
  PlayArrow,
  ThumbUp,
  ThumbDown,
} from '@mui/icons-material';
import { autonomyApi, AutonomousAction } from '../services/pipelineApi';

const GlassCard = styled(Card)(() => ({
  background: 'rgba(13, 27, 42, 0.4)',
  backdropFilter: 'blur(20px) saturate(180%)',
  border: '1px solid rgba(0, 102, 255, 0.2)',
  borderRadius: '24px',
  padding: '24px',
}));

const ActionCard = styled(Box)(() => ({
  padding: '16px',
  background: 'rgba(255, 255, 255, 0.03)',
  borderRadius: '12px',
  border: '1px solid rgba(255, 255, 255, 0.1)',
  marginBottom: '12px',
}));

const AutonomyStatus: React.FC = () => {
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autonomyEnabled, setAutonomyEnabled] = useState(false);
  const [autonomousActions, setAutonomousActions] = useState<AutonomousAction[]>([]);
  const [approvalActions, setApprovalActions] = useState<AutonomousAction[]>([]);
  const [safetyEnabled, setSafetyEnabled] = useState(true);

  const fetchStatus = async () => {
    try {
      const data = await autonomyApi.getAutonomyStatus();
      setAutonomyEnabled(data.autonomy_enabled);
      setSafetyEnabled(data.safety_checks_enabled);
      setAutonomousActions(data.actions_to_execute || []);
      setApprovalActions(data.actions_requiring_approval || []);
      setError(null);
    } catch (err) {
      setError('Failed to load autonomy status');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'LOW': return '#00E676';
      case 'MEDIUM': return '#FFD700';
      case 'HIGH': return '#FF3366';
      default: return '#00E5FF';
    }
  };

  if (error) {
    return (
      <GlassCard>
        <Alert severity="error">{error}</Alert>
      </GlassCard>
    );
  }

  return (
    <GlassCard>
      <CardContent>
        <Box style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 24 }}>
          <Box style={{ display: 'flex', alignItems: 'center' }}>
            <SmartToy style={{ marginRight: 8, color: '#00E5FF', fontSize: 32 }} />
            <Box>
              <Typography variant="h5" style={{ color: 'white', fontWeight: 'bold' }}>
                Autonomous Actions
              </Typography>
              <Typography variant="caption" style={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                AI-Powered Decision Making
              </Typography>
            </Box>
          </Box>
          <Box>
            <Chip
              icon={autonomyEnabled ? <CheckCircle /> : <Warning />}
              label={autonomyEnabled ? 'ACTIVE' : 'INACTIVE'}
              sx={{
                background: autonomyEnabled 
                  ? 'linear-gradient(135deg, #00E676 0%, #00E5FF 100%)'
                  : 'linear-gradient(135deg, #FF3366 0%, #FF9E6D 100%)',
                color: 'white',
                fontWeight: 'bold',
                mr: 1,
              }}
            />
            {safetyEnabled && (
              <Chip
                icon={<Security />}
                label="SAFE MODE"
                sx={{
                  background: 'rgba(255, 215, 0, 0.2)',
                  color: '#FFD700',
                  border: '1px solid #FFD700',
                }}
              />
            )}
          </Box>
        </Box>

        {autonomousActions.length > 0 && (
          <Box style={{ marginBottom: 24 }}>
            <Typography variant="subtitle2" style={{ color: 'rgba(255, 255, 255, 0.7)', marginBottom: 12 }}>
              Auto-Executing Actions
            </Typography>
            {autonomousActions.map((action, idx) => (
              <ActionCard key={idx}>
                <Box style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Box style={{ flex: 1 }}>
                    <Box style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                      <PlayArrow style={{ color: '#00E676', marginRight: 8 }} />
                      <Typography variant="body1" style={{ color: 'white', fontWeight: 500 }}>
                        {action.action}
                      </Typography>
                    </Box>
                    <Box style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
                      <Chip
                        label={`Risk: ${action.risk_level}`}
                        size="small"
                        sx={{
                          background: getRiskColor(action.risk_level),
                          color: 'white',
                          fontSize: 11,
                          height: 20,
                        }}
                      />
                      <Chip
                        label={`Confidence: ${(action.confidence * 100).toFixed(0)}%`}
                        size="small"
                        sx={{
                          background: 'rgba(0, 229, 255, 0.2)',
                          color: '#00E5FF',
                          fontSize: 11,
                          height: 20,
                        }}
                      />
                    </Box>
                  </Box>
                  <CheckCircle style={{ color: '#00E676', fontSize: 24 }} />
                </Box>
              </ActionCard>
            ))}
          </Box>
        )}

        {approvalActions.length > 0 && (
          <Box>
            <Typography variant="subtitle2" style={{ color: 'rgba(255, 255, 255, 0.7)', marginBottom: 12 }}>
              Awaiting Approval
            </Typography>
            {approvalActions.map((action, idx) => (
              <ActionCard key={idx}>
                <Box style={{ marginBottom: 12 }}>
                  <Box style={{ display: 'flex', alignItems: 'center', marginBottom: 8 }}>
                    <Warning style={{ color: '#FFD700', marginRight: 8 }} />
                    <Typography variant="body1" style={{ color: 'white', fontWeight: 500 }}>
                      {action.action}
                    </Typography>
                  </Box>
                  <Typography variant="caption" style={{ color: 'rgba(255, 255, 255, 0.6)' }}>
                    {action.expected_impact}
                  </Typography>
                  <Box style={{ display: 'flex', gap: 8, marginTop: 8 }}>
                    <Chip
                      label={`Risk: ${action.risk_level}`}
                      size="small"
                      sx={{ background: getRiskColor(action.risk_level), color: 'white' }}
                    />
                    <Chip
                      label={`Confidence: ${(action.confidence * 100).toFixed(0)}%`}
                      size="small"
                      sx={{ background: 'rgba(0, 229, 255, 0.2)', color: '#00E5FF' }}
                    />
                  </Box>
                </Box>
                <Divider sx={{ my: 1, borderColor: 'rgba(255, 255, 255, 0.1)' }} />
                <Box style={{ display: 'flex', gap: 8 }}>
                  <Button
                    variant="contained"
                    size="small"
                    startIcon={<ThumbUp />}
                    sx={{
                      background: 'linear-gradient(135deg, #00E676 0%, #00E5FF 100%)',
                      color: 'white',
                    }}
                  >
                    Approve
                  </Button>
                  <Button
                    variant="outlined"
                    size="small"
                    startIcon={<ThumbDown />}
                    sx={{
                      borderColor: '#FF3366',
                      color: '#FF3366',
                      '&:hover': {
                        borderColor: '#FF3366',
                        background: 'rgba(255, 51, 102, 0.1)',
                      },
                    }}
                  >
                    Reject
                  </Button>
                </Box>
              </ActionCard>
            ))}
          </Box>
        )}

        {autonomousActions.length === 0 && approvalActions.length === 0 && (
          <Alert severity="info" sx={{ background: 'rgba(0, 229, 255, 0.1)', color: '#00E5FF' }}>
            No autonomous actions at this time. System is monitoring in real-time.
          </Alert>
        )}
      </CardContent>
    </GlassCard>
  );
};

export default AutonomyStatus;
