import React, { useState, useEffect, useCallback, useRef } from 'react';
import {
  Box,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  IconButton,
  Chip,
  TextField,
  Select,
  MenuItem,
  FormControl,
  InputLabel,
  Badge,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Divider,
  Stack,
  Alert,
  Tooltip,
  Accordion,
  AccordionSummary,
  AccordionDetails,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow
} from '@mui/material';
import {
  Error as ErrorIcon,
  Warning as WarningIcon,
  Info as InfoIcon,
  CheckCircle as CheckCircleIcon,
  Close as CloseIcon,
  FilterList as FilterIcon,
  Search as SearchIcon,
  Notifications as NotificationsIcon,
  NotificationsActive as NotificationsActiveIcon,
  Delete as DeleteIcon,
  Check as CheckIcon,
  ExpandMore as ExpandMoreIcon,
  Psychology as PsychologyIcon,
  Timeline as TimelineIcon,
  Settings as SettingsIcon,
  Insights as InsightsIcon
} from '@mui/icons-material';
import { useTheme } from '@mui/material/styles';

interface NotificationAction {
  label: string;
  action: string;
  params?: Record<string, any>;
}

interface Notification {
  id: string;
  timestamp: string;
  category: string; // ML_DEFECT_PREDICTION, ML_SENSOR_ANOMALY, RL_RECOMMENDATION, CRITICAL_DEFECT, etc.
  priority: string; // CRITICAL, HIGH, MEDIUM, LOW
  title: string;
  message: string;
  source: string; // LSTM_Model, GNN_Model, RL_Agent, ML_MODEL, SENSOR, SYSTEM
  actions?: NotificationAction[];
  acknowledged: boolean;
  resolved: boolean;
  resolution_notes?: string;
  metadata?: Record<string, any>;
}

const API_BASE_URL = 'http://localhost:8000';

// Priority colors defined within component using theme
const priorityIcons: Record<string, React.ReactElement> = {
  CRITICAL: <ErrorIcon />,
  HIGH: <WarningIcon />,
  MEDIUM: <InfoIcon />,
  LOW: <CheckCircleIcon />
};

const categoryLabels: Record<string, string> = {
  CRITICAL_DEFECT: 'Критический дефект',
  PARAMETER_ANOMALY: 'Аномалия параметров',
  ML_PREDICTION_WARNING: 'Предупреждение ML',
  MAINTENANCE_REMINDER: 'Напоминание о обслуживании',
  SYSTEM_HEALTH: 'Здоровье системы',
  ML_DEFECT_PREDICTION: '🤖 LSTM Прогноз дефекта',
  ML_SENSOR_ANOMALY: '🧠 GNN Аномалия датчиков',
  RL_RECOMMENDATION: '🎮 RL Рекомендация'
};

// Extended category icons
const categoryIcons: Record<string, React.ReactElement> = {
  ML_DEFECT_PREDICTION: <TimelineIcon />,
  ML_SENSOR_ANOMALY: <InsightsIcon />,
  RL_RECOMMENDATION: <PsychologyIcon />,
  CRITICAL_DEFECT: <ErrorIcon />,
  PARAMETER_ANOMALY: <SettingsIcon />,
  ML_PREDICTION_WARNING: <WarningIcon />,
  MAINTENANCE_REMINDER: <SettingsIcon />,
  SYSTEM_HEALTH: <CheckCircleIcon />
};

const NotificationsCenter: React.FC = () => {
  const theme = useTheme();
  
  const priorityColors: Record<string, string> = {
    CRITICAL: theme.palette.error.main,
    HIGH: theme.palette.warning.main,
    MEDIUM: theme.palette.info.main,
    LOW: theme.palette.success.main
  };
  
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  
  // Filters
  const [priorityFilter, setPriorityFilter] = useState<string>('ALL');
  const [categoryFilter, setCategoryFilter] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState('');
  const [showOnlyUnacknowledged, setShowOnlyUnacknowledged] = useState(false);
  
  // Detail dialog
  const [selectedNotification, setSelectedNotification] = useState<Notification | null>(null);
  const [detailDialogOpen, setDetailDialogOpen] = useState(false);
  
  // Sound alerts
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const lastNotificationId = useRef<string | null>(null);

  // Fetch notifications
  const fetchNotifications = useCallback(async () => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      if (priorityFilter !== 'ALL') params.append('priority', priorityFilter);
      if (categoryFilter !== 'ALL') params.append('category', categoryFilter);

      const response = await fetch(`${API_BASE_URL}/api/notifications/active?${params.toString()}`);
      
      if (!response.ok) {
        throw new Error(`Failed to fetch notifications: ${response.statusText}`);
      }

      const data = await response.json();
      setNotifications(data.notifications || []);

      // Play sound for new critical notifications
      if (soundEnabled && data.notifications && data.notifications.length > 0) {
        const latestNotification = data.notifications[0];
        if (
          latestNotification.priority === 'CRITICAL' &&
          !latestNotification.acknowledged &&
          latestNotification.id !== lastNotificationId.current
        ) {
          playAlertSound();
          lastNotificationId.current = latestNotification.id;
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Unknown error occurred');
      console.error('Error fetching notifications:', err);
    } finally {
      setLoading(false);
    }
  }, [priorityFilter, categoryFilter, soundEnabled]);

  useEffect(() => {
    fetchNotifications();
    const interval = setInterval(fetchNotifications, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, [fetchNotifications]);

  // Initialize audio
  useEffect(() => {
    audioRef.current = new Audio('/alert.mp3'); // Ensure alert.mp3 is in public folder
    audioRef.current.volume = 0.5;
  }, []);

  const playAlertSound = () => {
    if (audioRef.current && soundEnabled) {
      audioRef.current.play().catch(err => console.warn('Audio playback failed:', err));
    }
  };

  const acknowledgeNotification = async (notificationId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/notifications/${notificationId}/acknowledge`, {
        method: 'POST'
      });

      if (!response.ok) {
        throw new Error('Failed to acknowledge notification');
      }

      // Update local state
      setNotifications(prev =>
        prev.map(n => (n.id === notificationId ? { ...n, acknowledged: true } : n))
      );
    } catch (err) {
      console.error('Error acknowledging notification:', err);
    }
  };

  const deleteNotification = async (notificationId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/notifications/${notificationId}`, {
        method: 'DELETE'
      });

      if (!response.ok) {
        throw new Error('Failed to delete notification');
      }

      // Remove from local state
      setNotifications(prev => prev.filter(n => n.id !== notificationId));
    } catch (err) {
      console.error('Error deleting notification:', err);
    }
  };

  const executeAction = async (action: NotificationAction, notificationId: string) => {
    try {
      console.log('Executing action:', action);
      
      // Handle specific actions
      if (action.action === 'apply_recommendation') {
        // Apply RL recommendation
        await fetch(`${API_BASE_URL}/api/knowledge-graph/enrich/human-decision`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            notification_id: notificationId,
            decision: 'applied',
            defect_type: selectedNotification?.metadata?.defect_type || 'unknown',
            notes: 'Applied via notification center'
          })
        });
      } else if (action.action === 'adjust_parameters') {
        // Adjust parameters action - would typically open a parameter adjustment dialog
        console.log('Adjust parameters action triggered');
      } else if (action.action === 'apply_rl_recommendation') {
        // Apply RL recommendation from sensor anomaly
        await fetch(`${API_BASE_URL}/api/knowledge-graph/enrich/human-decision`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json'
          },
          body: JSON.stringify({
            notification_id: notificationId,
            decision: 'applied',
            defect_type: 'sensor_anomaly',
            notes: 'Applied RL recommendation for sensor anomaly'
          })
        });
      } else if (action.action === 'inspect_sensors') {
        // Inspect sensors action
        console.log('Inspect sensors action triggered');
      } else if (action.action === 'dismiss') {
        // Dismiss action
        await acknowledgeNotification(notificationId);
      }
      
      // Acknowledge the notification after action
      await acknowledgeNotification(notificationId);
    } catch (err) {
      console.error('Error executing action:', err);
    }
  };

  const openDetailDialog = (notification: Notification) => {
    setSelectedNotification(notification);
    setDetailDialogOpen(true);
  };

  const closeDetailDialog = () => {
    setDetailDialogOpen(false);
    setSelectedNotification(null);
  };

  // Filter notifications
  const filteredNotifications = notifications.filter(notification => {
    if (showOnlyUnacknowledged && notification.acknowledged) return false;
    if (searchQuery && !notification.message.toLowerCase().includes(searchQuery.toLowerCase()) &&
        !notification.title.toLowerCase().includes(searchQuery.toLowerCase())) {
      return false;
    }
    return true;
  });

  const unacknowledgedCount = notifications.filter(n => !n.acknowledged).length;

  // Format timestamp for display
  const formatTimestamp = (timestamp: string) => {
    return new Date(timestamp).toLocaleString('ru-RU');
  };

  // Render parameter snapshot table
  const renderParameterSnapshot = (parameters: Record<string, any>) => {
    if (!parameters) return null;
    
    // Define optimal ranges for context
    const optimalRanges: Record<string, [number, number]> = {
      furnace_temperature: [1520, 1570],
      belt_speed: [140, 160],
      mold_temp: [300, 330],
      quality_score: [0.9, 1.0],
      furnace_pressure: [10, 20],
      forming_pressure: [30, 70],
      annealing_temp: [500, 700],
      cooling_rate: [2, 5]
    };
    
    return (
      <TableContainer>
        <Table size="small">
          <TableHead>
            <TableRow>
              <TableCell>Параметр</TableCell>
              <TableCell align="right">Значение</TableCell>
              <TableCell align="right">Оптимальный диапазон</TableCell>
              <TableCell align="center">Статус</TableCell>
            </TableRow>
          </TableHead>
          <TableBody>
            {Object.entries(parameters).map(([key, value]) => {
              const range = optimalRanges[key];
              let status = '';
              let statusColor = 'default';
              
              if (range && typeof value === 'number') {
                if (value >= range[0] && value <= range[1]) {
                  status = '✓ ОК';
                  statusColor = 'success';
                } else if (value >= range[0] * 0.95 && value <= range[1] * 1.05) {
                  status = '~ Приемлемо';
                  statusColor = 'warning';
                } else {
                  status = '✗ Отклонение';
                  statusColor = 'error';
                }
              }
              
              return (
                <TableRow key={key}>
                  <TableCell>{key}</TableCell>
                  <TableCell align="right">{typeof value === 'number' ? value.toFixed(2) : value}</TableCell>
                  <TableCell align="right">
                    {range ? `${range[0]}-${range[1]}` : 'N/A'}
                  </TableCell>
                  <TableCell align="center">
                    <Chip 
                      label={status || 'N/A'} 
                      size="small" 
                      color={statusColor as any}
                      variant="outlined"
                    />
                  </TableCell>
                </TableRow>
              );
            })}
          </TableBody>
        </Table>
      </TableContainer>
    );
  };

  // Render LSTM analysis details
  const renderLSTMAnalysis = (metadata: Record<string, any>) => {
    if (!metadata || !metadata.defect_probabilities) return null;
    
    // Sort defects by probability in descending order
    const sortedDefects = Object.entries(metadata.defect_probabilities)
      .sort((a, b) => (b[1] as number) - (a[1] as number));
    
    // Get top 3 defects
    const topDefects = sortedDefects.slice(0, 3);
    
    return (
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>🤖 Анализ LSTM</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Typography variant="subtitle2" gutterBottom>
            Вероятности дефектов (топ-3):
          </Typography>
          <TableContainer>
            <Table size="small">
              <TableHead>
                <TableRow>
                  <TableCell>Тип дефекта</TableCell>
                  <TableCell align="right">Вероятность</TableCell>
                  <TableCell align="right">Уровень риска</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {topDefects.map(([defectType, probability]) => {
                  const prob = probability as number;
                  let riskLevel = '';
                  let riskColor: 'success' | 'warning' | 'error' = 'success';
                  
                  if (prob > 0.3) {
                    riskLevel = 'Высокий';
                    riskColor = 'error';
                  } else if (prob > 0.15) {
                    riskLevel = 'Средний';
                    riskColor = 'warning';
                  } else {
                    riskLevel = 'Низкий';
                    riskColor = 'success';
                  }
                  
                  return (
                    <TableRow key={defectType}>
                      <TableCell>
                        {defectType}
                        {defectType === metadata.defect_type && (
                          <Chip 
                            label="Основной" 
                            size="small" 
                            color="primary" 
                            variant="outlined" 
                            sx={{ ml: 1 }}
                          />
                        )}
                      </TableCell>
                      <TableCell align="right">{(prob * 100).toFixed(2)}%</TableCell>
                      <TableCell align="right">
                        <Chip 
                          label={riskLevel} 
                          size="small" 
                          color={riskColor}
                          variant="outlined"
                        />
                      </TableCell>
                    </TableRow>
                  );
                })}
              </TableBody>
            </Table>
          </TableContainer>
          
          {metadata.parameters_snapshot && (
            <>
              <Typography variant="subtitle2" gutterBottom sx={{ mt: 2 }}>
                Снимок параметров при прогнозировании:
              </Typography>
              {renderParameterSnapshot(metadata.parameters_snapshot)}
            </>
          )}
        </AccordionDetails>
      </Accordion>
    );
  };

  // Render RL recommendations
  const renderRLRecommendations = (metadata: Record<string, any>) => {
    if (!metadata || !metadata.recommendations) return null;
    
    const recommendations = Array.isArray(metadata.recommendations) 
      ? metadata.recommendations 
      : [metadata.recommendations];
    
    return (
      <Accordion>
        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
          <Typography>🎮 Рекомендации RL агента</Typography>
        </AccordionSummary>
        <AccordionDetails>
          <Stack spacing={2}>
            {recommendations.map((rec: any, index: number) => (
              <Paper key={index} variant="outlined" sx={{ p: 2 }}>
                <Typography variant="subtitle2" gutterBottom>
                  {rec.action || rec.text || 'Рекомендация'}
                </Typography>
                <Stack direction="row" spacing={1} flexWrap="wrap">
                  {rec.confidence && (
                    <Chip 
                      label={`Доверие: ${(rec.confidence * 100).toFixed(1)}%`} 
                      size="small" 
                      color={rec.confidence > 0.8 ? "success" : rec.confidence > 0.6 ? "warning" : "default"}
                    />
                  )}
                  {rec.priority && (
                    <Chip 
                      label={`Приоритет: ${rec.priority}`} 
                      size="small" 
                      color={rec.priority === 'HIGH' ? "error" : rec.priority === 'MEDIUM' ? "warning" : "info"}
                    />
                  )}
                  {rec.parameter && (
                    <Chip 
                      label={`Параметр: ${rec.parameter}`} 
                      size="small" 
                      variant="outlined"
                    />
                  )}
                </Stack>
                {rec.expected_impact && (
                  <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                    Ожидаемый эффект: {rec.expected_impact}
                  </Typography>
                )}
                {rec.justification && (
                  <Typography variant="body2" sx={{ mt: 1, fontStyle: 'italic' }}>
                    Обоснование: {rec.justification}
                  </Typography>
                )}
                {(rec.current_value !== undefined || rec.suggested_value !== undefined) && (
                  <Typography variant="body2" sx={{ mt: 1 }}>
                    Текущее значение: {rec.current_value !== undefined ? rec.current_value.toFixed(1) : 'N/A'} → 
                    Рекомендуемое: {rec.suggested_value !== undefined ? rec.suggested_value.toFixed(1) : 'N/A'}
                  </Typography>
                )}
              </Paper>
            ))}
          </Stack>
        </AccordionDetails>
      </Accordion>
    );
  };

  return (
    // @ts-expect-error MUI sx prop complexity limit
    <Box sx={{ p: 3 }}>
      <Paper elevation={3} sx={{ p: 3 }}>
        {/* Header */}
        <Stack direction="row" justifyContent="space-between" alignItems="center" mb={3}>
          <Stack direction="row" alignItems="center" spacing={2}>
            <Badge badgeContent={unacknowledgedCount} color="error">
              <NotificationsActiveIcon sx={{ fontSize: 40, color: theme.palette.primary.main }} />
            </Badge>
            <Typography variant="h4">
              Центр уведомлений
            </Typography>
          </Stack>

          <Stack direction="row" spacing={2}>
            <Tooltip title={soundEnabled ? 'Отключить звук' : 'Включить звук'}>
              <IconButton onClick={() => setSoundEnabled(!soundEnabled)}>
                {soundEnabled ? <NotificationsActiveIcon /> : <NotificationsIcon />}
              </IconButton>
            </Tooltip>
            <Button
              variant={showOnlyUnacknowledged ? 'contained' : 'outlined'}
              onClick={() => setShowOnlyUnacknowledged(!showOnlyUnacknowledged)}
            >
              Только непрочитанные ({unacknowledgedCount})
            </Button>
          </Stack>
        </Stack>

        {/* Filters */}
        <Stack direction="row" spacing={2} mb={3}>
          <TextField
            placeholder="Поиск..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            InputProps={{
              startAdornment: <SearchIcon sx={{ mr: 1, color: theme.palette.text.secondary }} />
            }}
            sx={{ flexGrow: 1 }}
          />

          <FormControl sx={{ minWidth: 150 }}>
            <InputLabel>Приоритет</InputLabel>
            <Select
              value={priorityFilter}
              onChange={(e) => setPriorityFilter(e.target.value)}
              label="Приоритет"
            >
              <MenuItem value="ALL">Все</MenuItem>
              <MenuItem value="CRITICAL">Критический</MenuItem>
              <MenuItem value="HIGH">Высокий</MenuItem>
              <MenuItem value="MEDIUM">Средний</MenuItem>
              <MenuItem value="LOW">Низкий</MenuItem>
            </Select>
          </FormControl>

          <FormControl sx={{ minWidth: 200 }}>
            <InputLabel>Категория</InputLabel>
            <Select
              value={categoryFilter}
              onChange={(e) => setCategoryFilter(e.target.value)}
              label="Категория"
            >
              <MenuItem value="ALL">Все</MenuItem>
              <MenuItem value="ML_DEFECT_PREDICTION">🤖 LSTM Прогноз дефекта</MenuItem>
              <MenuItem value="ML_SENSOR_ANOMALY">🧠 GNN Аномалия датчиков</MenuItem>
              <MenuItem value="RL_RECOMMENDATION">🎮 RL Рекомендация</MenuItem>
              <MenuItem value="CRITICAL_DEFECT">Критический дефект</MenuItem>
              <MenuItem value="PARAMETER_ANOMALY">Аномалия параметров</MenuItem>
              <MenuItem value="ML_PREDICTION_WARNING">Предупреждение ML</MenuItem>
              <MenuItem value="MAINTENANCE_REMINDER">Обслуживание</MenuItem>
              <MenuItem value="SYSTEM_HEALTH">Здоровье системы</MenuItem>
            </Select>
          </FormControl>
        </Stack>

        {/* Error alert */}
        {error && (
          <Alert severity="error" sx={{ mb: 2 }}>
            {error}
          </Alert>
        )}

        {/* Notifications list */}
        <List sx={{ maxHeight: 600, overflow: 'auto' }}>
          {filteredNotifications.length === 0 ? (
            <Box sx={{ textAlign: 'center', py: 4 }}>
              <Typography variant="h6" color="textSecondary">
                {loading ? 'Загрузка уведомлений...' : 'Нет уведомлений'}
              </Typography>
            </Box>
          ) : (
            filteredNotifications.map((notification, index) => (
              <React.Fragment key={notification.id}>
                {index > 0 && <Divider />}
                <ListItem
                  sx={{
                    backgroundColor: notification.acknowledged
                      ? 'transparent'
                      : `${priorityColors[notification.priority]}10`,
                    borderLeft: `4px solid ${priorityColors[notification.priority]}`,
                    mb: 1,
                    borderRadius: 1,
                    '&:hover': { backgroundColor: `${theme.palette.primary.main}08` }
                  }}
                >
                  <ListItemIcon>
                    {React.cloneElement(priorityIcons[notification.priority], {
                      sx: { color: priorityColors[notification.priority], fontSize: 32 }
                    })}
                  </ListItemIcon>

                  <ListItemText
                    primary={
                      <Stack direction="row" alignItems="center" spacing={1}>
                        <Typography variant="subtitle1" fontWeight="bold">
                          {notification.title}
                        </Typography>
                        <Chip
                          label={notification.priority}
                          size="small"
                          sx={{
                            backgroundColor: priorityColors[notification.priority],
                            color: 'white',
                            fontSize: '0.7rem'
                          }}
                        />
                        <Chip
                          label={categoryLabels[notification.category]}
                          size="small"
                          variant="outlined"
                          sx={{ fontSize: '0.7rem' }}
                        />
                      </Stack>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" color="textSecondary" sx={{ mt: 0.5 }}>
                          {notification.message}
                        </Typography>
                        <Typography variant="caption" color="textSecondary" sx={{ mt: 0.5 }}>
                          {formatTimestamp(notification.timestamp)} • {notification.source}
                        </Typography>
                        
                        {notification.actions && notification.actions.length > 0 && (
                          <Stack direction="row" spacing={1} sx={{ mt: 1 }}>
                            {notification.actions.map((action, idx) => (
                              <Button
                                key={idx}
                                size="small"
                                variant="outlined"
                                onClick={() => executeAction(action, notification.id)}
                              >
                                {action.label}
                              </Button>
                            ))}
                          </Stack>
                        )}
                      </Box>
                    }
                    onClick={() => openDetailDialog(notification)}
                    sx={{ cursor: 'pointer' }}
                  />

                  <Stack direction="row" spacing={1}>
                    {!notification.acknowledged && (
                      <Tooltip title="Подтвердить">
                        <IconButton
                          size="small"
                          onClick={() => acknowledgeNotification(notification.id)}
                          sx={{ color: theme.palette.success.main }}
                        >
                          <CheckIcon />
                        </IconButton>
                      </Tooltip>
                    )}
                    <Tooltip title="Удалить">
                      <IconButton
                        size="small"
                        onClick={() => deleteNotification(notification.id)}
                        sx={{ color: theme.palette.error.main }}
                      >
                        <DeleteIcon />
                      </IconButton>
                    </Tooltip>
                  </Stack>
                </ListItem>
              </React.Fragment>
            ))
          )}
        </List>
      </Paper>

      {/* Detail Dialog */}
      <Dialog
        open={detailDialogOpen}
        onClose={closeDetailDialog}
        maxWidth="md"
        fullWidth
      >
        {selectedNotification && (
          <>
            <DialogTitle>
              <Stack direction="row" alignItems="center" spacing={2}>
                {priorityIcons[selectedNotification.priority]}
                <Typography variant="h6">{selectedNotification.title}</Typography>
                <Chip
                  label={selectedNotification.priority}
                  size="small"
                  sx={{ backgroundColor: priorityColors[selectedNotification.priority], color: 'white' }}
                />
              </Stack>
            </DialogTitle>
            <DialogContent dividers>
              <Typography variant="body1" paragraph>
                {selectedNotification.message}
              </Typography>
              
              <Divider sx={{ my: 2 }} />
              
              <Stack spacing={2}>
                <Typography variant="subtitle2" color="textSecondary">
                  Категория: {categoryLabels[selectedNotification.category]}
                </Typography>
                <Typography variant="subtitle2" color="textSecondary">
                  Источник: {selectedNotification.source}
                </Typography>
                <Typography variant="subtitle2" color="textSecondary">
                  Время: {formatTimestamp(selectedNotification.timestamp)}
                </Typography>
                <Typography variant="subtitle2" color="textSecondary">
                  Статус: {selectedNotification.acknowledged ? 'Подтверждено' : 'Не подтверждено'}
                </Typography>
                
                {/* Metadata section */}
                {selectedNotification.metadata && Object.keys(selectedNotification.metadata).length > 0 && (
                  <>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="h6">Детали уведомления</Typography>
                    
                    {/* Parameter snapshot */}
                    {selectedNotification.metadata.parameters_snapshot && (
                      <>
                        <Typography variant="subtitle2" gutterBottom>
                          Снимок параметров:
                        </Typography>
                        {renderParameterSnapshot(selectedNotification.metadata.parameters_snapshot)}
                      </>
                    )}
                    
                    {/* LSTM Analysis */}
                    {selectedNotification.category === 'ML_DEFECT_PREDICTION' && 
                     renderLSTMAnalysis(selectedNotification.metadata)}
                    
                    {/* RL Recommendations */}
                    {(selectedNotification.category === 'ML_DEFECT_PREDICTION' || 
                      selectedNotification.category === 'RL_RECOMMENDATION') && 
                     renderRLRecommendations(selectedNotification.metadata)}
                  </>
                )}
                
                {selectedNotification.resolution_notes && (
                  <>
                    <Divider sx={{ my: 2 }} />
                    <Typography variant="subtitle2" fontWeight="bold">
                      Примечания к решению:
                    </Typography>
                    <Typography variant="body2">{selectedNotification.resolution_notes}</Typography>
                  </>
                )}
              </Stack>
            </DialogContent>
            <DialogActions>
              {!selectedNotification.acknowledged && (
                <Button
                  onClick={() => {
                    acknowledgeNotification(selectedNotification.id);
                    closeDetailDialog();
                  }}
                  variant="contained"
                  color="success"
                >
                  Подтвердить
                </Button>
              )}
              <Button onClick={closeDetailDialog}>Закрыть</Button>
            </DialogActions>
          </>
        )}
      </Dialog>
    </Box>
  );
};

export default NotificationsCenter;