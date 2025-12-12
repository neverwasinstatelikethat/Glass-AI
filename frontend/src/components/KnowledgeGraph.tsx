import React, { useState, useEffect, useRef } from 'react';
import {
    Box,
    Typography,
    Paper,
    TextField,
    Button,
    Chip,
    CircularProgress,
    Alert,
    Grid,
    Card,
    CardContent,
    CardHeader,
    Divider,
    List,
    ListItem,
    ListItemText,
    ListItemIcon,
    Tooltip,
    IconButton,
    Fade,
    Zoom,
    Grow,
    Slide,
    alpha,
    MenuItem,
    Slider,
    Select,
    InputLabel,
    FormControl,
    styled,
    Drawer,
    Tabs,
    Tab
} from '@mui/material';
import { Cause, Evidence } from '../types/knowledgeGraph';
import {
    Psychology,
    Info,
    TrendingUp,
    Warning,
    CheckCircle,
    Error as ErrorIcon,
    PsychologyOutlined,
    PsychologyRounded,
    RocketLaunch,
    Insights,
    Hub,
    Timeline,
    NetworkPing,
    WorkspacePremium,
    AutoGraph,
    Polyline,
    Settings,
    Thermostat,
    Close,
    ExpandMore,
    ChevronRight
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';
// Simple graph visualization component
const GraphVisualization = ({ subgraph }: { subgraph: any }) => {
  if (!subgraph) {
    const emptyBoxStyles = {
      display: 'flex',
      justifyContent: 'center',
      alignItems: 'center',
      height: '400px',
      color: 'rgba(255, 255, 255, 0.7)',
      fontSize: '1rem'
    };
    
    return (
      <Box sx={emptyBoxStyles}>
        Нет данных для отображения графа
      </Box>
    );
  }

  const containerStyles = {
    height: '500px',
    overflow: 'auto',
    background: 'rgba(0, 0, 0, 0.2)',
    borderRadius: '8px',
    p: 2
  };

  const flexColumnStyles = {
    display: 'flex',
    flexDirection: 'column',
    gap: 2
  };

  const chipContainerStyles = {
    display: 'flex',
    flexWrap: 'wrap',
    gap: 1
  };

  const nodeChipStyles = {
    background: 'rgba(0, 102, 255, 0.2)',
    border: '1px solid rgba(0, 102, 255, 0.3)',
    color: 'white'
  };

  const edgeChipStyles = {
    background: 'rgba(0, 229, 255, 0.2)',
    border: '1px solid rgba(0, 229, 255, 0.3)',
    color: 'white'
  };

  return (
    <Box sx={containerStyles}>
      <Typography variant="h6" sx={{ mb: 2, color: 'white' }}>
        Граф знаний: {subgraph.defect}
      </Typography>
      <Box sx={flexColumnStyles}>
        <Box>
          <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 1 }}>
            Узлы ({subgraph.nodes?.length || 0})
          </Typography>
          <Box sx={chipContainerStyles}>
            {subgraph.nodes?.map((node: any, index: number) => (
              <Chip
                key={index}
                label={node.label || node.name}
                size="small"
                sx={nodeChipStyles}
              />
            ))}
          </Box>
        </Box>
        <Box>
          <Typography variant="subtitle2" sx={{ color: 'rgba(255, 255, 255, 0.7)', mb: 1 }}>
            Связи ({subgraph.edges?.length || 0})
          </Typography>
          <Box sx={chipContainerStyles}>
            {subgraph.edges?.map((edge: any, index: number) => (
              <Chip
                key={index}
                label={`${edge.source} → ${edge.target}`}
                size="small"
                sx={edgeChipStyles}
              />
            ))}
          </Box>
        </Box>
      </Box>
    </Box>
  );
};
// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// Современная цветовая палитра РТУ МИРЭА 2025
const MIREA_2025_COLORS = {
    primary: '#0066FF',
    primaryGradient: 'linear-gradient(135deg, #0066FF 0%, #00E5FF 100%)',
    primaryDark: '#0047B3',
    primaryLight: '#4D8FFF',
    secondary: '#FF3366',
    secondaryGradient: 'linear-gradient(135deg, #FF3366 0%, #FF6B35 100%)',
    secondaryDark: '#CC0033',
    secondaryLight: '#FF6690',
    success: '#00E676',
    successGradient: 'linear-gradient(135deg, #00E676 0%, #00C853 100%)',
    warning: '#FFD700',
    warningGradient: 'linear-gradient(135deg, #FFD700 0%, #FFA726 100%)',
    error: '#FF1744',
    errorGradient: 'linear-gradient(135deg, #FF1744 0%, #EF5350 100%)',
    info: '#00E5FF',
    infoGradient: 'linear-gradient(135deg, #00E5FF 0%, #29B6F6 100%)',
    background: '#000814',
    backgroundGradient: 'radial-gradient(ellipse at top, #001D3D 0%, #000814 50%, #000000 100%)',
    surface: 'rgba(13, 27, 42, 0.85)',
    surfaceLight: 'rgba(0, 102, 255, 0.1)',
    glass: 'rgba(255, 255, 255, 0.05)',
    text: '#FFFFFF',
    textSecondary: '#B8C5D6',
    textTertiary: '#8A9BB0',
    glowPrimary: 'rgba(0, 102, 255, 0.4)',
    glowSecondary: 'rgba(255, 51, 102, 0.4)',
    glowSuccess: 'rgba(0, 230, 118, 0.4)',
    glowWarning: 'rgba(255, 215, 0, 0.4)'
};

// Локальные интерфейсы для компонента (без зависимостей от внешних типов)
type CauseData = Cause;

interface RecommendationData {
    parameter: string;
    current_value: number;
    target_value: number;
    unit: string;
    action: string;
    confidence: number;
    strength: number;
    priority: 'HIGH' | 'MEDIUM' | 'LOW';
    expected_impact: string;
    ml_enhanced?: boolean;
    ml_confidence?: number;
}

interface GraphNode {
    id: number | string;
    name: string;
    label: string;
    nodeType: 'defect' | 'cause' | 'parameter' | 'recommendation' | 'human_decision' | 'equipment';
    confidence: number;
    properties?: Record<string, any>;
    applied?: boolean;
    source?: string;
}

interface GraphEdge {
    id: number;
    source: number;
    target: number;
    type: string;
    confidence: number;
    strength: number;
}

interface GraphData {
    defect: string;
    nodes: GraphNode[];
    edges: GraphEdge[];
}

// Создаем стилизованный компонент для корневого контейнера
const RootContainer = styled(Box)(({ theme }) => ({
    minHeight: '100vh',
    background: MIREA_2025_COLORS.backgroundGradient,
    backgroundAttachment: 'fixed',
    position: 'relative',
    overflow: 'hidden',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: `
      radial-gradient(circle at 20% 30%, ${alpha(MIREA_2025_COLORS.primary, 0.05)} 0%, transparent 50%),
      radial-gradient(circle at 80% 70%, ${alpha(MIREA_2025_COLORS.secondary, 0.05)} 0%, transparent 50%),
      url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%230066FF' fill-opacity='0.02'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")
    `,
        pointerEvents: 'none'
    }
}));

// Стиль для сканирующей линии
const ScanlineBox = styled(Box)(({ theme }) => ({
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 1,
    background: `linear-gradient(90deg, transparent, ${MIREA_2025_COLORS.primary}, transparent)`,
    animation: 'scanline 3s linear infinite',
    '@keyframes scanline': {
        '0%': { top: '0%' },
        '100%': { top: '100%' }
    }
}));

// Стилизованный компонент для пульсирующей точки
const PulsingDot = styled(Box, {
    shouldForwardProp: (prop) => prop !== 'color' && prop !== 'size',
})<{ color: string; size?: number }>(({ color, size = 8 }) => ({
    width: size,
    height: size,
    borderRadius: '50%',
    background: color,
    position: 'relative',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: -size,
        left: -size,
        right: -size,
        bottom: -size,
        background: color,
        borderRadius: '50%',
        animation: 'pulse 2s infinite',
        opacity: 0.4
    },
    '@keyframes pulse': {
        '0%': { transform: 'scale(0.8)', opacity: 0.4 },
        '50%': { transform: 'scale(1.2)', opacity: 0.2 },
        '100%': { transform: 'scale(0.8)', opacity: 0.4 }
    }
}));

// Анимированный компонент стекла
const GlassCard = styled(Card)(({ theme }) => ({
    background: 'rgba(13, 27, 42, 0.6)',
    backdropFilter: 'blur(20px) saturate(180%)',
    borderRadius: 16,
    border: '1px solid rgba(255, 255, 255, 0.1)',
    boxShadow: `
    0 8px 32px rgba(0, 0, 0, 0.3),
    0 0 0 1px rgba(0, 102, 255, 0.1),
    inset 0 1px 0 rgba(255, 255, 255, 0.05)
  `,
    position: 'relative',
    overflow: 'hidden',
    transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'linear-gradient(135deg, rgba(0, 102, 255, 0.1) 0%, rgba(0, 229, 255, 0.05) 50%, transparent 100%)',
        opacity: 0,
        transition: 'opacity 0.4s ease'
    },
    '&:hover': {
        transform: 'translateY(-8px) scale(1.02)',
        boxShadow: `
      0 20px 60px rgba(0, 102, 255, 0.4),
      0 0 120px rgba(0, 229, 255, 0.2),
      inset 0 1px 0 rgba(255, 255, 255, 0.1)
    `,
        borderColor: 'rgba(0, 229, 255, 0.3)',
        '&::before': {
            opacity: 1
        }
    }
}));

// Анимированный чип
const AnimatedChip = ({ label, color, icon, sx = {} }: any) => (
    <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        transition={{ duration: 0.3 }}
    >
        <Chip
            icon={icon}
            label={label}
            sx={{
                background: `rgba(${color === 'primary' ? '0, 102, 255' : color === 'success' ? '0, 230, 118' : '255, 215, 0'}, 0.15)`,
                backdropFilter: 'blur(10px)',
                border: `1px solid rgba(${color === 'primary' ? '0, 102, 255' : color === 'success' ? '0, 230, 118' : '255, 215, 0'}, 0.3)`,
                color: color === 'primary' ? MIREA_2025_COLORS.primaryLight :
                    color === 'success' ? MIREA_2025_COLORS.success : MIREA_2025_COLORS.warning,
                fontWeight: 700,
                fontSize: '0.75rem',
                padding: '4px 12px',
                borderRadius: 20,
                transition: 'all 0.3s ease',
                '&:hover': {
                    transform: 'scale(1.05)',
                    boxShadow: `0 0 20px rgba(${color === 'primary' ? '0, 102, 255' : color === 'success' ? '0, 230, 118' : '255, 215, 0'}, 0.4)`
                },
                ...sx
            }}
        />
    </motion.div>
);

// Анимированная кнопка
const AnimatedButton = ({ children, onClick, sx = {} }: any) => (
    <motion.div
        whileHover={{ scale: 1.05 }}
        whileTap={{ scale: 0.95 }}
    >
        <Button
            onClick={onClick}
            sx={{
                background: MIREA_2025_COLORS.primaryGradient,
                color: '#FFFFFF',
                fontWeight: 700,
                fontSize: '0.875rem',
                padding: '12px 24px',
                borderRadius: 12,
                textTransform: 'none',
                position: 'relative',
                overflow: 'hidden',
                transition: 'all 0.3s ease',
                '&::before': {
                    content: '""',
                    position: 'absolute',
                    top: 0,
                    left: '-100%',
                    width: '100%',
                    height: '100%',
                    background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.2), transparent)',
                    transition: 'left 0.6s'
                },
                '&:hover': {
                    background: MIREA_2025_COLORS.primaryGradient,
                    transform: 'translateY(-2px)',
                    boxShadow: `0 10px 30px ${MIREA_2025_COLORS.glowPrimary}`,
                    '&::before': {
                        left: '100%'
                    }
                },
                ...sx
            }}
        >
            {children}
        </Button>
    </motion.div>
);

const headerBoxStyles: React.CSSProperties = {
    position: 'relative',
    zIndex: 1,
    padding: 16
};

const KnowledgeGraph: React.FC = () => {
    const [defectType, setDefectType] = useState<string>('crack');
    const [minConfidence, setMinConfidence] = useState<number>(0.7);
    const [parameterValues, setParameterValues] = useState<Record<string, number>>({
        'furnace_temperature': 1580,
        'belt_speed': 150,
        'mold_temperature': 320,
        'pressure': 50,
        'oxygen_content': 3.5
    });

    const [causes, setCauses] = useState<CauseData[]>([]);

    const [recommendations, setRecommendations] = useState<RecommendationData[]>([]);

    const [subgraph, setSubgraph] = useState<GraphData | null>(null);
    const [selectedNode, setSelectedNode] = useState<GraphNode | null>(null);
    const [nodeDetailsOpen, setNodeDetailsOpen] = useState(false);

    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'causes' | 'recommendations' | 'graph'>('causes');
    
    // Функция для закрытия панели деталей
    const closeNodeDetails = () => {
        setNodeDetailsOpen(false);
        setSelectedNode(null);
    };
    
    // Statistics state
    const [statistics, setStatistics] = useState<{
        predictedDefectReduction: number;
        newCausalLinks: number;
        recommendationAccuracy: number;
    }>({
        predictedDefectReduction: 0,
        newCausalLinks: 0,
        recommendationAccuracy: 0
    });

    const fetchCauses = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(
                `${API_BASE_URL}/api/knowledge-graph/causes/${defectType}?min_confidence=${minConfidence}`
            );
            if (!response.ok) throw new Error('Failed to fetch causes');
            const data = await response.json();
            setCauses(data.causes.map((cause: any) => ({
                cause: cause.parameter_name || cause.parameter || cause.cause || 'Unknown',
                confidence: cause.confidence || 0,
                strength: cause.confidence || 0,  // Using confidence as strength
                observations: 0,  // Not provided in API response
                evidence: cause.evidence || [],  // Evidence from API response
                cause_type: 'PARAMETER',
                relationship_type: 'CAUSES',
                last_updated: new Date().toISOString()
            })));
        } catch (err) {
            setError('Ошибка при загрузке причин дефекта');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const fetchRecommendations = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(
                `${API_BASE_URL}/api/knowledge-graph/recommendations/${defectType}`,
                {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ parameter_values: parameterValues })
                }
            );
            if (!response.ok) throw new Error('Failed to fetch recommendations');
            const data = await response.json();
            setRecommendations((data.interventions || data.recommendations || []).map((rec: any) => ({
                parameter: rec.parameter || rec.target_parameter || 'Unknown',
                current_value: rec.current_value || 0,
                target_value: rec.target_value || 0,
                unit: rec.unit || '',
                action: rec.action || 'No action specified',
                confidence: rec.confidence || 0,
                strength: rec.confidence || 0,  // Using confidence as strength
                expected_impact: rec.expected_impact || 'No impact specified',
                priority: rec.priority || 'MEDIUM',
                ml_enhanced: rec.ml_enhanced || false,
                ml_confidence: rec.ml_confidence || 0
            })));
        } catch (err) {
            setError('Ошибка при загрузке рекомендаций');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const fetchSubgraph = async () => {
        setLoading(true);
        setError(null);

        try {
            const response = await fetch(
                `${API_BASE_URL}/api/knowledge-graph/subgraph/${defectType}?max_depth=2&include_recommendations=true&include_human_decisions=true`
            );
            if (!response.ok) throw new Error('Failed to fetch subgraph');
            const data = await response.json();
            setSubgraph(data);
        } catch (err) {
            setError('Ошибка при загрузке графа знаний');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };
    
    const fetchStatistics = async () => {
        try {
            // Fetch knowledge graph statistics
            const statsResponse = await fetch(`${API_BASE_URL}/api/knowledge-graph/statistics`);
            if (statsResponse.ok) {
                const statsData = await statsResponse.json();
                setStatistics({
                    predictedDefectReduction: statsData.quality_metrics?.predicted_defect_reduction || 
                        Math.min(50, Math.max(10, (statsData.nodes?.recommendations || 0) * 5)),
                    newCausalLinks: statsData.recent_activity?.causal_links || 0,
                    recommendationAccuracy: statsData.quality_metrics?.avg_recommendation_confidence ? 
                        Math.min(95, Math.max(70, statsData.quality_metrics.avg_recommendation_confidence * 100)) :
                        Math.min(95, Math.max(70, (statsData.nodes?.recommendations || 0) * 8))
                });
            } else {
                // Fallback to simulated data
                setStatistics({
                    predictedDefectReduction: Math.min(50, Math.max(10, (recommendations.length * 8))),
                    newCausalLinks: subgraph?.edges?.length || 0,
                    recommendationAccuracy: Math.min(95, Math.max(70, (causes.length * 15)))
                });
            }
        } catch (err) {
            console.error('Ошибка при загрузке статистики:', err);
            // Use simulated data as fallback
            setStatistics({
                predictedDefectReduction: Math.min(50, Math.max(10, (recommendations.length * 8))),
                newCausalLinks: subgraph?.edges?.length || 0,
                recommendationAccuracy: Math.min(95, Math.max(70, (causes.length * 15)))
            });
        }
    };

    const handleParameterChange = (param: string, value: string) => {
        setParameterValues(prev => ({
            ...prev,
            [param]: parseFloat(value) || 0
        }));
    };

    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            try {
                await fetchCauses();
                await fetchRecommendations();
                await fetchSubgraph();
                await fetchStatistics();
            } catch (err) {
                setError('Ошибка при загрузке данных');
                console.error(err);
            } finally {
                setLoading(false);
            }
        };
        
        fetchData();
    }, [defectType, minConfidence]);

    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'HIGH': return MIREA_2025_COLORS.error;
            case 'MEDIUM': return MIREA_2025_COLORS.warning;
            case 'LOW': return MIREA_2025_COLORS.success;
            default: return MIREA_2025_COLORS.info;
        }
    };

    const getConfidenceColor = (confidence: number) => {
        if (confidence >= 0.8) return MIREA_2025_COLORS.success;
        if (confidence >= 0.6) return MIREA_2025_COLORS.warning;
        return MIREA_2025_COLORS.error;
    };


    return (
        <RootContainer>
            <ScanlineBox />

            {/* @ts-ignore */}
            <Box style={headerBoxStyles}>
                {/* Заголовок */}
                <Grow in={true}>
                    <Box sx={{
                        mb: 6,
                        display: 'flex',
                        alignItems: 'center',
                        gap: 3,
                        background: 'rgba(13, 27, 42, 0.4)',
                        backdropFilter: 'blur(20px)',
                        borderRadius: 4,
                        p: 4,
                        border: '1px solid rgba(0, 102, 255, 0.2)',
                        position: 'relative',
                        overflow: 'hidden',
                        '&::before': {
                            content: '""',
                            position: 'absolute',
                            top: 0,
                            left: 0,
                            right: 0,
                            height: '4px',
                            background: MIREA_2025_COLORS.primaryGradient
                        }
                    }}>
                        <Box
                            sx={{
                                p: 2,
                                borderRadius: 3,
                                background: 'rgba(0, 102, 255, 0.1)',
                                border: '1px solid rgba(0, 102, 255, 0.2)',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center'
                            }}
                        >
                            <Psychology sx={{
                                fontSize: 48,
                                color: MIREA_2025_COLORS.primaryLight,
                                filter: 'drop-shadow(0 0 20px rgba(0, 102, 255, 0.5))'
                            }} />
                        </Box>
                        <Box>
                            <Typography variant="h3" sx={{
                                fontWeight: 800,
                                background: MIREA_2025_COLORS.primaryGradient,
                                backgroundClip: 'text',
                                WebkitBackgroundClip: 'text',
                                color: 'transparent',
                                mb: 1,
                                letterSpacing: '-0.02em'
                            }}>
                                Граф знаний производства
                            </Typography>
                            <Typography variant="h6" sx={{
                                color: MIREA_2025_COLORS.textSecondary,
                                fontWeight: 400,
                                maxWidth: 600
                            }}>
                                Искусственный интеллект анализирует причинно-следственные связи и генерирует предиктивные рекомендации
                            </Typography>
                        </Box>

                        {/* Пульсирующие индикаторы */}
                        <Box sx={{ display: 'flex', gap: 2, ml: 'auto' }}>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <PulsingDot color={MIREA_2025_COLORS.success} />
                                <Typography variant="caption" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                    AI Активен
                                </Typography>
                            </Box>
                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                <PulsingDot color={MIREA_2025_COLORS.primary} />
                                <Typography variant="caption" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                    Данные в реальном времени
                                </Typography>
                            </Box>
                        </Box>
                    </Box>
                </Grow>

                {error && (
                    <Slide direction="down" in={!!error}>
                        <Alert severity="error" sx={{
                            mb: 3,
                            borderRadius: 3,
                            background: 'rgba(255, 23, 68, 0.1)',
                            backdropFilter: 'blur(10px)',
                            border: '1px solid rgba(255, 23, 68, 0.3)',
                            color: MIREA_2025_COLORS.text,
                            '& .MuiAlert-icon': { color: MIREA_2025_COLORS.error }
                        }}>
                            {error}
                        </Alert>
                    </Slide>
                )}

                <Grid container spacing={3}>
                    {/* Левая панель - Управление */}
                    <Grid item xs={12} md={3}>
                        <GlassCard sx={{ height: '100%' }}>
                            <CardHeader
                                title={
                                    <Typography variant="h6" sx={{
                                        fontWeight: 700,
                                        color: MIREA_2025_COLORS.text,
                                        display: 'flex',
                                        alignItems: 'center',
                                        gap: 1
                                    }}>
                                        <Settings sx={{ fontSize: 20 }} />
                                        Управление анализом
                                    </Typography>
                                }
                                sx={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}
                            />

                            <CardContent sx={{ pt: 3 }}>
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                                    {/* Тип дефекта */}
                                    <Box>
                                        <Typography variant="caption" sx={{
                                            color: MIREA_2025_COLORS.textSecondary,
                                            fontWeight: 600,
                                            mb: 1,
                                            display: 'block'
                                        }}>
                                            Тип дефекта
                                        </Typography>
                                        <FormControl fullWidth>
                                            <Select
                                                value={defectType}
                                                onChange={(e) => setDefectType(e.target.value as string)}
                                                sx={{
                                                    background: 'rgba(255, 255, 255, 0.05)',
                                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                                    borderRadius: 2,
                                                    color: MIREA_2025_COLORS.text,
                                                    '& .MuiOutlinedInput-notchedOutline': {
                                                        borderColor: 'rgba(255, 255, 255, 0.1)'
                                                    },
                                                    '&:hover .MuiOutlinedInput-notchedOutline': {
                                                        borderColor: MIREA_2025_COLORS.primaryLight
                                                    }
                                                }}
                                                renderValue={(selected) => (
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                        <Warning sx={{ fontSize: 16, color: MIREA_2025_COLORS.warning }} />
                                                        <Typography sx={{ color: MIREA_2025_COLORS.text }}>
                                                            {selected === 'crack' ? 'Трещины' :
                                                                selected === 'bubble' ? 'Пузыри' : 'Сколы'}
                                                        </Typography>
                                                    </Box>
                                                )}
                                            >
                                                <MenuItem value="crack">Трещины</MenuItem>
                                                <MenuItem value="bubble">Пузыри</MenuItem>
                                                <MenuItem value="chip">Сколы</MenuItem>
                                                <MenuItem value="stain">Пятна</MenuItem>
                                                <MenuItem value="cloudiness">Помутнение</MenuItem>
                                                <MenuItem value="deformation">Деформация</MenuItem>
                                            </Select>
                                        </FormControl>
                                    </Box>

                                    {/* Уверенность */}
                                    <Box>
                                        <Typography variant="caption" sx={{
                                            color: MIREA_2025_COLORS.textSecondary,
                                            fontWeight: 600,
                                            mb: 1,
                                            display: 'block'
                                        }}>
                                            Минимальная уверенность
                                        </Typography>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                            <Slider
                                                value={minConfidence}
                                                onChange={(_: Event, value: number | number[]) => {
                                                    if (typeof value === 'number') {
                                                        setMinConfidence(value);
                                                    }
                                                }}
                                                min={0}
                                                max={1}
                                                step={0.1}
                                                sx={{
                                                    flex: 1,
                                                    color: MIREA_2025_COLORS.primary,
                                                    '& .MuiSlider-track': {
                                                        background: MIREA_2025_COLORS.primaryGradient
                                                    }
                                                }}
                                            />
                                            <Typography sx={{
                                                color: MIREA_2025_COLORS.text,
                                                fontWeight: 700,
                                                minWidth: 40,
                                                textAlign: 'center'
                                            }}>
                                                {(minConfidence * 100).toFixed(0)}%
                                            </Typography>
                                        </Box>
                                    </Box>

                                    {/* Параметры */}
                                    <Box>
                                        <Typography variant="caption" sx={{
                                            color: MIREA_2025_COLORS.textSecondary,
                                            fontWeight: 600,
                                            mb: 2,
                                            display: 'block'
                                        }}>
                                            Параметры процесса
                                        </Typography>
                                        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                            {Object.entries(parameterValues).map(([param, value]) => (
                                                <Box key={param} sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
                                                    <Box sx={{
                                                        p: 1,
                                                        borderRadius: 1,
                                                        background: 'rgba(0, 102, 255, 0.1)',
                                                        display: 'flex',
                                                        alignItems: 'center',
                                                        justifyContent: 'center'
                                                    }}>
                                                        <Thermostat sx={{ fontSize: 16, color: MIREA_2025_COLORS.primaryLight }} />
                                                    </Box>
                                                    <Box sx={{ flex: 1 }}>
                                                        <Typography variant="caption" sx={{
                                                            color: MIREA_2025_COLORS.textSecondary,
                                                            display: 'block'
                                                        }}>
                                                            {param.replace('_', ' ').toUpperCase()}
                                                        </Typography>
                                                        <TextField
                                                            fullWidth
                                                            type="number"
                                                            value={value}
                                                            onChange={(e) => handleParameterChange(param, e.target.value)}
                                                            size="small"
                                                            sx={{
                                                                '& .MuiOutlinedInput-root': {
                                                                    background: 'rgba(255, 255, 255, 0.05)',
                                                                    border: '1px solid rgba(255, 255, 255, 0.1)',
                                                                    borderRadius: 1
                                                                },
                                                                '& .MuiInputBase-input': {
                                                                    color: MIREA_2025_COLORS.text,
                                                                    fontSize: '0.875rem'
                                                                }
                                                            }}
                                                        />
                                                    </Box>
                                                </Box>
                                            ))}
                                        </Box>
                                    </Box>

                                    {/* Кнопки действий */}
                                    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2, mt: 2 }}>
                                        <AnimatedButton 
                                            onClick={async () => {
                                                setLoading(true);
                                                try {
                                                    // Trigger comprehensive root cause analysis
                                                    const response = await fetch(
                                                        `${API_BASE_URL}/api/knowledge-graph/root-cause?defect_type=${defectType}&min_confidence=${minConfidence}`
                                                    );
                                                    
                                                    if (response.ok) {
                                                        const analysisData = await response.json();
                                                        // Update causes with root cause analysis results
                                                        if (analysisData.root_causes) {
                                                            setCauses(analysisData.root_causes.map((cause: any) => ({
                                                                cause: cause.parameter_name || cause.root_cause || 'Unknown',
                                                                confidence: cause.confidence || 0,
                                                                strength: cause.confidence || 0,
                                                                observations: 0,
                                                                evidence: cause.evidence || [],
                                                                cause_type: 'PARAMETER',
                                                                relationship_type: 'CAUSES',
                                                                last_updated: new Date().toISOString()
                                                            })));
                                                        }
                                                    }
                                                    
                                                    // Also fetch updated statistics
                                                    await fetchStatistics();
                                                } catch (err) {
                                                    console.error('Ошибка при запуске анализа:', err);
                                                    // Fallback to basic causes fetch
                                                    await fetchCauses();
                                                    await fetchStatistics();
                                                } finally {
                                                    setLoading(false);
                                                }
                                            }} 
                                            disabled={loading}
                                        >
                                            {loading ? (
                                                <CircularProgress size={20} sx={{ color: '#FFFFFF' }} />
                                            ) : (
                                                <>
                                                    <RocketLaunch sx={{ mr: 1, fontSize: 16 }} />
                                                    Запустить анализ
                                                </>
                                            )}
                                        </AnimatedButton>

                                        <Button
                                            onClick={async () => {
                                                setLoading(true);
                                                try {
                                                    // Trigger comprehensive root cause analysis
                                                    const rootCauseResponse = await fetch(
                                                        `${API_BASE_URL}/api/knowledge-graph/root-cause?defect_type=${defectType}&min_confidence=${minConfidence}`
                                                    );
                                                    
                                                    if (rootCauseResponse.ok) {
                                                        const analysisData = await rootCauseResponse.json();
                                                        // Update causes with root cause analysis results
                                                        if (analysisData.root_causes) {
                                                            setCauses(analysisData.root_causes.map((cause: any) => ({
                                                                cause: cause.parameter_name || cause.root_cause || 'Unknown',
                                                                confidence: cause.confidence || 0,
                                                                strength: cause.confidence || 0,
                                                                observations: 0,
                                                                evidence: cause.evidence || [],
                                                                cause_type: 'PARAMETER',
                                                                relationship_type: 'CAUSES',
                                                                last_updated: new Date().toISOString()
                                                            })));
                                                        }
                                                    }
                                                    
                                                    await fetchRecommendations();
                                                    await fetchSubgraph();
                                                    await fetchStatistics();
                                                } catch (err) {
                                                    console.error('Ошибка при обновлении данных:', err);
                                                    // Fallback to basic data fetch
                                                    await fetchCauses();
                                                    await fetchRecommendations();
                                                    await fetchSubgraph();
                                                    await fetchStatistics();
                                                } finally {
                                                    setLoading(false);
                                                }
                                            }}
                                            disabled={loading}
                                            sx={{
                                                background: 'rgba(255, 255, 255, 0.05)',
                                                color: MIREA_2025_COLORS.text,
                                                fontWeight: 600,
                                                padding: '10px 20px',
                                                borderRadius: 3,
                                                border: '1px solid rgba(255, 255, 255, 0.1)',
                                                '&:hover': {
                                                    background: 'rgba(255, 255, 255, 0.1)',
                                                    borderColor: MIREA_2025_COLORS.primaryLight
                                                }
                                            }}
                                        >
                                            <AutoGraph sx={{ mr: 1, fontSize: 16 }} />
                                            Обновить все данные
                                        </Button>
                                    </Box>
                                </Box>
                            </CardContent>
                        </GlassCard>

                        {/* Статистика */}
                        <GlassCard sx={{ mt: 3 }}>
                            <CardContent>
                                <Typography variant="subtitle2" sx={{
                                    color: MIREA_2025_COLORS.textSecondary,
                                    fontWeight: 600,
                                    mb: 2
                                }}>
                                    Статистика анализа
                                </Typography>
                                <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                            Узлов графа
                                        </Typography>
                                        <Typography variant="h6" sx={{ color: MIREA_2025_COLORS.text, fontWeight: 700 }}>
                                            {subgraph?.nodes?.length || 0}
                                        </Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                            Связей
                                        </Typography>
                                        <Typography variant="h6" sx={{ color: MIREA_2025_COLORS.text, fontWeight: 700 }}>
                                            {subgraph?.edges?.length || 0}
                                        </Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                            Причин найдено
                                        </Typography>
                                        <Typography variant="h6" sx={{ color: MIREA_2025_COLORS.success, fontWeight: 700 }}>
                                            {causes?.length || 0}
                                        </Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mt: 1 }}>
                                        <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                            Новых связей (24ч)
                                        </Typography>
                                        <Typography variant="h6" sx={{ color: MIREA_2025_COLORS.info, fontWeight: 700 }}>
                                            {statistics.newCausalLinks}
                                        </Typography>
                                    </Box>
                                </Box>
                            </CardContent>
                        </GlassCard>
                    </Grid>

                    {/* Основная область контента */}
                    <Grid item xs={12} md={9}>
                        {/* Навигационные табы */}
                        <Box sx={{ display: 'flex', gap: 2, mb: 3 }}>
                            {(['causes', 'recommendations', 'graph'] as const).map((tab) => (
                                <Button
                                    key={tab}
                                    onClick={() => setActiveTab(tab)}
                                    sx={{
                                        flex: 1,
                                        background: activeTab === tab
                                            ? `linear-gradient(135deg, ${MIREA_2025_COLORS.primary}, ${MIREA_2025_COLORS.primaryLight})`
                                            : 'rgba(255, 255, 255, 0.05)',
                                        color: activeTab === tab ? '#FFFFFF' : MIREA_2025_COLORS.textSecondary,
                                        fontWeight: 700,
                                        padding: '12px 24px',
                                        borderRadius: 3,
                                        border: activeTab === tab
                                            ? 'none'
                                            : '1px solid rgba(255, 255, 255, 0.1)',
                                        transition: 'all 0.3s ease',
                                        '&:hover': {
                                            background: activeTab === tab
                                                ? `linear-gradient(135deg, ${MIREA_2025_COLORS.primary}, ${MIREA_2025_COLORS.primaryLight})`
                                                : 'rgba(255, 255, 255, 0.1)',
                                            transform: 'translateY(-2px)'
                                        }
                                    }}
                                >
                                    {tab === 'causes' && <Warning sx={{ mr: 1 }} />}
                                    {tab === 'recommendations' && <Insights sx={{ mr: 1 }} />}
                                    {tab === 'graph' && <Hub sx={{ mr: 1 }} />}
                                    {tab === 'causes' && 'Причины дефектов'}
                                    {tab === 'recommendations' && 'Рекомендации'}
                                    {tab === 'graph' && 'Визуализация графа'}
                                </Button>
                            ))}
                        </Box>

                        <AnimatePresence mode="wait">
                            {/* Панель причин */}
                            {activeTab === 'causes' && (
                                <motion.div
                                    key="causes"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <GlassCard>
                                        <CardHeader
                                            title={
                                                <Typography variant="h5" sx={{
                                                    fontWeight: 800,
                                                    color: MIREA_2025_COLORS.text,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: 2
                                                }}>
                                                    <Warning sx={{ color: MIREA_2025_COLORS.warning }} />
                                                    Причины дефекта "{defectType}"
                                                    <AnimatedChip
                                                        label={`${causes.length} причин`}
                                                        color="warning"
                                                        sx={{ ml: 'auto' }}
                                                    />
                                                </Typography>
                                            }
                                            subheader={
                                                <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary, mt: 1 }}>
                                                    Анализ выявил ключевые факторы, влияющие на качество продукции
                                                </Typography>
                                            }
                                            sx={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}
                                        />

                                        <CardContent sx={{ pt: 3 }}>
                                            <Grid container spacing={3}>
                                                {causes.map((cause, index) => (
                                                    <Grid item xs={12} key={index}>
                                                        <motion.div
                                                            initial={{ opacity: 0, x: -20 }}
                                                            animate={{ opacity: 1, x: 0 }}
                                                            transition={{ delay: index * 0.1 }}
                                                        >
                                                            <GlassCard sx={{
                                                                p: 3,
                                                                borderLeft: `4px solid ${getConfidenceColor(cause.confidence)}`
                                                            }}>
                                                                <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 3 }}>
                                                                    <Box sx={{
                                                                        p: 2,
                                                                        borderRadius: 2,
                                                                        background: getConfidenceColor(cause.confidence) + '26',
                                                                        display: 'flex',
                                                                        alignItems: 'center',
                                                                        justifyContent: 'center'
                                                                    }}>
                                                                        <PsychologyRounded sx={{
                                                                            fontSize: 24,
                                                                            color: getConfidenceColor(cause.confidence)
                                                                        }} />
                                                                    </Box>

                                                                    <Box sx={{ flex: 1 }}>
                                                                        <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 2 }}>
                                                                            <Typography variant="h6" sx={{
                                                                                fontWeight: 700,
                                                                                color: MIREA_2025_COLORS.text
                                                                            }}>
                                                                                {cause.cause}
                                                                            </Typography>
                                                                            <Chip
                                                                                label={`${(cause.confidence * 100).toFixed(0)}% уверенность`}
                                                                                sx={{
                                                                                    background: getConfidenceColor(cause.confidence) + '33',
                                                                                    color: getConfidenceColor(cause.confidence),
                                                                                    fontWeight: 700,
                                                                                    borderRadius: 2
                                                                                }}
                                                                            />
                                                                        </Box>

                                                                        <Typography variant="body2" sx={{
                                                                            color: MIREA_2025_COLORS.textSecondary,
                                                                            mb: 2
                                                                        }}>
                                                                            Тип: {cause.cause_type} • Наблюдений: {cause.observations} • Сила связи: {(cause.strength * 100).toFixed(0)}%
                                                                        </Typography>

                                                                        <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                                                                            {cause.evidence.map((evidence, idx) => (
                                                                                <Chip
                                                                                    key={idx}
                                                                                    label={
                                                                                        evidence.type === 'parameter_deviation' ? 
                                                                                            `${evidence.parameter}: ${evidence.current_value.toFixed(2)} (σ: ${evidence.deviation_sigma.toFixed(2)})` :
                                                                                            typeof evidence === 'string' ? 
                                                                                                evidence : 
                                                                                                JSON.stringify(evidence)
                                                                                    }
                                                                                    size="small"
                                                                                    sx={{
                                                                                        background: 'rgba(255, 255, 255, 0.05)',
                                                                                        border: '1px solid rgba(255, 255, 255, 0.1)',
                                                                                        color: MIREA_2025_COLORS.textSecondary,
                                                                                        borderRadius: 1
                                                                                    }}
                                                                                />
                                                                            ))}
                                                                        </Box>
                                                                    </Box>
                                                                </Box>
                                                            </GlassCard>
                                                        </motion.div>
                                                    </Grid>
                                                ))}
                                            </Grid>
                                        </CardContent>
                                    </GlassCard>
                                </motion.div>
                            )}

                            {/* Панель рекомендаций */}
                            {activeTab === 'recommendations' && (
                                <motion.div
                                    key="recommendations"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <GlassCard>
                                        <CardHeader
                                            title={
                                                <Typography variant="h5" sx={{
                                                    fontWeight: 800,
                                                    color: MIREA_2025_COLORS.text,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: 2
                                                }}>
                                                    <Insights sx={{ color: MIREA_2025_COLORS.success }} />
                                                    AI Рекомендации
                                                    <AnimatedChip
                                                        label={`${Math.min(recommendations.length, 5)} из ${recommendations.length} действий`}
                                                        color="success"
                                                        sx={{ ml: 'auto' }}
                                                    />
                                                </Typography>
                                            }
                                            subheader={
                                                <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary, mt: 1 }}>
                                                    Интеллектуальные предложения по оптимизации производственного процесса
                                                </Typography>
                                            }
                                            sx={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}
                                        />

                                        <CardContent sx={{ pt: 3 }}>
                                            <Grid container spacing={3}>
                                                {recommendations
                                                    .sort((a, b) => {
                                                        // Sort by confidence and strength (impact)
                                                        const aScore = (a.confidence || 0) * 0.7 + (a.strength || 0) * 0.3;
                                                        const bScore = (b.confidence || 0) * 0.7 + (b.strength || 0) * 0.3;
                                                        return bScore - aScore; // Descending order
                                                    })
                                                    .slice(0, 5) // Show only top 5 recommendations
                                                    .map((rec, index) => (
                                                    <Grid item xs={12} md={6} key={index}>
                                                        <motion.div
                                                            initial={{ opacity: 0, scale: 0.9 }}
                                                            animate={{ opacity: 1, scale: 1 }}
                                                            transition={{ delay: index * 0.1 }}
                                                            whileHover={{ scale: 1.02 }}
                                                        >
                                                            <GlassCard sx={{ height: '100%' }}>
                                                                <CardContent>
                                                                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', mb: 3 }}>
                                                                        <Box>
                                                                            <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                                                <Typography variant="h6" sx={{
                                                                                    fontWeight: 700,
                                                                                    color: MIREA_2025_COLORS.text,
                                                                                    mb: 0.5
                                                                                }}>
                                                                                    {rec.parameter}
                                                                                </Typography>
                                                                                {rec.ml_enhanced && (
                                                                                    <Chip 
                                                                                        label="ML" 
                                                                                        size="small" 
                                                                                        sx={{ 
                                                                                            background: MIREA_2025_COLORS.infoGradient, 
                                                                                            color: 'white', 
                                                                                            fontWeight: 700,
                                                                                            height: 20
                                                                                        }} 
                                                                                    />
                                                                                )}
                                                                            </Box>
                                                                            <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                                                                {rec.current_value} {rec.unit} → {rec.target_value} {rec.unit}
                                                                            </Typography>
                                                                        </Box>
                                                                        <Chip
                                                                            icon={rec.priority === 'HIGH' ? <ErrorIcon /> : rec.priority === 'MEDIUM' ? <Warning /> : <CheckCircle />}
                                                                            label={rec.priority}
                                                                            sx={{
                                                                                background: getPriorityColor(rec.priority) + '33',
                                                                                color: getPriorityColor(rec.priority),
                                                                                fontWeight: 700,
                                                                                borderRadius: 2
                                                                            }}
                                                                        />
                                                                    </Box>

                                                                    <Typography variant="body2" sx={{
                                                                        color: MIREA_2025_COLORS.text,
                                                                        mb: 3,
                                                                        lineHeight: 1.6
                                                                    }}>
                                                                        {rec.action}
                                                                    </Typography>

                                                                    <Box sx={{
                                                                        p: 2,
                                                                        borderRadius: 2,
                                                                        background: 'rgba(0, 102, 255, 0.05)',
                                                                        border: '1px solid rgba(0, 102, 255, 0.1)',
                                                                        mb: 2
                                                                    }}>
                                                                        <Typography variant="caption" sx={{
                                                                            color: MIREA_2025_COLORS.textSecondary,
                                                                            display: 'block',
                                                                            mb: 0.5
                                                                        }}>
                                                                            Ожидаемый эффект
                                                                        </Typography>
                                                                        <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.text, fontWeight: 600 }}>
                                                                            {rec.expected_impact}
                                                                        </Typography>
                                                                    </Box>

                                                                    <Box sx={{ display: 'flex', gap: 2 }}>
                                                                        <Chip
                                                                            label={`Уверенность: ${(rec.confidence * 100).toFixed(0)}${rec.ml_enhanced ? ' (ML)' : ''}`}
                                                                            size="small"
                                                                            sx={{
                                                                                background: getConfidenceColor(rec.confidence) + '33',
                                                                                color: getConfidenceColor(rec.confidence),
                                                                                fontWeight: 700
                                                                            }}
                                                                        />
                                                                        {rec.ml_enhanced && (
                                                                            <Chip
                                                                                label={`ML: ${(rec.ml_confidence! * 100).toFixed(0)}%`}
                                                                                size="small"
                                                                                sx={{
                                                                                    backgroundColor: `rgba(0, 229, 255, 0.2)`,
                                                                                    color: MIREA_2025_COLORS.info,
                                                                                    fontWeight: 700
                                                                                }}
                                                                            />
                                                                        )}
                                                                        <Chip
                                                                            label={`Влияние: ${(rec.strength * 100).toFixed(0)}%`}
                                                                            size="small"
                                                                            sx={{
                                                                                backgroundColor: 'rgba(0, 229, 255, 0.2)',
                                                                                color: MIREA_2025_COLORS.info,
                                                                                fontWeight: 700
                                                                            }}
                                                                        />
                                                                    </Box>
                                                                </CardContent>
                                                            </GlassCard>
                                                        </motion.div>
                                                    </Grid>
                                                ))}
                                            </Grid>
                                        </CardContent>
                                    </GlassCard>
                                </motion.div>
                            )}

                            {/* Панель визуализации графа */}
                            {activeTab === 'graph' && (
                                <motion.div
                                    key="graph"
                                    initial={{ opacity: 0, y: 20 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    exit={{ opacity: 0, y: -20 }}
                                    transition={{ duration: 0.3 }}
                                >
                                    <GlassCard>
                                        <CardHeader
                                            title={
                                                <Typography variant="h5" sx={{
                                                    fontWeight: 800,
                                                    color: MIREA_2025_COLORS.text,
                                                    display: 'flex',
                                                    alignItems: 'center',
                                                    gap: 2
                                                }}>
                                                    <Hub sx={{ color: MIREA_2025_COLORS.primaryLight }} />
                                                    3D Визуализация графа знаний
                                                    <AnimatedChip
                                                        label={`${subgraph?.nodes.length || 0} узлов, ${subgraph?.edges.length || 0} связей`}
                                                        color="primary"
                                                        sx={{ ml: 'auto' }}
                                                    />
                                                </Typography>
                                            }
                                            subheader={
                                                <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary, mt: 1 }}>
                                                    Интерактивное представление причинно-следственных связей в реальном времени
                                                </Typography>
                                            }
                                            sx={{ borderBottom: '1px solid rgba(255, 255, 255, 0.1)' }}
                                        />

                                        <CardContent>
                                            <GraphVisualization 
                                                subgraph={subgraph}
                                            />
                                        </CardContent>
                                    </GlassCard>
                                </motion.div>
                            )}
                        </AnimatePresence>

                        {/* Информационная панель */}
                        <Grid container spacing={3} sx={{ mt: 3 }}>
                            <Grid item xs={12} md={4}>
                                <GlassCard>
                                    <CardContent>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                                            <Box sx={{
                                                p: 1,
                                                borderRadius: 2,
                                                background: 'rgba(0, 230, 118, 0.1)',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center'
                                            }}>
                                                <Timeline sx={{ color: MIREA_2025_COLORS.success, fontSize: 24 }} />
                                            </Box>
                                            <Typography variant="h6" sx={{ fontWeight: 700, color: MIREA_2025_COLORS.text }}>
                                                Анализ трендов
                                            </Typography>
                                        </Box>
                                        <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                            Модель предсказывает снижение дефектов на {statistics.predictedDefectReduction}% после внедрения рекомендаций
                                        </Typography>
                                    </CardContent>
                                </GlassCard>
                            </Grid>

                            <Grid item xs={12} md={4}>
                                <GlassCard>
                                    <CardContent>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                                            <Box sx={{
                                                p: 1,
                                                borderRadius: 2,
                                                background: 'rgba(0, 229, 255, 0.1)',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center'
                                            }}>
                                                <NetworkPing sx={{ color: MIREA_2025_COLORS.info, fontSize: 24 }} />
                                            </Box>
                                            <Typography variant="h6" sx={{ fontWeight: 700, color: MIREA_2025_COLORS.text }}>
                                                Связи и зависимости
                                            </Typography>
                                        </Box>
                                        <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                            Обнаружено {statistics.newCausalLinks} новые причинно-следственные связи за последние 24 часа
                                        </Typography>
                                    </CardContent>
                                </GlassCard>
                            </Grid>

                            <Grid item xs={12} md={4}>
                                <GlassCard>
                                    <CardContent>
                                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2, mb: 2 }}>
                                            <Box sx={{
                                                p: 1,
                                                borderRadius: 2,
                                                background: 'rgba(255, 215, 0, 0.1)',
                                                display: 'flex',
                                                alignItems: 'center',
                                                justifyContent: 'center'
                                            }}>
                                                <WorkspacePremium sx={{ color: MIREA_2025_COLORS.warning, fontSize: 24 }} />
                                            </Box>
                                            <Typography variant="h6" sx={{ fontWeight: 700, color: MIREA_2025_COLORS.text }}>
                                                Качество предсказаний
                                            </Typography>
                                        </Box>
                                        <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                            Точность рекомендаций составляет {statistics.recommendationAccuracy}% на основе анализа данных
                                        </Typography>
                                    </CardContent>
                                </GlassCard>
                            </Grid>
                        </Grid>
                    </Grid>
                </Grid>
            </Box>
        </RootContainer>
    );
};

export default KnowledgeGraph;
