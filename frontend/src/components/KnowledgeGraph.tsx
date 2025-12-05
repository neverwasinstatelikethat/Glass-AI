import React, { useState, useEffect } from 'react';
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
    styled
} from '@mui/material';
import {
    Psychology,
    Info,
    TrendingUp,
    Warning,
    CheckCircle,
    Error,
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
    Thermostat
} from '@mui/icons-material';
import { motion, AnimatePresence } from 'framer-motion';

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
interface CauseData {
    cause: string;
    confidence: number;
    strength: number;
    observations: number;
    evidence: string[];
    cause_type: string;
}

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
}

interface GraphNode {
    id: number;
    name: string;
    label: string;
    nodeType: 'defect' | 'cause' | 'parameter';
    confidence: number;
    properties?: Record<string, any>;
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

    const [causes, setCauses] = useState<CauseData[]>([
        {
            cause: "Высокая температура печи",
            confidence: 0.92,
            strength: 0.85,
            observations: 145,
            evidence: ["Температура превышает 1600°C", "Быстрый нагрев"],
            cause_type: "ТЕМПЕРАТУРНЫЙ"
        },
        {
            cause: "Неравномерное охлаждение",
            confidence: 0.78,
            strength: 0.72,
            observations: 89,
            evidence: ["Разность температур >50°C", "Ветер в цеху"],
            cause_type: "ТЕПЛОВОЙ"
        },
        {
            cause: "Недостаточное давление",
            confidence: 0.65,
            strength: 0.68,
            observations: 67,
            evidence: ["Давление <45 bar", "Колебания давления"],
            cause_type: "МЕХАНИЧЕСКИЙ"
        }
    ]);

    const [recommendations, setRecommendations] = useState<RecommendationData[]>([
        {
            parameter: "Температура печи",
            current_value: 1580,
            target_value: 1550,
            unit: "°C",
            action: "Постепенно снизить температуру на 30°C в течение 2 часов",
            confidence: 0.88,
            strength: 0.82,
            priority: "HIGH",
            expected_impact: "Снижение трещин на 45%"
        },
        {
            parameter: "Скорость ленты",
            current_value: 150,
            target_value: 145,
            unit: "м/мин",
            action: "Уменьшить скорость конвейера на 5 м/мин",
            confidence: 0.75,
            strength: 0.68,
            priority: "MEDIUM",
            expected_impact: "Улучшение качества на 25%"
        }
    ]);

    const [subgraph, setSubgraph] = useState<GraphData | null>({
        defect: "crack",
        nodes: Array.from({ length: 15 }, (_, i) => ({
            id: i,
            name: `Узел ${i}`,
            label: `Узел ${i}`,
            nodeType: i % 3 === 0 ? 'defect' : i % 3 === 1 ? 'cause' : 'parameter',
            confidence: 0.7 + Math.random() * 0.3,
            properties: {}
        })),
        edges: Array.from({ length: 20 }, (_, i) => ({
            id: i,
            source: Math.floor(Math.random() * 15),
            target: Math.floor(Math.random() * 15),
            type: 'relation',
            confidence: 0.5 + Math.random() * 0.5,
            strength: 0.5 + Math.random() * 0.5
        }))
    });

    const [loading, setLoading] = useState<boolean>(false);
    const [error, setError] = useState<string | null>(null);
    const [activeTab, setActiveTab] = useState<'causes' | 'recommendations' | 'graph'>('causes');

    const fetchCauses = async () => {
        setLoading(true);
        setError(null);

        try {
            // В реальном приложении здесь будет вызов API
            await new Promise(resolve => setTimeout(resolve, 1000));
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
            await new Promise(resolve => setTimeout(resolve, 1000));
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
            await new Promise(resolve => setTimeout(resolve, 1000));
        } catch (err) {
            setError('Ошибка при загрузке графа знаний');
            console.error(err);
        } finally {
            setLoading(false);
        }
    };

    const handleParameterChange = (param: string, value: string) => {
        setParameterValues(prev => ({
            ...prev,
            [param]: parseFloat(value) || 0
        }));
    };

    useEffect(() => {
        fetchCauses();
        fetchRecommendations();
        fetchSubgraph();
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

    // Анимированные узлы для визуализации графа
    const GraphVisualization = () => (
        <Box sx={{ position: 'relative', width: '100%', height: 400, overflow: 'hidden', borderRadius: 4 }}>
            {/* Фоновый градиент */}
            <Box
                sx={{
                    position: 'absolute',
                    top: 0,
                    left: 0,
                    right: 0,
                    bottom: 0,
                    background: `radial-gradient(circle at 30% 20%, ${alpha(MIREA_2025_COLORS.primary, 0.1)} 0%, transparent 50%),
                      radial-gradient(circle at 70% 80%, ${alpha(MIREA_2025_COLORS.secondary, 0.1)} 0%, transparent 50%)`,
                    animation: 'float 20s ease-in-out infinite',
                    '@keyframes float': {
                        '0%, 100%': { transform: 'translate(0, 0)' },
                        '50%': { transform: 'translate(20px, 20px)' }
                    }
                }}
            />

            {/* Анимированные линии */}
            {Array.from({ length: 20 }).map((_, i) => (
                <motion.div
                    key={i}
                    style={{
                        position: 'absolute',
                        top: `${Math.random() * 100}%`,
                        left: `${Math.random() * 100}%`,
                        width: 2,
                        height: 50 + Math.random() * 100,
                        background: `linear-gradient(to bottom, transparent, ${MIREA_2025_COLORS.primaryLight}, transparent)`,
                        transform: `rotate(${Math.random() * 360}deg)`,
                        opacity: 0.3
                    }}
                    animate={{
                        y: [0, -50, 0],
                        opacity: [0.3, 0.6, 0.3]
                    }}
                    transition={{
                        duration: 3 + Math.random() * 4,
                        repeat: Infinity,
                        delay: i * 0.1
                    }}
                />
            ))}

            {/* Узлы графа */}
            {subgraph?.nodes.slice(0, 8).map((node, i) => {
                const angle = (i / 8) * Math.PI * 2;
                const radius = 120;
                const x = Math.cos(angle) * radius + 200;
                const y = Math.sin(angle) * radius + 200;

                return (
                    <motion.div
                        key={node.id}
                        style={{
                            position: 'absolute',
                            left: x,
                            top: y,
                            width: 40 + (node.confidence || 0) * 40,
                            height: 40 + (node.confidence || 0) * 40,
                            background: node.nodeType === 'defect'
                                ? MIREA_2025_COLORS.errorGradient
                                : node.nodeType === 'cause'
                                    ? MIREA_2025_COLORS.warningGradient
                                    : MIREA_2025_COLORS.primaryGradient,
                            borderRadius: '50%',
                            display: 'flex',
                            alignItems: 'center',
                            justifyContent: 'center',
                            color: '#FFFFFF',
                            fontWeight: 'bold',
                            fontSize: 12,
                            cursor: 'pointer',
                            boxShadow: `0 0 30px ${node.nodeType === 'defect' ? MIREA_2025_COLORS.error : node.nodeType === 'cause' ? MIREA_2025_COLORS.warning : MIREA_2025_COLORS.primary}80`
                        }}
                        animate={{
                            scale: [1, 1.1, 1],
                            boxShadow: [
                                `0 0 20px ${node.nodeType === 'defect' ? MIREA_2025_COLORS.error : node.nodeType === 'cause' ? MIREA_2025_COLORS.warning : MIREA_2025_COLORS.primary}40`,
                                `0 0 40px ${node.nodeType === 'defect' ? MIREA_2025_COLORS.error : node.nodeType === 'cause' ? MIREA_2025_COLORS.warning : MIREA_2025_COLORS.primary}60`,
                                `0 0 20px ${node.nodeType === 'defect' ? MIREA_2025_COLORS.error : node.nodeType === 'cause' ? MIREA_2025_COLORS.warning : MIREA_2025_COLORS.primary}40`
                            ]
                        }}
                        transition={{
                            duration: 2,
                            repeat: Infinity,
                            delay: i * 0.2
                        }}
                        whileHover={{ scale: 1.2 }}
                    >
                        {node.label}
                    </motion.div>
                );
            })}

            {/* Центральный узел */}
            <motion.div
                style={{
                    position: 'absolute',
                    left: 200,
                    top: 200,
                    width: 80,
                    height: 80,
                    background: MIREA_2025_COLORS.primaryGradient,
                    borderRadius: '50%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    color: '#FFFFFF',
                    fontWeight: 'bold',
                    fontSize: 16,
                    boxShadow: `0 0 60px ${MIREA_2025_COLORS.primary}B3`
                }}
                animate={{
                    scale: [1, 1.05, 1],
                    rotate: [0, 360]
                }}
                transition={{
                    scale: {
                        duration: 2,
                        repeat: Infinity
                    },
                    rotate: {
                        duration: 20,
                        repeat: Infinity,
                        ease: "linear"
                    }
                }}
            >
                <Psychology sx={{ fontSize: 32 }} />
            </motion.div>
        </Box>
    );

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
                                        <AnimatedButton onClick={fetchCauses} disabled={loading}>
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
                                            onClick={() => {
                                                fetchCauses();
                                                fetchRecommendations();
                                                fetchSubgraph();
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
                                            156
                                        </Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                            Связей
                                        </Typography>
                                        <Typography variant="h6" sx={{ color: MIREA_2025_COLORS.text, fontWeight: 700 }}>
                                            892
                                        </Typography>
                                    </Box>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                                        <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                            Точность модели
                                        </Typography>
                                        <Typography variant="h6" sx={{ color: MIREA_2025_COLORS.success, fontWeight: 700 }}>
                                            94.2%
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
                                                                                    label={evidence}
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
                                                        label={`${recommendations.length} действий`}
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
                                                {recommendations.map((rec, index) => (
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
                                                                            <Typography variant="h6" sx={{
                                                                                fontWeight: 700,
                                                                                color: MIREA_2025_COLORS.text,
                                                                                mb: 0.5
                                                                            }}>
                                                                                {rec.parameter}
                                                                            </Typography>
                                                                            <Typography variant="body2" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                                                                {rec.current_value} {rec.unit} → {rec.target_value} {rec.unit}
                                                                            </Typography>
                                                                        </Box>
                                                                        <Chip
                                                                            icon={rec.priority === 'HIGH' ? <Error /> : rec.priority === 'MEDIUM' ? <Warning /> : <CheckCircle />}
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
                                                                            label={`Уверенность: ${(rec.confidence * 100).toFixed(0)}%`}
                                                                            size="small"
                                                                            sx={{
                                                                                background: getConfidenceColor(rec.confidence) + '33',
                                                                                color: getConfidenceColor(rec.confidence),
                                                                                fontWeight: 700
                                                                            }}
                                                                        />
                                                                        <Chip
                                                                            label={`Влияние: ${(rec.strength * 100).toFixed(0)}%`}
                                                                            size="small"
                                                                            sx={{
                                                                                background: 'rgba(0, 229, 255, 0.2)',
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
                                            <GraphVisualization />

                                            <Box sx={{ mt: 4, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                                                <Box sx={{ display: 'flex', gap: 3 }}>
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                        <Box sx={{
                                                            width: 12,
                                                            height: 12,
                                                            borderRadius: '50%',
                                                            background: MIREA_2025_COLORS.errorGradient
                                                        }} />
                                                        <Typography variant="caption" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                                            Дефекты
                                                        </Typography>
                                                    </Box>
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                        <Box sx={{
                                                            width: 12,
                                                            height: 12,
                                                            borderRadius: '50%',
                                                            background: MIREA_2025_COLORS.warningGradient
                                                        }} />
                                                        <Typography variant="caption" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                                            Причины
                                                        </Typography>
                                                    </Box>
                                                    <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                        <Box sx={{
                                                            width: 12,
                                                            height: 12,
                                                            borderRadius: '50%',
                                                            background: MIREA_2025_COLORS.primaryGradient
                                                        }} />
                                                        <Typography variant="caption" sx={{ color: MIREA_2025_COLORS.textSecondary }}>
                                                            Параметры
                                                        </Typography>
                                                    </Box>
                                                </Box>

                                                <Button
                                                    startIcon={<Polyline />}
                                                    onClick={fetchSubgraph}
                                                    disabled={loading}
                                                    sx={{
                                                        background: 'rgba(0, 102, 255, 0.1)',
                                                        color: MIREA_2025_COLORS.primaryLight,
                                                        fontWeight: 600,
                                                        borderRadius: 3,
                                                        border: '1px solid rgba(0, 102, 255, 0.3)',
                                                        '&:hover': {
                                                            background: 'rgba(0, 102, 255, 0.2)'
                                                        }
                                                    }}
                                                >
                                                    {loading ? 'Обновление...' : 'Обновить граф'}
                                                </Button>
                                            </Box>
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
                                            Модель предсказывает снижение дефектов на 34% после внедрения рекомендаций
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
                                            Обнаружено 23 новые причинно-следственные связи за последние 24 часа
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
                                            Точность рекомендаций увеличилась на 12% после обучения на новых данных
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
