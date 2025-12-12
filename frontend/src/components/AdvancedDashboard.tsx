import React, { useState, useEffect } from 'react';
import {
    Card,
    CardContent,
    Typography,
    Grid,
    Chip,
    Button,
    Box,
    LinearProgress,
    Avatar,
    Paper,
    Fade,
    Zoom,
    Grow,
    Slide,
    useTheme,
    alpha,
    IconButton,
    Tooltip,
    Skeleton
} from '@mui/material';
import * as MUIIcons from '@mui/icons-material';
import { styled } from '@mui/material/styles';
import {
    TrendingUp,
    Warning,
    CheckCircle,
    Factory,
    Speed,
    Thermostat,
    DeviceHub,
    Psychology,
    LocalFireDepartment,
    Science,
    BubbleChart,
    Timeline,
    AutoAwesome,
    ArrowUpward,
    ArrowDownward,
    Refresh,
    FilterList,
    MoreVert,
    School,
    RocketLaunch,
    Insights,
    Palette,
    WaterDrop,
    Diamond,
    Visibility,
    TrendingFlat,
    PlayCircle,
    PauseCircle
} from '@mui/icons-material';
import {
    LineChart,
    Line,
    PieChart,
    Pie,
    Cell,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip as RechartsTooltip,
    Legend,
    ResponsiveContainer,
    Area,
    AreaChart,
    RadialBarChart,
    RadialBar,
    PolarAngleAxis,
    PolarGrid,
    PolarRadiusAxis
} from 'recharts';
import type { Theme } from '@mui/material/styles';

import useDashboardData from '../hooks/useDashboardData';
import { useWebSocketStream } from '../hooks/useWebSocketStream';
import { useNotifications } from '../hooks/useNotifications';
// Removed unused imports
// import ExplainabilityPanel from './ExplainabilityPanel';
// import MetricsMonitor from './MetricsMonitor';
// import AutonomyStatus from './AutonomyStatus';

// Расширенная палитра РТУ МИРЭА 2025
const MIREA_2025_COLORS = {
    primary: '#0066FF',
    primaryGradient: 'linear-gradient(135deg, #0066FF 0%, #00E5FF 100%)',
    primaryGlass: 'rgba(0, 102, 255, 0.15)',
    secondary: '#FF3366',
    secondaryGradient: 'linear-gradient(135deg, #FF3366 0%, #FF9E6D 100%)',
    tertiary: '#00E676',
    tertiaryGradient: 'linear-gradient(135deg, #00E676 0%, #6EFFB2 100%)',
    warning: '#FFD700',
    warningGradient: 'linear-gradient(135deg, #FFD700 0%, #FFEE4D 100%)',
    error: '#FF1744',
    errorGradient: 'linear-gradient(135deg, #FF1744 0%, #FF5C7C 100%)',
    info: '#00E5FF',
    infoGradient: 'linear-gradient(135deg, #00E5FF 0%, #6EFFFF 100%)',
    dark: '#000814',
    darkSurface: 'rgba(13, 27, 42, 0.7)',
    glassSurface: 'rgba(255, 255, 255, 0.05)',
    lightSurface: 'rgba(255, 255, 255, 0.1)',
    textPrimary: '#FFFFFF',
    textSecondary: 'rgba(255, 255, 255, 0.8)',
    textTertiary: 'rgba(255, 255, 255, 0.6)',
    accentPurple: '#9D4EDD',
    accentCyan: '#00E5FF',
    accentPink: '#FF3366',
    accentYellow: '#FFD700',
};

// Стилизованные компоненты с glassmorphism эффектом
const GlassCard = styled(Card)(({ theme }) => ({
    background: 'rgba(13, 27, 42, 0.4)',
    backdropFilter: 'blur(20px) saturate(180%)',
    border: '1px solid rgba(0, 102, 255, 0.2)',
    borderRadius: '24px',
    position: 'relative',
    overflow: 'hidden',
    transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: '-100%',
        width: '100%',
        height: '100%',
        background: 'linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.1), transparent)',
        transition: 'left 0.7s',
    },
    '&:hover': {
        transform: 'translateY(-8px)',
        borderColor: 'rgba(0, 229, 255, 0.5)',
        boxShadow: `0 20px 60px rgba(0, 102, 255, 0.3),
      0 0 120px rgba(0, 229, 255, 0.2),
      inset 0 1px 0 rgba(255, 255, 255, 0.1)`,
        '&::before': {
            left: '100%',
        },
    },
}));

const HolographicBadge = styled(Box)(({ theme }) => ({
    position: 'relative',
    padding: '8px 16px',
    background: 'linear-gradient(135deg, rgba(0, 102, 255, 0.2), rgba(0, 229, 255, 0.1))',
    border: '1px solid rgba(0, 229, 255, 0.3)',
    borderRadius: '12px',
    backdropFilter: 'blur(10px)',
    overflow: 'hidden',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: '-50%',
        left: '-50%',
        width: '200%',
        height: '200%',
        background: 'linear-gradient(45deg, transparent 30%, rgba(0, 229, 255, 0.1) 50%, transparent 70%)',
        animation: 'hologramMove 3s linear infinite',
    },
    '@keyframes hologramMove': {
        '0%': { transform: 'rotate(0deg) translateX(-25%)' },
        '100%': { transform: 'rotate(360deg) translateX(-25%)' },
    },
}));

const DecorativeElement = styled(Box)(({ theme }) => ({
    position: 'absolute',
    top: -20,
    right: -20,
    width: 80,
    height: 80,
    borderRadius: '50%',
    opacity: 0.5,
    zIndex: 0
}));

const CardHeaderContainer = styled(Box)(({ theme }) => ({
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing(2)
}));

const TrendContainer = styled(Box)(({ theme }) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(1),
    marginTop: theme.spacing(2)
}));

const TrendValueContainer = styled(Box)(({ theme }) => ({
    display: 'flex',
    alignItems: 'center',
    padding: theme.spacing(0.75, 1.5),
    borderRadius: '12px'
}));

// Стили для AnimatedKPICard (вынесены наружу)
const TitleTypography = styled(Typography)({
    color: MIREA_2025_COLORS.textTertiary,
    fontWeight: 600,
    letterSpacing: '0.05em',
    textTransform: 'uppercase'
});

const ValueTypography = styled(Typography)({
    fontWeight: 800,
    marginBottom: 8,
    fontSize: '3.5rem',
    lineHeight: 1,
});

const TrendCaption = styled(Typography)<{ direction: 'up' | 'down' | 'stable' }>(({ direction }) => ({
    fontWeight: 700,
    color: direction === 'up' ? '#00E676' : direction === 'down' ? '#FF1744' : '#00E5FF'
}));

const TrendTimeCaption = styled(Typography)({
    color: MIREA_2025_COLORS.textTertiary
});

const LinearProgressStyled = styled(LinearProgress)({
    marginTop: 24,
    height: 8,
    borderRadius: 4,
    background: 'rgba(255, 255, 255, 0.1)',
    '& .MuiLinearProgress-bar': {
        borderRadius: 4,
        position: 'relative',
        overflow: 'hidden',
        '&::after': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'linear-gradient(90deg, transparent, rgba(255, 255, 255, 0.3), transparent)',
            animation: 'shimmer 2s infinite',
        },
    },
    '@keyframes shimmer': {
        '0%': { transform: 'translateX(-100%)' },
        '100%': { transform: 'translateX(100%)' },
    },
});

const AvatarStyled = styled(Avatar)({
    width: 56,
    height: 56,
    backdropFilter: 'blur(10px)',
});

interface AnimatedKPICardProps {
    title: string;
    value: string | number;
    icon: React.ReactElement;
    color: string;
    trend?: { value: number; direction: 'up' | 'down' | 'stable' };
    unit?: string;
    delay?: number;
}

const AnimatedKPICard: React.FC<AnimatedKPICardProps> = ({
    title,
    value,
    icon,
    color,
    trend,
    unit = '',
    delay = 0
}) => {
    const [animatedValue, setAnimatedValue] = useState(0);

    useEffect(() => {
        if (typeof value === 'number') {
            const duration = 1500;
            const steps = 60;
            const increment = value / steps;
            let current = 0;
            const timer = setInterval(() => {
                current += increment;
                if (current >= value) {
                    setAnimatedValue(value);
                    clearInterval(timer);
                } else {
                    setAnimatedValue(current);
                }
            }, duration / steps);

            return () => clearInterval(timer);
        }
    }, [value]);

    // Create dynamic styled components
    const AvatarContent = styled(Avatar)({
        width: 56,
        height: 56,
        backdropFilter: 'blur(10px)',
        background: `linear-gradient(135deg, ${color}20, ${color}05)`,
        border: `2px solid ${color}40`,
    });

    const LinearProgressBar = styled(LinearProgress)({
        marginTop: 24,
        height: 8,
        borderRadius: 4,
        background: 'rgba(255, 255, 255, 0.1)',
        '& .MuiLinearProgress-bar': {
            borderRadius: 4,
            position: 'relative',
            overflow: 'hidden',
            background: `linear-gradient(90deg, ${color}, ${color}CC)`,
        },
    });

    const CustomValueTypography = styled(Typography)({
        fontWeight: 800,
        marginBottom: 8,
        fontSize: '3.5rem',
        lineHeight: 1,
        background: color,
        WebkitBackgroundClip: 'text',
        WebkitTextFillColor: 'transparent',
    });

    return (
        <Grow in timeout={800} style={{ transitionDelay: `${delay}ms` }}>
            <GlassCard>
                <CardContent style={{ padding: '28px', position: 'relative', zIndex: 1 }}>
                    <CardHeaderContainer>
                        <TitleTypography variant="subtitle2">
                            {title}
                        </TitleTypography>
                        <AvatarContent>
                            {React.cloneElement(icon, {
                                sx: {
                                    fontSize: 28,
                                    background: color,
                                    WebkitBackgroundClip: 'text',
                                    WebkitTextFillColor: 'transparent',
                                }
                            })}
                        </AvatarContent>
                    </CardHeaderContainer>

                    <CustomValueTypography variant="h2">
                        {typeof value === 'number' ? animatedValue.toFixed(1) : value}{unit}
                    </CustomValueTypography>

                    {trend && (
                        <TrendContainer>
                            <TrendValueContainer style={{
                                background: trend.direction === 'up'
                                    ? 'rgba(0, 230, 118, 0.15)'
                                    : trend.direction === 'down'
                                        ? 'rgba(255, 23, 68, 0.15)'
                                        : 'rgba(0, 229, 255, 0.15)',
                                border: trend.direction === 'up'
                                    ? '1px solid rgba(0, 230, 118, 0.3)'
                                    : trend.direction === 'down'
                                        ? '1px solid rgba(255, 23, 68, 0.3)'
                                        : '1px solid rgba(0, 229, 255, 0.3)',
                            }}>
                                {trend.direction === 'up' ? (
                                    <ArrowUpward style={{ fontSize: 16, color: '#00E676', marginRight: 4 }} />
                                ) : trend.direction === 'down' ? (
                                    <ArrowDownward style={{ fontSize: 16, color: '#FF1744', marginRight: 4 }} />
                                ) : (
                                    <TrendingFlat style={{ fontSize: 16, color: '#00E5FF', marginRight: 4 }} />
                                )}
                                <TrendCaption variant="caption" direction={trend.direction}>
                                    {Math.abs(trend.value)}%
                                </TrendCaption>
                            </TrendValueContainer>
                            <TrendTimeCaption variant="caption">
                                за последний час
                            </TrendTimeCaption>
                        </TrendContainer>
                    )}

                    <LinearProgressBar
                        variant="determinate"
                        value={typeof value === 'number' ? Math.min(100, Number(value)) : 0}
                    />
                </CardContent>

                {/* Декоративные элементы */}
                <DecorativeElement style={{
                    background: `radial-gradient(circle, ${color}30 0%, transparent 70%)`,
                }} />
            </GlassCard>
        </Grow>
    );
};

const ChartCardHeaderContainer = styled(Box)(({ theme }) => ({
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: theme.spacing(3)
}));

interface InteractiveChartCardProps {
    title: string;
    subtitle?: string;
    icon: React.ReactElement;
    children: React.ReactNode;
    action?: React.ReactNode;
}

// Стили для InteractiveChartCard (вынесены наружу)
const FlexContainer = styled(Box)({
    flex: '1 1 auto'
});

const IconBox = styled(Box)({
    display: 'flex',
    alignItems: 'center',
    gap: 12
});

const ChartTitleTypography = styled(Typography)({
    fontWeight: 700,
    color: MIREA_2025_COLORS.textPrimary,
    background: MIREA_2025_COLORS.primaryGradient,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
});

const ChartSubtitleTypography = styled(Typography)({
    color: MIREA_2025_COLORS.textTertiary
});

const InteractiveChartCard: React.FC<InteractiveChartCardProps> = ({
    title,
    subtitle,
    icon,
    children,
    action
}) => {
    return (
        <Slide in timeout={1000} direction="up">
            <GlassCard style={{ height: '100%' }}>
                <CardContent style={{ padding: '28px', height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <ChartCardHeaderContainer>
                        <IconBox>
                            {icon}
                            <ChartTitleTypography variant="h6">
                                {title}
                            </ChartTitleTypography>
                        </IconBox>
                        {subtitle && (
                            <ChartSubtitleTypography variant="body2">
                                {subtitle}
                            </ChartSubtitleTypography>
                        )}
                    </ChartCardHeaderContainer>
                    <FlexContainer>
                        {children}
                    </FlexContainer>
                </CardContent>
            </GlassCard>
        </Slide>
    );
};

const StatusChips = styled(Box)({
    display: 'flex',
    flexWrap: 'wrap',
    gap: '12px',
    marginTop: '24px',
    justifyContent: 'center'
});

const DefectIndicator = styled(Box)({
    width: 8,
    height: 8,
    borderRadius: '50%'
});

interface PerformanceRadialChartProps {
    data: { name: string; value: number; color: string }[];
}

const PerformanceRadialChart: React.FC<PerformanceRadialChartProps> = ({ data }) => {
    const total = data.reduce((sum, item) => sum + item.value, 0);

    const IconButtonStyled = styled(IconButton)({
        background: 'rgba(0, 102, 255, 0.1)',
        '&:hover': { background: 'rgba(0, 102, 255, 0.2)' }
    });

    const ColoredDefectIndicator = styled(DefectIndicator)<{ color: string }>(({ color }) => ({
        background: color,
        boxShadow: `0 0 8px ${color}`
    }));

    const ChipBox = styled(Box)({
        display: 'flex',
        alignItems: 'center',
        gap: 4
    });

    return (
        <InteractiveChartCard
            title="Распределение дефектов"
            subtitle="В реальном времени"
            icon={<BubbleChart sx={{ color: MIREA_2025_COLORS.accentPurple, fontSize: 28 }} />}
            action={
                <Tooltip title="Показать детали">
                    <IconButtonStyled>
                        <MoreVert sx={{ color: MIREA_2025_COLORS.textSecondary }} />
                    </IconButtonStyled>
                </Tooltip>
            }
        >
            <ResponsiveContainer width="100%" height={250}>
                <RadialBarChart
                    innerRadius="20%"
                    outerRadius="90%"
                    data={data}
                    startAngle={180}
                    endAngle={-180}
                >
                    <PolarGrid
                        stroke={MIREA_2025_COLORS.glassSurface}
                        strokeWidth={1}
                    />
                    <PolarAngleAxis
                        type="number"
                        domain={[0, 100]}
                        tick={false}
                    />
                    <RadialBar
                        dataKey="value"
                        background={{ fill: 'rgba(255, 255, 255, 0.05)' }}
                        cornerRadius={8}
                    >
                        {data.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                        ))}
                    </RadialBar>
                </RadialBarChart>
            </ResponsiveContainer>

            <StatusChips>
                {data.map((item, index) => (
                    <Chip
                        key={index}
                        label={
                            <ChipBox>
                                <ColoredDefectIndicator color={item.color} />
                                <Typography variant="caption" sx={{ fontWeight: 600 }}>
                                    {item.name}: {item.value} ({((item.value / total) * 100).toFixed(1)}%)
                                </Typography>
                            </ChipBox>
                        }
                        style={{
                            backgroundColor: 'rgba(255, 255, 255, 0.05)',
                            border: `1px solid ${item.color}40`,
                            backdropFilter: 'blur(10px)',
                        }}
                        classes={{ label: 'MuiChip-label' }}
                    />
                ))}
            </StatusChips>
        </InteractiveChartCard>
    );
};

interface PerformanceTrendChartProps {
    data: { time: string; quality: number; defects: number }[];
}

const PerformanceTrendChart: React.FC<PerformanceTrendChartProps> = ({ data }) => {
    const [isPlaying, setIsPlaying] = useState(true);

    const IconButtonStyled = styled(IconButton)({
        background: 'rgba(0, 102, 255, 0.1)',
        '&:hover': { background: 'rgba(0, 102, 255, 0.2)' }
    });

    return (
        <InteractiveChartCard
            title="Динамика производства"
            subtitle="Качество vs Дефекты"
            icon={<Timeline sx={{ color: MIREA_2025_COLORS.primary, fontSize: 28 }} />}
            action={
                <Tooltip title={isPlaying ? 'Пауза' : 'Воспроизвести'}>
                    <IconButtonStyled
                        onClick={() => setIsPlaying(!isPlaying)}
                    >
                        {isPlaying ?
                            <PauseCircle sx={{ color: MIREA_2025_COLORS.primary }} /> :
                            <PlayCircle sx={{ color: MIREA_2025_COLORS.primary }} />
                        }
                    </IconButtonStyled>
                </Tooltip>
            }
        >
            <ResponsiveContainer width="100%" height={250}>
                <AreaChart data={data}>
                    <defs>
                        <linearGradient id="colorQuality" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={MIREA_2025_COLORS.tertiary} stopOpacity={0.8} />
                            <stop offset="95%" stopColor={MIREA_2025_COLORS.tertiary} stopOpacity={0.1} />
                        </linearGradient>
                        <linearGradient id="colorDefects" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={MIREA_2025_COLORS.error} stopOpacity={0.8} />
                            <stop offset="95%" stopColor={MIREA_2025_COLORS.error} stopOpacity={0.1} />
                        </linearGradient>
                    </defs>
                    <CartesianGrid
                        strokeDasharray="3 3"
                        stroke={MIREA_2025_COLORS.glassSurface}
                        strokeWidth={0.5}
                    />
                    <XAxis
                        dataKey="time"
                        stroke={MIREA_2025_COLORS.textTertiary}
                        fontSize={12}
                        tickLine={false}
                        axisLine={{ stroke: MIREA_2025_COLORS.glassSurface }}
                    />
                    <YAxis
                        stroke={MIREA_2025_COLORS.textTertiary}
                        fontSize={12}
                        tickLine={false}
                        axisLine={{ stroke: MIREA_2025_COLORS.glassSurface }}
                    />
                    <RechartsTooltip
                        contentStyle={{
                            backgroundColor: MIREA_2025_COLORS.darkSurface,
                            border: `1px solid ${MIREA_2025_COLORS.glassSurface}`,
                            borderRadius: '12px',
                            backdropFilter: 'blur(20px)',
                            color: MIREA_2025_COLORS.textPrimary,
                            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.3)',
                        }}
                        formatter={(value: number) => [value.toFixed(1), 'Значение']}
                    />
                    <Area
                        type="monotone"
                        dataKey="quality"
                        stroke={MIREA_2025_COLORS.tertiary}
                        fill="url(#colorQuality)"
                        strokeWidth={3}
                        dot={{ stroke: MIREA_2025_COLORS.tertiary, strokeWidth: 2, r: 4 }}
                        activeDot={{ r: 6, strokeWidth: 0 }}
                        name="Качество (%)"
                        animationDuration={isPlaying ? 1500 : 0}
                    />
                    <Area
                        type="monotone"
                        dataKey="defects"
                        stroke={MIREA_2025_COLORS.error}
                        fill="url(#colorDefects)"
                        strokeWidth={3}
                        dot={{ stroke: MIREA_2025_COLORS.error, strokeWidth: 2, r: 4 }}
                        activeDot={{ r: 6, strokeWidth: 0 }}
                        name="Дефекты (шт)"
                        animationDuration={isPlaying ? 1500 : 0}
                    />
                </AreaChart>
            </ResponsiveContainer>
        </InteractiveChartCard>
    );
};

interface RealTimeMetricsPanelProps {
    metrics: {
        name: string;
        value: number;
        unit: string;
        max: number;
        icon: string; // Changed from React.ReactElement to string
        trend?: 'up' | 'down' | 'stable';
    }[];
}

// Стили для RealTimeMetricsPanel (вынесены наружу)
const RealTimeHeaderContainer = styled(Box)({
    display: 'flex',
    alignItems: 'center',
    gap: 16,
    marginBottom: 24
});

const RealTimeMetricsIcon = styled(Box)({
    width: 56,
    height: 56,
    borderRadius: '16px',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    background: MIREA_2025_COLORS.primaryGradient,
    boxShadow: `0 8px 32px ${MIREA_2025_COLORS.primary}40`,
});

const RealTimeMetricsCarousel = styled(Box)({
    marginBottom: 24
});

const RealTimeMetricSlide = styled(Box)({
    textAlign: 'center'
});

const RealTimeMetricValueContainer = styled(Box)({
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 16,
    marginBottom: 16
});

const RealTimeMetricValueText = styled(Typography)({
    fontWeight: 800,
    background: MIREA_2025_COLORS.infoGradient,
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
});

const RealTimeMetricNameText = styled(Typography)({
    color: MIREA_2025_COLORS.textPrimary,
    marginBottom: 8
});

const RealTimePanelTitle = styled(Typography)({
    fontWeight: 700,
    color: MIREA_2025_COLORS.textPrimary,
    marginBottom: 4
});

const RealTimePanelSubtitle = styled(Typography)({
    color: MIREA_2025_COLORS.textTertiary
});

const RealTimeIndicatorContainer = styled(Box)({
    display: 'flex',
    justifyContent: 'center',
    gap: 8,
    marginBottom: 24
});

const RealTimeMetricsList = styled(Box)({
    display: 'flex',
    flexDirection: 'column',
    gap: 16
});

const RealTimeMetricRow = styled(Box)({
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8
});

const RealTimeMetricIconRow = styled(Box)({
    display: 'flex',
    alignItems: 'center',
    gap: 8
});

const RealTimeMetricsPanel: React.FC<RealTimeMetricsPanelProps> = ({ metrics }) => {
    const [currentMetric, setCurrentMetric] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setCurrentMetric((prev) => (prev + 1) % metrics.length);
        }, 5000);

        return () => clearInterval(interval);
    }, [metrics.length]);

    // Компоненты, зависящие от пропсов, остаются внутри
    const IndicatorDot = styled(Box)<{ active: boolean }>(({ active }) => ({
        width: 8,
        height: 8,
        borderRadius: '50%',
        cursor: 'pointer',
        transition: 'all 0.3s',
        background: active ? MIREA_2025_COLORS.primary : MIREA_2025_COLORS.glassSurface,
        '&:hover': {
            background: MIREA_2025_COLORS.primary,
            transform: 'scale(1.2)',
        },
    }));

    const MetricValueDisplay = styled(Typography)<{ critical: boolean; warning: boolean }>(({ critical, warning }) => ({
        fontWeight: 700,
        color: critical ? MIREA_2025_COLORS.error : warning ? MIREA_2025_COLORS.warning : MIREA_2025_COLORS.tertiary
    }));

    const LinearProgressBar = styled(LinearProgress)<{ critical: boolean; warning: boolean }>(({ critical, warning }) => ({
        height: 6,
        borderRadius: 3,
        background: 'rgba(255, 255, 255, 0.1)',
        '& .MuiLinearProgress-bar': {
            borderRadius: 3,
            background: critical
                ? MIREA_2025_COLORS.errorGradient
                : warning
                    ? MIREA_2025_COLORS.warningGradient
                    : MIREA_2025_COLORS.tertiaryGradient,
        },
    }));

    const MetricItem = styled(Box)<{ active: boolean }>(({ active }) => ({
        opacity: active ? 1 : 0.6
    }));

    return (
        <Slide in timeout={1200} direction="right">
            <GlassCard style={{ height: '100%' }}>
                <CardContent style={{ padding: '28px', height: '100%' }}>
                    <RealTimeHeaderContainer>
                        <RealTimeMetricsIcon>
                            <Speed style={{ fontSize: 28, color: '#FFFFFF' }} />
                        </RealTimeMetricsIcon>
                        <Box>
                            <RealTimePanelTitle variant="h6">
                                Метрики в реальном времени
                            </RealTimePanelTitle>
                            <RealTimePanelSubtitle variant="body2">
                                Автоматическое обновление
                            </RealTimePanelSubtitle>
                        </Box>
                    </RealTimeHeaderContainer>

                    <RealTimeMetricsCarousel>
                        {metrics.map((metric, index) => (
                            <Fade in={index === currentMetric} key={index} timeout={500}>
                                <RealTimeMetricSlide>
                                    <RealTimeMetricValueContainer>
                                        {/* Updated to handle MUI icon names */}
                                        {React.createElement(MUIIcons[metric.icon as keyof typeof MUIIcons] || MUIIcons.Help, { 
                                            style: { fontSize: 32 } 
                                        })}
                                        <RealTimeMetricValueText variant="h3">
                                            {metric.value} {metric.unit}
                                        </RealTimeMetricValueText>
                                    </RealTimeMetricValueContainer>
                                    <RealTimeMetricNameText variant="h6">
                                        {metric.name}
                                    </RealTimeMetricNameText>
                                </RealTimeMetricSlide>
                            </Fade>
                        ))}
                    </RealTimeMetricsCarousel>

                    <RealTimeIndicatorContainer>
                        {metrics.map((_, index) => (
                            <IndicatorDot
                                key={index}
                                onClick={() => setCurrentMetric(index)}
                                active={index === currentMetric}
                            />
                        ))}
                    </RealTimeIndicatorContainer>

                    <RealTimeMetricsList>
                        {metrics.map((metric, index) => {
                            const percentage = (metric.value / metric.max) * 100;
                            const isWarning = percentage > 80;
                            const isCritical = percentage > 95;

                            return (
                                <MetricItem key={index} active={index === currentMetric}>
                                    <RealTimeMetricRow>
                                        <RealTimeMetricIconRow>
                                            {/* Updated to handle MUI icon names */}
                                            {React.createElement(MUIIcons[metric.icon as keyof typeof MUIIcons] || MUIIcons.Help, { 
                                                style: { fontSize: 32, color: MIREA_2025_COLORS.textSecondary } 
                                            })}
                                            <Typography variant="body2" style={{ color: MIREA_2025_COLORS.textSecondary }}>
                                                {metric.name}
                                            </Typography>
                                        </RealTimeMetricIconRow>
                                        <MetricValueDisplay 
                                            variant="body2" 
                                            critical={isCritical}
                                            warning={isWarning}
                                        >
                                            {metric.value} {metric.unit}
                                        </MetricValueDisplay>
                                    </RealTimeMetricRow>
                                    <LinearProgressBar
                                        variant="determinate"
                                        value={percentage}
                                        critical={isCritical}
                                        warning={isWarning}
                                    />
                                </MetricItem>
                            );
                        })}
                    </RealTimeMetricsList>
                </CardContent>
            </GlassCard>
        </Slide>
    );
};

interface AIRecommendation {
    text: string;
    priority: 'high' | 'medium' | 'low';
    impact: number;
    icon: string; // Changed from React.ReactElement to string
}

interface AIInsightsPanelProps {
    recommendations: AIRecommendation[];
}

const AIInsightsPanel: React.FC<AIInsightsPanelProps> = ({ recommendations }) => {
    const getPriorityColor = (priority: string) => {
        switch (priority) {
            case 'high': return MIREA_2025_COLORS.error;
            case 'medium': return MIREA_2025_COLORS.warning;
            case 'low': return MIREA_2025_COLORS.info;
            default: return MIREA_2025_COLORS.info;
        }
    };

    const getPriorityLabel = (priority: string) => {
        switch (priority) {
            case 'high': return 'Критично';
            case 'medium': return 'Важно';
            case 'low': return 'Рекомендация';
            default: return 'Рекомендация';
        }
    };

    const AIHeader = styled(Box)({
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        marginBottom: 24
    });

    const AIIcon = styled(Box)({
        position: 'relative',
        width: 56,
        height: 56,
        borderRadius: '16px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        background: MIREA_2025_COLORS.secondaryGradient,
        boxShadow: `0 8px 32px ${MIREA_2025_COLORS.secondary}40`,
    });

    const AIContentFlexContainer = styled(Box)({
        flex: '1 1 auto',
        overflowY: 'auto',
        paddingRight: 8
    });

    const AIButtonStyled = styled(Button)({
        marginTop: 24,
        padding: '12px 24px',
        background: MIREA_2025_COLORS.primaryGradient,
        borderRadius: '12px',
        fontWeight: 700,
        fontSize: 16,
        textTransform: 'none',
        '&:hover': {
            background: `linear-gradient(135deg, ${MIREA_2025_COLORS.secondary}, ${MIREA_2025_COLORS.primary})`,
            transform: 'translateY(-2px)',
            boxShadow: `0 12px 32px ${MIREA_2025_COLORS.primary}40`,
        },
    });

    const AIPanelTitle = styled(Typography)({
        fontWeight: 700,
        color: MIREA_2025_COLORS.textPrimary
    });

    const AIPanelSubtitle = styled(Typography)({
        color: MIREA_2025_COLORS.textTertiary
    });

    const AIBadgeText = styled(Typography)({
        fontWeight: 800,
        color: MIREA_2025_COLORS.accentCyan,
        letterSpacing: '0.1em'
    });

    return (
        <Fade in timeout={1400}>
            <GlassCard style={{ height: '100%' }}>
                <CardContent style={{ padding: '28px', height: '100%', display: 'flex', flexDirection: 'column' }}>
                    <AIHeader>
                        <AIIcon>
                            <div
                                style={{
                                    content: '',
                                    position: 'absolute',
                                    inset: -2,
                                    background: MIREA_2025_COLORS.secondaryGradient,
                                    borderRadius: '18px',
                                    zIndex: -1,
                                    filter: 'blur(8px)',
                                    opacity: 0.6,
                                }}
                            />
                            <Psychology style={{ fontSize: 28, color: '#FFFFFF' }} />
                        </AIIcon>
                        <Box style={{ flex: '1 1 auto' }}>
                            <Box style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                                <AIPanelTitle variant="h6">
                                    Интеллектуальные рекомендации
                                </AIPanelTitle>
                                <HolographicBadge>
                                    <AIBadgeText variant="caption">
                                        ИИ АКТИВЕН
                                    </AIBadgeText>
                                </HolographicBadge>
                            </Box>
                            <AIPanelSubtitle variant="body2">
                                На основе машинного обучения
                            </AIPanelSubtitle>
                        </Box>
                    </AIHeader>

                    <AIContentFlexContainer>
                        {recommendations.map((rec, index) => {
                            const color = getPriorityColor(rec.priority);

                            const PaperStyled = styled(Paper)({
                                padding: '20px',
                                marginBottom: 16,
                                background: 'rgba(255, 255, 255, 0.03)',
                                border: `1px solid ${color}30`,
                                borderRadius: '16px',
                                transition: 'all 0.3s',
                                '&:hover': {
                                    transform: 'translateX(8px)',
                                    background: 'rgba(255, 255, 255, 0.05)',
                                    borderColor: `${color}50`,
                                    boxShadow: `0 8px 24px ${color}20`,
                                },
                            });

                            const AvatarStyled = styled(Avatar)({
                                width: 48,
                                height: 48,
                                background: `linear-gradient(135deg, ${color}, ${color}CC)`,
                                boxShadow: `0 4px 16px ${color}40`,
                            });

                            const ButtonOutlined = styled(Button)({
                                borderColor: `${color}40`,
                                color: color,
                                '&:hover': {
                                    borderColor: color,
                                    background: `${color}10`,
                                },
                            });

                            const ButtonContained = styled(Button)({
                                background: `linear-gradient(135deg, ${color}, ${color}CC)`,
                                '&:hover': {
                                    background: `linear-gradient(135deg, ${color}CC, ${color})`,
                                },
                            });

                            const ContentBox = styled(Box)({
                                display: 'flex',
                                gap: 16,
                                alignItems: 'flex-start'
                            });

                            const ContentInnerBox = styled(Box)({
                                flex: '1 1 auto'
                            });

                            const ChipBox = styled(Box)({
                                display: 'flex',
                                alignItems: 'center',
                                gap: 8,
                                marginBottom: 8
                            });

                            const ButtonBox = styled(Box)({
                                display: 'flex',
                                justifyContent: 'flex-end',
                                gap: 8
                            });

                            const RecommendationText = styled(Typography)({
                                color: MIREA_2025_COLORS.textSecondary,
                                lineHeight: 1.6,
                                marginBottom: 12
                            });

                            return (
                                <Grow in timeout={800 + index * 200} key={index}>
                                    <PaperStyled elevation={0}>
                                        <ContentBox>
                                            {/* Updated to handle MUI icon names */}
                                            <AvatarStyled>
                                                {React.createElement(MUIIcons[rec.icon as keyof typeof MUIIcons] || MUIIcons.Help, { 
                                                    style: { fontSize: 24 } 
                                                })}
                                            </AvatarStyled>
                                            <ContentInnerBox>
                                                <ChipBox>
                                                    <Chip
                                                        label={getPriorityLabel(rec.priority)}
                                                        size="small"
                                                        style={{
                                                            backgroundColor: `${color}20`,
                                                            color: color,
                                                            fontWeight: 700,
                                                            fontSize: 11,
                                                            border: `1px solid ${color}40`,
                                                            backdropFilter: 'blur(10px)',
                                                        }}
                                                    />
                                                    <Chip
                                                        label={`Влияние: ${rec.impact}%`}
                                                        size="small"
                                                        style={{
                                                            backgroundColor: 'rgba(0, 102, 255, 0.1)',
                                                            color: MIREA_2025_COLORS.primary,
                                                            fontWeight: 600,
                                                            fontSize: 11,
                                                            border: '1px solid rgba(0, 102, 255, 0.3)',
                                                        }}
                                                    />
                                                </ChipBox>
                                                <RecommendationText variant="body1">
                                                    {rec.text}
                                                </RecommendationText>
                                                <ButtonBox>
                                                    <ButtonOutlined
                                                        size="small"
                                                        variant="outlined"
                                                    >
                                                        Детали
                                                    </ButtonOutlined>
                                                    <ButtonContained
                                                        size="small"
                                                        variant="contained"
                                                    >
                                                        Применить
                                                    </ButtonContained>
                                                </ButtonBox>
                                            </ContentInnerBox>
                                        </ContentBox>
                                    </PaperStyled>
                                </Grow>
                            );
                        })}
                    </AIContentFlexContainer>

                    <AIButtonStyled
                        variant="contained"
                        fullWidth
                        startIcon={<AutoAwesome />}
                    >
                        Применить все рекомендации
                    </AIButtonStyled>
                </CardContent>
            </GlassCard>
        </Fade>
    );
};

const GradientLine = styled(Box)({
    content: '""',
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: '2px',
    background: 'linear-gradient(90deg, #0066FF, #00E5FF, #FF3366)',
    animation: 'gradientMove 3s linear infinite',
    '@keyframes gradientMove': {
        '0%': { backgroundPosition: '0% 50%' },
        '50%': { backgroundPosition: '100% 50%' },
        '100%': { backgroundPosition: '0% 50%' },
    },
});

const UniversityHeaderContainer = styled(Box)(({ theme }) => ({
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(2),
    padding: theme.spacing(3),
    position: 'relative',
    overflow: 'hidden'
}));

const UniversityAvatar = styled(Avatar)({
    width: 64,
    height: 64,
    background: MIREA_2025_COLORS.primaryGradient,
    boxShadow: `0 8px 32px ${MIREA_2025_COLORS.primary}40`,
    border: '2px solid rgba(0, 229, 255, 0.5)',
    fontSize: 32,
    fontWeight: 900,
});

const UniversityTitle = styled(Typography)({
    fontWeight: 900,
    background: 'linear-gradient(135deg, #FFFFFF 0%, #00E5FF 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    marginBottom: '0.5rem'
});

const UniversitySubtitle = styled(Typography)({
    color: MIREA_2025_COLORS.textTertiary,
    display: 'flex',
    alignItems: 'center',
    gap: 8
});

const ActionButtonContainer = styled(Box)({
    display: 'flex',
    gap: 12
});

// Add this constant before the UniversityHeader component
const universityHeaderStyles: React.CSSProperties = {
    flex: 1
};

const UniversityHeader: React.FC = () => {
    const StyledIconButton = styled(IconButton)({
        background: 'rgba(0, 102, 255, 0.1)',
        '&:hover': { background: 'rgba(0, 102, 255, 0.2)' }
    });

    return (
        <UniversityHeaderContainer>
            <GradientLine />
            <UniversityAvatar>
                <School />
            </UniversityAvatar>
            {/* @ts-ignore */}
            <Box style={universityHeaderStyles}>
                <UniversityTitle variant="h4">
                    РТУ МИРЭА | Glass AI
                </UniversityTitle>
                <UniversitySubtitle variant="subtitle1">
                    <RocketLaunch style={{ fontSize: 16 }} />
                    Интеллектуальная система контроля качества стекла
                </UniversitySubtitle>
            </Box>
            <ActionButtonContainer>
                <Tooltip title="Обновить данные">
                    <StyledIconButton>
                        <Refresh style={{ color: MIREA_2025_COLORS.primary }} />
                    </StyledIconButton>
                </Tooltip>
                <Tooltip title="Фильтры">
                    <StyledIconButton>
                        <FilterList style={{ color: MIREA_2025_COLORS.primary }} />
                    </StyledIconButton>
                </Tooltip>
            </ActionButtonContainer>
        </UniversityHeaderContainer>
    );
};

const DashboardContainer = styled(Box)({
    flexGrow: 1,
    minHeight: '100vh',
    background: 'radial-gradient(ellipse at top, #001D3D 0%, #000814 50%, #000000 100%)',
    position: 'relative',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: `
      radial-gradient(circle at 20% 30%, rgba(0, 102, 255, 0.15) 0%, transparent 40%),
      radial-gradient(circle at 80% 20%, rgba(0, 229, 255, 0.1) 0%, transparent 40%),
      radial-gradient(circle at 40% 80%, rgba(255, 51, 102, 0.08) 0%, transparent 40%)
    `,
        pointerEvents: 'none',
    },
});

const ContentContainer = styled(Box)(({ theme }) => ({
    padding: theme.spacing(4),
    maxWidth: '1920px',
    margin: '0 auto',
    position: 'relative',
    zIndex: 1,
}));

const GridContainer = styled(Grid)(({ theme }) => ({
    marginBottom: theme.spacing(4),
}));

const StatusIndicator = styled(Box)({
    width: 12,
    height: 12,
    borderRadius: '50%',
    background: '#00E676',
    boxShadow: '0 0 12px #00E676',
    animation: 'pulse 2s infinite',
    '@keyframes pulse': {
        '0%, 100%': { opacity: 1 },
        '50%': { opacity: 0.5 },
    },
});

const StatusCardHeader = styled(Box)({
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'space-between',
    marginBottom: 24
});

const StatusTitle = styled(Typography)({
    fontWeight: 700,
    background: 'linear-gradient(135deg, #FF3366, #FF9E6D)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
});

const StatusText = styled(Typography)({
    color: MIREA_2025_COLORS.textSecondary,
    textAlign: 'center',
    lineHeight: 1.8
});

const AdvancedDashboard: React.FC = () => {
    const { data, loading, error } = useDashboardData();
    const { wsData, isConnected } = useWebSocketStream();
    const { notifications, unacknowledgedCount } = useNotifications(wsData.defectAlerts);
    
    // Calculate total defects from aggregation
    const totalDefectsFromWs = React.useMemo(() => {
        return Object.values(wsData.defectAggregation).reduce((sum, count) => sum + count, 0);
    }, [wsData.defectAggregation]);
    
    // Build defect distribution from WebSocket data
    const defectDistribution = React.useMemo(() => {
        const agg = wsData.defectAggregation;
        const hasData = Object.keys(agg).length > 0 && Object.values(agg).some(v => v > 0);
        
        if (hasData) {
            return [
                { name: 'Трещины', value: agg['crack'] || 0, color: MIREA_2025_COLORS.error },
                { name: 'Пузыри', value: agg['bubble'] || 0, color: MIREA_2025_COLORS.warning },
                { name: 'Сколы', value: agg['chip'] || 0, color: MIREA_2025_COLORS.tertiary },
                { name: 'Помутнение', value: agg['cloudiness'] || 0, color: MIREA_2025_COLORS.info },
                { name: 'Деформация', value: agg['deformation'] || 0, color: MIREA_2025_COLORS.accentPurple },
                { name: 'Пятна', value: agg['stain'] || 0, color: MIREA_2025_COLORS.accentCyan }
            ].filter(d => d.value > 0);
        }
        
        // Use API data if no WebSocket data
        return data?.defectDistribution || [
            { name: 'Трещины', value: 0, color: MIREA_2025_COLORS.error },
            { name: 'Пузыри', value: 0, color: MIREA_2025_COLORS.warning },
            { name: 'Сколы', value: 0, color: MIREA_2025_COLORS.tertiary }
        ];
    }, [wsData.defectAggregation, data?.defectDistribution]);
    
    // Build real-time metrics from WebSocket parameters
    const realTimeMetrics = React.useMemo(() => {
        if (wsData.parameters) {
            const p = wsData.parameters;
            return [
                {
                    name: 'Температура печи',
                    value: Math.round(p.furnace.temperature),
                    unit: '°C',
                    max: 1600,
                    trend: 'stable' as const,
                    icon: 'LocalFireDepartment'
                },
                {
                    name: 'Давление печи',
                    value: Math.round(p.furnace.pressure * 10) / 10,
                    unit: 'кПа',
                    max: 50,
                    trend: 'stable' as const,
                    icon: 'WaterDrop'
                },
                {
                    name: 'Скорость ленты',
                    value: Math.round(p.forming.speed),
                    unit: 'м/мин',
                    max: 200,
                    trend: 'down' as const,
                    icon: 'Speed'
                },
                {
                    name: 'Температура формы',
                    value: Math.round(p.forming.mold_temp),
                    unit: '°C',
                    max: 400,
                    trend: 'stable' as const,
                    icon: 'Thermostat'
                }
            ];
        }
        return data?.realTimeMetrics || [];
    }, [wsData.parameters, data?.realTimeMetrics]);
    
    // Build AI recommendations from WebSocket or API data
    const aiRecommendations = React.useMemo(() => {
        if (wsData.recommendations.length > 0) {
            return wsData.recommendations.map((rec: any) => ({
                text: rec.action || rec.description || rec.text || 'Рекомендация RL агента',
                priority: (rec.priority || 'medium').toLowerCase() as 'high' | 'medium' | 'low',
                impact: Math.round((rec.confidence || rec.expected_improvement || 0.7) * 100),
                icon: rec.parameter === 'furnace_temperature' ? 'LocalFireDepartment' : 
                      rec.parameter === 'belt_speed' ? 'Speed' :
                      rec.parameter === 'mold_temp' ? 'Thermostat' :
                      rec.parameter === 'energy_consumption' ? 'Factory' : 'Psychology'
            }));
        }
        return data?.aiRecommendations || [];
    }, [wsData.recommendations, data?.aiRecommendations]);
    
    // Memoized KPI values - use WebSocket quality metrics if available
    const kpiData = React.useMemo(() => {
        // Priority: WebSocket qualityMetrics > calculated from defects > API data
        const wsQuality = wsData.qualityMetrics;
        
        if (wsQuality) {
            return {
                qualityRate: wsQuality.qualityRate,
                defectCount: wsQuality.defectCount,
                unitsProduced: wsQuality.unitsProduced,
                uptime: isConnected ? (data?.kpiData?.uptime || 98.5) : 0
            };
        }
        
        // Fallback: calculate from defect aggregation
        if (totalDefectsFromWs > 0) {
            // Use a dynamic base that reflects actual production rather than hardcoded 1000
            const baseUnits = Math.max(100, totalDefectsFromWs * 10); // Adjust this formula as needed
            const qualityRate = Math.max(85, ((baseUnits - totalDefectsFromWs) / baseUnits) * 100);
            return {
                qualityRate: Math.round(qualityRate * 10) / 10,
                defectCount: totalDefectsFromWs,
                unitsProduced: baseUnits - totalDefectsFromWs, // Actual good units
                uptime: isConnected ? (data?.kpiData?.uptime || 98.5) : 0
            };
        }
        
        // Fallback: use API data without hardcoded defaults
        return {
            qualityRate: data?.kpiData?.qualityRate || 0,
            defectCount: data?.kpiData?.defectCount || 0,
            unitsProduced: data?.kpiData?.unitsProduced || 0,
            uptime: isConnected ? (data?.kpiData?.uptime || 0) : 0
        };
    }, [wsData.qualityMetrics, totalDefectsFromWs, data?.kpiData, isConnected]);
    
    // Performance data from API (updates every 10 minutes via polling)
    const performanceData = data?.performanceData || [
        { time: '--:--', quality: 0, defects: 0 }
    ];
    
    // Last update indicator
    const lastUpdateTime = React.useMemo(() => {
        if (wsData.lastUpdate) {
            return new Date(wsData.lastUpdate).toLocaleTimeString('ru-RU');
        }
        return null;
    }, [wsData.lastUpdate]);

    const StatusIconRow = styled(Box)({
        display: 'flex',
        alignItems: 'center',
        gap: 8
    });

    return (
        <DashboardContainer>
            <UniversityHeader />

            <ContentContainer>
                <GridContainer container spacing={3}>
                    <Grid item xs={12} sm={6} md={3}>
                        <AnimatedKPICard
                            title="Уровень качества"
                            value={kpiData.qualityRate}
                            unit="%"
                            icon={<CheckCircle />}
                            color={MIREA_2025_COLORS.tertiaryGradient}
                            trend={{ value: 2.3, direction: 'up' }}
                            delay={100}
                        />
                    </Grid>

                    <Grid item xs={12} sm={6} md={3}>
                        <AnimatedKPICard
                            title="Обнаружено дефектов"
                            value={kpiData.defectCount}
                            icon={<Warning />}
                            color={MIREA_2025_COLORS.errorGradient}
                            trend={{ value: 5.1, direction: 'down' }}
                            delay={200}
                        />
                    </Grid>

                    <Grid item xs={12} sm={6} md={3}>
                        <AnimatedKPICard
                            title="Произведено единиц"
                            value={kpiData.unitsProduced}
                            icon={<Factory />}
                            color={MIREA_2025_COLORS.infoGradient}
                            trend={{ value: 8.7, direction: 'up' }}
                            delay={300}
                        />
                    </Grid>

                    <Grid item xs={12} sm={6} md={3}>
                        <AnimatedKPICard
                            title="Время работы"
                            value={kpiData.uptime}
                            unit="%"
                            icon={<DeviceHub />}
                            color={MIREA_2025_COLORS.primaryGradient}
                            trend={{ value: 1.2, direction: 'up' }}
                            delay={400}
                        />
                    </Grid>
                </GridContainer>

                <GridContainer container spacing={3}>
                    <Grid item xs={12} md={7}>
                        <PerformanceTrendChart data={performanceData} />
                    </Grid>
                    <Grid item xs={12} md={5}>
                        <PerformanceRadialChart data={defectDistribution} />
                    </Grid>
                </GridContainer>





                {/* Дополнительный контент */}
                <Grid container spacing={3}>
                    <Grid item xs={12}>
                        <GlassCard>
                            <CardContent style={{ padding: 24 }}>
                                <StatusCardHeader>
                                    <StatusTitle variant="h6">
                                        Навигация по системе
                                    </StatusTitle>
                                </StatusCardHeader>
                                <StatusText variant="body1" style={{ marginBottom: 20 }}>
                                    Добро пожаловать в интеллектуальную систему контроля качества стекла. Ниже представлена структура интерфейса:
                                </StatusText>
                                
                                <Grid container spacing={2}>
                                    <Grid item xs={12} md={6}>
                                        <Box style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                                            <AvatarStyled style={{ marginRight: 16, background: 'linear-gradient(135deg, #0066FF, #00E5FF)' }}>
                                                <Timeline />
                                            </AvatarStyled>
                                            <Box>
                                                <Typography variant="h6" style={{ color: '#FFFFFF', marginBottom: 4 }}>
                                                    Аналитика и отчеты
                                                </Typography>
                                                <Typography variant="body2" style={{ color: MIREA_2025_COLORS.textTertiary }}>
                                                    Подробная аналитика, тренды дефектов, корреляции и KPI метрики
                                                </Typography>
                                            </Box>
                                        </Box>
                                    </Grid>
                                    
                                    <Grid item xs={12} md={6}>
                                        <Box style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                                            <AvatarStyled style={{ marginRight: 16, background: 'linear-gradient(135deg, #FF3366, #FF9E6D)' }}>
                                                <Psychology />
                                            </AvatarStyled>
                                            <Box>
                                                <Typography variant="h6" style={{ color: '#FFFFFF', marginBottom: 4 }}>
                                                    Рекомендации ИИ
                                                </Typography>
                                                <Typography variant="body2" style={{ color: MIREA_2025_COLORS.textTertiary }}>
                                                    Интеллектуальные рекомендации для оптимизации процесса производства
                                                </Typography>
                                            </Box>
                                        </Box>
                                    </Grid>
                                    
                                    <Grid item xs={12} md={6}>
                                        <Box style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                                            <AvatarStyled style={{ marginRight: 16, background: 'linear-gradient(135deg, #00E676, #6EFFB2)' }}>
                                                <DeviceHub />
                                            </AvatarStyled>
                                            <Box>
                                                <Typography variant="h6" style={{ color: '#FFFFFF', marginBottom: 4 }}>
                                                    Граф знаний
                                                </Typography>
                                                <Typography variant="body2" style={{ color: MIREA_2025_COLORS.textTertiary }}>
                                                    Визуализация причинно-следственных связей между параметрами и дефектами
                                                </Typography>
                                            </Box>
                                        </Box>
                                    </Grid>
                                    
                                    <Grid item xs={12} md={6}>
                                        <Box style={{ display: 'flex', alignItems: 'center', marginBottom: 16 }}>
                                            <AvatarStyled style={{ marginRight: 16, background: 'linear-gradient(135deg, #9D4EDD, #E0AAFF)' }}>
                                                <Factory />
                                            </AvatarStyled>
                                            <Box>
                                                <Typography variant="h6" style={{ color: '#FFFFFF', marginBottom: 4 }}>
                                                    Цифровой двойник
                                                </Typography>
                                                <Typography variant="body2" style={{ color: MIREA_2025_COLORS.textTertiary }}>
                                                    3D визуализация производственной линии и моделирование процессов
                                                </Typography>
                                            </Box>
                                        </Box>
                                    </Grid>
                                </Grid>
                                
                                <StatusText variant="body1" style={{ marginTop: 20, fontStyle: 'italic' }}>
                                    Используйте навигационное меню слева для перехода к нужному разделу системы.
                                </StatusText>
                            </CardContent>
                        </GlassCard>
                    </Grid>
                </Grid>
            </ContentContainer>
        </DashboardContainer>
    );
};

export default AdvancedDashboard;