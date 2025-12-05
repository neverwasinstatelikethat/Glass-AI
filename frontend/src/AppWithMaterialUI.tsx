import React, { useState } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { ThemeProvider, createTheme, styled } from '@mui/material/styles';
import { CssBaseline, Box, Paper, Typography, Chip } from '@mui/material';
import MIREANavigation from './components/MIREANavigation';
import AdvancedDashboard from './components/AdvancedDashboard';
import ARVisualization from './components/ARVisualization';
import DigitalTwin3D from './components/DigitalTwin3D';
import KnowledgeGraph from './components/KnowledgeGraph';
import { Factory, Science, Timeline, Warning } from '@mui/icons-material';

const theme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#0066FF',
            light: '#4D8FFF',
            dark: '#0047B3',
        },
        secondary: {
            main: '#FF3366',
            light: '#FF6690',
            dark: '#CC0033',
        },
        success: {
            main: '#00E676',
            light: '#69F0AE',
            dark: '#00C853',
        },
        warning: {
            main: '#FFD700',
            light: '#FFE44D',
            dark: '#FFC400',
        },
        error: {
            main: '#FF1744',
            light: '#FF5C7C',
            dark: '#D50000',
        },
        info: {
            main: '#00E5FF',
            light: '#6EFFFF',
            dark: '#00B8D4',
        },
        background: {
            default: '#000814',
            paper: 'rgba(13, 27, 42, 0.85)',
        },
        text: {
            primary: '#FFFFFF',
            secondary: '#B8C5D6',
        },
    },
    typography: {
        fontFamily: [
            'Inter',
            '-apple-system',
            'BlinkMacSystemFont',
            '"Segoe UI"',
            'Roboto',
            '"Helvetica Neue"',
            'Arial',
            'sans-serif',
        ].join(','),
        h1: {
            fontWeight: 800,
            letterSpacing: '-0.02em',
        },
        h2: {
            fontWeight: 800,
            letterSpacing: '-0.015em',
        },
        h3: {
            fontWeight: 700,
            letterSpacing: '-0.01em',
        },
        h4: {
            fontWeight: 700,
        },
        h5: {
            fontWeight: 600,
        },
        h6: {
            fontWeight: 600,
        },
        button: {
            fontWeight: 700,
            textTransform: 'none',
            letterSpacing: '0.02em',
        },
    },
    components: {
        MuiCssBaseline: {
            styleOverrides: {
                body: {
                    background: 'radial-gradient(ellipse at top, #001D3D 0%, #000814 50%, #000000 100%)',
                    backgroundAttachment: 'fixed',
                    '&::-webkit-scrollbar': {
                        width: 10,
                    },
                    '&::-webkit-scrollbar-track': {
                        background: 'rgba(0, 8, 20, 0.5)',
                    },
                    '&::-webkit-scrollbar-thumb': {
                        background: 'linear-gradient(180deg, #0066FF, #00E5FF)',
                        borderRadius: 10,
                        '&:hover': {
                            background: 'linear-gradient(180deg, #4D8FFF, #6EFFFF)',
                        },
                    },
                },
            },
        },
        MuiCard: {
            styleOverrides: {
                root: {
                    borderRadius: 20,
                    background: 'rgba(13, 27, 42, 0.6)',
                    backdropFilter: 'blur(20px) saturate(180%)',
                    border: '1px solid rgba(0, 102, 255, 0.15)',
                    boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5), 0 0 80px rgba(0, 102, 255, 0.1), inset 0 1px 0 rgba(255, 255, 255, 0.1)',
                    transition: 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)',
                    position: 'relative',
                    overflow: 'hidden',
                    '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: 0,
                        left: '-100%',
                        width: '100%',
                        height: '100%',
                        background: 'linear-gradient(90deg, transparent, rgba(0, 102, 255, 0.2), transparent)',
                        transition: 'left 0.5s',
                    },
                    '&:hover': {
                        transform: 'translateY(-12px) scale(1.02)',
                        boxShadow: '0 20px 60px rgba(0, 102, 255, 0.4), 0 0 100px rgba(0, 229, 255, 0.3), inset 0 1px 0 rgba(255, 255, 255, 0.2)',
                        borderColor: 'rgba(0, 229, 255, 0.5)',
                        '&::before': {
                            left: '100%',
                        },
                    },
                },
            },
        },
        MuiButton: {
            styleOverrides: {
                root: {
                    borderRadius: 14,
                    padding: '14px 32px',
                    fontWeight: 700,
                    textTransform: 'none',
                    fontSize: '1rem',
                    position: 'relative',
                    overflow: 'hidden',
                    transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
                    '&::before': {
                        content: '""',
                        position: 'absolute',
                        top: '50%',
                        left: '50%',
                        width: '0',
                        height: '0',
                        borderRadius: '50%',
                        background: 'rgba(255, 255, 255, 0.3)',
                        transform: 'translate(-50%, -50%)',
                        transition: 'width 0.6s, height 0.6s',
                    },
                    '&:hover::before': {
                        width: '300px',
                        height: '300px',
                    },
                },
                contained: {
                    background: 'linear-gradient(135deg, #0066FF 0%, #00E5FF 100%)',
                    boxShadow: '0 8px 24px rgba(0, 102, 255, 0.4), 0 0 40px rgba(0, 229, 255, 0.2)',
                    '&:hover': {
                        background: 'linear-gradient(135deg, #4D8FFF 0%, #6EFFFF 100%)',
                        boxShadow: '0 12px 36px rgba(0, 102, 255, 0.6), 0 0 60px rgba(0, 229, 255, 0.4)',
                        transform: 'translateY(-4px) scale(1.05)',
                    },
                },
            },
        },
        MuiChip: {
            styleOverrides: {
                root: {
                    borderRadius: 10,
                    fontWeight: 700,
                    fontSize: '0.85rem',
                    padding: '4px 8px',
                    backdropFilter: 'blur(10px)',
                    transition: 'all 0.3s ease',
                    '&:hover': {
                        transform: 'scale(1.1)',
                    },
                },
            },
        },
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundImage: 'none',
                    background: 'rgba(13, 27, 42, 0.6)',
                    backdropFilter: 'blur(20px)',
                },
            },
        },
    },
    shape: {
        borderRadius: 16,
    },
});

// --- STYLE CONSTANTS ---
const factoryIconSx = {
    fontSize: 56,
    color: '#00E5FF',
};

const timelineIconSx = {
    fontSize: 56,
    color: '#0066FF',
};

const warningIconSx = {
    fontSize: 56,
    color: '#FFD700',
};

const digitalTwinTitleSx = {
    fontWeight: 800,
    color: '#FFFFFF',
    marginBottom: '0.5rem',
};

const analyticsTitleSx = {
    fontWeight: 800,
    color: '#FFFFFF',
    marginBottom: '0.5rem',
};

const alertsTitleSx = {
    fontWeight: 800,
    color: '#FFFFFF',
    marginBottom: '0.5rem',
};

const subtitleSx = {
    color: '#B8C5D6',
};

const scienceIconSx = {
    fontSize: 100,
    color: '#0066FF',
    marginBottom: '2rem',
};

const warningLargeIconSx = {
    fontSize: 100,
    color: '#FFD700',
    marginBottom: '2rem',
};

const analyticsPaperTitleSx = {
    marginBottom: '2rem',
    color: '#FFFFFF',
    fontWeight: 800,
};

const alertsPaperTitleSx = {
    marginBottom: '2rem',
    color: '#FFFFFF',
    fontWeight: 800,
};

const chipSx = {
    backgroundColor: 'rgba(0, 229, 255, 0.2)',
    color: '#00E5FF',
    fontWeight: 800,
    fontSize: '1rem',
    padding: '24px 16px',
    border: '2px solid #00E5FF',
    boxShadow: '0 0 30px rgba(0, 229, 255, 0.5)',
};

const alertsChipSx = {
    backgroundColor: 'rgba(255, 215, 0, 0.2)',
    color: '#FFD700',
    fontWeight: 800,
    fontSize: '1rem',
    padding: '24px 16px',
    border: '2px solid #FFD700',
    boxShadow: '0 0 30px rgba(255, 215, 0, 0.5)',
};

// --- STYLED COMPONENTS ---

const StyledAppLayout = styled(Box)({
    display: 'flex',
    minHeight: '100vh',
});

const StyledContentArea = styled(Box)(({ theme }) => ({
    flexGrow: 1,
    background: 'radial-gradient(ellipse at top, #001D3D 0%, #000814 50%, #000000 100%)',
    minHeight: '100vh',
    position: 'relative',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: 'url("data:image/svg+xml,%3Csvg width=\'60\' height=\'60\' viewBox=\'0 0 60 60\' xmlns=\'http://www.w3.org/2000/svg\'%3E%3Cg fill=\'none\' fill-rule=\'evenodd\'%3E%3Cg fill=\'%230066FF\' fill-opacity=\'0.05\'%3E%3Cpath d=\'M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z\'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")',
        opacity: 0.3,
        pointerEvents: 'none',
    },
}));

const StyledPageContainer = styled(Box)(({ theme }) => ({
    padding: '24px',
    background: 'radial-gradient(ellipse at center, rgba(0, 102, 255, 0.05) 0%, transparent 70%)',
}));

const StyledHeaderBox = styled(Box)(({ theme }) => ({
    marginBottom: '24px',
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    padding: '24px',
    background: 'linear-gradient(135deg, rgba(0, 102, 255, 0.1) 0%, rgba(0, 229, 255, 0.05) 100%)',
    borderRadius: '4px',
    border: '1px solid rgba(0, 102, 255, 0.3)',
    backdropFilter: 'blur(10px)',
}));

const StyledAnalyticsContainer = styled(Box)(({ theme }) => ({
    padding: theme.spacing(3),
}));

const StyledAnalyticsHeader = styled(Box)(({ theme }) => ({
    marginBottom: theme.spacing(3),
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(2),
    padding: theme.spacing(3),
    background: 'linear-gradient(135deg, rgba(0, 102, 255, 0.1) 0%, rgba(255, 51, 102, 0.05) 100%)',
    borderRadius: theme.spacing(0.5),
    border: '1px solid rgba(0, 102, 255, 0.3)',
    backdropFilter: 'blur(10px)',
}));

const StyledAnalyticsPaper = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(4),
    textAlign: 'center',
    background: 'linear-gradient(135deg, rgba(13, 27, 42, 0.8) 0%, rgba(0, 29, 61, 0.6) 100%)',
    border: '2px solid rgba(0, 102, 255, 0.3)',
    borderRadius: theme.spacing(0.5),
    backdropFilter: 'blur(20px)',
    position: 'relative',
    overflow: 'hidden',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: '-50%',
        left: '-50%',
        width: '200%',
        height: '200%',
        background: 'radial-gradient(circle, rgba(0, 102, 255, 0.1) 0%, transparent 70%)',
        animation: 'rotate 20s linear infinite',
    },
    '@keyframes rotate': {
        '0%': { transform: 'rotate(0deg)' },
        '100%': { transform: 'rotate(360deg)' },
    },
}));

const StyledAlertsContainer = styled(Box)(({ theme }) => ({
    padding: theme.spacing(3),
}));

const StyledAlertsHeader = styled(Box)(({ theme }) => ({
    marginBottom: theme.spacing(3),
    display: 'flex',
    alignItems: 'center',
    gap: theme.spacing(2),
    padding: theme.spacing(3),
    background: 'linear-gradient(135deg, rgba(255, 215, 0, 0.1) 0%, rgba(255, 23, 68, 0.05) 100%)',
    borderRadius: theme.spacing(0.5),
    border: '1px solid rgba(255, 215, 0, 0.3)',
    backdropFilter: 'blur(10px)',
}));

const StyledAlertsPaper = styled(Paper)(({ theme }) => ({
    padding: theme.spacing(4),
    textAlign: 'center',
    background: 'linear-gradient(135deg, rgba(13, 27, 42, 0.8) 0%, rgba(61, 29, 0, 0.4) 100%)',
    border: '2px solid rgba(255, 215, 0, 0.3)',
    borderRadius: theme.spacing(0.5),
    backdropFilter: 'blur(20px)',
}));

// --- COMPONENTS ---

const DashboardPage = () => <AdvancedDashboard />;

const ARViewPage = () => (
    <div style={{ height: 'calc(100vh - 100px)' }}>
        <ARVisualization />
    </div>
);

const DigitalTwinHeader = () => (
    <StyledHeaderBox>
        <Factory sx={factoryIconSx} />
        <div style={{ color: '#FFFFFF' }}>
            <Typography variant="h3" sx={digitalTwinTitleSx}>
                Цифровой двойник производства
            </Typography>
            <Typography variant="h6" sx={subtitleSx}>
                3D визуализация производственной линии в реальном времени
            </Typography>
        </div>
    </StyledHeaderBox>
);

const DigitalTwinPage = () => (
    <StyledPageContainer>
        <DigitalTwinHeader />
        <DigitalTwin3D />
    </StyledPageContainer>
);

const KnowledgeGraphPage = () => (
    <div style={{ height: '100vh', overflow: 'auto' }}>
        <KnowledgeGraph />
    </div>
);

const AnalyticsHeader = () => (
    <StyledAnalyticsHeader>
        <Timeline sx={timelineIconSx} />
        <div style={{ color: '#FFFFFF' }}>
            <Typography variant="h3" sx={analyticsTitleSx}>
                Аналитика и отчеты
            </Typography>
            <Typography variant="h6" sx={subtitleSx}>
                Углубленный анализ производственных данных и трендов
            </Typography>
        </div>
    </StyledAnalyticsHeader>
);

const AnalyticsPage = () => (
    <StyledAnalyticsContainer>
        <AnalyticsHeader />
        <StyledAnalyticsPaper>
            <Science sx={scienceIconSx} />
            <Typography variant="h4" sx={analyticsPaperTitleSx}>
                Модуль расширенной аналитики
            </Typography>
            <Typography variant="h6" sx={{ color: '#B8C5D6', mb: 3, lineHeight: 1.8 }}>
                Здесь будут представлены детальные отчеты, прогнозные модели и статистический анализ
            </Typography>
            <Chip
                label="В РАЗРАБОТКЕ"
                sx={chipSx}

            />
        </StyledAnalyticsPaper>
    </StyledAnalyticsContainer>
);

const AlertsHeader = () => (
    <StyledAlertsHeader>
        <Warning sx={warningIconSx} />
        <div style={{ color: '#FFFFFF' }}>
            <Typography variant="h3" sx={alertsTitleSx}>
                Система оповещений
            </Typography>
            <Typography variant="h6" sx={subtitleSx}>
                Мониторинг и управление критическими событиями
            </Typography>
        </div>
    </StyledAlertsHeader>
);

const AlertsPage = () => (
    <StyledAlertsContainer>
        <AlertsHeader />
        <StyledAlertsPaper>
            <Warning sx={warningLargeIconSx} />
            <Typography variant="h4" sx={alertsPaperTitleSx}>
                Система уведомлений в реальном времени
            </Typography>
            <Typography variant="h6" sx={{ color: '#B8C5D6', mb: 3, lineHeight: 1.8 }}>
                Интеллектуальная система отслеживания аномалий и критических событий
            </Typography>
            <Chip
                label="В РАЗРАБОТКЕ"
                sx={alertsChipSx}

            />
        </StyledAlertsPaper>
    </StyledAlertsContainer>
);

// --- MAIN COMPONENT ---

function App() {
    return (
        <ThemeProvider theme={theme}>
            <CssBaseline />
            <Router>
                <StyledAppLayout>
                    <MIREANavigation />
                    <StyledContentArea>
                        <Routes>
                            <Route path="/" element={<DashboardPage />} />
                            <Route path="/dashboard" element={<DashboardPage />} />
                            <Route path="/digital-twin" element={<DigitalTwinPage />} />
                            <Route path="/3d-view" element={<DigitalTwinPage />} />
                            <Route path="/ar-view" element={<ARViewPage />} />
                            <Route path="/analytics" element={<AnalyticsPage />} />
                            <Route path="/alerts" element={<AlertsPage />} />
                            <Route path="/knowledge-graph" element={<KnowledgeGraphPage />} />
                        </Routes>
                    </StyledContentArea>
                </StyledAppLayout>
            </Router>
        </ThemeProvider>
    );
}

export default App;