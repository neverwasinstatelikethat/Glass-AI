import React from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import {
    Drawer,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Divider,
    Typography,
    Box,
    Chip,
    styled,
    Avatar
} from '@mui/material';
import {
    Dashboard,
    Assessment,
    Warning,
    Visibility,
    Memory,
    AccountCircle,
    Settings,
    Home,
    Factory,
    ThreeDRotation,
    Psychology,
    TrendingUp,
    Insights,
    Speed,
    Circle
} from '@mui/icons-material';

interface MIREANavigationProps {
    activePage?: string;
    onPageChange?: (page: string) => void;
}

const NavigationDrawer = styled(Drawer)(() => ({
    width: 300,
    flexShrink: 0,
    '& .MuiDrawer-paper': {
        width: 300,
        boxSizing: 'border-box',
        background: 'linear-gradient(180deg, rgba(0, 8, 20, 0.95) 0%, rgba(0, 29, 61, 0.95) 100%)',
        backdropFilter: 'blur(20px)',
        borderRight: 'none',
        boxShadow: '8px 0 40px rgba(0, 102, 255, 0.3), inset -1px 0 0 rgba(0, 102, 255, 0.2)',
        position: 'relative',
        overflow: 'hidden',
        '&::before': {
            content: '""',
            position: 'absolute',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'radial-gradient(circle at 20% 50%, rgba(0, 102, 255, 0.1) 0%, transparent 50%), radial-gradient(circle at 80% 80%, rgba(0, 229, 255, 0.1) 0%, transparent 50%)',
            pointerEvents: 'none',
        },
        '&::-webkit-scrollbar': {
            width: 8,
        },
        '&::-webkit-scrollbar-track': {
            background: 'rgba(0, 8, 20, 0.5)',
            borderRadius: 10,
            margin: '8px 0',
        },
        '&::-webkit-scrollbar-thumb': {
            background: 'linear-gradient(180deg, #0066FF, #00E5FF)',
            borderRadius: 10,
            '&:hover': {
                background: 'linear-gradient(180deg, #4D8FFF, #6EFFFF)',
            },
        },
    },
}));

const NavigationHeader = styled(Box)(() => ({
    padding: '32px 24px',
    background: 'linear-gradient(135deg, rgba(0, 102, 255, 0.2) 0%, rgba(0, 229, 255, 0.1) 100%)',
    borderBottom: '2px solid rgba(0, 102, 255, 0.3)',
    position: 'relative',
    overflow: 'hidden',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: '-50%',
        left: '-50%',
        width: '200%',
        height: '200%',
        background: 'radial-gradient(circle, rgba(0, 229, 255, 0.15) 0%, transparent 70%)',
        animation: 'rotate 15s linear infinite',
    },
    '@keyframes rotate': {
        '0%': { transform: 'rotate(0deg)' },
        '100%': { transform: 'rotate(360deg)' },
    },
}));

const LogoContainer = styled(Box)(() => ({
    display: 'flex',
    alignItems: 'center',
    gap: '16px',
    marginBottom: '20px',
    position: 'relative',
    zIndex: 1,
}));

const LogoAvatar = styled(Avatar)(() => ({
    width: 64,
    height: 64,
    background: 'linear-gradient(135deg, #0066FF 0%, #00E5FF 100%)',
    boxShadow: '0 8px 32px rgba(0, 102, 255, 0.5), 0 0 60px rgba(0, 229, 255, 0.3)',
    border: '3px solid rgba(0, 229, 255, 0.5)',
    fontSize: 32,
    fontWeight: 900,
}));

const TitleBox = styled(Box)(() => ({
    flex: 1,
}));

const AppTitle = styled(Typography)(() => ({
    fontWeight: 900,
    fontSize: '1.8rem',
    background: 'linear-gradient(135deg, #FFFFFF 0%, #00E5FF 100%)',
    WebkitBackgroundClip: 'text',
    WebkitTextFillColor: 'transparent',
    textShadow: '0 0 40px rgba(0, 229, 255, 0.5)',
    letterSpacing: '0.05em',
}));

const AppSubtitle = styled(Typography)(() => ({
    color: '#B8C5D6',
    fontWeight: 600,
    fontSize: '0.95rem',
    opacity: 0.9,
}));

const UniversityBadge = styled(Box)(() => ({
    textAlign: 'center',
    padding: '12px 16px',
    background: 'rgba(0, 102, 255, 0.15)',
    borderRadius: '12px',
    border: '1px solid rgba(0, 229, 255, 0.3)',
    position: 'relative',
    zIndex: 1,
}));

const UniversityText = styled(Typography)(() => ({
    color: '#00E5FF',
    fontWeight: 800,
    fontSize: '0.85rem',
    letterSpacing: '0.15em',
    textTransform: 'uppercase',
    textShadow: '0 0 20px rgba(0, 229, 255, 0.6)',
}));

const NavigationList = styled(List)(() => ({
    padding: '24px 16px',
    overflowY: 'auto',
    '&::-webkit-scrollbar': {
        width: 8,
    },
    '&::-webkit-scrollbar-track': {
        background: 'rgba(0, 8, 20, 0.5)',
        borderRadius: 10,
    },
    '&::-webkit-scrollbar-thumb': {
        background: 'linear-gradient(180deg, #0066FF, #00E5FF)',
        borderRadius: 10,
        '&:hover': {
            background: 'linear-gradient(180deg, #4D8FFF, #6EFFFF)',
        },
    },
}));

const StyledListItem = styled(ListItem)<{ active?: boolean }>(({ active }) => ({
    borderRadius: 16,
    marginBottom: 8,
    padding: '14px 18px',
    background: active
        ? 'linear-gradient(135deg, rgba(0, 102, 255, 0.3) 0%, rgba(0, 229, 255, 0.2) 100%)'
        : 'transparent',
    border: active
        ? '2px solid rgba(0, 229, 255, 0.5)'
        : '2px solid transparent',
    boxShadow: active
        ? '0 8px 32px rgba(0, 102, 255, 0.4), inset 0 0 20px rgba(0, 229, 255, 0.1)'
        : 'none',
    transition: 'all 0.4s cubic-bezier(0.34, 1.56, 0.64, 1)',
    position: 'relative',
    overflow: 'hidden',
    '&::before': {
        content: '""',
        position: 'absolute',
        top: 0,
        left: active ? 0 : '-100%',
        width: '100%',
        height: '100%',
        background: 'linear-gradient(90deg, transparent, rgba(0, 229, 255, 0.2), transparent)',
        transition: 'left 0.6s',
    },
    '&:hover': {
        background: 'linear-gradient(135deg, rgba(0, 102, 255, 0.2) 0%, rgba(0, 229, 255, 0.1) 100%)',
        border: '2px solid rgba(0, 229, 255, 0.4)',
        transform: 'translateX(8px) scale(1.02)',
        boxShadow: '0 12px 40px rgba(0, 102, 255, 0.5), inset 0 0 30px rgba(0, 229, 255, 0.15)',
        '&::before': {
            left: '100%',
        },
    },
}));

const StyledListItemIcon = styled(ListItemIcon)<{ active?: boolean }>(({ active }) => ({
    minWidth: 48,
    color: active ? '#00E5FF' : '#B8C5D6',
    transition: 'all 0.3s ease',
    '& .MuiSvgIcon-root': {
        fontSize: 28,
        filter: active ? 'drop-shadow(0 0 12px rgba(0, 229, 255, 0.8))' : 'none',
    },
}));

const StyledListItemText = styled(ListItemText)<{ active?: boolean }>(({ active }) => ({
    '& .MuiTypography-root': {
        fontWeight: active ? 800 : 600,
        fontSize: '1rem',
        color: active ? '#FFFFFF' : '#B8C5D6',
        textShadow: active ? '0 0 20px rgba(0, 229, 255, 0.5)' : 'none',
        transition: 'all 0.3s ease',
    },
}));

const StyledChip = styled(Chip)<{ chiptype?: 'new' | 'alert' }>(({ chiptype }) => ({
    height: 24,
    fontSize: '0.75rem',
    fontWeight: 800,
    background: chiptype === 'new'
        ? 'linear-gradient(135deg, #FF3366 0%, #FF6690 100%)'
        : 'linear-gradient(135deg, #FFD700 0%, #FFC400 100%)',
    color: '#FFFFFF',
    border: 'none',
    boxShadow: chiptype === 'new'
        ? '0 4px 16px rgba(255, 51, 102, 0.5), 0 0 20px rgba(255, 51, 102, 0.3)'
        : '0 4px 16px rgba(255, 215, 0, 0.5), 0 0 20px rgba(255, 215, 0, 0.3)',
    animation: 'pulse 2s ease-in-out infinite',
    '@keyframes pulse': {
        '0%, 100%': { transform: 'scale(1)' },
        '50%': { transform: 'scale(1.05)' },
    },
}));

const BottomSection = styled(Box)(() => ({
    marginTop: 'auto',
    padding: '16px',
}));

const StatusCard = styled(Box)(() => ({
    padding: '20px',
    borderRadius: 16,
    background: 'linear-gradient(135deg, rgba(0, 102, 255, 0.15) 0%, rgba(0, 229, 255, 0.1) 100%)',
    border: '2px solid rgba(0, 229, 255, 0.3)',
    marginBottom: 16,
    backdropFilter: 'blur(10px)',
    boxShadow: '0 8px 32px rgba(0, 102, 255, 0.3), inset 0 0 20px rgba(0, 229, 255, 0.1)',
}));

const StatusHeader = styled(Box)(() => ({
    display: 'flex',
    alignItems: 'center',
    gap: 12,
    marginBottom: 16,
}));

const StatusIcon = styled(Box)(() => ({
    width: 48,
    height: 48,
    borderRadius: 12,
    background: 'linear-gradient(135deg, #0066FF 0%, #00E5FF 100%)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    boxShadow: '0 4px 20px rgba(0, 102, 255, 0.5), 0 0 40px rgba(0, 229, 255, 0.3)',
}));

const StatusTextBox = styled(Box)(() => ({
    flex: 1,
}));

const StatusTitle = styled(Typography)(() => ({
    fontWeight: 800,
    color: '#FFFFFF',
    fontSize: '1rem',
    textShadow: '0 0 20px rgba(0, 229, 255, 0.3)',
}));

const StatusSubtitle = styled(Typography)(() => ({
    color: '#B8C5D6',
    fontSize: '0.85rem',
    fontWeight: 500,
}));

const StatusChips = styled(Box)(() => ({
    display: 'flex',
    gap: 8,
    flexWrap: 'wrap',
}));

const MetricChip = styled(Chip)(() => ({
    background: 'rgba(0, 230, 118, 0.2)',
    border: '1px solid rgba(0, 230, 118, 0.5)',
    color: '#00E676',
    fontWeight: 700,
    fontSize: '0.8rem',
    boxShadow: '0 4px 16px rgba(0, 230, 118, 0.3)',
    '& .MuiChip-icon': {
        color: '#00E676',
    },
}));

const AIChip = styled(Chip)(() => ({
    background: 'rgba(0, 229, 255, 0.2)',
    border: '1px solid rgba(0, 229, 255, 0.5)',
    color: '#00E5FF',
    fontWeight: 700,
    fontSize: '0.8rem',
    boxShadow: '0 4px 16px rgba(0, 229, 255, 0.3)',
    '& .MuiChip-icon': {
        color: '#00E5FF',
    },
}));

const SettingsItem = styled(ListItem)(() => ({
    borderRadius: 12,
    marginBottom: 8,
    padding: '12px 16px',
    transition: 'all 0.3s ease',
    '&:hover': {
        background: 'rgba(0, 102, 255, 0.15)',
        transform: 'translateX(8px)',
        boxShadow: '0 4px 20px rgba(0, 102, 255, 0.3)',
    },
}));

const MIREANavigation: React.FC<MIREANavigationProps> = ({ activePage, onPageChange }) => {
    const navigate = useNavigate();
    const location = useLocation();
    
    const handleNavigation = (path: string) => {
        navigate(path);
        if (onPageChange) onPageChange(path);
    };
    
    // Determine active page based on current location
    const currentActivePage = activePage || location.pathname;
    const navItems = [
        { id: '/', text: 'Главная', icon: <Home />, badge: null, badgeType: null },
        { id: '/dashboard', text: 'Панель управления', icon: <Dashboard />, badge: null, badgeType: null },
        { id: '/analytics', text: 'Аналитика', icon: <Assessment />, badge: null, badgeType: null },
        { id: '/alerts', text: 'Оповещения', icon: <Warning />, badge: '3', badgeType: 'alert' },
        { id: '/knowledge-graph', text: 'Граф знаний', icon: <Psychology />, badge: 'NEW', badgeType: 'new' },
        { id: '/3d-view', text: '3D Визуализация', icon: <ThreeDRotation />, badge: 'NEW', badgeType: 'new' },
        { id: '/ar-view', text: 'AR Режим', icon: <Visibility />, badge: 'NEW', badgeType: 'new' },
        { id: '/ai-insights', text: 'ИИ Аналитика', icon: <Insights />, badge: null, badgeType: null },
        { id: '/performance', text: 'Производительность', icon: <TrendingUp />, badge: null, badgeType: null },
        { id: '/digital-twin', text: 'Цифровой двойник', icon: <Factory />, badge: null, badgeType: null },
        { id: '/system', text: 'Система', icon: <Memory />, badge: null, badgeType: null },
    ];

    return (
        <NavigationDrawer variant="permanent" anchor="left">
            <NavigationHeader>
                <LogoContainer>
                    <LogoAvatar>
                        GA
                    </LogoAvatar>
                    <TitleBox>
                        <AppTitle>
                            Glass AI
                        </AppTitle>
                        <AppSubtitle>
                            Система контроля
                        </AppSubtitle>
                    </TitleBox>
                </LogoContainer>
                <UniversityBadge>
                    <UniversityText>
                        РТУ МИРЭА
                    </UniversityText>
                </UniversityBadge>
            </NavigationHeader>

            <Divider sx={{ borderColor: 'rgba(0, 102, 255, 0.2)', borderWidth: 1 }} />

            <NavigationList>
                {navItems.map((item) => {
                    const isActive = currentActivePage === item.id;

                    return (
                        <StyledListItem
                            key={item.id}
                            onClick={() => handleNavigation(item.id)}
                            active={isActive}
                        >
                            <StyledListItemIcon active={isActive}>
                                {item.icon}
                            </StyledListItemIcon>
                            <StyledListItemText
                                primary={item.text}
                                active={isActive}
                            />
                            {item.badge && (
                                <StyledChip
                                    label={item.badge}
                                    size="small"
                                    chiptype={item.badgeType as 'new' | 'alert'}
                                />
                            )}
                        </StyledListItem>
                    );
                })}
            </NavigationList>

            <BottomSection>
                <Divider sx={{ borderColor: 'rgba(0, 102, 255, 0.2)', borderWidth: 1, mb: 2 }} />

                <StatusCard>
                    <StatusHeader>
                        <StatusIcon>
                            <Circle sx={{
                                color: '#00E676',
                                fontSize: 24,
                                filter: 'drop-shadow(0 0 8px rgba(0, 230, 118, 0.8))',
                                animation: 'blink 2s ease-in-out infinite',
                                '@keyframes blink': {
                                    '0%, 100%': { opacity: 1 },
                                    '50%': { opacity: 0.5 },
                                },
                            }} />
                        </StatusIcon>
                        <StatusTextBox>
                            <StatusTitle>
                                Статус системы
                            </StatusTitle>
                            <StatusSubtitle>
                                Все системы в норме
                            </StatusSubtitle>
                        </StatusTextBox>
                    </StatusHeader>
                    <StatusChips>
                        <MetricChip
                            icon={<Speed />}
                            label="98.5%"
                            size="small"
                        />
                        <AIChip
                            icon={<Insights />}
                            label="ИИ активен"
                            size="small"
                        />
                    </StatusChips>
                </StatusCard>

                <SettingsItem onClick={() => {}}>
                    <ListItemIcon sx={{ color: '#B8C5D6', minWidth: 44 }}>
                        <AccountCircle sx={{ fontSize: 26 }} />
                    </ListItemIcon>
                    <ListItemText
                        primary="Профиль"
                        primaryTypographyProps={{
                            sx: {
                                fontSize: '0.95rem',
                                fontWeight: 600,
                                color: '#B8C5D6'
                            }
                        }}
                    />
                </SettingsItem>

                <SettingsItem onClick={() => {}}>
                    <ListItemIcon sx={{ color: '#B8C5D6', minWidth: 44 }}>
                        <Settings sx={{ fontSize: 26 }} />
                    </ListItemIcon>
                    <ListItemText
                        primary="Настройки"
                        primaryTypographyProps={{
                            sx: {
                                fontSize: '0.95rem',
                                fontWeight: 600,
                                color: '#B8C5D6'
                            }
                        }}
                    />
                </SettingsItem>
            </BottomSection>
        </NavigationDrawer>
    );
};

export default MIREANavigation;