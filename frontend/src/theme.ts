// theme.ts
import { createTheme } from '@mui/material/styles';
import { ruRU } from '@mui/material/locale';

const theme = createTheme({
  palette: {
    mode: 'dark',
    primary: {
      main: '#00D4FF',
      light: '#66E8FF',
      dark: '#00A3CC',
      contrastText: '#000',
    },
    secondary: {
      main: '#FF3D71',
      light: '#FF7096',
      dark: '#C7003A',
    },
    background: {
      default: '#0B0E17',
      paper: 'rgba(17, 25, 40, 0.75)',
    },
    error: { main: '#FF453A' },
    warning: { main: '#FFB340' },
    success: { main: '#32D74B' },
    info: { main: '#64D2FF' },
    text: {
      primary: '#FFFFFF',
      secondary: '#B0B8D1',
    },
    divider: 'rgba(100, 210, 255, 0.12)',
  },
  typography: {
    fontFamily: '"SF Pro Display", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: { fontWeight: 800, letterSpacing: '-0.02em' },
    h2: { fontWeight: 800, letterSpacing: '-0.01em' },
    h3: { fontWeight: 700 },
    h4: { fontWeight: 700 },
    h5: { fontWeight: 600 },
    button: { fontWeight: 700, textTransform: 'none', letterSpacing: '0.5px' },
  },
  shape: { borderRadius: 16 },
  components: {
    MuiCssBaseline: {
      styleOverrides: {
        body: {
          background: 'linear-gradient(135deg, #0B0E17 0%, #1A2332 100%)',
          backgroundAttachment: 'fixed',
          scrollbarColor: '#00D4FF #0B0E17',
          '&::-webkit-scrollbar': { width: 8 },
          '&::-webkit-scrollbar-track': { background: '#0B0E17' },
          '&::-webkit-scrollbar-thumb': { background: '#00D4FF', borderRadius: 4 },
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          background: 'rgba(17, 25, 40, 0.65)',
          backdropFilter: 'blur(16px)',
          border: '1px solid rgba(100, 210, 255, 0.2)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.4), 0 0 32px rgba(0, 212, 255, 0.1)',
          transition: 'all 0.4s cubic-bezier(0.25, 0.8, 0.25, 1)',
          '&:hover': {
            transform: 'translateY(-12px)',
            boxShadow: '0 24px 64px rgba(0, 0, 0, 0.6), 0 0 48px rgba(0, 212, 255, 0.3)',
            borderColor: 'rgba(100, 210, 255, 0.5)',
          },
        },
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          borderRadius: 12,
          padding: '12px 28px',
          fontWeight: 700,
          textTransform: 'none',
          boxShadow: '0 4px 20px rgba(0, 212, 255, 0.25)',
          '&:hover': {
            boxShadow: '0 8px 32px rgba(0, 212, 255, 0.4)',
            transform: 'translateY(-2px)',
          },
        },
        containedPrimary: {
          background: 'linear-gradient(135deg, #00D4FF, #0099CC)',
          color: '#000',
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none',
          background: 'rgba(17, 25, 40, 0.65)',
          backdropFilter: 'blur(16px)',
        },
      },
    },
  },
}, ruRU);

export default theme;