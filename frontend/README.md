# Glass AI Frontend

Enhanced frontend for the Glass Production Predictive Analytics System with RTU MIREA styling.

## Features

### 🎨 MIREA-Themed Interface
- Custom dark theme with RTU MIREA colors (#00579c blue and #e91e63 pink accents)
- Modern gradient backgrounds and glass-morphism effects
- Responsive design with mobile-friendly navigation

### 📊 Advanced Dashboard
- Real-time KPI monitoring with animated cards
- Interactive charts for performance trends and defect distribution
- Live metrics visualization with progress indicators
- AI-powered recommendations from the Reinforcement Learning optimizer

### 🏭 3D Digital Twin Visualization
- Interactive 3D model of the glass production line using Three.js
- Real-time visualization of furnace, forming line, and conveyor systems
- Particle effects for defect visualization
- Dynamic lighting based on temperature and operational status

### 📱 AR Interface
- Augmented Reality visualization overlay
- Real-time sensor data display
- Interactive controls for camera modes and zoom
- Defect warning indicators

### 📈 Analytics & Predictions
- Predictive analytics dashboard
- Machine learning model insights
- Historical data visualization
- Performance forecasting

### ⚠️ Alert Management
- Priority-based alert system
- Real-time notifications
- Alert history and resolution tracking

## Technical Stack

- **React** with TypeScript for component-based architecture
- **Material-UI** for consistent, accessible UI components
- **Three.js** with React Three Fiber for 3D visualizations
- **Recharts** for data visualization
- **Framer Motion** for smooth animations
- **WebSocket** for real-time data streaming
- **CSS3** with gradients and glass-morphism effects

## Installation

```bash
npm install
```

## Development

```bash
npm start
```

The application will be available at http://localhost:3000

## Building for Production

```bash
npm run build
```

## Environment Variables

Create a `.env` file in the frontend directory with the following variables:

```
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws
```

## Architecture

The frontend is organized into the following components:

- `AppWithMaterialUI.tsx` - Main application component with routing
- `components/DigitalTwin3D.tsx` - 3D visualization of the production line
- `components/AdvancedDashboard.tsx` - Main dashboard with KPIs and charts
- `components/ARVisualization.tsx` - AR interface with overlay controls
- `components/MIREANavigation.tsx` - Navigation sidebar with MIREA styling
- `theme.ts` - Custom MIREA-themed Material-UI theme

## API Integration

The frontend connects to the backend API at the configured URL and uses WebSocket for real-time updates. Key endpoints include:

- `/api/sensors/data` - Sensor data ingestion
- `/api/defects` - Defect reporting
- `/api/predictions/request` - Prediction requests
- `/api/alerts/active` - Active alerts
- `/api/quality/metrics` - Quality metrics
- `/ws` - WebSocket endpoint for real-time updates

## Styling Guidelines

The interface follows RTU MIREA's visual identity:
- Primary color: #00579c (MIREA blue)
- Secondary color: #e91e63 (accent pink)
- Dark theme with gradient backgrounds
- Glass-morphism effects for cards and UI elements
- Smooth animations and transitions

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a pull request

## License

This project is part of the Glass Production Predictive Analytics System developed for RTU MIREA.