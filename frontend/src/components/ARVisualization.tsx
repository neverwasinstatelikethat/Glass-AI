import React, { useState, useEffect } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, Environment, Html } from '@react-three/drei';
import * as THREE from 'three';
import { 
  Box, 
  Typography, 
  Fab, 
  IconButton, 
  Slider, 
  FormControl, 
  InputLabel, 
  Select, 
  MenuItem, 
  Checkbox, 
  FormControlLabel, 
  FormGroup,
  Paper,
  Chip
} from '@mui/material';
import useWebSocket from '../hooks/useWebSocket';
import { systemApi } from '../services/api';
import { 
  PlayArrow, 
  Pause, 
  RotateLeft,
  ZoomIn, 
  ZoomOut, 
  Visibility, 
  VisibilityOff,
  Settings,
  Warning,
  Thermostat,
  Speed
} from '@mui/icons-material';

// Цветовая палитра РТУ МИРЭА
const MIREA_COLORS = {
  primary: '#0066CC',
  secondary: '#FF6B35',
  success: '#00C853',
  warning: '#FFA726',
  error: '#EF5350',
  info: '#29B6F6',
  background: '#0A1929',
  surfaceLight: '#2A3A4F',
  text: '#FFFFFF'
};

// Types
interface SensorData {
  furnace: {
    temperature: number;
    pressure: number;
    meltLevel: number;
  };
  forming: {
    beltSpeed: number;
    moldTemp: number;
    coolingRate: number;
  };
  defects: Record<string, number>;
}

interface DefectData {
  position: [number, number, number];
  type: string;
  severity: number;
}

// Печь для стекла
const GlassFurnace: React.FC<{ 
  temperature: number; 
  meltLevel: number;
  defects: DefectData[];
  showDefects: boolean;
}> = ({ temperature, meltLevel, defects, showDefects }) => {
  const getColor = (temp: number) => {
    const normalized = Math.min(1, Math.max(0, (temp - 1400) / 200));
    return new THREE.Color(
      Math.min(1, normalized * 2),
      Math.max(0, 1 - Math.abs(normalized - 0.5) * 2),
      Math.max(0, (1 - normalized) * 2)
    );
  };

  return (
    <group position={[0, 0, 0]}>
      <mesh position={[0, 2, 0]}>
        <boxGeometry args={[4, 4, 4]} />
        <meshStandardMaterial 
          color={getColor(temperature)} 
          transparent 
          opacity={0.8}
          emissive={getColor(temperature)}
          emissiveIntensity={0.2}
        />
      </mesh>
      
      <mesh position={[0, 0.5, 0]}>
        <cylinderGeometry args={[1.5, 1.8, meltLevel * 0.5, 32]} />
        <meshStandardMaterial 
          color="#ff9800" 
          transparent 
          opacity={0.7}
        />
      </mesh>
      
      {showDefects && defects.map((defect, index) => (
        <mesh key={index} position={defect.position}>
          <sphereGeometry args={[defect.severity * 0.1, 16, 16]} />
          <meshStandardMaterial 
            color={defect.type === 'bubble' ? '#00bcd4' : defect.type === 'crack' ? '#f44336' : '#ffeb3b'} 
            emissive={defect.type === 'bubble' ? '#00bcd4' : defect.type === 'crack' ? '#f44336' : '#ffeb3b'}
            emissiveIntensity={0.3}
          />
          <Html distanceFactor={10}>
            <div style={{ 
              background: 'rgba(0,0,0,0.7)', 
              padding: '4px 8px', 
              borderRadius: '4px',
              color: 'white',
              fontSize: '10px'
            }}>
              {defect.type === 'bubble' ? 'Пузырь' : defect.type === 'crack' ? 'Трещина' : 'Скол'}
            </div>
          </Html>
        </mesh>
      ))}
    </group>
  );
};

// Формовочная линия
const FormingLine: React.FC<{ 
  beltSpeed: number; 
  moldTemp: number;
  showLabels: boolean;
}> = ({ beltSpeed, moldTemp, showLabels }) => {
  return (
    <group position={[6, 0, 0]}>
      <mesh position={[0, 0.5, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <boxGeometry args={[8, 0.1, 2]} />
        <meshStandardMaterial color="#795548" />
      </mesh>
      
      <mesh position={[0, 1.5, 0]}>
        <boxGeometry args={[2, 2, 2]} />
        <meshStandardMaterial 
          color="#607d8b" 
          emissive="#607d8b"
          emissiveIntensity={moldTemp > 320 ? 0.2 : 0}
        />
      </mesh>
      
      {showLabels && (
        <>
          <Html position={[0, 3, 0]} center>
            <div style={{ 
              background: 'rgba(0,0,0,0.7)', 
              padding: '4px 8px', 
              borderRadius: '4px',
              color: 'white'
            }}>
              Скорость ленты: {beltSpeed} м/мин
            </div>
          </Html>
          <Html position={[0, 2.5, 0]} center>
            <div style={{ 
              background: 'rgba(0,0,0,0.7)', 
              padding: '4px 8px', 
              borderRadius: '4px',
              color: 'white'
            }}>
              Температура формы: {moldTemp}°C
            </div>
          </Html>
        </>
      )}
    </group>
  );
};

// Конвейерная система
const ConveyorSystem: React.FC<{ 
  speed: number;
  showLabels: boolean;
}> = ({ speed, showLabels }) => {
  return (
    <group position={[-6, 0, 0]}>
      <mesh position={[0, 0.5, 0]} rotation={[Math.PI / 2, 0, 0]}>
        <boxGeometry args={[8, 0.1, 2]} />
        <meshStandardMaterial color="#9e9e9e" />
      </mesh>
      
      {[...Array(5)].map((_, i) => (
        <mesh key={i} position={[-3 + i * 1.5, 1, 0]}>
          <cylinderGeometry args={[0.3, 0.3, 2, 32]} />
          <meshStandardMaterial color="#616161" />
        </mesh>
      ))}
      
      {showLabels && (
        <Html position={[0, 2, 0]} center>
          <div style={{ 
            background: 'rgba(0,0,0,0.7)', 
            padding: '4px 8px', 
            borderRadius: '4px',
            color: 'white'
          }}>
            Скорость конвейера: {speed} м/мин
          </div>
        </Html>
      )}
    </group>
  );
};

// Главная 3D сцена
const GlassProductionScene: React.FC<{ 
  sensorData: SensorData;
  showDefects: boolean;
  showLabels: boolean;
}> = ({ sensorData, showDefects, showLabels }) => {
  const generateDefects = (): DefectData[] => {
    const defects: DefectData[] = [];
    Object.entries(sensorData.defects).forEach(([type, count]) => {
      for (let i = 0; i < count; i++) {
        defects.push({
          position: [
            (Math.random() - 0.5) * 3,
            2 + Math.random() * 2,
            (Math.random() - 0.5) * 3
          ] as [number, number, number],
          type,
          severity: Math.random()
        });
      }
    });
    return defects;
  };

  const defects = generateDefects();

  return (
    <>
      <ambientLight intensity={0.5} />
      <pointLight position={[10, 10, 10]} intensity={1} />
      <spotLight
        position={[0, 10, 0]}
        angle={0.3}
        penumbra={1}
        intensity={1}
        castShadow
      />
      
      <GlassFurnace 
        temperature={sensorData.furnace.temperature} 
        meltLevel={sensorData.furnace.meltLevel}
        defects={defects}
        showDefects={showDefects}
      />
      
      <FormingLine 
        beltSpeed={sensorData.forming.beltSpeed} 
        moldTemp={sensorData.forming.moldTemp}
        showLabels={showLabels}
      />
      
      <ConveyorSystem 
        speed={sensorData.forming.beltSpeed * 0.8} 
        showLabels={showLabels}
      />
      
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
      />
      
      <PerspectiveCamera makeDefault position={[0, 5, 15]} fov={50} />
      <Environment preset="city" />
    </>
  );
};

// AR наложение
const AROverlay: React.FC<{ 
  sensorData: SensorData; 
  defects: Record<string, number>; 
  showOverlay: boolean;
}> = ({ sensorData, defects, showOverlay }) => {
  if (!showOverlay) return null;

  return (
    <div style={{
      position: 'absolute',
      top: 0,
      left: 0,
      width: '100%',
      height: '100%',
      pointerEvents: 'none',
      zIndex: 1000
    }}>
      <Paper
        elevation={4}
        style={{
          position: 'absolute',
          top: '20%',
          left: '10%',
          background: 'rgba(0, 0, 0, 0.8)',
          padding: '16px',
          borderRadius: '8px',
          border: `2px solid ${MIREA_COLORS.secondary}`
        }}
      >
        <Typography variant="body2" style={{ color: MIREA_COLORS.text }}>
          Печь
        </Typography>
        <Typography variant="h6" style={{ color: MIREA_COLORS.secondary, fontWeight: 700 }}>
          {sensorData.furnace?.temperature?.toFixed(0) || 0}°C
        </Typography>
      </Paper>

      <Paper
        elevation={4}
        style={{
          position: 'absolute',
          top: '20%',
          right: '10%',
          background: 'rgba(0, 0, 0, 0.8)',
          padding: '16px',
          borderRadius: '8px',
          border: `2px solid ${MIREA_COLORS.success}`
        }}
      >
        <Typography variant="body2" style={{ color: MIREA_COLORS.text }}>
          Формовочная линия
        </Typography>
        <Typography variant="h6" style={{ color: MIREA_COLORS.success, fontWeight: 700 }}>
          {sensorData.forming?.beltSpeed?.toFixed(0) || 0} м/мин
        </Typography>
      </Paper>

      {Object.entries(defects).map(([type, count], index) => (
        count > 5 && (
          <Paper
            key={type}
            elevation={4}
            style={{
              position: 'absolute',
              bottom: `${20 + index * 15}%`,
              left: '50%',
              transform: 'translateX(-50%)',
              background: 'rgba(255, 0, 0, 0.9)',
              padding: '16px',
              borderRadius: '8px',
              border: `2px solid ${MIREA_COLORS.error}`,
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}
          >
            <Warning style={{ color: MIREA_COLORS.text }} />
            <Typography variant="body2" style={{ color: MIREA_COLORS.text }}>
              Обнаружено {type === 'bubbles' ? 'пузырей' : type === 'cracks' ? 'трещин' : 'сколов'}: {count}
            </Typography>
          </Paper>
        )
      ))}
    </div>
  );
};

// Панель управления
interface ControlPanelProps {
  isPlaying: boolean;
  onPlayPause: () => void;
  onReset: () => void;
  cameraMode: string;
  onCameraModeChange: (mode: string) => void;
  zoomLevel: number;
  onZoomChange: (zoom: number) => void;
  showDefects: boolean;
  onShowDefectsChange: (show: boolean) => void;
  showLabels: boolean;
  onShowLabelsChange: (show: boolean) => void;
}

const ControlPanel: React.FC<ControlPanelProps> = ({ 
  isPlaying, 
  onPlayPause, 
  onReset,
  cameraMode,
  onCameraModeChange,
  zoomLevel,
  onZoomChange,
  showDefects,
  onShowDefectsChange,
  showLabels,
  onShowLabelsChange
}) => {
  return (
    <Paper
      elevation={8}
      style={{
        position: 'absolute', 
        bottom: '20px', 
        left: '50%', 
        transform: 'translateX(-50%)',
        display: 'flex',
        gap: '16px',
        alignItems: 'center',
        background: `linear-gradient(135deg, ${MIREA_COLORS.background}E6, ${MIREA_COLORS.surfaceLight}E6)`,
        backdropFilter: 'blur(10px)',
        padding: '16px',
        borderRadius: '32px',
        zIndex: 1001,
        border: `1px solid ${MIREA_COLORS.primary}60`
      }}
    >
      <Fab 
        size="small" 
        style={{
          backgroundColor: isPlaying ? MIREA_COLORS.secondary : MIREA_COLORS.primary,
        }}
        onClick={onPlayPause}
      >
        {isPlaying ? <Pause /> : <PlayArrow />}
      </Fab>
      
      <Fab size="small" style={{ backgroundColor: MIREA_COLORS.surfaceLight }} onClick={onReset}>
        <RotateLeft />
      </Fab>
      
      <IconButton onClick={() => onZoomChange(Math.max(1, zoomLevel - 0.5))}>
        <ZoomOut style={{ color: MIREA_COLORS.text }} />
      </IconButton>
      
      <div style={{ width: '100px' }}>
        <Slider
          value={zoomLevel}
          onChange={(_, value) => onZoomChange(value as number)}
          min={1}
          max={3}
          step={0.1}
          style={{ 
            color: MIREA_COLORS.primary
          }}
        />
      </div>
      
      <IconButton onClick={() => onZoomChange(Math.min(3, zoomLevel + 0.5))}>
        <ZoomIn style={{ color: MIREA_COLORS.text }} />
      </IconButton>
      
      <div style={{ minWidth: '120px', marginRight: '16px' }}>
        <FormControl size="small" style={{ width: '100%' }}>
          <InputLabel style={{ color: MIREA_COLORS.text }}>Камера</InputLabel>
          <Select
            value={cameraMode}
            label="Камера"
            onChange={(e) => onCameraModeChange(e.target.value as string)}
            style={{
              color: MIREA_COLORS.text,
            }}
            sx={{
              '& .MuiOutlinedInput-notchedOutline': {
                borderColor: MIREA_COLORS.primary
              }
            }}
          >
            <MenuItem value="orbit">Орбита</MenuItem>
            <MenuItem value="fixed">Фиксированная</MenuItem>
            <MenuItem value="follow">Следящая</MenuItem>
          </Select>
        </FormControl>
      </div>
      
      <FormGroup row>
        <FormControlLabel
          control={
            <Checkbox
              checked={showDefects}
              onChange={(e) => onShowDefectsChange(e.target.checked)}
              style={{
                color: MIREA_COLORS.text,
              }}
              sx={{
                '&.Mui-checked': {
                  color: MIREA_COLORS.primary
                }
              }}
            />
          }
          label={<Typography style={{ color: MIREA_COLORS.text }}>Дефекты</Typography>}
        />
        <FormControlLabel
          control={
            <Checkbox
              checked={showLabels}
              onChange={(e) => onShowLabelsChange(e.target.checked)}
              style={{
                color: MIREA_COLORS.text,
              }}
              sx={{
                '&.Mui-checked': {
                  color: MIREA_COLORS.primary
                }
              }}
            />
          }
          label={<Typography style={{ color: MIREA_COLORS.text }}>Метки</Typography>}
        />
      </FormGroup>
    </Paper>
  );
};

// Главный компонент AR визуализации
const ARVisualization: React.FC = () => {
  const [sensorData, setSensorData] = useState<SensorData>({
    furnace: {
      temperature: 1520,
      pressure: 2.5,
      meltLevel: 2.4
    },
    forming: {
      beltSpeed: 155,
      moldTemp: 325,
      coolingRate: 1.2
    },
    defects: {
      bubbles: 12,
      cracks: 8,
      chips: 5
    }
  });
  
  const [isPlaying, setIsPlaying] = useState(true);
  const [cameraMode, setCameraMode] = useState('orbit');
  const [zoomLevel, setZoomLevel] = useState(1);
  const [showDefects, setShowDefects] = useState(true);
  const [showLabels, setShowLabels] = useState(true);
  const [showOverlay, setShowOverlay] = useState(true);
  
  // Initialize WebSocket connection
  const wsUrl = `${process.env.REACT_APP_WS_URL || 'ws://localhost:8000/ws'}`;
  const { isConnected, sendMessage } = useWebSocket(wsUrl, {
    onMessage: (data) => {
      // Handle real-time updates from WebSocket
      if (data.type === 'sensor_update' && data.data) {
        // Update sensor data with real values from backend
        setSensorData(prev => ({
          ...prev,
          furnace: {
            ...prev.furnace,
            temperature: data.data.sensors?.furnace_temperature || prev.furnace.temperature,
            pressure: data.data.sensors?.furnace_pressure || prev.furnace.pressure,
            meltLevel: data.data.sensors?.melt_level || prev.furnace.meltLevel
          },
          forming: {
            ...prev.forming,
            beltSpeed: data.data.sensors?.belt_speed || prev.forming.beltSpeed,
            moldTemp: data.data.sensors?.mold_temperature || prev.forming.moldTemp,
            coolingRate: data.data.sensors?.cooling_rate || prev.forming.coolingRate
          }
        }));
      } else if (data.type === 'realtime_update' && data.data) {
        // Handle general real-time updates
        console.log('Real-time update received:', data.data);
      }
    },
    onOpen: () => {
      console.log('WebSocket connected to backend');
      // Subscribe to updates when connection is established
      sendMessage({ type: 'subscribe', topics: ['sensors', 'realtime'] });
    },
    onClose: () => {
      console.log('WebSocket disconnected from backend');
    },
    onError: (error) => {
      console.error('WebSocket error:', error);
    }
  });
  
  // Fetch initial data from backend API
  useEffect(() => {
    const fetchInitialData = async () => {
      try {
        // Try to get digital twin state for initial values
        const digitalTwinData = await systemApi.getDigitalTwinState();
        if (digitalTwinData.state) {
          setSensorData(prev => ({
            ...prev,
            furnace: {
              ...prev.furnace,
              temperature: digitalTwinData.state.furnace_temperature || prev.furnace.temperature,
              meltLevel: (digitalTwinData.state.melt_level / 1000) || prev.furnace.meltLevel // Convert units
            },
            forming: {
              ...prev.forming,
              beltSpeed: digitalTwinData.state.forming_belt_speed || prev.forming.beltSpeed,
            }
          }));
        }
      } catch (error) {
        console.error('Error fetching initial data:', error);
        // Continue with default values
      }
    };
    
    fetchInitialData();
  }, []);

  return (
    <div style={{ width: '100%', height: '100%', position: 'relative' }}>
      <Canvas
        camera={{ position: [0, 5, 15], fov: 50 }}
        style={{ background: `linear-gradient(to bottom, ${MIREA_COLORS.background}, ${MIREA_COLORS.surfaceLight})` }}
      >
        <GlassProductionScene 
          sensorData={sensorData}
          showDefects={showDefects}
          showLabels={showLabels}
        />
      </Canvas>
      
      <AROverlay 
        sensorData={sensorData}
        defects={sensorData.defects}
        showOverlay={showOverlay}
      />
      
      <ControlPanel
        isPlaying={isPlaying}
        onPlayPause={() => setIsPlaying(!isPlaying)}
        onReset={() => {
          setSensorData({
            furnace: {
              temperature: 1520,
              pressure: 2.5,
              meltLevel: 2.4
            },
            forming: {
              beltSpeed: 155,
              moldTemp: 325,
              coolingRate: 1.2
            },
            defects: {
              bubbles: 12,
              cracks: 8,
              chips: 5
            }
          });
        }}
        cameraMode={cameraMode}
        onCameraModeChange={setCameraMode}
        zoomLevel={zoomLevel}
        onZoomChange={setZoomLevel}
        showDefects={showDefects}
        onShowDefectsChange={setShowDefects}
        showLabels={showLabels}
        onShowLabelsChange={setShowLabels}
      />
      
      <Fab 
        size="small" 
        style={{
          position: 'absolute', 
          top: '20px', 
          right: '20px',
          zIndex: 1001,
          backgroundColor: showOverlay ? MIREA_COLORS.primary : MIREA_COLORS.surfaceLight,
        }}
        onClick={() => setShowOverlay(!showOverlay)}
      >
        {showOverlay ? <VisibilityOff /> : <Visibility />}
      </Fab>
      
      <Fab 
        size="small" 
        style={{
          position: 'absolute', 
          top: '20px', 
          right: '80px',
          zIndex: 1001,
          backgroundColor: MIREA_COLORS.surfaceLight
        }}
      >
        <Settings />
      </Fab>
    </div>
  );
};

export default ARVisualization;