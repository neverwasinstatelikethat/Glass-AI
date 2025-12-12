import React, { useRef, useState, useEffect, useMemo, useCallback } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text } from '@react-three/drei';
import * as THREE from 'three';
import { 
  Box, Chip, Paper, Typography, Slider, Button, 
  Grid, Card, CardContent, Switch, FormControlLabel, Divider,
  Stack, LinearProgress, Tooltip, IconButton, Alert, Collapse
} from '@mui/material';
import { 
  PlayArrow, Pause, Refresh, TrendingUp, Science,
  LocalFireDepartment, Thermostat, Speed, Warning, CheckCircle,
  Visibility, VisibilityOff, Info, ExpandMore, ExpandLess
} from '@mui/icons-material';

// Интерфейсы типов
interface MIREAColors {
  primary: string;
  primaryDark: string;
  primaryLight: string;
  secondary: string;
  secondaryDark: string;
  accent: string;
  success: string;
  warning: string;
  error: string;
  background: string;
  backgroundLight: string;
  surface: string;
  surfaceLight: string;
  text: string;
  textSecondary: string;
  glass: string;
}

interface WebSocketData {
  parameters: any;
  defectAlerts: any[];
  isConnected: boolean;
}

interface FurnaceProps {
  temperature: number;
  thermalField: number[][] | null;
  showField: boolean;
  defects: { [key: string]: number };
}

interface ThermalFieldProps {
  field: number[][];
  position: [number, number, number];
}

interface FormingLineProps {
  speed: number;
  moldTemp: number;
  glass: boolean;
  defects: { [key: string]: number };
}

interface DefectParticlesProps {
  defects: { [key: string]: number };
  showDefects: boolean;
  furnacePosition: [number, number, number];
  moldPositions: Array<[number, number, number]>;
}

interface DigitalTwinSceneProps {
  furnaceTemp: number;
  moldTemp: number;
  beltSpeed: number;
  defects: { [key: string]: number };
  thermalField: number[][] | null;
  showThermalField: boolean;
  showDefects: boolean;
}

// Цветовая палитра РТУ МИРЭА
const MIREA_COLORS: MIREAColors = {
  primary: '#0066CC',
  primaryDark: '#004C99',
  primaryLight: '#3385D6',
  secondary: '#FF6B35',
  secondaryDark: '#CC5529',
  accent: '#00D9FF',
  success: '#00C853',
  warning: '#FFA726',
  error: '#EF5350',
  background: '#0A0E1A',
  backgroundLight: '#141824',
  surface: '#1E2330',
  surfaceLight: '#2A3142',
  text: '#FFFFFF',
  textSecondary: '#B0B8C8',
  glass: 'rgba(255, 255, 255, 0.05)'
};

// Hook для WebSocket подключения
const useWebSocket = (url: string): WebSocketData => {
  const [wsData, setWsData] = useState<WebSocketData>({
    parameters: null,
    defectAlerts: [],
    isConnected: false
  });

  useEffect(() => {
    const ws = new WebSocket(url);
    
    ws.onopen = () => {
      console.log('WebSocket connected');
      setWsData(prev => ({ ...prev, isConnected: true }));
    };
    
    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data);
        
        if (message.type === 'parameter_update') {
          setWsData(prev => ({
            ...prev,
            parameters: message.data
          }));
        } else if (message.type === 'defect_alert') {
          setWsData(prev => ({
            ...prev,
            defectAlerts: [...prev.defectAlerts.slice(-19), message.data]
          }));
        }
      } catch (err) {
        console.error('WebSocket parse error:', err);
      }
    };
    
    ws.onclose = () => {
      console.log('WebSocket disconnected');
      setWsData(prev => ({ ...prev, isConnected: false }));
    };
    
    return () => ws.close();
  }, [url]);

  return wsData;
};

// Компонент: Дефекты на поверхности печи
const FurnaceDefects = ({ defects, position }: { defects: { [key: string]: number }, position: [number, number, number] }) => {
  const groupRef = useRef<THREE.Group>(null);
  
  const particles = useMemo(() => {
    const totalDefects = Object.values(defects).reduce((sum: number, val: number) => sum + val, 0);
    const count = Math.min(50, Math.floor(totalDefects * 100));
    
    const defectColors = {
      crack: MIREA_COLORS.error,
      bubble: MIREA_COLORS.warning,
      chip: MIREA_COLORS.secondary,
      cloudiness: '#AAAAAA',
      deformation: '#9C27B0',
      stress: '#E91E63'
    };
    
    return Array.from({ length: count }, (_, i) => {
      // Генерация на поверхности прямоугольного параллелепипеда (4x3x3)
      const face = Math.floor(Math.random() * 6);
      let x = 0, y = 0, z = 0;
      
      switch(face) {
        case 0: // front
          x = (Math.random() - 0.5) * 4;
          y = (Math.random() - 0.5) * 3;
          z = 1.5;
          break;
        case 1: // back
          x = (Math.random() - 0.5) * 4;
          y = (Math.random() - 0.5) * 3;
          z = -1.5;
          break;
        case 2: // right
          x = 2;
          y = (Math.random() - 0.5) * 3;
          z = (Math.random() - 0.5) * 3;
          break;
        case 3: // left
          x = -2;
          y = (Math.random() - 0.5) * 3;
          z = (Math.random() - 0.5) * 3;
          break;
        case 4: // top
          x = (Math.random() - 0.5) * 4;
          y = 1.5;
          z = (Math.random() - 0.5) * 3;
          break;
        case 5: // bottom
          x = (Math.random() - 0.5) * 4;
          y = -1.5;
          z = (Math.random() - 0.5) * 3;
          break;
      }
      
      const defectTypes = Object.keys(defects);
      const selectedType = defectTypes[Math.floor(Math.random() * defectTypes.length)];
      
      return {
        position: [x, y, z] as [number, number, number],
        color: (defectColors as any)[selectedType] || '#FFFFFF',
        size: 0.03 + Math.random() * 0.05,
        type: selectedType
      };
    });
  }, [defects]);
  
  useFrame((state) => {
    if (groupRef.current) {
      // Легкое мерцание дефектов
      groupRef.current.children.forEach((child, i) => {
        if (child instanceof THREE.Mesh) {
          const intensity = 0.5 + 0.3 * Math.sin(state.clock.elapsedTime * 3 + i);
          child.material.emissiveIntensity = intensity;
        }
      });
    }
  });
  
  return (
    <group ref={groupRef} position={position}>
      {particles.map((particle, i) => (
        <mesh key={i} position={particle.position}>
          <sphereGeometry args={[particle.size]} />
          <meshStandardMaterial 
            color={particle.color}
            emissive={particle.color}
            emissiveIntensity={0.5}
            transparent
            opacity={0.7}
          />
        </mesh>
      ))}
    </group>
  );
};

// Компонент: Дефекты на формах
const MoldDefects = ({ defects, positions }: { defects: { [key: string]: number }, positions: Array<[number, number, number]> }) => {
  const groupRef = useRef<THREE.Group>(null);
  
  const particles = useMemo(() => {
    const totalDefects = Object.values(defects).reduce((sum: number, val: number) => sum + val, 0);
    const particlesPerMold = Math.min(15, Math.floor(totalDefects * 30 / positions.length));
    
    const defectColors = {
      crack: MIREA_COLORS.error,
      bubble: MIREA_COLORS.warning,
      chip: MIREA_COLORS.secondary,
      cloudiness: '#AAAAAA',
      deformation: '#9C27B0',
      stress: '#E91E63'
    };
    
    const allParticles: Array<{
      position: [number, number, number];
      color: string;
      size: number;
      type: string;
    }> = [];
    
    positions.forEach((moldPos, moldIndex) => {
      for (let i = 0; i < particlesPerMold; i++) {
        const angle = Math.random() * Math.PI * 2;
        const height = (Math.random() - 0.5) * 0.8;
        const radius = 0.6;
        
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        
        const defectTypes = Object.keys(defects);
        const selectedType = defectTypes[Math.floor(Math.random() * defectTypes.length)];
        
        allParticles.push({
          position: [
            moldPos[0] + x,
            moldPos[1] + height + 0.5,
            moldPos[2] + z
          ],
          color: (defectColors as any)[selectedType] || '#FFFFFF',
          size: 0.02 + Math.random() * 0.03,
          type: selectedType
        });
      }
    });
    
    return allParticles;
  }, [defects, positions]);
  
  useFrame((state) => {
    if (groupRef.current) {
      // Легкое вращение дефектов вокруг форм
      groupRef.current.rotation.y += 0.001;
    }
  });
  
  return (
    <group ref={groupRef}>
      {particles.map((particle, i) => (
        <mesh key={i} position={particle.position}>
          <sphereGeometry args={[particle.size]} />
          <meshStandardMaterial 
            color={particle.color}
            emissive={particle.color}
            emissiveIntensity={0.6}
            transparent
            opacity={0.8}
          />
        </mesh>
      ))}
    </group>
  );
};

// Компонент: Печь с температурной визуализацией
const EnhancedFurnace = ({ temperature, thermalField, showField, defects }: FurnaceProps) => {
  const furnaceRef = useRef<THREE.Mesh>(null);
  const doorRef = useRef<THREE.Mesh>(null);
  const [glowIntensity, setGlowIntensity] = useState(0);
  
  useEffect(() => {
    const intensity = Math.min(1, Math.max(0, (temperature - 1400) / 300));
    setGlowIntensity(intensity);
  }, [temperature]);
  
  useFrame((state) => {
    if (furnaceRef.current && glowIntensity > 0.7) {
      const pulse = Math.sin(state.clock.elapsedTime * 2) * 0.03;
      furnaceRef.current.scale.set(1 + pulse, 1, 1 + pulse);
    }
  });
  
  return (
    <group position={[-6, 1.5, 0]}>
      {/* Основной корпус печи */}
      <mesh ref={furnaceRef} position={[0, 0, 0]} castShadow receiveShadow>
        <boxGeometry args={[4, 3, 3]} />
        <meshStandardMaterial 
          color={MIREA_COLORS.secondary}
          emissive={MIREA_COLORS.secondary}
          emissiveIntensity={glowIntensity * 0.7}
          metalness={0.3}
          roughness={0.6}
        />
      </mesh>
      
      {/* Дверца печи */}
      <mesh ref={doorRef} position={[0, 0, 1.6]} castShadow>
        <boxGeometry args={[2.5, 2, 0.15]} />
        <meshStandardMaterial 
          color={MIREA_COLORS.error}
          emissive={MIREA_COLORS.error}
          emissiveIntensity={glowIntensity * 0.9}
          metalness={0.5}
          roughness={0.3}
        />
      </mesh>
      
      {/* Окно печи */}
      <mesh position={[0, 0.5, 1.65]}>
        <boxGeometry args={[1, 0.8, 0.05]} />
        <meshStandardMaterial 
          color="#FFD700"
          emissive="#FFD700"
          emissiveIntensity={glowIntensity}
          transparent
          opacity={0.9}
        />
      </mesh>
      
      {/* Горелки */}
      {[...Array(3)].map((_, i) => (
        <group key={i} position={[-1.5 + i * 1.5, -1.2, 0]}>
          <mesh rotation={[Math.PI / 2, 0, 0]}>
            <cylinderGeometry args={[0.15, 0.15, 0.8]} />
            <meshStandardMaterial color="#333333" metalness={0.8} roughness={0.2} />
          </mesh>
          {/* Flame effect */}
          <mesh position={[0, 0, 0.6]} rotation={[Math.PI, 0, 0]}>
            <coneGeometry args={[0.2, 0.6, 8]} />
            <meshStandardMaterial 
              color="#FF4500"
              emissive="#FF4500"
              emissiveIntensity={glowIntensity * 1.2}
              transparent
              opacity={0.8}
            />
          </mesh>
        </group>
      ))}
      
      {/* Temperature label */}
      <Text
        position={[0, 2.5, 0]}
        color={MIREA_COLORS.accent}
        fontSize={0.4}
        fontWeight={700}
        anchorX="center"
        anchorY="middle"
      >
        {temperature.toFixed(0)}°C
      </Text>
      
      {/* Status indicator */}
      <mesh position={[-1.8, 1.3, 1.6]}>
        <sphereGeometry args={[0.15]} />
        <meshStandardMaterial 
          color={temperature > 1600 ? MIREA_COLORS.error : MIREA_COLORS.success}
          emissive={temperature > 1600 ? MIREA_COLORS.error : MIREA_COLORS.success}
          emissiveIntensity={0.8}
        />
      </mesh>
      
      {/* Дефекты на печи */}
      <FurnaceDefects defects={defects} position={[0, 0, 0]} />
    </group>
  );
};

// Компонент: Визуализация температурного поля
const ThermalFieldVisualization = ({ field, position }: ThermalFieldProps) => {
  const meshRef = useRef<THREE.Group>(null);
  
  useFrame(() => {
    if (meshRef.current) {
      meshRef.current.rotation.y += 0.001;
    }
  });
  
  return (
    <group ref={meshRef} position={position}>
      {field.map((row, i) =>
        row.map((temp, j) => {
          const normalizedTemp = (temp - 1000) / 1000;
          const color = new THREE.Color().setHSL(
            (1 - normalizedTemp) * 0.6, 
            1, 
            0.5
          );
          
          return (
            <mesh 
              key={`${i}-${j}`}
              position={[
                (i - field.length / 2) * 0.3,
                (j - row.length / 2) * 0.3,
                0
              ]}
            >
              <sphereGeometry args={[0.08]} />
              <meshStandardMaterial 
                color={color}
                emissive={color}
                emissiveIntensity={normalizedTemp * 0.5}
                transparent
                opacity={0.6}
              />
            </mesh>
          );
        })
      )}
    </group>
  );
};

// Компонент: Формовочная линия с анимацией
const FormingLine = ({ speed, moldTemp, glass, defects }: FormingLineProps) => {
  const beltRef = useRef<THREE.Mesh>(null);
  const glassRef = useRef<THREE.Mesh>(null);
  const [beltOffset, setBeltOffset] = useState(0);
  
  const moldPositions: Array<[number, number, number]> = [
    [-4, 0.5, 0],
    [-1.5, 0.5, 0],
    [1, 0.5, 0],
    [3.5, 0.5, 0]
  ];
  
  useFrame((state) => {
    if (beltRef.current) {
      const delta = state.clock.getDelta();
      setBeltOffset((prev) => (prev + speed * delta * 0.05) % 10);
      beltRef.current.position.x = -4 + beltOffset;
    }
    
    // Анимация стекла
    if (glassRef.current && glass) {
      const pulse = Math.sin(state.clock.elapsedTime * 3) * 0.02;
      glassRef.current.scale.set(1, 1 + pulse, 1);
    }
  });
  
  return (
    <group position={[4, 0.5, 0]}>
      {/* Основание конвейера */}
      <mesh position={[0, -0.3, 0]} castShadow receiveShadow>
        <boxGeometry args={[10, 0.4, 2]} />
        <meshStandardMaterial 
          color={MIREA_COLORS.surfaceLight}
          metalness={0.7}
          roughness={0.3}
        />
      </mesh>
      
      {/* Движущаяся лента */}
      <mesh 
        ref={beltRef}
        position={[0, 0, 0]} 
        castShadow
      >
        <boxGeometry args={[1.5, 0.05, 1.8]} />
        <meshStandardMaterial 
          color={MIREA_COLORS.primary}
          metalness={0.6}
          roughness={0.4}
        />
      </mesh>
      
      {/* Формы для стекла */}
      {moldPositions.map((pos, i) => {
        const tempColor = moldTemp > 350 ? MIREA_COLORS.error : MIREA_COLORS.warning;
        const intensity = moldTemp > 350 ? 0.4 : 0.1;
        
        return (
          <group key={i} position={pos}>
            <mesh castShadow>
              <cylinderGeometry args={[0.6, 0.6, 0.8]} />
              <meshStandardMaterial 
                color={tempColor}
                emissive={tempColor}
                emissiveIntensity={intensity}
                metalness={0.5}
                roughness={0.5}
              />
            </mesh>
            
            {/* Glass piece */}
            {glass && i === 1 && (
              <mesh 
                ref={glassRef}
                position={[0, 0.5, 0]}
                castShadow
              >
                <boxGeometry args={[1, 0.3, 1]} />
                <meshPhysicalMaterial 
                  color={MIREA_COLORS.accent}
                  transparent
                  opacity={0.7}
                  metalness={0.1}
                  roughness={0.1}
                  transmission={0.9}
                  thickness={0.5}
                />
              </mesh>
            )}
            
            <Text
              position={[0, 1.5, 0]}
              color={MIREA_COLORS.textSecondary}
              fontSize={0.25}
              anchorX="center"
            >
              {moldTemp.toFixed(0)}°C
            </Text>
          </group>
        );
      })}
      
      {/* Дефекты на формах */}
      <MoldDefects defects={defects} positions={moldPositions} />
      
      {/* Speed indicator */}
      <Text
        position={[0, -1, 0]}
        color={MIREA_COLORS.accent}
        fontSize={0.35}
        fontWeight={700}
        anchorX="center"
      >
        {speed.toFixed(0)} м/мин
      </Text>
    </group>
  );
};

// Компонент: Освещение и фон
const SceneLighting = () => {
  return (
    <>
      <ambientLight intensity={0.2} />
      <pointLight 
        position={[10, 10, 10]} 
        intensity={1.5} 
        castShadow 
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      <pointLight 
        position={[-10, 5, -10]} 
        intensity={0.8} 
        color={MIREA_COLORS.primary} 
      />
      <spotLight
        position={[0, 15, 0]}
        angle={0.4}
        penumbra={1}
        intensity={2}
        castShadow
        color={MIREA_COLORS.accent}
      />
      
      {/* Grid floor */}
      <mesh 
        rotation={[-Math.PI / 2, 0, 0]} 
        position={[0, -0.5, 0]} 
        receiveShadow
      >
        <planeGeometry args={[40, 40]} />
        <meshStandardMaterial 
          color={MIREA_COLORS.background}
          metalness={0.3}
          roughness={0.7}
        />
      </mesh>
      
      {/* Grid lines */}
      <gridHelper 
        args={[40, 40, MIREA_COLORS.primary, MIREA_COLORS.surfaceLight]} 
        position={[0, -0.49, 0]}
      />
    </>
  );
};

// Компонент: 3D сцена
const DigitalTwinScene = ({ 
  furnaceTemp, 
  moldTemp, 
  beltSpeed, 
  defects,
  thermalField,
  showThermalField,
  showDefects
}: DigitalTwinSceneProps) => {
  return (
    <>
      <SceneLighting />
      
      <EnhancedFurnace 
        temperature={furnaceTemp} 
        thermalField={thermalField}
        showField={showThermalField}
        defects={defects}
      />
      
      <FormingLine 
        speed={beltSpeed} 
        moldTemp={moldTemp}
        glass={true}
        defects={defects}
      />
      
      {/* Factory building outline */}
      <mesh position={[0, 4, -5]} receiveShadow>
        <boxGeometry args={[25, 8, 1]} />
        <meshStandardMaterial 
          color={MIREA_COLORS.surfaceLight}
          metalness={0.5}
          roughness={0.5}
          transparent
          opacity={0.3}
        />
      </mesh>
      
      {/* Title */}
      <Text
        position={[0, 7, 0]}
        color={MIREA_COLORS.primary}
        fontSize={0.8}
        fontWeight={900}
        anchorX="center"
        anchorY="middle"
      >
        РТУ МИРЭА • Digital Twin
      </Text>
      
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
        maxPolarAngle={Math.PI / 2}
        minDistance={5}
        maxDistance={30}
      />
    </>
  );
};

// Главный компонент
const EnhancedDigitalTwin3D = () => {
  const wsData = useWebSocket('ws://localhost:8000/ws/realtime');
  
  const [realState, setRealState] = useState({
    furnaceTemp: 1520,
    moldTemp: 320,
    beltSpeed: 150,
    defects: { crack: 0.1, bubble: 0.05, chip: 0.03, cloudiness: 0.02, deformation: 0.01, stress: 0.015 }
  });
  
  const [whatIfState, setWhatIfState] = useState({
    furnaceTemp: 1520,
    moldTemp: 320,
    beltSpeed: 150
  });
  
  const [whatIfDefects, setWhatIfDefects] = useState<{ [key: string]: number }>({});
  const [shadowMode, setShadowMode] = useState(false);
  const [whatIfMode, setWhatIfMode] = useState(false);
  const [isSimulating, setIsSimulating] = useState(false);
  const [showThermalField, setShowThermalField] = useState(true);
  const [showDefects, setShowDefects] = useState(true);
  const [showPhysics, setShowPhysics] = useState(false);
  const [thermalField, setThermalField] = useState<number[][] | null>(null);
  
  useEffect(() => {
    if (wsData.parameters) {
      setRealState(prev => ({
        ...prev,
        furnaceTemp: wsData.parameters?.furnace?.temperature || prev.furnaceTemp,
        moldTemp: wsData.parameters?.forming?.mold_temp || prev.moldTemp,
        beltSpeed: wsData.parameters?.forming?.speed || prev.beltSpeed,
        defects: wsData.parameters?.defects || prev.defects
      }));
    }
  }, [wsData.parameters]);
  
  useEffect(() => {
    // Generate synthetic thermal field
    const field = Array.from({ length: 10 }, (_, i) =>
      Array.from({ length: 10 }, (_, j) => {
        const centerDist = Math.sqrt(
          Math.pow(i - 5, 2) + Math.pow(j - 5, 2)
        );
        return 1000 + (realState.furnaceTemp - 1000) * Math.exp(-centerDist / 5);
      })
    );
    setThermalField(field);
  }, [realState.furnaceTemp]);
  
  const simulateWhatIf = async () => {
    setIsSimulating(true);
    
    try {
      const response = await fetch('http://localhost:8000/api/digital-twin/what-if', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          parameter_changes: {
            furnace_temperature: whatIfState.furnaceTemp,
            belt_speed: whatIfState.beltSpeed,
            mold_temperature: whatIfState.moldTemp
          }
        })
      });
      
      const data = await response.json();
      
      if (data.impact_analysis) {
        const impact = data.impact_analysis;
        const newDefects: { [key: string]: number } = {};
        Object.keys(realState.defects).forEach(key => {
          newDefects[key] = Math.max(0, Math.min(1, 
            (realState.defects as any)[key] * (1 + (impact.defect_rate_change_percent || 0) / 100)
          ));
        });
        setWhatIfDefects(newDefects);
      }
    } catch (err) {
      console.error('What-If simulation error:', err);
      // Fallback to simple calculation
      const tempDev = Math.abs(whatIfState.furnaceTemp - 1520) / 30;
      const speedDev = Math.abs(whatIfState.beltSpeed - 150) / 10;
      
      setWhatIfDefects({
        crack: Math.min(tempDev * 0.3, 0.5),
        bubble: Math.min(tempDev * 0.2, 0.4),
        chip: Math.min(speedDev * 0.25, 0.35),
        cloudiness: Math.min(tempDev * 0.15, 0.3),
        deformation: Math.min(speedDev * 0.2, 0.3),
        stress: Math.min(tempDev * 0.25, 0.4)
      });
    }
    
    setTimeout(() => setIsSimulating(false), 800);
  };
  
  const resetWhatIf = () => {
    setWhatIfState({
      furnaceTemp: realState.furnaceTemp,
      moldTemp: realState.moldTemp,
      beltSpeed: realState.beltSpeed
    });
    setWhatIfDefects({});
  };
  
  const handleSliderChange = (key: 'furnaceTemp' | 'moldTemp' | 'beltSpeed', value: number | number[]) => {
    const numValue = Array.isArray(value) ? value[0] : value;
    setWhatIfState(prev => ({ ...prev, [key]: numValue }));
  };
  
  const displayState = whatIfMode ? 
    { ...whatIfState, defects: Object.keys(whatIfDefects).length > 0 ? whatIfDefects : realState.defects } : 
    realState;

  return (
    <Box sx={{ 
      width: '100%', 
      minHeight: '100vh',
      background: `linear-gradient(135deg, ${MIREA_COLORS.background} 0%, ${MIREA_COLORS.backgroundLight} 100%)`,
      p: 2
    }}>
      {/* Header */}
      <Card sx={{ 
        mb: 2, 
        background: `linear-gradient(135deg, ${MIREA_COLORS.surface}CC 0%, ${MIREA_COLORS.surfaceLight}CC 100%)`,
        backdropFilter: 'blur(20px)',
        border: `1px solid ${MIREA_COLORS.primary}40`,
        boxShadow: `0 8px 32px ${MIREA_COLORS.primary}20`
      }}>
        <CardContent>
          <Grid container spacing={2} alignItems="center">
            <Grid item xs={12} md={6}>
              <Typography variant="h5" sx={{ 
                color: MIREA_COLORS.text,
                fontWeight: 900,
                background: `linear-gradient(90deg, ${MIREA_COLORS.primary}, ${MIREA_COLORS.accent})`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                display: 'flex',
                alignItems: 'center',
                gap: 1
              }}>
                🏭 Цифровой Двойник • РТУ МИРЭА
              </Typography>
              <Typography variant="body2" sx={{ color: MIREA_COLORS.textSecondary, mt: 0.5 }}>
                Интеллектуальная система мониторинга производства стекла
              </Typography>
            </Grid>
            
            <Grid item xs={12} md={6}>
              <Stack direction="row" spacing={1} justifyContent="flex-end" flexWrap="wrap">
                <Chip
                  icon={wsData.isConnected ? <CheckCircle /> : <Warning />}
                  label={wsData.isConnected ? 'Подключено' : 'Отключено'}
                  color={wsData.isConnected ? 'success' : 'error'}
                  size="small"
                  sx={{ 
                    fontWeight: 700,
                    background: wsData.isConnected ? MIREA_COLORS.success : MIREA_COLORS.error,
                    color: MIREA_COLORS.text
                  }}
                />
                
                <Box sx={{ display: 'flex', alignItems: 'center' }}>
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={shadowMode} 
                        onChange={(e) => {
                          setShadowMode(e.target.checked);
                          if (e.target.checked) setWhatIfMode(false);
                        }}
                        sx={{
                          '& .MuiSwitch-switchBase.Mui-checked': {
                            color: MIREA_COLORS.primary
                          }
                        }}
                      />
                    }
                    label="Shadow Mode"
                    sx={{ mr: 1 }}
                  />
                  
                  <FormControlLabel
                    control={
                      <Switch 
                        checked={whatIfMode} 
                        onChange={(e) => {
                          setWhatIfMode(e.target.checked);
                          if (e.target.checked) setShadowMode(false);
                        }}
                        sx={{
                          '& .MuiSwitch-switchBase.Mui-checked': {
                            color: MIREA_COLORS.secondary
                          }
                        }}
                      />
                    }
                    label="What-If"
                  />
                </Box>
              </Stack>
            </Grid>
          </Grid>
        </CardContent>
      </Card>
      
      <Grid container spacing={2}>
        {/* 3D Visualization */}
        <Grid item xs={12} md={whatIfMode ? 8 : 12}>
          <Paper
            elevation={8}
            sx={{
              height: '700px',
              position: 'relative',
              overflow: 'hidden',
              background: MIREA_COLORS.background,
              border: `2px solid ${whatIfMode ? MIREA_COLORS.secondary : MIREA_COLORS.primary}60`,
              borderRadius: 2,
              boxShadow: `0 8px 32px ${MIREA_COLORS.primary}30`
            }}
          >
            <Canvas
              shadows
              camera={{ position: [12, 8, 12], fov: 60 }}
            >
              <DigitalTwinScene 
                furnaceTemp={displayState.furnaceTemp} 
                moldTemp={displayState.moldTemp} 
                beltSpeed={displayState.beltSpeed} 
                defects={displayState.defects}
                thermalField={thermalField}
                showThermalField={showThermalField}
                showDefects={showDefects}
              />
            </Canvas>
            
            {/* Overlay Controls */}
            <Stack
              spacing={1}
              sx={{
                position: 'absolute',
                top: 16,
                right: 16,
                background: `${MIREA_COLORS.surface}CC`,
                backdropFilter: 'blur(10px)',
                p: 1.5,
                borderRadius: 2,
                border: `1px solid ${MIREA_COLORS.primary}40`,
                display: 'flex',
                flexDirection: 'column'
              }}
            >
              <Tooltip title="Тепловое поле">
                <IconButton 
                  size="small" 
                  onClick={() => setShowThermalField(!showThermalField)}
                  sx={{ 
                    color: showThermalField ? MIREA_COLORS.accent : MIREA_COLORS.textSecondary,
                    '&:hover': { background: `${MIREA_COLORS.primary}20` }
                  }}
                >
                  <Thermostat />
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Дефекты">
                <IconButton 
                  size="small" 
                  onClick={() => setShowDefects(!showDefects)}
                  sx={{ 
                    color: showDefects ? MIREA_COLORS.error : MIREA_COLORS.textSecondary,
                    '&:hover': { background: `${MIREA_COLORS.primary}20` }
                  }}
                >
                  {showDefects ? <Visibility /> : <VisibilityOff />}
                </IconButton>
              </Tooltip>
              
              <Tooltip title="Физика">
                <IconButton 
                  size="small" 
                  onClick={() => setShowPhysics(!showPhysics)}
                  sx={{ 
                    color: showPhysics ? MIREA_COLORS.primary : MIREA_COLORS.textSecondary,
                    '&:hover': { background: `${MIREA_COLORS.primary}20` }
                  }}
                >
                  <Science />
                </IconButton>
              </Tooltip>
            </Stack>
            
            {/* Metrics Overlay */}
            <Box
              sx={{
                position: 'absolute',
                bottom: 16,
                left: 16,
                background: `${MIREA_COLORS.surface}E6`,
                backdropFilter: 'blur(15px)',
                p: 2,
                borderRadius: 2,
                border: `1px solid ${MIREA_COLORS.primary}60`,
                minWidth: 280
              }}
            >
              <Typography variant="caption" sx={{ color: MIREA_COLORS.textSecondary, fontWeight: 700, display: 'block', mb: 1 }}>
                📊 Параметры производства
              </Typography>
              
              <Stack spacing={1}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body2" sx={{ color: MIREA_COLORS.text, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <LocalFireDepartment fontSize="small" /> Печь:
                  </Typography>
                  <Chip 
                    label={`${displayState.furnaceTemp}°C`}
                    size="small"
                    sx={{ 
                      background: `linear-gradient(90deg, ${MIREA_COLORS.secondary}, ${MIREA_COLORS.error})`,
                      color: MIREA_COLORS.text,
                      fontWeight: 700
                    }}
                  />
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body2" sx={{ color: MIREA_COLORS.text, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Thermostat fontSize="small" /> Форма:
                  </Typography>
                  <Chip 
                    label={`${displayState.moldTemp}°C`}
                    size="small"
                    sx={{ 
                      background: MIREA_COLORS.warning,
                      color: MIREA_COLORS.background,
                      fontWeight: 700
                    }}
                  />
                </Box>
                
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <Typography variant="body2" sx={{ color: MIREA_COLORS.text, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                    <Speed fontSize="small" /> Скорость:
                  </Typography>
                  <Chip 
                    label={`${displayState.beltSpeed} м/мин`}
                    size="small"
                    sx={{ 
                      background: MIREA_COLORS.primary,
                      color: MIREA_COLORS.text,
                      fontWeight: 700
                    }}
                  />
                </Box>
              </Stack>
              
              <Divider sx={{ my: 1.5, borderColor: `${MIREA_COLORS.primary}40` }} />
              
              <Typography variant="caption" sx={{ color: MIREA_COLORS.textSecondary, fontWeight: 700, display: 'block', mb: 1 }}>
                ⚠️ Дефекты:
              </Typography>
              
              <Stack spacing={0.5}>
                {Object.entries(displayState.defects).map(([type, prob]) => (
                  <Box key={type} sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Typography variant="caption" sx={{ color: MIREA_COLORS.textSecondary, minWidth: 90, textTransform: 'capitalize' }}>
                      {type}:
                    </Typography>
                    <LinearProgress 
                      variant="determinate" 
                      value={prob * 100}
                      sx={{ 
                        flex: 1,
                        height: 6,
                        borderRadius: 3,
                        background: `${MIREA_COLORS.glass}40`,
                        '& .MuiLinearProgress-bar': {
                          background: prob > 0.3 ? MIREA_COLORS.error : prob > 0.15 ? MIREA_COLORS.warning : MIREA_COLORS.success,
                          borderRadius: 3
                        }
                      }}
                    />
                    <Typography variant="caption" sx={{ color: MIREA_COLORS.text, fontWeight: 700, minWidth: 40, textAlign: 'right' }}>
                      {(prob * 100).toFixed(1)}%
                    </Typography>
                  </Box>
                ))}
              </Stack>
            </Box>
          </Paper>
        </Grid>
        
        {/* What-If Control Panel */}
        {whatIfMode && (
          <Grid item xs={12} md={4}>
            <Paper
              elevation={8}
              sx={{
                height: '700px',
                overflow: 'auto',
                background: `linear-gradient(180deg, ${MIREA_COLORS.surface}F2 0%, ${MIREA_COLORS.surfaceLight}F2 100%)`,
                backdropFilter: 'blur(20px)',
                p: 3,
                border: `2px solid ${MIREA_COLORS.secondary}80`,
                borderRadius: 2,
                boxShadow: `0 8px 32px ${MIREA_COLORS.secondary}30`
              }}
            >
              <Typography variant="h6" sx={{ 
                color: MIREA_COLORS.text, 
                mb: 2, 
                fontWeight: 900,
                background: `linear-gradient(90deg, ${MIREA_COLORS.secondary}, ${MIREA_COLORS.warning})`,
                WebkitBackgroundClip: 'text',
                WebkitTextFillColor: 'transparent',
                display: 'flex',
                alignItems: 'center',
                gap: 1
              }}>
                🧪 What-If Анализ
              </Typography>
              
              <Divider sx={{ mb: 3, borderColor: `${MIREA_COLORS.secondary}40` }} />
              
              {/* Temperature Control */}
              <Box sx={{ mb: 4 }}>
                <Typography variant="body2" sx={{ color: MIREA_COLORS.text, mb: 1, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <LocalFireDepartment fontSize="small" /> Температура печи: {whatIfState.furnaceTemp}°C
                </Typography>
                <Slider
                  value={whatIfState.furnaceTemp}
                  onChange={(_, value) => handleSliderChange('furnaceTemp', value)}
                  min={1400}
                  max={1700}
                  step={5}
                  marks={[
                    { value: 1400, label: '1400' },
                    { value: 1520, label: '1520' },
                    { value: 1700, label: '1700' }
                  ]}
                  sx={{ 
                    color: MIREA_COLORS.secondary,
                    '& .MuiSlider-markLabel': {
                      color: MIREA_COLORS.textSecondary,
                      fontSize: '0.7rem'
                    }
                  }}
                />
              </Box>
              
              {/* Mold Temperature */}
              <Box sx={{ mb: 4 }}>
                <Typography variant="body2" sx={{ color: MIREA_COLORS.text, mb: 1, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Thermostat fontSize="small" /> Температура формы: {whatIfState.moldTemp}°C
                </Typography>
                <Slider
                  value={whatIfState.moldTemp}
                  onChange={(_, value) => handleSliderChange('moldTemp', value)}
                  min={280}
                  max={380}
                  step={5}
                  marks={[
                    { value: 280, label: '280' },
                    { value: 320, label: '320' },
                    { value: 380, label: '380' }
                  ]}
                  sx={{ 
                    color: MIREA_COLORS.warning,
                    '& .MuiSlider-markLabel': {
                      color: MIREA_COLORS.textSecondary,
                      fontSize: '0.7rem'
                    }
                  }}
                />
              </Box>
              
              {/* Belt Speed */}
              <Box sx={{ mb: 4 }}>
                <Typography variant="body2" sx={{ color: MIREA_COLORS.text, mb: 1, fontWeight: 700, display: 'flex', alignItems: 'center', gap: 0.5 }}>
                  <Speed fontSize="small" /> Скорость ленты: {whatIfState.beltSpeed} м/мин
                </Typography>
                <Slider
                  value={whatIfState.beltSpeed}
                  onChange={(_, value) => handleSliderChange('beltSpeed', value)}
                  min={100}
                  max={200}
                  step={5}
                  marks={[
                    { value: 100, label: '100' },
                    { value: 150, label: '150' },
                    { value: 200, label: '200' }
                  ]}
                  sx={{ 
                    color: MIREA_COLORS.primary,
                    '& .MuiSlider-markLabel': {
                      color: MIREA_COLORS.textSecondary,
                      fontSize: '0.7rem'
                    }
                  }}
                />
              </Box>
              
              {/* Action Buttons */}
              <Stack direction="row" spacing={1} sx={{ mb: 3 }}>
                <Button
                  variant="contained"
                  fullWidth
                  onClick={simulateWhatIf}
                  disabled={isSimulating}
                  startIcon={isSimulating ? null : <TrendingUp />}
                  sx={{
                    background: `linear-gradient(45deg, ${MIREA_COLORS.secondary} 30%, ${MIREA_COLORS.warning} 90%)`,
                    fontWeight: 700,
                    boxShadow: `0 4px 16px ${MIREA_COLORS.secondary}40`,
                    '&:hover': {
                      background: `linear-gradient(45deg, ${MIREA_COLORS.secondaryDark} 30%, ${MIREA_COLORS.warning} 90%)`,
                      boxShadow: `0 6px 20px ${MIREA_COLORS.secondary}60`
                    },
                    '&.Mui-disabled': {
                      background: MIREA_COLORS.surfaceLight,
                      color: MIREA_COLORS.textSecondary
                    }
                  }}
                >
                  {isSimulating ? 'Анализ...' : 'Симулировать'}
                </Button>
                
                <Button
                  variant="outlined"
                  onClick={resetWhatIf}
                  startIcon={<Refresh />}
                  sx={{ 
                    borderColor: MIREA_COLORS.text, 
                    color: MIREA_COLORS.text,
                    fontWeight: 700,
                    '&:hover': {
                      borderColor: MIREA_COLORS.primary,
                      background: `${MIREA_COLORS.primary}20`
                    }
                  }}
                >
                  Сброс
                </Button>
              </Stack>
              
              <Divider sx={{ my: 2, borderColor: `${MIREA_COLORS.secondary}40` }} />
              
              {/* Results */}
              {Object.keys(whatIfDefects).length > 0 && (
                <Box>
                  <Typography variant="body2" sx={{ color: MIREA_COLORS.text, mb: 2, fontWeight: 700 }}>
                    📊 Прогноз дефектов:
                  </Typography>
                  
                  <Stack spacing={1.5}>
                    {Object.entries(whatIfDefects).map(([type, prob]) => (
                      <Card 
                        key={type}
                        sx={{ 
                          bgcolor: `${prob > 0.3 ? MIREA_COLORS.error : prob > 0.15 ? MIREA_COLORS.warning : MIREA_COLORS.success}15`, 
                          border: '1px solid', 
                          borderColor: prob > 0.3 ? MIREA_COLORS.error : prob > 0.15 ? MIREA_COLORS.warning : MIREA_COLORS.success,
                          boxShadow: `0 4px 12px ${prob > 0.3 ? MIREA_COLORS.error : prob > 0.15 ? MIREA_COLORS.warning : MIREA_COLORS.success}20`
                        }}
                      >
                        <CardContent sx={{ py: 1.5 }}>
                          <Typography variant="caption" sx={{ color: MIREA_COLORS.textSecondary, textTransform: 'capitalize' }}>
                            {type}
                          </Typography>
                          <Typography variant="h6" sx={{ 
                            color: prob > 0.3 ? MIREA_COLORS.error : prob > 0.15 ? MIREA_COLORS.warning : MIREA_COLORS.success, 
                            fontWeight: 900 
                          }}>
                            {(prob * 100).toFixed(1)}%
                          </Typography>
                        </CardContent>
                      </Card>
                    ))}
                  </Stack>
                  
                  {/* Recommendation */}
                  <Alert 
                    severity={Object.values(whatIfDefects).some(v => v > 0.3) ? "error" : "success"}
                    sx={{ 
                      mt: 2,
                      background: `${Object.values(whatIfDefects).some(v => v > 0.3) ? MIREA_COLORS.error : MIREA_COLORS.success}15`,
                      border: '1px solid',
                      borderColor: Object.values(whatIfDefects).some(v => v > 0.3) ? MIREA_COLORS.error : MIREA_COLORS.success,
                      '& .MuiAlert-icon': {
                        color: Object.values(whatIfDefects).some(v => v > 0.3) ? MIREA_COLORS.error : MIREA_COLORS.success
                      }
                    }}
                  >
                    <Typography variant="caption" sx={{ fontWeight: 700 }}>
                      {Object.values(whatIfDefects).some(v => v > 0.3)
                        ? '⚠️ Параметры выходят за оптимальные значения. Высокий риск дефектов!'
                        : '✅ Параметры в норме. Качество продукции стабильно.'}
                    </Typography>
                  </Alert>
                </Box>
              )}
            </Paper>
          </Grid>
        )}
      </Grid>
      
      {/* Physics Panel */}
      <Collapse in={showPhysics}>
        <Card sx={{ 
          mt: 2,
          background: `${MIREA_COLORS.surface}F2`,
          backdropFilter: 'blur(20px)',
          border: `1px solid ${MIREA_COLORS.primary}40`,
          boxShadow: `0 8px 32px ${MIREA_COLORS.primary}20`
        }}>
          <CardContent>
            <Typography variant="h6" sx={{ 
              color: MIREA_COLORS.text, 
              mb: 2, 
              fontWeight: 900,
              display: 'flex',
              alignItems: 'center',
              gap: 1
            }}>
              <Science sx={{ color: MIREA_COLORS.primary }} />
              Физические параметры
            </Typography>
            
            <Grid container spacing={2}>
              <Grid item xs={12} md={3}>
                <Paper sx={{ p: 2, background: `${MIREA_COLORS.surfaceLight}CC`, border: `1px solid ${MIREA_COLORS.accent}40`, height: '100%' }}>
                  <Typography variant="caption" sx={{ color: MIREA_COLORS.textSecondary }}>
                    Вязкость
                  </Typography>
                  <Typography variant="h5" sx={{ color: MIREA_COLORS.accent, fontWeight: 900 }}>
                    {(1000 + Math.random() * 500).toFixed(0)} Pa·s
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Paper sx={{ p: 2, background: `${MIREA_COLORS.surfaceLight}CC`, border: `1px solid ${MIREA_COLORS.warning}40`, height: '100%' }}>
                  <Typography variant="caption" sx={{ color: MIREA_COLORS.textSecondary }}>
                    Теплопроводность
                  </Typography>
                  <Typography variant="h5" sx={{ color: MIREA_COLORS.warning, fontWeight: 900 }}>
                    1.4 W/(m·K)
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Paper sx={{ p: 2, background: `${MIREA_COLORS.surfaceLight}CC`, border: `1px solid ${MIREA_COLORS.error}40`, height: '100%' }}>
                  <Typography variant="caption" sx={{ color: MIREA_COLORS.textSecondary }}>
                    Тепловое расширение
                  </Typography>
                  <Typography variant="h5" sx={{ color: MIREA_COLORS.error, fontWeight: 900 }}>
                    9×10⁻⁶ K⁻¹
                  </Typography>
                </Paper>
              </Grid>
              
              <Grid item xs={12} md={3}>
                <Paper sx={{ p: 2, background: `${MIREA_COLORS.surfaceLight}CC`, border: `1px solid ${MIREA_COLORS.success}40`, height: '100%' }}>
                  <Typography variant="caption" sx={{ color: MIREA_COLORS.textSecondary }}>
                    Плотность
                  </Typography>
                  <Typography variant="h5" sx={{ color: MIREA_COLORS.success, fontWeight: 900 }}>
                    2500 kg/m³
                  </Typography>
                </Paper>
              </Grid>
            </Grid>
          </CardContent>
        </Card>
      </Collapse>
    </Box>
  );
};

export default EnhancedDigitalTwin3D;