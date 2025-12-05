// @ts-nocheck
import React, { useRef, useState, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Box as DreiBox, Sphere, Cylinder, Plane } from '@react-three/drei';
import * as THREE from 'three';
import { Box, Chip, Paper, Typography } from '@mui/material';
import { systemApi } from '../services/api';

// Цветовая палитра РТУ МИРЭА
const MIREA_COLORS = {
  primary: '#0066CC',
  secondary: '#FF6B35',
  success: '#00C853',
  warning: '#FFA726',
  error: '#EF5350',
  info: '#29B6F6',
  background: '#0A1929',
  backgroundLight: '#1A2332',
  text: '#FFFFFF'
};

// Здание завода
const FactoryBuilding = () => {
  return (
    <group>
      <DreiBox position={[0, 2.5, 0]} args={[20, 5, 10]} castShadow receiveShadow>
        <meshStandardMaterial color={MIREA_COLORS.primary} />
      </DreiBox>
      
      <DreiBox position={[0, 5.5, 0]} args={[22, 1, 12]} castShadow>
        <meshStandardMaterial color={MIREA_COLORS.primary} />
      </DreiBox>
      
      <Cylinder position={[8, 4, -3]} args={[0.5, 0.5, 4]} castShadow>
        <meshStandardMaterial color="#555555" />
      </Cylinder>
      
      {Array.from({ length: 8 }).map((_, i) => (
        <DreiBox 
          key={i} 
          position={[-8 + i * 2.5, 3, 5.1]} 
          args={[1.5, 2, 0.2]} 
          castShadow
        >
          <meshStandardMaterial 
            color={MIREA_COLORS.info} 
            emissive={MIREA_COLORS.info} 
            emissiveIntensity={0.2} 
          />
        </DreiBox>
      ))}
    </group>
  );
};

// Печь
const Furnace = ({ temperature }: { temperature: number }) => {
  const furnaceRef = useRef<THREE.Mesh>(null);
  const [glowIntensity, setGlowIntensity] = useState(0);
  
  useEffect(() => {
    const intensity = Math.min(1, Math.max(0, (temperature - 1400) / 300));
    setGlowIntensity(intensity);
  }, [temperature]);
  
  useFrame(() => {
    if (furnaceRef.current) {
      if (glowIntensity > 0.7) {
        furnaceRef.current.scale.x = 1 + Math.sin(Date.now() * 0.005) * 0.05;
        furnaceRef.current.scale.z = 1 + Math.sin(Date.now() * 0.005) * 0.05;
      }
    }
  });
  
  return (
    <group position={[-5, 1, 0]}>
      <DreiBox ref={furnaceRef} position={[0, 1, 0]} args={[3, 2, 3]} castShadow receiveShadow>
        <meshStandardMaterial 
          color={MIREA_COLORS.secondary}
          emissive={MIREA_COLORS.secondary}
          emissiveIntensity={glowIntensity * 0.5}
        />
      </DreiBox>
      
      <DreiBox position={[0, 1, 1.6]} args={[2, 1, 0.2]} castShadow>
        <meshStandardMaterial 
          color={MIREA_COLORS.error}
          emissive={MIREA_COLORS.error}
          emissiveIntensity={glowIntensity * 0.8}
        />
      </DreiBox>
      
      <Text
        position={[0, 3, 0]}
        color="white"
        fontSize={0.5}
        anchorX="center"
        anchorY="middle"
      >
        {temperature.toFixed(0)}°C
      </Text>
    </group>
  );
};

// Формовочная линия
const FormingLine = ({ speed, moldTemp }: { speed: number; moldTemp: number }) => {
  const beltRef = useRef<THREE.Mesh>(null);
  const [beltOffset, setBeltOffset] = useState(0);
  
  useFrame(() => {
    if (beltRef.current) {
      setBeltOffset((prev) => (prev + speed * 0.01) % 10);
      beltRef.current.position.z = beltOffset;
    }
  });
  
  return (
    <group position={[5, 0.5, 0]}>
      <DreiBox position={[0, 0, 0]} args={[8, 0.2, 2]} castShadow receiveShadow>
        <meshStandardMaterial color={MIREA_COLORS.info} />
      </DreiBox>
      
      <DreiBox 
        ref={beltRef}
        position={[0, 0.11, 0]} 
        args={[7.8, 0.02, 1.8]} 
        castShadow
      >
        <meshStandardMaterial color={MIREA_COLORS.info} />
      </DreiBox>
      
      {Array.from({ length: 3 }).map((_, i) => (
        <group key={i} position={[-3 + i * 3, 1, 0]}>
          <Cylinder args={[0.5, 0.5, 1]} castShadow>
            <meshStandardMaterial 
              color={moldTemp > 350 ? MIREA_COLORS.error : MIREA_COLORS.info}
              emissive={moldTemp > 350 ? MIREA_COLORS.error : MIREA_COLORS.info}
              emissiveIntensity={moldTemp > 350 ? 0.3 : 0.1}
            />
          </Cylinder>
          <Text
            position={[0, 1.8, 0]}
            color="white"
            fontSize={0.3}
            anchorX="center"
            anchorY="middle"
          >
            {moldTemp.toFixed(0)}°C
          </Text>
        </group>
      ))}
    </group>
  );
};

// Частицы дефектов
const DefectParticles = ({ defects }: { defects: Record<string, number> }) => {
  const particlesRef = useRef<THREE.Group>(null);
  const totalDefects = Object.values(defects).reduce((sum, val) => sum + val, 0);
  
  useFrame(() => {
    if (particlesRef.current) {
      particlesRef.current.rotation.y += 0.005;
    }
  });
  
  return (
    <group ref={particlesRef} position={[0, 3, 0]}>
      {Array.from({ length: Math.min(50, totalDefects * 10) }).map((_, i) => {
        const angle = (i / 50) * Math.PI * 2;
        const radius = 2 + Math.random() * 3;
        const x = Math.cos(angle) * radius;
        const z = Math.sin(angle) * radius;
        const y = Math.random() * 2;
        
        let color = "#ffffff";
        const defectTypes = Object.keys(defects);
        if (defectTypes.length > 0) {
          const defectType = defectTypes[Math.floor(Math.random() * defectTypes.length)];
          if (defectType === "crack") color = MIREA_COLORS.error;
          else if (defectType === "bubble") color = MIREA_COLORS.warning;
          else if (defectType === "chip") color = MIREA_COLORS.success;
        }
        
        return (
          <Sphere 
            key={i} 
            position={[x, y, z]} 
            args={[0.05]} 
            castShadow
          >
            <meshStandardMaterial 
              color={color} 
              emissive={color} 
              emissiveIntensity={0.2}
            />
          </Sphere>
        );
      })}
    </group>
  );
};

// Главная 3D сцена
const DigitalTwinScene = ({ 
  furnaceTemp, 
  moldTemp, 
  beltSpeed, 
  defects 
}: { 
  furnaceTemp: number; 
  moldTemp: number; 
  beltSpeed: number; 
  defects: Record<string, number>;
}) => {
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[10, 10, 10]} intensity={1} castShadow />
      <pointLight position={[-10, 5, -10]} intensity={0.5} color={MIREA_COLORS.info} />
      <spotLight
        position={[0, 10, 0]}
        angle={0.3}
        penumbra={1}
        intensity={1}
        castShadow
        shadow-mapSize-width={2048}
        shadow-mapSize-height={2048}
      />
      
      <Plane 
        rotation={[-Math.PI / 2, 0, 0]} 
        position={[0, -0.5, 0]} 
        args={[30, 30]} 
        receiveShadow
      >
        <meshStandardMaterial color={MIREA_COLORS.backgroundLight} />
      </Plane>
      
      <FactoryBuilding />
      
      <Furnace temperature={furnaceTemp} />
      <FormingLine speed={beltSpeed} moldTemp={moldTemp} />
      
      <DefectParticles defects={defects} />
      
      <Text
        position={[0, 6, 0]}
        color="white"
        fontSize={0.8}
        anchorX="center"
        anchorY="middle"
      >
        Цифровой двойник • Линия производства стекла
      </Text>
      
      <OrbitControls 
        enablePan={true}
        enableZoom={true}
        enableRotate={true}
      />
    </>
  );
};

// Главный компонент
const DigitalTwin3D = () => {
  // State for real-time data
  const [furnaceTemp, setFurnaceTemp] = useState<number>(1500);
  const [moldTemp, setMoldTemp] = useState<number>(320);
  const [beltSpeed, setBeltSpeed] = useState<number>(150);
  const [defects, setDefects] = useState<Record<string, number>>({ crack: 0.2, bubble: 0.15, chip: 0.05 });
  const [isConnected, setIsConnected] = useState(false);
  
  // Fetch real data from backend
  useEffect(() => {
    const fetchData = async () => {
      try {
        const data = await systemApi.getDigitalTwinState();
        if (data.data) {
          setFurnaceTemp(data.data.furnace_temperature || 1500);
          setMoldTemp(data.data.melt_level || 320);
          setBeltSpeed(data.data.belt_speed || 150);
          // Convert predictions to defects format
          setDefects(data.data.defects || {
            crack: 0.1,
            bubble: 0.05,
            chip: 0.02
          });
          setIsConnected(true);
        }
      } catch (error) {
        console.error('Error fetching digital twin state:', error);
        // Keep using default values
        setIsConnected(false);
      }
    };
    
    fetchData();
    
    // Poll for updates every 5 seconds
    const interval = setInterval(fetchData, 5000);
    return () => clearInterval(interval);
  }, []);

  // @ts-ignore
  return (
    // @ts-ignore
    <Box sx={{ width: '100%', height: '600px', position: 'relative' }}>
      <Canvas
        shadows
        camera={{ position: [10, 5, 10], fov: 50 }}
        style={{ background: `linear-gradient(to bottom, ${MIREA_COLORS.background}, ${MIREA_COLORS.backgroundLight})` } as React.CSSProperties}
      >
        <DigitalTwinScene 
          furnaceTemp={furnaceTemp} 
          moldTemp={moldTemp} 
          beltSpeed={beltSpeed} 
          defects={defects} 
        />
      </Canvas>
      
      <Chip
        label={isConnected ? '🔗 Подключено' : '⚠️ Отключено'}
        sx={{
          position: 'absolute',
          top: 16,
          right: 16,
          bgcolor: isConnected ? MIREA_COLORS.success : MIREA_COLORS.error,
          color: MIREA_COLORS.text,
          fontWeight: 700,
          fontSize: 12,
          px: 2,
          py: 1
        }}
      />
      
      <Paper
        elevation={4}
        sx={{
          position: 'absolute',
          bottom: 16,
          left: 16,
          background: 'rgba(0, 0, 0, 0.8)',
          backdropFilter: 'blur(10px)',
          p: 2,
          borderRadius: 2,
          border: `1px solid ${MIREA_COLORS.primary}60`
        }}
      >
        <Typography variant="caption" sx={{ color: MIREA_COLORS.text, display: 'block', mb: 1, fontWeight: 700 }}>
          Легенда дефектов:
        </Typography>
        <Box sx={{ display: 'flex', flexDirection: 'column', gap: 0.5 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: MIREA_COLORS.error }} />
            <Typography variant="caption" sx={{ color: MIREA_COLORS.text }}>Трещины</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: MIREA_COLORS.warning }} />
            <Typography variant="caption" sx={{ color: MIREA_COLORS.text }}>Пузыри</Typography>
          </Box>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Box sx={{ width: 12, height: 12, borderRadius: '50%', bgcolor: MIREA_COLORS.success }} />
            <Typography variant="caption" sx={{ color: MIREA_COLORS.text }}>Сколы</Typography>
          </Box>
        </Box>
      </Paper>
    </Box>
  );
};

export default DigitalTwin3D;