import { useRef, useMemo } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { Sphere, MeshDistortMaterial, Float, Stars } from '@react-three/drei';
import * as THREE from 'three';

/* ─── Animated Neural Core ─── */
function NeuralCore() {
  const meshRef = useRef();
  useFrame(({ clock }) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = clock.elapsedTime * 0.15;
      meshRef.current.rotation.z = Math.sin(clock.elapsedTime * 0.3) * 0.1;
    }
  });

  return (
    <Float speed={1.5} rotationIntensity={0.3} floatIntensity={0.8}>
      <Sphere ref={meshRef} args={[1.8, 64, 64]} position={[0, 0, 0]}>
        <MeshDistortMaterial
          color="#00e5ff"
          attach="material"
          distort={0.35}
          speed={2}
          roughness={0.1}
          metalness={0.9}
          transparent
          opacity={0.7}
        />
      </Sphere>
    </Float>
  );
}

/* ─── Data Particle Ring ─── */
function DataRing({ count = 200, radius = 3.5 }) {
  const points = useRef();

  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const angle = (i / count) * Math.PI * 2;
      const r = radius + (Math.random() - 0.5) * 0.6;
      pos[i * 3] = Math.cos(angle) * r;
      pos[i * 3 + 1] = (Math.random() - 0.5) * 0.8;
      pos[i * 3 + 2] = Math.sin(angle) * r;
    }
    return pos;
  }, [count, radius]);

  const colors = useMemo(() => {
    const col = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const isFraud = Math.random() < 0.04;
      if (isFraud) {
        col[i * 3] = 0.94; col[i * 3 + 1] = 0.27; col[i * 3 + 2] = 0.27;
      } else {
        col[i * 3] = 0; col[i * 3 + 1] = 0.9; col[i * 3 + 2] = 1;
      }
    }
    return col;
  }, [count]);

  useFrame(({ clock }) => {
    if (points.current) {
      points.current.rotation.y = clock.elapsedTime * 0.08;
    }
  });

  return (
    <points ref={points}>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" count={count} array={positions} itemSize={3} />
        <bufferAttribute attach="attributes-color" count={count} array={colors} itemSize={3} />
      </bufferGeometry>
      <pointsMaterial size={0.06} vertexColors transparent opacity={0.9} sizeAttenuation />
    </points>
  );
}

/* ─── Connection Lines ─── */
function ConnectionLines() {
  const linesRef = useRef();
  const lineCount = 30;

  const positions = useMemo(() => {
    const pos = [];
    for (let i = 0; i < lineCount; i++) {
      const theta1 = Math.random() * Math.PI * 2;
      const phi1 = Math.random() * Math.PI;
      const r1 = 1.8;
      const x1 = r1 * Math.sin(phi1) * Math.cos(theta1);
      const y1 = r1 * Math.cos(phi1);
      const z1 = r1 * Math.sin(phi1) * Math.sin(theta1);

      const theta2 = Math.random() * Math.PI * 2;
      const r2 = 3.5 + (Math.random() - 0.5) * 0.4;
      const x2 = Math.cos(theta2) * r2;
      const y2 = (Math.random() - 0.5) * 0.6;
      const z2 = Math.sin(theta2) * r2;

      pos.push(new THREE.Vector3(x1, y1, z1));
      pos.push(new THREE.Vector3(x2, y2, z2));
    }
    return pos;
  }, []);

  useFrame(({ clock }) => {
    if (linesRef.current) {
      linesRef.current.rotation.y = clock.elapsedTime * 0.05;
    }
  });

  return (
    <group ref={linesRef}>
      {Array.from({ length: lineCount }).map((_, i) => (
        <line key={i}>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={2}
              array={new Float32Array([
                positions[i * 2].x, positions[i * 2].y, positions[i * 2].z,
                positions[i * 2 + 1].x, positions[i * 2 + 1].y, positions[i * 2 + 1].z,
              ])}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#8b5cf6" transparent opacity={0.2} />
        </line>
      ))}
    </group>
  );
}

/* ─── Main Scene Export ─── */
export default function NeuralScene() {
  return (
    <div className="canvas-container">
      <Canvas camera={{ position: [0, 2, 7], fov: 50 }}>
        <ambientLight intensity={0.3} />
        <pointLight position={[10, 10, 10]} intensity={0.8} color="#00e5ff" />
        <pointLight position={[-10, -5, 5]} intensity={0.4} color="#8b5cf6" />

        <Stars radius={50} depth={30} count={2000} factor={3} saturation={0} fade speed={1} />

        <NeuralCore />
        <DataRing count={300} radius={3.5} />
        <ConnectionLines />
      </Canvas>

      <div className="canvas-overlay">
        <h3>RL Neural Decision Engine</h3>
        <p>Deep Q-Network processing incoming transaction stream</p>
      </div>
    </div>
  );
}
