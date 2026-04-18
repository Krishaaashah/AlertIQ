import { motion } from 'framer-motion';
import {
  RadarChart, Radar, PolarGrid, PolarAngleAxis, PolarRadiusAxis,
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
  AreaChart, Area, Legend, LineChart, Line, Cell
} from 'recharts';
import { PIPELINE_METRICS, TRAINING_CURVE_DATA, BENCHMARKS } from '../data/simulationData';

const CHART_COLORS = {
  cyan: '#00e5ff',
  purple: '#8b5cf6',
  green: '#10b981',
  red: '#ef4444',
  amber: '#f59e0b',
  blue: '#3b82f6',
};

/* ─── Custom Tooltip ─── */
const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: 'rgba(17, 24, 39, 0.95)',
      border: '1px solid rgba(0, 229, 255, 0.2)',
      borderRadius: '8px',
      padding: '10px 14px',
      fontSize: '0.8rem',
    }}>
      <p style={{ color: '#94a3b8', marginBottom: '4px' }}>{label}</p>
      {payload.map((entry, i) => (
        <p key={i} style={{ color: entry.color, fontWeight: 600 }}>
          {entry.name}: {typeof entry.value === 'number' ? entry.value.toFixed(4) : entry.value}
        </p>
      ))}
    </div>
  );
};

/* ─── Policy Comparison Chart ─── */
export function PolicyComparisonChart() {
  const data = PIPELINE_METRICS.phase4.policyComparison.map(d => ({
    ...d,
    fraudCatch: +(d.fraudCatch * 100).toFixed(2),
    fpReduction: +(d.fpReduction * 100).toFixed(2),
    workload: +(d.workload * 100).toFixed(2),
  }));

  return (
    <motion.div className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
      <div className="card-header">
        <span className="card-title">Policy Comparison — RL vs Static Thresholds</span>
        <span className="card-badge badge-info">6 policies</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <BarChart data={data} barGap={2}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
          <XAxis dataKey="policy" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0, 100]} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
          <Bar dataKey="fraudCatch" name="Fraud Catch %" fill={CHART_COLORS.red} radius={[4, 4, 0, 0]} />
          <Bar dataKey="fpReduction" name="FP Reduction %" fill={CHART_COLORS.cyan} radius={[4, 4, 0, 0]} />
          <Bar dataKey="workload" name="Workload Saved %" fill={CHART_COLORS.green} radius={[4, 4, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

/* ─── Training Progress Chart ─── */
export function TrainingCurveChart() {
  return (
    <motion.div className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
      <div className="card-header">
        <span className="card-title">DQN Training Convergence</span>
        <span className="card-badge badge-success">100 episodes</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <LineChart data={TRAINING_CURVE_DATA}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
          <XAxis dataKey="episode" tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'Episode', position: 'insideBottom', offset: -5, style: { fill: '#64748b', fontSize: 11 } }} />
          <YAxis yAxisId="left" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis yAxisId="right" orientation="right" tick={{ fill: '#94a3b8', fontSize: 11 }} domain={[0, 1]} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
          <Line yAxisId="right" type="monotone" dataKey="fraudCatch" name="Fraud Catch Rate" stroke={CHART_COLORS.red} strokeWidth={2} dot={{ r: 4 }} />
          <Line yAxisId="right" type="monotone" dataKey="fpReduction" name="FP Reduction Rate" stroke={CHART_COLORS.green} strokeWidth={2} dot={{ r: 4 }} />
        </LineChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

/* ─── Benchmark Radar ─── */
export function BenchmarkRadar() {
  const data = Object.values(BENCHMARKS).map(b => ({
    metric: b.label,
    ours: b.lowerBetter ? (1 - b.ours) * 100 : b.ours * 100,
    industry: b.lowerBetter ? (1 - b.industry) * 100 : b.industry * 100,
  }));

  return (
    <motion.div className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }}>
      <div className="card-header">
        <span className="card-title">AlertIQ vs Industry Benchmarks</span>
        <span className="card-badge badge-success">Exceeds All</span>
      </div>
      <ResponsiveContainer width="100%" height={300}>
        <RadarChart data={data}>
          <PolarGrid stroke="rgba(148,163,184,0.1)" />
          <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <PolarRadiusAxis tick={{ fill: '#64748b', fontSize: 9 }} domain={[0, 100]} />
          <Radar name="AlertIQ" dataKey="ours" stroke={CHART_COLORS.cyan} fill={CHART_COLORS.cyan} fillOpacity={0.2} strokeWidth={2} />
          <Radar name="Industry Avg" dataKey="industry" stroke={CHART_COLORS.amber} fill={CHART_COLORS.amber} fillOpacity={0.1} strokeWidth={2} strokeDasharray="5 5" />
          <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
          <Tooltip content={<CustomTooltip />} />
        </RadarChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

/* ─── Drift Detection Chart ─── */
export function DriftChart() {
  const data = PIPELINE_METRICS.phase5.driftResults.map(d => ({
    intensity: `${d.intensity.toFixed(1)}`,
    psi: d.avgPsi,
    drifted: d.drifted,
    severity: d.severity,
  }));

  return (
    <motion.div className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.5 }}>
      <div className="card-header">
        <span className="card-title">Drift Detection Sensitivity (PSI)</span>
        <span className="card-badge badge-warning">{PIPELINE_METRICS.phase5.featuresMonitored} features</span>
      </div>
      <ResponsiveContainer width="100%" height={280}>
        <AreaChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
          <XAxis dataKey="intensity" tick={{ fill: '#94a3b8', fontSize: 11 }} label={{ value: 'Drift Intensity', position: 'insideBottom', offset: -5, style: { fill: '#64748b', fontSize: 11 } }} />
          <YAxis tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <Tooltip content={<CustomTooltip />} />
          <Legend wrapperStyle={{ fontSize: '0.75rem' }} />
          <Area type="monotone" dataKey="psi" name="Avg PSI" fill={CHART_COLORS.purple} fillOpacity={0.2} stroke={CHART_COLORS.purple} strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </motion.div>
  );
}

/* ─── Workload Savings Donut/Bar ─── */
export function WorkloadChart() {
  const data = [
    { label: 'Auto-Suppressed', value: 97.51, color: CHART_COLORS.green },
    { label: 'Human Review', value: 2.49, color: CHART_COLORS.amber },
  ];

  return (
    <motion.div className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }}>
      <div className="card-header">
        <span className="card-title">Analyst Workload Distribution</span>
      </div>
      <ResponsiveContainer width="100%" height={200}>
        <BarChart data={data} layout="vertical" barSize={28}>
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(148,163,184,0.08)" />
          <XAxis type="number" domain={[0, 100]} tick={{ fill: '#94a3b8', fontSize: 11 }} />
          <YAxis dataKey="label" type="category" tick={{ fill: '#94a3b8', fontSize: 11 }} width={120} />
          <Tooltip content={<CustomTooltip />} />
          <Bar dataKey="value" name="Percentage" radius={[0, 6, 6, 0]}>
            {data.map((entry, index) => (
              <Cell key={`cell-${index}`} fill={entry.color} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </motion.div>
  );
}
