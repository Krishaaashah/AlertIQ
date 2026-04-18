import { useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, Activity, Brain, BookOpen, TrendingUp, ArrowUp, ArrowDown, AlertTriangle } from 'lucide-react';
import NeuralScene from './components/NeuralScene';
import LiveFeed from './components/LiveFeed';
import { PolicyComparisonChart, WorkloadChart } from './components/Charts';
import ResearchPage from './components/ResearchPage';
import { PIPELINE_METRICS } from './data/simulationData';
import './index.css';

const PAGES = ['dashboard', 'research'];

function StatCard({ label, value, subtitle, icon: Icon, color, delay = 0 }) {
  return (
    <motion.div
      className="card"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, type: 'spring', stiffness: 100 }}
    >
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
        <div>
          <div className="stat-label">{label}</div>
          <div className="stat-value" style={{ color }}>{value}</div>
          {subtitle && <div className="stat-change positive"><ArrowUp size={12} /> {subtitle}</div>}
        </div>
        <div style={{
          width: 42, height: 42, borderRadius: 'var(--radius-sm)',
          background: `${color}15`, display: 'flex', alignItems: 'center', justifyContent: 'center'
        }}>
          <Icon size={20} color={color} />
        </div>
      </div>
    </motion.div>
  );
}

export default function App() {
  const [page, setPage] = useState('dashboard');
  const [liveStats, setLiveStats] = useState({
    totalProcessed: 0,
    totalSuppressed: 0,
    moneySaved: 0,
    fraudBlocked: 0,
  });

  const handleNewTransaction = useCallback((tx) => {
    setLiveStats(prev => ({
      totalProcessed: prev.totalProcessed + 1,
      totalSuppressed: prev.totalSuppressed + (tx.action === 'suppress' ? 1 : 0),
      moneySaved: prev.moneySaved + (tx.action === 'suppress' ? 15 : 0), // $15 per analyst review saved
      fraudBlocked: prev.fraudBlocked + (tx.isFraud && tx.action === 'escalate' ? tx.amount : 0),
    }));
  }, []);

  const p = PIPELINE_METRICS;

  return (
    <div className="app">
      {/* ─── Navbar ─── */}
      <nav className="navbar">
        <div className="navbar-logo">
          <div className="logo-icon">IQ</div>
          <span className="logo-text">AlertIQ</span>
        </div>

        <div className="navbar-links">
          <button
            className={`nav-link ${page === 'dashboard' ? 'active' : ''}`}
            onClick={() => setPage('dashboard')}
          >
            <Activity size={14} style={{ marginRight: 6 }} />
            Command Center
          </button>
          <button
            className={`nav-link ${page === 'research' ? 'active' : ''}`}
            onClick={() => setPage('research')}
          >
            <BookOpen size={14} style={{ marginRight: 6 }} />
            Research & Metrics
          </button>
        </div>

        <div className="nav-status">
          <div className="pulse-dot"></div>
          RL Agent Active
        </div>
      </nav>

      {/* ─── Main Content ─── */}
      <main className="main-content">
        <AnimatePresence mode="wait">
          {page === 'dashboard' ? (
            <motion.div key="dashboard" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              {/* Hero: 3D Scene + Live Feed */}
              <div className="grid-hero">
                <div>
                  <div className="section-header">
                    <h2>RL Decision Engine</h2>
                    <p>Deep Q-Network processing live transaction stream</p>
                  </div>
                  <NeuralScene />
                </div>
                <div>
                  <div className="section-header">
                    <h2>Transaction Stream</h2>
                    <p>Real-time alert triage by the RL agent</p>
                  </div>
                  <LiveFeed onNewTransaction={handleNewTransaction} />
                </div>
              </div>

              {/* Stat Cards */}
              <div className="grid-4" style={{ marginBottom: '1.5rem' }}>
                <StatCard
                  label="Fraud Catch Rate"
                  value={`${(p.phase4.dqn.fraudCatchRate * 100).toFixed(1)}%`}
                  subtitle="Above 95% target"
                  icon={Shield}
                  color="var(--accent-cyan)"
                  delay={0.1}
                />
                <StatCard
                  label="FP Reduction"
                  value={`${(p.phase4.dqn.fpReduction * 100).toFixed(1)}%`}
                  subtitle="vs 45% industry avg"
                  icon={TrendingUp}
                  color="var(--accent-green)"
                  delay={0.15}
                />
                <StatCard
                  label="Analyst Time Saved"
                  value={`$${liveStats.moneySaved.toLocaleString()}`}
                  subtitle={`${liveStats.totalSuppressed} alerts auto-triaged`}
                  icon={Brain}
                  color="var(--accent-purple)"
                  delay={0.2}
                />
                <StatCard
                  label="Fraud Blocked"
                  value={`$${liveStats.fraudBlocked.toLocaleString()}`}
                  subtitle="Potential losses prevented"
                  icon={AlertTriangle}
                  color="var(--accent-red)"
                  delay={0.25}
                />
              </div>

              {/* Charts */}
              <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
                <PolicyComparisonChart />
                <WorkloadChart />
              </div>

              {/* Pipeline Overview */}
              <motion.div className="card" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
                <div className="card-header">
                  <span className="card-title">System Architecture — 5-Phase Pipeline</span>
                  <span className="card-badge badge-success">All Online</span>
                </div>
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '8px' }}>
                  {[
                    { phase: 'P1', name: 'Data Ingestion', stat: `${(p.phase1.totalTransactions / 1e6).toFixed(1)}M txns`, color: 'var(--accent-blue)' },
                    { phase: 'P2', name: 'Alert Generation', stat: `${(p.phase2.alertRate * 100).toFixed(0)}% flagged`, color: 'var(--accent-amber)' },
                    { phase: 'P3', name: 'ML Suppression', stat: `AUC ${p.phase3.rocAuc.toFixed(3)}`, color: 'var(--accent-purple)' },
                    { phase: 'P4', name: 'RL Agent', stat: `${(p.phase4.dqn.workloadSavings * 100).toFixed(0)}% saved`, color: 'var(--accent-cyan)' },
                    { phase: 'P5', name: 'Drift Monitor', stat: `${p.phase5.featuresMonitored} features`, color: 'var(--accent-green)' },
                  ].map((p, i) => (
                    <div key={i} style={{
                      background: `${p.color}08`,
                      border: `1px solid ${p.color}25`,
                      borderRadius: 'var(--radius-sm)',
                      padding: '14px',
                      textAlign: 'center',
                      position: 'relative',
                    }}>
                      <div style={{ fontFamily: 'var(--font-mono)', fontSize: '0.7rem', color: p.color, marginBottom: 4 }}>{p.phase}</div>
                      <div style={{ fontSize: '0.8rem', fontWeight: 600, marginBottom: 4 }}>{p.name}</div>
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>{p.stat}</div>
                      {i < 4 && (
                        <div style={{
                          position: 'absolute', right: -14, top: '50%', transform: 'translateY(-50%)',
                          color: 'var(--text-muted)', fontSize: '1rem', zIndex: 2
                        }}>→</div>
                      )}
                    </div>
                  ))}
                </div>
              </motion.div>
            </motion.div>
          ) : (
            <motion.div key="research" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
              <ResearchPage />
            </motion.div>
          )}
        </AnimatePresence>
      </main>
    </div>
  );
}
