import { motion } from 'framer-motion';
import { PIPELINE_METRICS, BENCHMARKS } from '../data/simulationData';
import { BenchmarkRadar, TrainingCurveChart, DriftChart } from './Charts';
import { Brain, Target, Zap, Layers, AlertTriangle, TrendingUp, Database, Shield } from 'lucide-react';

const fadeIn = { hidden: { opacity: 0, y: 20 }, visible: (i) => ({ opacity: 1, y: 0, transition: { delay: i * 0.1 } }) };

export default function ResearchPage() {
  const p = PIPELINE_METRICS;

  return (
    <div>
      {/* ─── MDP Formulation ─── */}
      <motion.div className="page-section" initial="hidden" animate="visible">
        <div className="section-header">
          <h2>MDP Formulation & Environment Design</h2>
          <p>The Markov Decision Process that drives the RL Alert Governance Agent</p>
        </div>

        <div className="info-panel">
          <h4>Problem Statement</h4>
          <p>
            Legacy financial security tools generate up to <code>99.6%</code> false positive alerts,
            costing banks millions in wasted analyst hours. AlertIQ uses Deep Reinforcement Learning
            to dynamically optimize the triage process, safely auto-suppressing benign alerts while
            ensuring <code>97.48%</code> of real fraud is escalated to human review.
          </p>
        </div>

        <div className="mdp-grid">
          <motion.div className="mdp-card" custom={0} variants={fadeIn}>
            <div className="mdp-icon"><Layers size={24} color="var(--accent-cyan)" /></div>
            <h4>State Space S</h4>
            <p>5D continuous vector: <code>[prob_fraud, rule_count, amount, fraud_rate, workload]</code></p>
          </motion.div>
          <motion.div className="mdp-card" custom={1} variants={fadeIn}>
            <div className="mdp-icon"><Target size={24} color="var(--accent-purple)" /></div>
            <h4>Action Space A</h4>
            <p>Binary discrete: <code>0 = Suppress</code> (auto-dismiss), <code>1 = Escalate</code> (human review)</p>
          </motion.div>
          <motion.div className="mdp-card" custom={2} variants={fadeIn}>
            <div className="mdp-icon"><Zap size={24} color="var(--accent-amber)" /></div>
            <h4>Discount Factor</h4>
            <p><code>γ = 0.99</code> — Agent optimizes for long-horizon economic cost, not just immediate reward</p>
          </motion.div>
          <motion.div className="mdp-card" custom={3} variants={fadeIn}>
            <div className="mdp-icon"><Brain size={24} color="var(--accent-green)" /></div>
            <h4>Model Architecture</h4>
            <p>Deep Q-Network (MLP) with 50K-capacity Replay Buffer and ε-greedy exploration</p>
          </motion.div>
        </div>
      </motion.div>

      {/* ─── Reward Structure ─── */}
      <motion.div className="page-section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.3 }}>
        <div className="section-header">
          <h2>Asymmetric Reward Function</h2>
          <p>Cost-based reward engineering to handle extreme class imbalance (0.39% fraud rate)</p>
        </div>

        <div className="info-panel" style={{ borderColor: 'rgba(239, 68, 68, 0.2)', background: 'rgba(239, 68, 68, 0.03)' }}>
          <h4 style={{ color: 'var(--accent-red)' }}>Key Innovation: Balanced Experience Replay</h4>
          <p>
            Standard RL agents fail catastrophically on imbalanced data because they see fraud in only <code>0.39%</code> of samples.
            AlertIQ forces <code>50/50 fraud:benign</code> ratio during training episodes, allowing the DQN to develop
            statistically significant Q-value boundaries for both classes while evaluating on the real-world imbalanced test set.
          </p>
        </div>

        <div className="reward-matrix">
          <div className="reward-cell positive">
            <div className="reward-value" style={{ color: 'var(--accent-green)' }}>+1.0</div>
            <div className="reward-label">Suppress Benign Alert</div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 2 }}>Analyst time saved</div>
          </div>
          <div className="reward-cell positive">
            <div className="reward-value" style={{ color: 'var(--accent-green)' }}>+5.0</div>
            <div className="reward-label">Escalate Real Fraud</div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 2 }}>Fraud successfully caught</div>
          </div>
          <div className="reward-cell negative">
            <div className="reward-value" style={{ color: 'var(--accent-amber)' }}>−0.5</div>
            <div className="reward-label">Escalate Benign Alert</div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 2 }}>Minor: analyst reviews harmless txn</div>
          </div>
          <div className="reward-cell catastrophic">
            <div className="reward-value" style={{ color: 'var(--accent-red)' }}>−500</div>
            <div className="reward-label">Suppress Real Fraud</div>
            <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 2 }}>CATASTROPHIC: fraud leak</div>
          </div>
        </div>
      </motion.div>

      {/* ─── Pipeline Metrics ─── */}
      <motion.div className="page-section" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.4 }}>
        <div className="section-header">
          <h2>End-to-End Pipeline Metrics</h2>
          <p>Quantitative evaluation across all 5 system phases</p>
        </div>

        <div className="card" style={{ marginBottom: '1rem' }}>
          <table className="metric-table">
            <thead>
              <tr>
                <th>Phase</th>
                <th>Metric</th>
                <th>Value</th>
                <th>Industry Benchmark</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td><Database size={14} style={{ marginRight: 6 }} />Phase 1</td>
                <td>Dataset Size</td>
                <td className="mono">6,362,620</td>
                <td className="mono">—</td>
                <td><span className="card-badge badge-info">PaySim</span></td>
              </tr>
              <tr>
                <td><AlertTriangle size={14} style={{ marginRight: 6 }} />Phase 2</td>
                <td>Alert FP Rate</td>
                <td className="mono">{(p.phase2.fpRate * 100).toFixed(2)}%</td>
                <td className="mono">95-99%</td>
                <td><span className="card-badge badge-warning">Realistic</span></td>
              </tr>
              <tr>
                <td rowSpan={2}><Shield size={14} style={{ marginRight: 6 }} />Phase 3</td>
                <td>ROC-AUC</td>
                <td className="mono" style={{ color: 'var(--accent-green)' }}>{p.phase3.rocAuc.toFixed(4)}</td>
                <td className="mono">≥ 0.85</td>
                <td><span className="card-badge badge-success">Exceeds</span></td>
              </tr>
              <tr>
                <td>Brier Score</td>
                <td className="mono" style={{ color: 'var(--accent-green)' }}>{p.phase3.brierScore.toFixed(4)}</td>
                <td className="mono">≤ 0.10</td>
                <td><span className="card-badge badge-success">Exceeds</span></td>
              </tr>
              <tr>
                <td rowSpan={3}><Brain size={14} style={{ marginRight: 6 }} />Phase 4</td>
                <td>Fraud Catch Rate</td>
                <td className="mono" style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{(p.phase4.dqn.fraudCatchRate * 100).toFixed(2)}%</td>
                <td className="mono">≥ 95%</td>
                <td><span className="card-badge badge-success">Exceeds</span></td>
              </tr>
              <tr>
                <td>FP Reduction</td>
                <td className="mono" style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{(p.phase4.dqn.fpReduction * 100).toFixed(2)}%</td>
                <td className="mono">30-60%</td>
                <td><span className="card-badge badge-success">SOTA</span></td>
              </tr>
              <tr>
                <td>Workload Savings</td>
                <td className="mono" style={{ color: 'var(--accent-cyan)', fontWeight: 700 }}>{(p.phase4.dqn.workloadSavings * 100).toFixed(2)}%</td>
                <td className="mono">30-60%</td>
                <td><span className="card-badge badge-success">SOTA</span></td>
              </tr>
              <tr>
                <td><TrendingUp size={14} style={{ marginRight: 6 }} />Phase 5</td>
                <td>Drift Detection</td>
                <td className="mono">{p.phase5.featuresMonitored} features</td>
                <td className="mono">PSI &gt; 0.1</td>
                <td><span className="card-badge badge-success">Active</span></td>
              </tr>
            </tbody>
          </table>
        </div>
      </motion.div>

      {/* ─── Charts ─── */}
      <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
        <TrainingCurveChart />
        <BenchmarkRadar />
      </div>

      <div className="grid-2" style={{ marginBottom: '1.5rem' }}>
        <DriftChart />
        <motion.div className="card" initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.6 }}>
          <div className="card-header">
            <span className="card-title">Benchmark Comparison Table</span>
          </div>
          <table className="metric-table">
            <thead>
              <tr><th>Metric</th><th>AlertIQ</th><th>Industry Target</th><th>Δ</th></tr>
            </thead>
            <tbody>
              {Object.values(BENCHMARKS).map((b, i) => {
                const oursDisplay = b.unit === '%' ? (b.ours * 100).toFixed(2) + '%' : b.ours.toFixed(4);
                const indDisplay = b.unit === '%' ? (b.industry * 100).toFixed(0) + '%' : b.industry.toFixed(2);
                const diff = b.lowerBetter
                  ? ((b.industry - b.ours) / b.industry * 100).toFixed(0)
                  : ((b.ours - b.industry) / b.industry * 100).toFixed(0);
                return (
                  <tr key={i}>
                    <td>{b.label}</td>
                    <td className="mono" style={{ color: 'var(--accent-cyan)', fontWeight: 600 }}>{oursDisplay}</td>
                    <td className="mono">{indDisplay}</td>
                    <td style={{ color: 'var(--accent-green)', fontWeight: 600 }}>+{diff}%</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </motion.div>
      </div>
    </div>
  );
}
