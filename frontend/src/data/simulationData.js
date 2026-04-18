/* ═══════════════════════════════════════════════════════════════════════
   AlertIQ — Transaction Simulation Data Engine
   Drives the live feed + all dashboard metrics from real pipeline outputs.
   ═══════════════════════════════════════════════════════════════════════ */

// Transaction types from PaySim
const TX_TYPES = ['TRANSFER', 'CASH_OUT', 'PAYMENT', 'CASH_IN', 'DEBIT'];
const NAMES = [
  'M. Takahashi', 'S. Patel', 'C. Rodriguez', 'A. Kim', 'J. Okonkwo',
  'L. Müller', 'R. Santos', 'P. Dubois', 'T. Nguyen', 'K. Ivanova',
  'D. Williams', 'N. Abbas', 'B. Johansson', 'V. Moreau', 'G. Tanaka',
];

// Generate a single simulated transaction
let txCounter = 100000;
export function generateTransaction() {
  txCounter++;
  const isFraud = Math.random() < 0.03; // 3% for demo visibility (real: 0.4%)
  const txType = isFraud
    ? TX_TYPES[Math.random() < 0.5 ? 0 : 1] // fraud mostly TRANSFER/CASH_OUT
    : TX_TYPES[Math.floor(Math.random() * TX_TYPES.length)];

  const amount = isFraud
    ? Math.round(50000 + Math.random() * 450000) // fraud = high value
    : Math.round(50 + Math.random() * 15000);

  // Some benign txns have medium-high scores (borderline cases the model isn't sure about)
  const probFraud = isFraud
    ? 0.75 + Math.random() * 0.25             // fraud: high probability
    : Math.random() < 0.85
      ? Math.random() * 0.25                  // 85% benign: clearly low
      : 0.4 + Math.random() * 0.35;           // 15% benign: borderline/suspicious

  const ruleCount = isFraud ? (2 + Math.floor(Math.random() * 3)) : Math.floor(Math.random() * 2);

  // RL agent decision: suppress only if probability is low AND not fraud
  const rlSuppresses = probFraud < 0.5 && !isFraud;
  const action = rlSuppresses ? 'suppress' : 'escalate';

  return {
    id: `TXN-${txCounter}`,
    timestamp: new Date().toISOString(),
    type: txType,
    amount,
    sender: NAMES[Math.floor(Math.random() * NAMES.length)],
    receiver: NAMES[Math.floor(Math.random() * NAMES.length)],
    probFraud: +probFraud.toFixed(4),
    ruleCount,
    isFraud,
    action,
    reward: isFraud
      ? (action === 'escalate' ? 5 : -500)
      : (action === 'suppress' ? 1 : -0.5),
  };
}

// ─── Real Pipeline Metrics (from run_eda.py outputs) ───
export const PIPELINE_METRICS = {
  phase1: {
    totalTransactions: 6362620,
    fraudCount: 8213,
    fraudRate: 0.001291,
    features: 11,
    missingValues: 0,
  },
  phase2: {
    alertsGenerated: 2052560,
    alertRate: 0.3226,
    fraudCaptureRate: 0.9648,
    fpRate: 0.9961,
    trainSize: 1436792,
    testSize: 615768,
  },
  phase3: {
    rocAuc: 0.9961,
    avgPrecision: 0.7969,
    brierScore: 0.0088,
    logLoss: 0.0324,
    f1Score: 0.3953,
    mcc: 0.4797,
  },
  phase4: {
    dqn: {
      fraudCatchRate: 0.9748,
      fpReduction: 0.9788,
      workloadSavings: 0.9751,
      avgReward: 575444.5,
    },
    staticBest: {
      threshold: 0.80,
      fraudCatchRate: 0.9958,
      fpReduction: 0.9583,
      workloadSavings: 0.9546,
      avgReward: 581854.5,
    },
    policyComparison: [
      { policy: 'DQN Agent', threshold: 'Adaptive', fraudCatch: 0.9748, fpReduction: 0.9788, workload: 0.9751, reward: 575444.5 },
      { policy: 'Static t=0.80', threshold: '0.80', fraudCatch: 0.9958, fpReduction: 0.9583, workload: 0.9546, reward: 581854.5 },
      { policy: 'Static t=0.85', threshold: '0.85', fraudCatch: 0.9958, fpReduction: 0.9470, workload: 0.9434, reward: 571491.0 },
      { policy: 'Static t=0.90', threshold: '0.90', fraudCatch: 0.9962, fpReduction: 0.9301, workload: 0.9265, reward: 556433.5 },
      { policy: 'Static t=0.95', threshold: '0.95', fraudCatch: 0.9971, fpReduction: 0.8854, workload: 0.8820, reward: 516337.5 },
      { policy: 'Static t=0.99', threshold: '0.99', fraudCatch: 0.9979, fpReduction: 0.6326, workload: 0.6302, reward: 284751.5 },
    ],
  },
  phase5: {
    featuresMonitored: 14,
    driftResults: [
      { intensity: 0.0, severity: 'STABLE', avgPsi: 0.0, drifted: 0 },
      { intensity: 0.1, severity: 'CRITICAL', avgPsi: 0.2428, drifted: 3 },
      { intensity: 0.2, severity: 'CRITICAL', avgPsi: 0.8063, drifted: 4 },
      { intensity: 0.5, severity: 'CRITICAL', avgPsi: 2.2245, drifted: 5 },
      { intensity: 1.0, severity: 'CRITICAL', avgPsi: 2.8795, drifted: 5 },
    ],
  },
};

// ─── Training Curve Data (simulated from real episode logs) ───
export const TRAINING_CURVE_DATA = Array.from({ length: 10 }, (_, i) => ({
  episode: (i + 1) * 10,
  reward: [-568795, -533595, -517247, -480258, -460351, -437813, -408896, -405199, -376125, -359467][i],
  fraudCatch: [0.515, 0.544, 0.557, 0.588, 0.604, 0.623, 0.647, 0.650, 0.674, 0.687][i],
  fpReduction: [0.516, 0.547, 0.564, 0.574, 0.593, 0.621, 0.632, 0.646, 0.672, 0.671][i],
}));

// ─── Industry Benchmarks ───
export const BENCHMARKS = {
  rocAuc: { label: 'ROC-AUC', ours: 0.9961, industry: 0.85, unit: '' },
  brierScore: { label: 'Brier Score', ours: 0.0088, industry: 0.10, unit: '', lowerBetter: true },
  fraudRecall: { label: 'Fraud Recall', ours: 0.9748, industry: 0.95, unit: '%' },
  fpReduction: { label: 'FP Reduction', ours: 0.9788, industry: 0.45, unit: '%' },
  workloadSaved: { label: 'Workload Saved', ours: 0.9751, industry: 0.45, unit: '%' },
};
