import { useState, useEffect, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Shield, ShieldAlert, ArrowDownCircle, TrendingUp } from 'lucide-react';
import { generateTransaction } from '../data/simulationData';

export default function LiveFeed({ onNewTransaction }) {
  const [transactions, setTransactions] = useState([]);
  const [stats, setStats] = useState({ total: 0, suppressed: 0, escalated: 0, fraudCaught: 0 });
  const feedRef = useRef(null);

  useEffect(() => {
    const interval = setInterval(() => {
      const tx = generateTransaction();
      setTransactions(prev => [tx, ...prev].slice(0, 50));
      setStats(prev => ({
        total: prev.total + 1,
        suppressed: prev.suppressed + (tx.action === 'suppress' ? 1 : 0),
        escalated: prev.escalated + (tx.action === 'escalate' ? 1 : 0),
        fraudCaught: prev.fraudCaught + (tx.isFraud && tx.action === 'escalate' ? 1 : 0),
      }));
      if (onNewTransaction) onNewTransaction(tx);
    }, 800);

    return () => clearInterval(interval);
  }, [onNewTransaction]);

  useEffect(() => {
    if (feedRef.current) {
      feedRef.current.scrollTop = 0;
    }
  }, [transactions]);

  const getIcon = (tx) => {
    if (tx.isFraud && tx.action === 'escalate') return <ShieldAlert size={16} />;
    if (tx.action === 'suppress') return <ArrowDownCircle size={16} />;
    return <Shield size={16} />;
  };

  const getIconClass = (tx) => {
    if (tx.isFraud) return 'feed-icon fraud-detected';
    if (tx.action === 'suppress') return 'feed-icon suppress';
    return 'feed-icon escalate';
  };

  const getItemClass = (tx) => {
    if (tx.isFraud) return 'feed-item fraud';
    if (tx.action === 'suppress') return 'feed-item suppressed';
    return 'feed-item escalated';
  };

  return (
    <div className="card" style={{ height: '100%' }}>
      <div className="card-header">
        <span className="card-title">Live Transaction Feed</span>
        <span className="card-badge badge-info">{stats.total} processed</span>
      </div>

      {/* Mini Stats */}
      <div style={{
        display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: '8px', marginBottom: '12px'
      }}>
        <div style={{
          background: 'rgba(16,185,129,0.08)', borderRadius: '8px', padding: '8px', textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-green)' }}>
            {stats.suppressed}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>SUPPRESSED</div>
        </div>
        <div style={{
          background: 'rgba(245,158,11,0.08)', borderRadius: '8px', padding: '8px', textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-amber)' }}>
            {stats.escalated}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>ESCALATED</div>
        </div>
        <div style={{
          background: 'rgba(239,68,68,0.08)', borderRadius: '8px', padding: '8px', textAlign: 'center'
        }}>
          <div style={{ fontSize: '1.1rem', fontWeight: 700, color: 'var(--accent-red)' }}>
            {stats.fraudCaught}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-muted)' }}>FRAUD CAUGHT</div>
        </div>
      </div>

      {/* Feed */}
      <div className="feed-container" ref={feedRef}>
        <AnimatePresence initial={false}>
          {transactions.map((tx) => (
            <motion.div
              key={tx.id}
              className={getItemClass(tx)}
              initial={{ opacity: 0, x: -20, height: 0 }}
              animate={{ opacity: 1, x: 0, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className={getIconClass(tx)}>{getIcon(tx)}</div>
              <div className="feed-details">
                <div className="feed-id">{tx.id}</div>
                <div className="feed-meta">
                  {tx.type} &middot; {tx.sender} → {tx.receiver}
                </div>
              </div>
              <div className="feed-amount">${tx.amount.toLocaleString()}</div>
              <div
                className="feed-prob"
                style={{
                  background: tx.probFraud > 0.7
                    ? 'rgba(239,68,68,0.15)'
                    : tx.probFraud > 0.4
                      ? 'rgba(245,158,11,0.15)'
                      : 'rgba(16,185,129,0.15)',
                  color: tx.probFraud > 0.7
                    ? 'var(--accent-red)'
                    : tx.probFraud > 0.4
                      ? 'var(--accent-amber)'
                      : 'var(--accent-green)',
                }}
              >
                {(tx.probFraud * 100).toFixed(1)}%
              </div>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </div>
  );
}
