import React, { useEffect, useState } from 'react';
import { getDashboardStats } from '../services/dashboardService';
import { getCurrentMintedTokens } from '../services/tokenService';
import { getAuditHistory, initiateNewAudit } from '../services/auditService';
import { useNavigate } from 'react-router-dom';

const Dashboard = () => {
  const navigate = useNavigate();
  const [stats, setStats] = useState({
    co2CapturedKg: 0,
    carbonCreditsGenerated: 0,
    activeSensors: 0,
    systemStatus: 'Loading...'
  });
  const [tokens, setTokens] = useState({ totalMinted: 0 });
  const [audits, setAudits] = useState([]);
  const [loading, setLoading] = useState(true);
  
  // Audit state
  const [auditStatus, setAuditStatus] = useState('idle'); // idle, processing, success, error
  const [auditResult, setAuditResult] = useState(null);

  const fetchDashboardData = async () => {
    try {
      const [dashStats, tokenStats, auditData] = await Promise.all([
        getDashboardStats(),
        getCurrentMintedTokens(),
        getAuditHistory()
      ]);
      setStats(dashStats);
      setTokens(tokenStats);
      setAudits(auditData || []);
    } catch (error) {
      console.error("Failed to load dashboard data:", error);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const handleStartAudit = async () => {
    setAuditStatus('processing');
    setAuditResult(null);
    try {
      const response = await initiateNewAudit({ timestamp: Date.now() });
      setAuditStatus('success');
      // Capture the audit result
      if (response && response.record) {
        setAuditResult(response.record);
      }
      // Refresh the dashboard data
      await fetchDashboardData();
    } catch (e) {
      console.error(e);
      setAuditStatus('error');
    }
  };

  if (loading) {
    return <div className="container" style={{ paddingTop: '50px' }}><h2>Loading Dashboard...</h2></div>;
  }

  const latestAudit = audits.length > 0 ? audits[0] : null;
  const recentAudits = audits.slice(0, 3);

  return (
    <div className="container" style={{ paddingTop: '40px' }}>
      
      {/* Header Section */}
      <div style={{ marginBottom: '40px', borderBottom: '1px solid var(--border-color)', paddingBottom: '20px' }}>
        <h1 style={{ margin: '0 0 10px 0', fontSize: '2rem', color: '#fff' }}>CUBEX Dashboard</h1>
        <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '1.1rem' }}>
          Carbon Capture Audit Platform. Monitor captured CO₂, generated credits and run audits.
        </p>
      </div>

      {/* KPI Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '20px', marginBottom: '40px' }}>
        <div className="glass-card" style={{ padding: '20px' }}>
          <h3 style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>CO₂ Captured</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#fff' }}>
            {stats.co2CapturedKg.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })} <span style={{ fontSize: '1rem', color: 'var(--text-secondary)', fontWeight: 'normal' }}>kg</span>
          </div>
        </div>
        <div className="glass-card" style={{ padding: '20px' }}>
          <h3 style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>Carbon Credits</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: 'var(--primary-color)' }}>
            {stats.carbonCreditsGenerated.toLocaleString()} <span style={{ fontSize: '1rem', color: 'var(--text-secondary)', fontWeight: 'normal' }}>Generated</span>
          </div>
        </div>
        <div className="glass-card" style={{ padding: '20px' }}>
          <h3 style={{ color: 'var(--text-secondary)', fontSize: '0.85rem', marginBottom: '8px', textTransform: 'uppercase', letterSpacing: '1px' }}>Audits</h3>
          <div style={{ fontSize: '2rem', fontWeight: 'bold', color: '#fff' }}>
            {audits.length} <span style={{ fontSize: '1rem', color: 'var(--text-secondary)', fontWeight: 'normal' }}>Completed</span>
          </div>
        </div>
      </div>

      {/* Middle Row: System / Latest Audit / New Audit */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(350px, 1fr))', gap: '20px', marginBottom: '40px' }}>
        
        {/* Audit Overview */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
          <h2 style={{ fontSize: '1.2rem', margin: 0, borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>Audit Overview</h2>
          
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '20px' }}>
            <div>
              <h3 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '10px' }}>Current Status</h3>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '8px' }}>
                <div style={{ width: '10px', height: '10px', borderRadius: '50%', background: stats.systemStatus === 'Normal' && auditStatus === 'idle' ? '#4caf50' : (auditStatus === 'processing' ? '#ff9800' : '#f44336') }}></div>
                <strong style={{ fontSize: '1rem' }}>
                  {auditStatus === 'processing' ? 'AUDIT IN PROGRESS' : (stats.systemStatus === 'Normal' ? 'SYSTEM READY' : 'SYSTEM ERROR')}
                </strong>
              </div>
              <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                {auditStatus === 'processing' ? 'Running anomaly detection...' : 'Ready for new audit'}
              </p>
            </div>
            
            <div>
              <h3 style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', textTransform: 'uppercase', marginBottom: '10px' }}>Latest Audit</h3>
              {latestAudit ? (
                <div>
                  <div style={{ fontWeight: 'bold', marginBottom: '4px' }}>{latestAudit.dataSetNumber}</div>
                  <div style={{ 
                    color: latestAudit.status === 'Normal' ? '#4caf50' : '#f44336', 
                    fontSize: '0.9rem',
                    fontWeight: 'bold',
                    marginBottom: '4px'
                  }}>
                    {latestAudit.status.toUpperCase()}
                  </div>
                  <div style={{ fontSize: '0.9rem', color: 'var(--text-secondary)' }}>
                    {latestAudit.creditsGenerated} credits
                  </div>
                </div>
              ) : (
                <p style={{ margin: 0, fontSize: '0.9rem', color: 'var(--text-secondary)' }}>No audits yet</p>
              )}
            </div>
          </div>
        </div>

        {/* New Audit Action */}
        <div className="glass-card" style={{ display: 'flex', flexDirection: 'column' }}>
          <h2 style={{ fontSize: '1.2rem', margin: 0, borderBottom: '1px solid var(--border-color)', paddingBottom: '10px', marginBottom: '20px' }}>New Audit</h2>
          
          {auditStatus === 'idle' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <p style={{ color: 'var(--text-secondary)', marginBottom: '20px', fontSize: '0.95rem', lineHeight: '1.5' }}>
                Run a fresh carbon-capture dataset through the anomaly detection and carbon-credit pipeline.
              </p>
              <button className="btn-primary" onClick={handleStartAudit} style={{ width: '100%', padding: '16px', fontSize: '1.1rem' }}>
                START NEW AUDIT
              </button>
            </div>
          )}

          {auditStatus === 'processing' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '15px', justifyContent: 'center' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#fff' }}>
                <span>Generating Data</span> <span style={{ color: 'var(--primary-color)' }}>✓</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#fff' }}>
                <span>Training Model</span> <span style={{ color: 'var(--primary-color)' }}>✓</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: '#fff' }}>
                <span>Analyzing Telemetry</span> <span style={{ color: '#ff9800' }}>◉</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)' }}>
                <span>Calculating Credits</span> <span>○</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)' }}>
                <span>Completed</span> <span>○</span>
              </div>
            </div>
          )}

          {auditStatus === 'success' && auditResult && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
              <h3 style={{ color: '#4caf50', margin: '0 0 15px 0' }}>AUDIT COMPLETE</h3>
              <div style={{ background: 'rgba(0,0,0,0.3)', padding: '15px', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                <div style={{ fontWeight: 'bold', marginBottom: '8px', fontSize: '1.1rem' }}>{auditResult.dataSetNumber}</div>
                <div style={{ color: auditResult.status === 'Normal' ? '#4caf50' : '#f44336', fontWeight: 'bold', marginBottom: '8px' }}>
                  {auditResult.status.toUpperCase()}
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', marginBottom: '4px' }}>
                  50 samples analyzed
                </div>
                <div style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>
                  {auditResult.creditsGenerated} credits generated
                </div>
              </div>
              <button 
                onClick={() => setAuditStatus('idle')} 
                style={{ background: 'transparent', color: 'var(--text-secondary)', border: '1px solid var(--text-secondary)', padding: '8px', borderRadius: '6px', marginTop: '15px', cursor: 'pointer' }}
              >
                Reset
              </button>
            </div>
          )}

          {auditStatus === 'error' && (
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', justifyContent: 'center', alignItems: 'center' }}>
              <h3 style={{ color: '#f44336', marginBottom: '10px' }}>Audit Failed</h3>
              <p style={{ color: 'var(--text-secondary)', textAlign: 'center', marginBottom: '20px' }}>An error occurred during the audit pipeline.</p>
              <button className="btn-primary" onClick={() => setAuditStatus('idle')} style={{ width: '100%' }}>Try Again</button>
            </div>
          )}
        </div>
      </div>

      {/* Bottom Row: Recent Audits */}
      <div className="glass-card">
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '20px', borderBottom: '1px solid var(--border-color)', paddingBottom: '10px' }}>
          <h2 style={{ fontSize: '1.2rem', margin: 0 }}>Recent Audits</h2>
          <button onClick={() => navigate('/history')} style={{ background: 'transparent', color: 'var(--primary-color)', border: 'none', fontSize: '0.95rem', cursor: 'pointer' }}>
            View full history →
          </button>
        </div>
        
        {recentAudits.length > 0 ? (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
            {recentAudits.map((audit) => (
              <div key={audit.id} style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', padding: '15px', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', alignItems: 'center' }}>
                <div style={{ fontWeight: '600' }}>{audit.dataSetNumber}</div>
                <div style={{ color: audit.status === 'Normal' ? '#4caf50' : '#f44336', fontWeight: 'bold' }}>{audit.status.toUpperCase()}</div>
                <div style={{ color: 'var(--text-secondary)', textAlign: 'right' }}>{audit.creditsGenerated} credits</div>
              </div>
            ))}
          </div>
        ) : (
          <p style={{ color: 'var(--text-secondary)' }}>No recent audits found.</p>
        )}
      </div>

    </div>
  );
};

export default Dashboard;
