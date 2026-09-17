import React, { useEffect, useState } from 'react';
import { getAuditHistory } from '../services/auditService';
import { getMintedTokenHistory } from '../services/tokenService';

const History = () => {
  const [audits, setAudits] = useState([]);
  const [mints, setMints] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const [auditData, mintData] = await Promise.all([
          getAuditHistory(),
          getMintedTokenHistory()
        ]);
        setAudits(auditData);
        setMints(mintData);
      } catch (error) {
        console.error("Failed to fetch history:", error);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, []);

  if (loading) {
    return <div className="container" style={{ paddingTop: '50px' }}><h2>Loading History...</h2></div>;
  }

  return (
    <div className="container" style={{ paddingTop: '40px', paddingBottom: '100px' }}>
      
      <div style={{ marginBottom: '40px', borderBottom: '1px solid var(--border-color)', paddingBottom: '20px' }}>
        <h1 style={{ margin: '0 0 10px 0', fontSize: '2rem', color: '#fff' }}>Audit History</h1>
        <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '1.1rem' }}>
          Review previous carbon-capture audits and their generated credits.
        </p>
      </div>
      
      <div className="glass-card" style={{ marginBottom: '50px', padding: 0, overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'rgba(0,0,0,0.3)', borderBottom: '1px solid var(--border-color)' }}>
                <th style={{ padding: '20px', color: 'var(--text-secondary)', fontWeight: '600', fontSize: '0.9rem', textTransform: 'uppercase' }}>S.No.</th>
                <th style={{ padding: '20px', color: 'var(--text-secondary)', fontWeight: '600', fontSize: '0.9rem', textTransform: 'uppercase' }}>Audit Date & Time</th>
                <th style={{ padding: '20px', color: 'var(--text-secondary)', fontWeight: '600', fontSize: '0.9rem', textTransform: 'uppercase' }}>Data Set Number</th>
                <th style={{ padding: '20px', color: 'var(--text-secondary)', fontWeight: '600', fontSize: '0.9rem', textTransform: 'uppercase' }}>Normal or Anomalies</th>
                <th style={{ padding: '20px', color: 'var(--text-secondary)', fontWeight: '600', fontSize: '0.9rem', textTransform: 'uppercase' }}>Carbon Credits Generated</th>
              </tr>
            </thead>
            <tbody>
              {audits.map((audit, index) => (
                <tr key={audit.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)', transition: 'background 0.2s' }}>
                  <td style={{ padding: '20px', color: 'var(--text-secondary)' }}>{String(index + 1).padStart(2, '0')}</td>
                  <td style={{ padding: '20px', color: 'var(--text-secondary)' }}>{audit.timestamp ? new Date(audit.timestamp).toLocaleString() : 'N/A'}</td>
                  <td style={{ padding: '20px', color: '#fff', fontWeight: '500' }}>{audit.dataSetNumber}</td>
                  <td style={{ padding: '20px' }}>
                    <span style={{ 
                      color: audit.status === 'Normal' ? '#4caf50' : '#f44336',
                      background: audit.status === 'Normal' ? 'rgba(76, 175, 80, 0.1)' : 'rgba(244, 67, 54, 0.1)',
                      padding: '6px 12px',
                      borderRadius: '6px',
                      fontSize: '0.85rem',
                      fontWeight: 'bold',
                      textTransform: 'uppercase'
                    }}>
                      {audit.status}
                    </span>
                  </td>
                  <td style={{ padding: '20px', fontWeight: 'bold', color: '#fff' }}>{audit.creditsGenerated}</td>
                </tr>
              ))}
              {audits.length === 0 && (
                <tr>
                  <td colSpan="5" style={{ padding: '30px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    No audit history found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ marginBottom: '20px' }}>
        <h2 style={{ fontSize: '1.5rem', margin: 0, color: '#fff' }}>Minted Token History</h2>
      </div>

      <div className="glass-card" style={{ padding: 0, overflow: 'hidden' }}>
        <div style={{ overflowX: 'auto' }}>
          <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left' }}>
            <thead>
              <tr style={{ background: 'rgba(0,0,0,0.3)', borderBottom: '1px solid var(--border-color)' }}>
                <th style={{ padding: '20px', color: 'var(--text-secondary)', fontWeight: '600', fontSize: '0.9rem', textTransform: 'uppercase' }}>Mint ID</th>
                <th style={{ padding: '20px', color: 'var(--text-secondary)', fontWeight: '600', fontSize: '0.9rem', textTransform: 'uppercase' }}>Date</th>
                <th style={{ padding: '20px', color: 'var(--text-secondary)', fontWeight: '600', fontSize: '0.9rem', textTransform: 'uppercase' }}>Source Audit</th>
                <th style={{ padding: '20px', color: 'var(--text-secondary)', fontWeight: '600', fontSize: '0.9rem', textTransform: 'uppercase' }}>Amount Minted</th>
              </tr>
            </thead>
            <tbody>
              {mints.map((mint) => (
                <tr key={mint.id} style={{ borderBottom: '1px solid rgba(255,255,255,0.05)' }}>
                  <td style={{ padding: '20px', color: '#fff' }}>{mint.transactionId || mint.id}</td>
                  <td style={{ padding: '20px', color: 'var(--text-secondary)' }}>{new Date(mint.mintedAt || mint.timestamp).toLocaleString()}</td>
                  <td style={{ padding: '20px', color: '#fff' }}>{mint.source || 'N/A'}</td>
                  <td style={{ padding: '20px', fontWeight: 'bold', color: 'var(--primary-color)' }}>+{mint.amount} CBX</td>
                </tr>
              ))}
              {mints.length === 0 && (
                <tr>
                  <td colSpan="4" style={{ padding: '30px', textAlign: 'center', color: 'var(--text-secondary)' }}>
                    No minted tokens found.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};

export default History;
