import React from 'react';
import { Link, useLocation } from 'react-router-dom';

const Navbar = () => {
  const location = useLocation();
  
  const getLinkStyle = (path) => ({
    color: location.pathname === path ? 'var(--primary-color)' : '#fff',
    fontSize: '0.95rem',
    textDecoration: 'none',
    fontWeight: location.pathname === path ? '600' : 'normal',
    padding: '8px 16px',
    borderRadius: '6px',
    background: location.pathname === path ? 'rgba(0, 255, 136, 0.1)' : 'transparent',
    transition: 'all 0.2s ease'
  });

  return (
    <nav style={{
      display: 'flex',
      justifyContent: 'space-between',
      alignItems: 'center',
      padding: '16px 40px',
      background: 'rgba(10, 10, 10, 0.9)',
      backdropFilter: 'blur(12px)',
      borderBottom: '1px solid var(--border-color)',
      position: 'sticky',
      top: 0,
      zIndex: 100
    }}>
      <div style={{ fontSize: '1.4rem', fontWeight: 'bold', letterSpacing: '1px' }}>
        <Link to="/" style={{ color: '#fff', textDecoration: 'none', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{ width: '20px', height: '20px', background: 'var(--primary-color)', borderRadius: '4px' }}></div>
          CUBEX
        </Link>
      </div>
      <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
        <Link to="/" style={getLinkStyle('/')}>Dashboard</Link>
        <Link to="/history" style={getLinkStyle('/history')}>Audit History</Link>
      </div>
    </nav>
  );
};

export default Navbar;
