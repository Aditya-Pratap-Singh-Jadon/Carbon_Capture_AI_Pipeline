import React from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Navbar from './components/Navbar';
import Dashboard from './pages/Dashboard';
import History from './pages/History';

function App() {
  return (
    <Router>
      <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
        <Navbar />
        <main style={{ flex: 1, paddingBottom: '50px' }}>
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
            <Route path="*" element={<Navigate to="/" />} />
          </Routes>
        </main>
        <footer style={{ 
          textAlign: 'center', 
          padding: '24px', 
          borderTop: '1px solid var(--border-color)',
          color: 'var(--text-secondary)',
          fontSize: '0.9rem',
          background: 'rgba(0,0,0,0.2)'
        }}>
          &copy; {new Date().getFullYear()} CUBEX Industrial Carbon Capture. All rights reserved.
        </footer>
      </div>
    </Router>
  );
}

export default App;
