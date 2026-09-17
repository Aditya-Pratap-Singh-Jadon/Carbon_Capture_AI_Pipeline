import axios from 'axios';

const isLocal =
  window.location.hostname === 'localhost' ||
  window.location.hostname === '127.0.0.1';

const API_URL = isLocal
  ? 'http://localhost:3001/api'
  : 'https://carbon-capture-ai-pipeline.onrender.com/api';

export const api = {
  // Stats
  getStats: () => axios.get(`${API_URL}/stats`),
  
  // User Balance
  getBalance: (userId) => axios.get(`${API_URL}/users/${userId}/balance`),
  
  // History
  getHistory: (userId) => axios.get(`${API_URL}/tokens/history`, { params: { userId } }),
  
  // Mint Tokens
  mintTokens: (data) => axios.post(`${API_URL}/tokens/mint`, data),
  
  // Update Balance (Add/Subtract)
  updateBalance: (data) => axios.post(`${API_URL}/users/update-balance`, data),
  
  // Blockchain Placeholders
  getPrice: () => axios.get(`${API_URL}/price`),
  
  // Audit Pipeline
  startNewAudit: () => axios.post(`${API_URL}/audit/new`),
  getAuditHistory: () => axios.get(`${API_URL}/audit/history`)
};