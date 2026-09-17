// dashboardService.js
// Provides the UI with current dashboard statistics.
// This is a data abstraction layer that will be connected to the Python backend in the future.

export const getDashboardStats = async () => {
  // TODO: Replace with real API call to Python backend
  // return await axios.get('/api/dashboard/stats');
  
  return new Promise((resolve) => {
    setTimeout(() => {
      resolve({
        co2CapturedKg: 1248.52,
        carbonCreditsGenerated: 1248,
        activeSensors: 6,
        systemStatus: 'Normal'
      });
    }, 300);
  });
};
