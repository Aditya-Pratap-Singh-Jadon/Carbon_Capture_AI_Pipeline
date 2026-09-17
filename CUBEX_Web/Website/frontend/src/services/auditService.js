import { api } from '../api';

export const getAuditHistory = async () => {
  try {
    const response = await api.getAuditHistory();
    return response.data.history || [];
  } catch (error) {
    console.error("Error fetching audit history", error);
    return [];
  }
};

export const initiateNewAudit = async (auditParams) => {
  const response = await api.startNewAudit();
  return response.data;
};
