// tokenService.js
// Abstracts the future Python-based token layer (replaces blockchain).
import { api } from '../api';

export const getCurrentMintedTokens = async () => {
  try {
    const response = await api.getStats();
    return {
      totalMinted: response.data?.stats?.totalMinted || 0,
      pendingMint: 0
    };
  } catch (error) {
    console.error("Error fetching minted tokens:", error);
    return { totalMinted: 0, pendingMint: 0 };
  }
};

export const getMintedTokenHistory = async () => {
  try {
    const response = await api.getHistory();
    return response.data?.history || [];
  } catch (error) {
    console.error("Error fetching token history:", error);
    return [];
  }
};
