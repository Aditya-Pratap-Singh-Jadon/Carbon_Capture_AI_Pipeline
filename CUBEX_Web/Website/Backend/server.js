require('dotenv').config();
const express = require('express');
const cors = require('cors');
const bodyParser = require('body-parser');
const tokenController = require('./controllers/tokenController');
const auditController = require('./controllers/auditController');

const app = express();
const PORT = process.env.PORT || 3001;

app.use(cors());
app.use(bodyParser.json());

app.get('/', (req, res) => {
    res.send('CubeX Exchange API is running');
});

const userBalances = {};
const mintHistory = [];

app.post('/api/tokens/mint', async (req, res) => {
  try {
    const { userId, amount, transactionId, source, timestamp } = req.body;
    
    // Validate request
    if (!userId || !amount || !transactionId) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields'
      });
    }
    
    // Check if transaction already processed (idempotency)
    const existingTxn = mintHistory.find(t => t.transactionId === transactionId);
    if (existingTxn) {
      return res.status(200).json({
        success: true,
        message: 'Transaction already processed',
        transaction: existingTxn,
        newBalance: userBalances[userId] || 0
      });
    }
    
    // Initialize user balance if needed
    if (!userBalances[userId]) {
      userBalances[userId] = 0;
    }
    
    // Mint tokens (add to balance)
    userBalances[userId] += amount;
    
    // Record transaction
    const transaction = {
      transactionId,
      userId,
      amount,
      source,
      timestamp,
      mintedAt: new Date().toISOString(),
      status: 'completed'
    };
    
    mintHistory.push(transaction);
    
    console.log(`✅ Minted ${amount} tokens for ${userId} (Transaction: ${transactionId})`);
    
    res.json({
      success: true,
      message: `Successfully minted ${amount} tokens`,
      transaction,
      newBalance: userBalances[userId]
    });
    
  } catch (error) {
    console.error('Error minting tokens:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.post('/api/users/update-balance', async (req, res) => {
  try {
    const { userId, tokens, value, operation, timestamp } = req.body;
    
    // Validate request
    if (!userId || tokens === undefined || !operation) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields'
      });
    }
    
    // Initialize user balance if needed
    if (!userBalances[userId]) {
      userBalances[userId] = 0;
    }
    
    // Update balance
    const oldBalance = userBalances[userId];
    
    if (operation === 'add') {
      userBalances[userId] += tokens;
    } else if (operation === 'subtract') {
      userBalances[userId] -= tokens;
      // Prevent negative balance
      if (userBalances[userId] < 0) {
        userBalances[userId] = 0;
      }
    } else {
      return res.status(400).json({
        success: false,
        error: 'Invalid operation. Use "add" or "subtract"'
      });
    }
    
    console.log(`💰 Updated balance for ${userId}: ${oldBalance} → ${userBalances[userId]} (${operation} ${tokens})`);
    
    res.json({
      success: true,
      message: `Balance updated successfully`,
      userId,
      operation,
      amount: tokens,
      value,
      oldBalance,
      newBalance: userBalances[userId],
      timestamp
    });
    
  } catch (error) {
    console.error('Error updating balance:', error);
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

app.get('/api/users/:userId/balance', (req, res) => {
  const { userId } = req.params;
  
  const balance = userBalances[userId] || 0;
  
  res.json({
    success: true,
    userId,
    balance,
    lastUpdated: new Date().toISOString()
  });
});

app.get('/api/tokens/history', (req, res) => {
  const { userId } = req.query;
  
  let history = mintHistory;
  
  if (userId) {
    history = mintHistory.filter(t => t.userId === userId);
  }
  
  res.json({
    success: true,
    count: history.length,
    history
  });
});

app.get('/api/stats', (req, res) => {
  const totalMinted = mintHistory.reduce((sum, t) => sum + t.amount, 0);
  const totalUsers = Object.keys(userBalances).length;
  const totalBalance = Object.values(userBalances).reduce((sum, b) => sum + b, 0);
  
  res.json({
    success: true,
    stats: {
      totalUsers,
      totalMinted,
      totalBalance,
      totalTransactions: mintHistory.length
    }
  });
});

app.get('/health', (req, res) => {
  res.json({
    success: true,
    service: 'Express Backend',
    timestamp: new Date().toISOString()
  });
});

app.get('/api/price', tokenController.getTokenPrice);
app.post('/api/buy', tokenController.buyTokens);
app.post('/api/sell', tokenController.sellTokens);
app.get('/api/history/:wallet', tokenController.getTransactionHistory);

// AUDIT INTEGRATION
app.post('/api/audit/new', auditController.startNewAudit);
app.get('/api/audit/history', auditController.getAuditHistory);

app.listen(PORT, () => {
  console.log('='.repeat(70));
  console.log('EXPRESS BACKEND - AUDIT ORCHESTRATOR ACTIVE');
  console.log('='.repeat(70));
  console.log(`Server running on http://localhost:${PORT}`);
  console.log('\nAvailable Endpoints:');
  console.log('  POST   /api/audit/new');
  console.log('  GET    /api/audit/history');
  console.log('='.repeat(70));
});
