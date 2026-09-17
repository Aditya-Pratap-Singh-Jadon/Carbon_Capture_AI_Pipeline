const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

// Resolve the root of the CUBEX Final project
const PROJECT_ROOT = path.resolve(__dirname, '../../../../');
const CUBEX_COM_DIR = path.join(PROJECT_ROOT, 'CUBEX_COM');
const CUBEX_ML_DIR = path.join(PROJECT_ROOT, 'CUBEX_ML');

const TRAINING_CSV = path.join(CUBEX_ML_DIR, 'data', 'synthetic', 'telemetry_train.csv');

// In-memory store for audit history (since we don't have a DB required)
let auditHistory = [];
let auditIdCounter = 1;

let isAuditRunning = false;

function spawnPromise(command, args, cwd, stdoutStream = null) {
  return new Promise((resolve, reject) => {
    console.log(`[EXEC] ${command} ${args.join(' ')} (CWD: ${cwd})`);
    
    const child = spawn(command, args, { cwd, shell: false });
    
    let output = '';
    let errorOutput = '';

    child.stdout.on('data', (data) => {
      output += data.toString();
      if (stdoutStream) {
        stdoutStream.write(data);
      }
    });

    child.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    child.on('close', (code) => {
      if (code === 0) {
        resolve(output);
      } else {
        reject(new Error(`Command failed with code ${code}\nStderr: ${errorOutput}\nStdout: ${output}`));
      }
    });
    
    child.on('error', (err) => {
      reject(err);
    });
  });
}

function spawnPiped(cmd1, args1, cwd1, cmd2, args2, cwd2) {
  return new Promise((resolve, reject) => {
    console.log(`[EXEC PIPED] ${cmd1} ${args1.join(' ')} | ${cmd2} ${args2.join(' ')}`);
    
    const p1 = spawn(cmd1, args1, { cwd: cwd1, shell: false });
    const p2 = spawn(cmd2, args2, { cwd: cwd2, shell: false });
    
    let output = '';
    let errorOutput = '';

    p1.stdout.pipe(p2.stdin);
    
    p1.stderr.on('data', (data) => {
      console.error(`[P1 STDERR]: ${data.toString()}`);
    });

    p2.stdout.on('data', (data) => {
      output += data.toString();
    });

    p2.stderr.on('data', (data) => {
      errorOutput += data.toString();
    });

    p2.on('close', (code) => {
      if (code === 0) {
        resolve(output);
      } else {
        reject(new Error(`Piped command failed with code ${code}\nStderr: ${errorOutput}`));
      }
    });
    
    p2.on('error', (err) => {
      reject(err);
    });
  });
}

exports.startNewAudit = async (req, res) => {
  if (isAuditRunning) {
    return res.status(409).json({ success: false, error: 'An audit is already running.' });
  }

  isAuditRunning = true;

  try {
    const pythonExe = 'python'; 

    // --- STEP 1: Generate Training Data ---
    console.log("--- STEP 1: Generating Training Data ---");
    const runTwinScript = path.join(CUBEX_COM_DIR, 'carbon_capture_digital_twin', 'scripts', 'run_twin.py');
    const writeStream = fs.createWriteStream(TRAINING_CSV);
    
    try {
      await spawnPromise(
        pythonExe,
        [runTwinScript, '--labeled', '--samples', '200'],
        PROJECT_ROOT, // run from root like user manual command
        writeStream
      );
    } catch (e) {
      writeStream.close();
      throw new Error(`TRAINING_DATA_FAILED: ${e.message}`);
    }
    writeStream.close();
    
    // Verify file exists and is not empty
    if (!fs.existsSync(TRAINING_CSV) || fs.statSync(TRAINING_CSV).size === 0) {
      throw new Error('TRAINING_DATA_FAILED: Output CSV is missing or empty.');
    }

    // --- STEP 2: Train Model ---
    console.log("--- STEP 2: Training Model ---");
    const trainScript = path.join(CUBEX_ML_DIR, 'scripts', 'train_anomaly_detector.py');
    const inputCsv = path.join(CUBEX_ML_DIR, 'data', 'synthetic', 'telemetry_train.csv');
    
    try {
      await spawnPromise(
        pythonExe,
        [trainScript, '--input', inputCsv],
        PROJECT_ROOT
      );
    } catch (e) {
      throw new Error(`TRAINING_FAILED: ${e.message}`);
    }
    
    // --- STEP 3: Live Data Generation + Inference ---
    console.log("--- STEP 3: Live Inference ---");
    const inferenceScript = path.join(CUBEX_ML_DIR, 'scripts', 'run_telemetry_inference.py');
    
    let inferenceRawOutput = '';
    try {
      inferenceRawOutput = await spawnPiped(
        pythonExe, [runTwinScript, '--samples', '50'], PROJECT_ROOT,
        pythonExe, [inferenceScript], PROJECT_ROOT
      );
    } catch (e) {
      throw new Error(`INFERENCE_FAILED: ${e.message}`);
    }
    
    // Parse output line by line
    const lines = inferenceRawOutput.split('\n').filter(l => l.trim().length > 0);
    let anomaliesDetected = 0;
    let totalProcessed = 0;

    for (const line of lines) {
      try {
        const result = JSON.parse(line);
        totalProcessed++;
        if (result.status === 'ANOMALOUS') {
          anomaliesDetected++;
        }
      } catch (e) {
        // Skip non-JSON log lines
      }
    }
    
    if (totalProcessed === 0) {
      throw new Error('INFERENCE_FAILED: No valid inference JSON output parsed.');
    }

    const overallStatus = anomaliesDetected > 0 ? 'Anomalies' : 'Normal';
    const creditsGenerated = 'Not yet integrated'; // Carbon credit logic isn't connected to telemetry pipeline yet

    // --- STEP 4: Save to History ---
    const dataSetNumber = `DATA-${String(auditIdCounter++).padStart(3, '0')}`;
    const newAuditRecord = {
      id: auditIdCounter - 1,
      dataSetNumber,
      status: overallStatus,
      creditsGenerated,
      timestamp: new Date().toISOString()
    };
    
    auditHistory.push(newAuditRecord);

    isAuditRunning = false;
    return res.json({
      success: true,
      auditResult: {
        status: overallStatus,
        anomalies: anomaliesDetected,
        totalSamples: totalProcessed,
        creditsGenerated
      },
      record: newAuditRecord
    });
    
  } catch (err) {
    isAuditRunning = false;
    console.error(err);
    const stageFailed = err.message.split(':')[0] || 'UNKNOWN_FAILURE';
    return res.status(500).json({
      success: false,
      error: err.message,
      stage: stageFailed
    });
  }
};

exports.getAuditHistory = (req, res) => {
  return res.json({ success: true, history: [...auditHistory].reverse() });
};
