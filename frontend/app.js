/**
 * mmWave Human Identification Platform - Frontend Application
 * 
 * Handles:
 * - File uploads
 * - API communication
 * - Real-time training visualization
 * - Progress tracking
 * - Results display
 */

// Configuration
const API_BASE_URL = window.location.origin;
const POLL_INTERVAL = 2000; // 2 seconds

// Global state
let currentJobId = null;
let selectedTask = 'classification';  // 'classification' | 'autoencoder'
let selectedModel = 'pointnet';
let pollingInterval = null;
let trainingChart = null;

// Workflow state
let workflowState = {
    hasData: false,
    isPreprocessed: false,
    isTraining: false
};

// ==================== Initialization ====================

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
    setupEventListeners();
    checkServerHealth();
    checkExistingData();
});

function initializeApp() {
    console.log('Initializing mmWave Training Platform...');
    setupChartDefaults();
    updateWorkflowButtons();
}

function setupChartDefaults() {
    if (typeof Chart !== 'undefined') {
        Chart.defaults.color = '#9ca3af';
        Chart.defaults.borderColor = 'rgba(255, 255, 255, 0.1)';
    }
}

// ==================== Event Listeners ====================

function setupEventListeners() {
    // Upload zone
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');

    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    // Preprocessing
    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);

    // Task selection
    document.querySelectorAll('.task-option').forEach(option => {
        option.addEventListener('click', () => selectTask(option.dataset.task));
    });

    // Model selection
    document.querySelectorAll('.model-option').forEach(option => {
        option.addEventListener('click', () => selectModel(option.dataset.model));
    });

    // Training
    document.getElementById('trainBtn').addEventListener('click', startTraining);

    // Evaluation
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);

    // Downloads
    document.getElementById('downloadModelBtn').addEventListener('click', downloadModel);
    document.getElementById('downloadReportBtn').addEventListener('click', downloadReport);

    // File List Toggle
    document.getElementById('toggleFileListBtn').addEventListener('click', toggleFileList);

    // Chart Controls
    document.querySelectorAll('.chart-controls input').forEach(checkbox => {
        checkbox.addEventListener('change', toggleChartDataset);
    });
}

// ==================== Server Health ====================

async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();

        if (data.status === 'healthy') {
            updateServerStatus(true);
        } else {
            updateServerStatus(false);
        }
    } catch (error) {
        console.error('Server health check failed:', error);
        updateServerStatus(false);
    }
}

function updateServerStatus(isHealthy) {
    const statusIndicator = document.getElementById('serverStatus');
    const statusDot = statusIndicator.querySelector('.status-dot');
    const statusText = statusIndicator.querySelector('span');

    if (isHealthy) {
        statusDot.style.background = '#10b981';
        statusText.textContent = 'Server Connected';
    } else {
        statusDot.style.background = '#ef4444';
        statusText.textContent = 'Server Disconnected';
    }
}

// ==================== File Upload ====================

function handleDragOver(e) {
    e.preventDefault();
    e.currentTarget.classList.add('dragover');
}

function handleDragLeave(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');
}

function handleDrop(e) {
    e.preventDefault();
    e.currentTarget.classList.remove('dragover');

    const files = Array.from(e.dataTransfer.files);
    uploadFiles(files);
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    uploadFiles(files);
}

async function uploadFiles(files) {
    if (files.length === 0) return;

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    try {
        showLoading('Uploading files...');

        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            showSuccess(`Uploaded ${data.count} files successfully`);
            updateUploadBadge(data.count);
            displayFileList(data.uploaded);
        } else {
            showError(`Upload failed: ${data.error}`);
        }
    } catch (error) {
        console.error('Upload error:', error);
        showError('Upload failed. Please try again.');
    } finally {
        hideLoading();
    }
}

function updateUploadBadge(count) {
    const badge = document.getElementById('uploadBadge');
    badge.textContent = `${count} files`;
}

function displayFileList(files) {
    const fileList = document.getElementById('fileList');
    const header = document.getElementById('fileListHeader');
    const countSpan = document.getElementById('fileCount');

    fileList.innerHTML = '';

    if (files.length > 0) {
        header.style.display = 'flex';
        countSpan.textContent = `${files.length} files uploaded`;
    } else {
        header.style.display = 'none';
    }

    files.forEach(filename => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <div class="file-name">${filename}</div>
            </div>
            <div class="file-size">✓ Ready</div>
        `;
        fileList.appendChild(fileItem);
    });
}

function toggleFileList() {
    const list = document.getElementById('fileList');
    const btn = document.getElementById('toggleFileListBtn');
    const icon = btn.querySelector('.icon');

    if (list.classList.contains('collapsed')) {
        list.classList.remove('collapsed');
        list.classList.add('expanded');
        btn.innerHTML = '<span class="icon">▲</span> Collapse All';
    } else {
        list.classList.remove('expanded');
        list.classList.add('collapsed');
        btn.innerHTML = '<span class="icon">▼</span> Show All';
    }
}

// ==================== Preprocessing ====================

async function preprocessData() {
    const btn = document.getElementById('preprocessBtn');
    const spinner = document.getElementById('preprocessSpinner');

    try {
        btn.classList.add('loading');
        btn.disabled = true;

        const params = {
            data: {
                num_points: parseInt(document.getElementById('numPoints').value),
                samples_per_mesh: parseInt(document.getElementById('samplesPerMesh').value),
                normalize_center: false,
                normalize_scale: false
            },
            augmentation: {
                rotation_range: parseInt(document.getElementById('rotationRange').value),
                translation_range: parseFloat(document.getElementById('translationRange').value),
                normalize: document.getElementById('normalize').value === 'true'
            }
        };

        const response = await fetch(`${API_BASE_URL}/api/preprocess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        const data = await response.json();

        if (data.success) {
            showSuccess('Data preprocessing completed successfully!');
        } else {
            showError(`Preprocessing failed: ${data.error}`);
        }
    } catch (error) {
        console.error('Preprocessing error:', error);
        showError('Preprocessing failed. Please try again.');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

// ==================== Task & Model Selection ====================

function selectTask(taskType) {
    selectedTask = taskType;

    document.querySelectorAll('.task-option').forEach(opt => opt.classList.remove('selected'));
    document.querySelector(`[data-task="${taskType}"]`).classList.add('selected');

    const classificationModels = document.getElementById('classificationModels');
    const autoencoderModels = document.getElementById('autoencoderModels');

    if (taskType === 'classification') {
        classificationModels.style.display = 'flex';
        autoencoderModels.style.display = 'none';
        selectedModel = 'pointnet';
        document.querySelectorAll('#classificationModels .model-option').forEach(opt => opt.classList.remove('selected'));
        document.querySelector('#classificationModels [data-model="pointnet"]').classList.add('selected');
    } else {
        classificationModels.style.display = 'none';
        autoencoderModels.style.display = 'flex';
        selectedModel = 'pointnet_ae';
        document.querySelectorAll('#autoencoderModels .model-option').forEach(opt => opt.classList.remove('selected'));
        document.querySelector('#autoencoderModels [data-model="pointnet_ae"]').classList.add('selected');
    }
}

function selectModel(modelType) {
    selectedModel = modelType;

    document.querySelectorAll('.model-option').forEach(option => {
        option.classList.remove('selected');
    });

    const target = document.querySelector(`[data-model="${modelType}"]`);
    if (target) target.classList.add('selected');
}

// ==================== Training ====================

async function startTraining() {
    const btn = document.getElementById('trainBtn');

    try {
        btn.classList.add('loading');
        btn.disabled = true;

        const params = {
            model_type: selectedModel,
            training: {
                batch_size: parseInt(document.getElementById('batchSize').value),
                num_epochs: parseInt(document.getElementById('numEpochs').value),
                learning_rate: parseFloat(document.getElementById('learningRate').value),
                weight_decay: 0.0001,
                early_stopping_patience: 20
            },
            model: {
                dropout: parseFloat(document.getElementById('dropout').value)
            }
        };

        const response = await fetch(`${API_BASE_URL}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        const data = await response.json();

        if (data.success) {
            currentJobId = data.job_id;
            showSuccess(`Training started! Model: ${data.model_type}`);
            showMonitorSection();
            startPollingTrainingStatus();
        } else {
            showError(`Training failed: ${data.error}`);
        }
    } catch (error) {
        console.error('Training error:', error);
        showError('Failed to start training. Please try again.');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

function showMonitorSection() {
    document.getElementById('monitorSection').style.display = 'block';
    document.getElementById('monitorSection').scrollIntoView({ behavior: 'smooth' });
    initializeTrainingChart();
}

function initializeTrainingChart() {
    if (trainingChart) {
        trainingChart.destroy();
    }
    const ctx = document.getElementById('trainingChart').getContext('2d');

    trainingChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: [],
            datasets: [
                {
                    label: 'Train Loss',
                    data: [],
                    borderColor: '#ef4444',
                    backgroundColor: 'rgba(239, 68, 68, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#f59e0b',
                    backgroundColor: 'rgba(245, 158, 11, 0.1)',
                    tension: 0.4
                },
                {
                    label: 'Train Acc',
                    data: [],
                    borderColor: '#10b981',
                    backgroundColor: 'rgba(16, 185, 129, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                },
                {
                    label: 'Val Acc',
                    data: [],
                    borderColor: '#00d4ff',
                    backgroundColor: 'rgba(0, 212, 255, 0.1)',
                    tension: 0.4,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: {
                        display: true,
                        text: 'Loss'
                    }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: {
                        display: true,
                        text: 'Accuracy (%)'
                    },
                    grid: {
                        drawOnChartArea: false
                    }
                }
            },
            plugins: {
                legend: {
                    display: true,
                    position: 'top'
                }
            }
        }
    });
}

function startPollingTrainingStatus() {
    // Clear any existing polling
    if (pollingInterval) {
        clearInterval(pollingInterval);
    }

    // Poll immediately, then every POLL_INTERVAL
    pollTrainingStatus();
    pollingInterval = setInterval(pollTrainingStatus, POLL_INTERVAL);
}

async function pollTrainingStatus() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/training/status/${currentJobId}`);
        const data = await response.json();

        if (data.success) {
            updateTrainingUI(data);

            // Stop polling if training is complete or failed
            if (data.status === 'completed' || data.status === 'failed') {
                clearInterval(pollingInterval);
                onTrainingComplete(data);
            }
        }
    } catch (error) {
        console.error('Polling error:', error);
    }
}

function updateTrainingUI(data) {
    const isAE = (data.model_type || selectedModel || '').includes('_ae');

    // Update status badge
    const statusBadge = document.getElementById('trainingStatus');
    statusBadge.textContent = data.status.charAt(0).toUpperCase() + data.status.slice(1);
    statusBadge.className = `badge badge-${data.status}`;

    // Update progress
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');
    const progressPercent = document.getElementById('progressPercent');

    progressFill.style.width = `${data.progress}%`;
    progressText.textContent = `Epoch ${data.current_epoch}/${data.total_epochs}`;
    progressPercent.textContent = `${Math.round(data.progress)}%`;

    // Show/hide accuracy for AE (AE uses loss only)
    const trainAccCard = document.getElementById('trainAccCard');
    const valAccCard = document.getElementById('valAccCard');
    const trainAccLabel = document.getElementById('trainAccLabel');
    const valAccLabel = document.getElementById('valAccLabel');
    if (trainAccCard) trainAccCard.style.display = isAE ? 'none' : 'block';
    if (valAccCard) valAccCard.style.display = isAE ? 'none' : 'block';
    if (trainAccLabel) trainAccLabel.style.display = isAE ? 'none' : 'inline-flex';
    if (valAccLabel) valAccLabel.style.display = isAE ? 'none' : 'inline-flex';

    // Update metrics
    if (data.metrics.train_loss.length > 0) {
        const latest = data.metrics.train_loss.length - 1;

        document.getElementById('trainLoss').textContent =
            data.metrics.train_loss[latest]?.toFixed(6) || '-';
        document.getElementById('valLoss').textContent =
            data.metrics.val_loss[latest]?.toFixed(6) || '-';

        if (!isAE) {
            document.getElementById('trainAcc').textContent =
                ((data.metrics.train_acc[latest] || 0) * 100).toFixed(2) + '%';
            document.getElementById('valAcc').textContent =
                ((data.metrics.val_acc[latest] || 0) * 100).toFixed(2) + '%';
        }

        // Update chart
        updateChart(data.metrics, isAE);
    }

    // Fetch and update logs
    fetchTrainingLogs();
}

function updateChart(metrics, isAE) {
    if (!trainingChart) return;

    const epochs = Array.from({ length: metrics.train_loss.length }, (_, i) => i + 1);

    trainingChart.data.labels = epochs;
    trainingChart.data.datasets[0].data = metrics.train_loss;
    trainingChart.data.datasets[1].data = metrics.val_loss;
    trainingChart.data.datasets[2].data = (metrics.train_acc || []).map(v => v * 100);
    trainingChart.data.datasets[3].data = (metrics.val_acc || []).map(v => v * 100);

    if (isAE) {
        trainingChart.setDatasetVisibility(2, false);
        trainingChart.setDatasetVisibility(3, false);
    }

    trainingChart.update('none');
}

function toggleChartDataset(e) {
    if (!trainingChart) return;

    const datasetIndex = parseInt(e.target.dataset.dataset);
    const isVisible = e.target.checked;

    trainingChart.setDatasetVisibility(datasetIndex, isVisible);
    trainingChart.update();
}

async function fetchTrainingLogs() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/training/logs/${currentJobId}`);
        const data = await response.json();

        if (data.success) {
            updateLogsDisplay(data.logs);
        }
    } catch (error) {
        console.error('Log fetch error:', error);
    }
}

function updateLogsDisplay(logs) {
    const logsContainer = document.getElementById('trainingLogs');
    logsContainer.innerHTML = logs.map(log =>
        `<div class="log-entry">${escapeHtml(log)}</div>`
    ).join('');

    // Auto-scroll to bottom
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

function onTrainingComplete(data) {
    if (data.status === 'completed') {
        showSuccess('Training completed successfully!');
        document.getElementById('resultsSection').style.display = 'block';
        document.getElementById('downloadSection').style.display = 'block';
        document.getElementById('resultsSection').scrollIntoView({ behavior: 'smooth' });
    } else if (data.status === 'failed') {
        showError(`Training failed: ${data.error}`);
    }
}

// ==================== Evaluation ====================

async function evaluateModel() {
    if (!currentJobId) return;

    const btn = document.getElementById('evaluateBtn');

    try {
        btn.classList.add('loading');
        btn.disabled = true;

        const response = await fetch(`${API_BASE_URL}/api/report/generate/${currentJobId}`, {
            method: 'POST'
        });

        const data = await response.json();

        if (data.success) {
            showSuccess('Evaluation report generated!');
            displayResults();
        } else {
            showError(`Evaluation failed: ${data.error}`);
        }
    } catch (error) {
        console.error('Evaluation error:', error);
        showError('Evaluation failed. Please try again.');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = false;
    }
}

async function displayResults() {
    if (!currentJobId) return;

    try {
        const vizResponse = await fetch(`${API_BASE_URL}/api/visualization/${currentJobId}`);
        const vizData = await vizResponse.json();

        if (vizData.success) {
            const statusResponse = await fetch(`${API_BASE_URL}/api/training/status/${currentJobId}`);
            const statusData = statusResponse.ok ? await statusResponse.json() : {};
            const modelType = statusData.model_type || selectedModel || '';
            const isAE = modelType.includes('_ae');

            const resultsGrid = document.getElementById('resultsGrid');
            if (isAE) {
                const bestValLoss = Math.min(...(vizData.metrics.val_loss || [1e10]));
                resultsGrid.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-label">Best Val Loss (Chamfer Dist)</div>
                        <div class="metric-value">${bestValLoss.toFixed(6)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Final Train Loss</div>
                        <div class="metric-value">${(vizData.metrics.train_loss[vizData.metrics.train_loss.length - 1] || 0).toFixed(6)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Epochs</div>
                        <div class="metric-value">${vizData.metrics.train_loss.length}</div>
                    </div>
                `;
            } else {
                resultsGrid.innerHTML = `
                    <div class="metric-card">
                        <div class="metric-label">Best Validation Accuracy</div>
                        <div class="metric-value">${(Math.max(...(vizData.metrics.val_acc || [0])) * 100).toFixed(2)}%</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Final Train Loss</div>
                        <div class="metric-value">${(vizData.metrics.train_loss[vizData.metrics.train_loss.length - 1] || 0).toFixed(4)}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Epochs</div>
                        <div class="metric-value">${vizData.metrics.train_loss.length}</div>
                    </div>
                `;
            }
        }
    } catch (error) {
        console.error('Results display error:', error);
    }
}

// ==================== Downloads ====================

async function downloadModel() {
    if (!currentJobId) return;

    try {
        // Use window.open to prevent page navigation/reload
        const url = `${API_BASE_URL}/api/download/model/${currentJobId}`;
        window.open(url, '_blank');
        showSuccess('Model download started!');
    } catch (error) {
        console.error('Download error:', error);
        showError('Download failed. Please try again.');
    }
}

async function downloadReport() {
    if (!currentJobId) return;

    try {
        // Use window.open to prevent page navigation/reload
        const url = `${API_BASE_URL}/api/download/report/${currentJobId}`;
        window.open(url, '_blank');
        showSuccess('Report download started!');
    } catch (error) {
        console.error('Download error:', error);
        showError('Download failed. Please try again.');
    }
}

// ==================== Utilities ====================

function showLoading(message) {
    console.log('Loading:', message);
}

function hideLoading() {
    console.log('Loading complete');
}

function showSuccess(message) {
    console.log('Success:', message);
    // Could add toast notifications here
}

function showError(message) {
    console.error('Error:', message);
    alert(message); // Simple error display, could be improved with toast
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ==================== Workflow Validation ====================

async function checkExistingData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/files`);
        const data = await response.json();

        if (data.success && data.count > 0) {
            workflowState.hasData = true;
            updateUploadBadge(data.count);

            // Show info message
            const infoBox = document.getElementById('dataStatusBox');
            const infoText = document.getElementById('dataStatusText');
            infoBox.className = 'info-box success';
            infoText.textContent = `Found ${data.count} existing mesh files in data/raw/. You can skip upload or add more files.`;
            infoBox.style.display = 'flex';

            // Display existing files
            displayFileList(data.files.map(f => f.name));
        } else {
            // No data found
            const infoBox = document.getElementById('dataStatusBox');
            const infoText = document.getElementById('dataStatusText');
            infoBox.className = 'info-box warning';
            infoText.textContent = 'No data files found. Please upload mesh files to begin.';
            infoBox.style.display = 'flex';

            updateUploadBadge(0);
        }

        updateWorkflowButtons();
    } catch (error) {
        console.error('Error checking data:', error);
        updateUploadBadge(0);
    }
}

function updateWorkflowButtons() {
    const preprocessBtn = document.getElementById('preprocessBtn');
    const trainBtn = document.getElementById('trainBtn');

    // Preprocess button: enabled if hasData
    preprocessBtn.disabled = !workflowState.hasData;

    // Train button: enabled if preprocessed
    trainBtn.disabled = !workflowState.isPreprocessed;
}

// Enhanced uploadFiles with progress bar
async function uploadFilesWithProgress(files) {
    if (files.length === 0) return;

    const progressContainer = document.getElementById('uploadProgress');
    const progressFill = document.getElementById('uploadProgressFill');
    const progressText = document.getElementById('uploadProgressText');
    const progressPercent = document.getElementById('uploadProgressPercent');

    progressContainer.style.display = 'block';

    const formData = new FormData();
    files.forEach(file => formData.append('files', file));

    try {
        // Simulate progress (real progress would need server-sent events or websockets)
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 10;
            if (progress > 90) {
                clearInterval(progressInterval);
            }
            progressFill.style.width = `${progress}%`;
            progressPercent.textContent = `${progress}%`;
        }, 200);

        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });

        clearInterval(progressInterval);
        progressFill.style.width = '100%';
        progressPercent.textContent = '100%';

        const data = await response.json();

        if (data.success) {
            workflowState.hasData = true;
            updateWorkflowButtons();

            showSuccess(`Uploaded ${data.count} files successfully`);
            updateUploadBadge(data.count);
            displayFileList(data.uploaded);

            // Update info box
            const infoBox = document.getElementById('dataStatusBox');
            const infoText = document.getElementById('dataStatusText');
            infoBox.className = 'info-box success';
            infoText.textContent = `Successfully uploaded ${data.count} files. Ready for preprocessing.`;
            infoBox.style.display = 'flex';

            setTimeout(() => {
                progressContainer.style.display = 'none';
                progressFill.style.width = '0%';
            }, 1500);
        } else {
            showError(`Upload failed: ${data.error}`);
            progressContainer.style.display = 'none';
        }
    } catch (error) {
        console.error('Upload error:', error);
        showError('Upload failed. Please try again.');
        progressContainer.style.display = 'none';
    }
}

// Override original uploadFiles
uploadFiles = uploadFilesWithProgress;

// Enhanced preprocessing with progress
async function preprocessDataWithProgress() {
    if (!workflowState.hasData) {
        showError('Please upload data first!');
        return;
    }

    const btn = document.getElementById('preprocessBtn');
    const progressContainer = document.getElementById('preprocessProgress');
    const progressFill = document.getElementById('preprocessProgressFill');
    const progressText = document.getElementById('preprocessProgressText');

    try {
        btn.classList.add('loading');
        btn.disabled = true;
        progressContainer.style.display = 'block';

        const params = {
            data: {
                num_points: parseInt(document.getElementById('numPoints').value),
                samples_per_mesh: parseInt(document.getElementById('samplesPerMesh').value),
                normalize_center: document.getElementById('normalizeCenter').value === 'true',
                normalize_scale: document.getElementById('normalize').value === 'true'
            },
            augmentation: {
                rotation_range: parseInt(document.getElementById('rotationRange').value),
                translation_range: parseFloat(document.getElementById('translationRange').value),
                normalize: document.getElementById('normalize').value === 'true'
            }
        };

        // Animate progress bar
        let progress = 0;
        const progressInterval = setInterval(() => {
            progress += 5;
            if (progress > 90) {
                clearInterval(progressInterval);
            }
            progressFill.style.width = `${progress}%`;
        }, 300);

        const response = await fetch(`${API_BASE_URL}/api/preprocess`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        clearInterval(progressInterval);
        progressFill.style.width = '100%';

        const data = await response.json();

        if (data.success) {
            workflowState.isPreprocessed = true;
            updateWorkflowButtons();

            showSuccess('Data preprocessing completed successfully!');
            progressText.textContent = 'Completed!';

            // Show permanent success message
            const statusBox = document.getElementById('preprocessStatusBox');
            const statusText = document.getElementById('preprocessStatusText');
            statusBox.className = 'info-box success';
            statusText.textContent = 'Data processing completed! You can now proceed to model training.';
            statusBox.style.display = 'flex';

            setTimeout(() => {
                progressContainer.style.display = 'none';
                progressFill.style.width = '50%';
            }, 2000);
        } else {
            showError(`Preprocessing failed: ${data.error}`);
            progressContainer.style.display = 'none';
        }
    } catch (error) {
        console.error('Preprocessing error:', error);
        showError('Preprocessing failed. Please try again.');
        progressContainer.style.display = 'none';
    } finally {
        btn.classList.remove('loading');
        btn.disabled = !workflowState.hasData; // Re-enable based on data state
    }
}

// Override original preprocessData  
preprocessData = preprocessDataWithProgress;

// Enhanced training check
async function startTrainingWithValidation() {
    if (!workflowState.isPreprocessed) {
        showError('Please preprocess data before training!');
        return;
    }

    workflowState.isTraining = true;

    // Call original startTraining logic
    const btn = document.getElementById('trainBtn');

    try {
        btn.classList.add('loading');
        btn.disabled = true;

        const params = {
            model_type: selectedModel,
            training: {
                batch_size: parseInt(document.getElementById('batchSize').value),
                num_epochs: parseInt(document.getElementById('numEpochs').value),
                learning_rate: parseFloat(document.getElementById('learningRate').value),
                weight_decay: 0.0001,
                early_stopping_patience: 20
            },
            model: {
                dropout: parseFloat(document.getElementById('dropout').value)
            }
        };

        const response = await fetch(`${API_BASE_URL}/api/train`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params)
        });

        const data = await response.json();

        if (data.success) {
            currentJobId = data.job_id;
            showSuccess(`Training started! Model: ${data.model_type}`);
            showMonitorSection();
            startPollingTrainingStatus();
        } else {
            showError(`Training failed: ${data.error}`);
            workflowState.isTraining = false;
        }
    } catch (error) {
        console.error('Training error:', error);
        showError('Failed to start training. Please try again.');
        workflowState.isTraining = false;
    } finally {
        btn.classList.remove('loading');
        btn.disabled = !workflowState.isPreprocessed;
    }
}

// Override original startTraining
startTraining = startTrainingWithValidation;
