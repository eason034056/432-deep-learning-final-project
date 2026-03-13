const API_BASE_URL = window.location.origin;
const POLL_INTERVAL = 2000;
const STAGE_ORDER = ['dataset', 'preprocess', 'configure', 'train', 'evaluate', 'export'];

let currentJobId = null;
let selectedTask = 'classification';
let selectedModel = 'pointnet';
let pollingInterval = null;
let trainingChart = null;
let currentConfig = null;

const workflowState = {
    activeStage: 'dataset',
    hasData: false,
    isPreprocessed: false,
    isTraining: false,
    trainingCompleted: false,
    reportGenerated: false,
    serverHealthy: false,
    files: [],
    latestMetrics: null,
    evaluationResults: null,
    trainingStatus: 'idle',
    totalEpochs: 0,
    currentEpoch: 0,
    runStartedAt: null,
    reportPath: null
};

document.addEventListener('DOMContentLoaded', () => {
    initializeApp();
});

async function initializeApp() {
    setupChartDefaults();
    setupEventListeners();
    updateModelVisibility();
    updateRunSummary();
    updateStageUI();
    renderContextPanel();

    await Promise.all([
        loadConfig(),
        checkServerHealth(),
        checkExistingData()
    ]);

    await loadLatestJob();
    resetViewToNextStage();
}

function setupChartDefaults() {
    if (typeof Chart !== 'undefined') {
        Chart.defaults.color = '#a6b8d0';
        Chart.defaults.borderColor = 'rgba(163, 186, 214, 0.12)';
        Chart.defaults.font.family = '"IBM Plex Sans", sans-serif';
    }
}

function setupEventListeners() {
    const uploadZone = document.getElementById('uploadZone');
    const fileInput = document.getElementById('fileInput');

    uploadZone.addEventListener('click', () => fileInput.click());
    uploadZone.addEventListener('dragover', handleDragOver);
    uploadZone.addEventListener('dragleave', handleDragLeave);
    uploadZone.addEventListener('drop', handleDrop);
    fileInput.addEventListener('change', handleFileSelect);

    document.getElementById('preprocessBtn').addEventListener('click', preprocessData);
    document.getElementById('trainBtn').addEventListener('click', startTraining);
    document.getElementById('evaluateBtn').addEventListener('click', evaluateModel);
    document.getElementById('downloadModelBtn').addEventListener('click', downloadModel);
    document.getElementById('downloadReportBtn').addEventListener('click', downloadReport);
    document.getElementById('toggleFileListBtn').addEventListener('click', toggleFileList);
    document.getElementById('refreshWorkspaceBtn').addEventListener('click', refreshWorkspace);
    document.getElementById('resetViewBtn').addEventListener('click', resetViewToNextStage);
    document.getElementById('viewCurrentRunBtn').addEventListener('click', () => setActiveStage('train'));

    document.querySelectorAll('[data-stage-target]').forEach(button => {
        button.addEventListener('click', () => setActiveStage(button.dataset.stageTarget));
    });

    document.querySelectorAll('[data-stage-open]').forEach(button => {
        button.addEventListener('click', () => setActiveStage(button.dataset.stageOpen));
    });

    document.querySelectorAll('.task-option').forEach(option => {
        option.addEventListener('click', () => selectTask(option.dataset.task));
    });

    document.querySelectorAll('.model-option').forEach(option => {
        option.addEventListener('click', () => selectModel(option.dataset.model));
    });

    document.querySelectorAll('.chart-controls input').forEach(checkbox => {
        checkbox.addEventListener('change', toggleChartDataset);
    });

    [
        'numPoints',
        'samplesPerMesh',
        'normalizeCenter',
        'normalize',
        'rotationRange',
        'translationRange',
        'batchSize',
        'numEpochs',
        'learningRate',
        'dropout'
    ].forEach(id => {
        const el = document.getElementById(id);
        el.addEventListener('input', refreshDerivedUi);
        el.addEventListener('change', refreshDerivedUi);
    });
}

async function loadConfig() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/config`);
        const data = await response.json();

        if (!data.success) return;

        currentConfig = data.config;
        const config = data.config;
        const dataCfg = config.data || {};
        const augCfg = config.augmentation || {};
        const trainCfg = config.training || {};
        const modelCfg = config.model || {};

        document.getElementById('numPoints').value = dataCfg.num_points ?? 500;
        document.getElementById('samplesPerMesh').value = dataCfg.samples_per_mesh ?? 100;
        document.getElementById('normalizeCenter').value = String(Boolean(dataCfg.normalize_center));
        document.getElementById('normalize').value = String(Boolean(dataCfg.normalize_scale));
        document.getElementById('rotationRange').value = augCfg.rotation_range ?? 360;
        document.getElementById('translationRange').value = augCfg.translation_range ?? 0;

        if (modelCfg.type && !modelCfg.type.endsWith('_ae')) {
            selectedModel = modelCfg.type;
        }

        syncTaskSpecificControls();
        syncSelectionUi();
        refreshDerivedUi();
    } catch (error) {
        console.error('Config load failed:', error);
    }
}

function getTaskForModel(modelType) {
    return modelType.includes('_ae') ? 'autoencoder' : 'classification';
}

function getDefaultModelForTask(taskType) {
    return taskType === 'autoencoder' ? 'pointnet_ae' : 'pointnet';
}

function getSelectedTrainConfig(config = currentConfig) {
    if (!config) {
        return {};
    }

    if (selectedTask === 'autoencoder') {
        return config.autoencoder?.train || {};
    }

    return config.training || {};
}

function getSelectedModelConfig(config = currentConfig) {
    if (!config) {
        return {};
    }

    if (selectedTask === 'autoencoder') {
        const aeKey = selectedModel.replace('_ae', '');
        return config.autoencoder?.[aeKey] || {};
    }

    return config.model || {};
}

function syncTaskSpecificControls() {
    const trainCfg = getSelectedTrainConfig();
    const modelCfg = getSelectedModelConfig();

    document.getElementById('batchSize').value = trainCfg.batch_size ?? 64;
    document.getElementById('numEpochs').value = trainCfg.num_epochs ?? 120;
    document.getElementById('learningRate').value = trainCfg.learning_rate ?? 0.001;
    document.getElementById('dropout').value = modelCfg.dropout ?? (selectedTask === 'autoencoder' ? 0.1 : 0.5);
}

async function checkServerHealth() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/health`);
        const data = await response.json();
        workflowState.serverHealthy = data.status === 'healthy';
        updateServerStatus(workflowState.serverHealthy);
    } catch (error) {
        console.error('Server health check failed:', error);
        workflowState.serverHealthy = false;
        updateServerStatus(false);
    }
}

function updateServerStatus(isHealthy) {
    const statusIndicator = document.getElementById('serverStatus');
    const statusDot = statusIndicator.querySelector('.status-dot');
    const statusText = statusIndicator.querySelector('span:last-child');

    if (isHealthy) {
        statusDot.style.background = '#58c79b';
        statusText.textContent = 'Server connected';
    } else {
        statusDot.style.background = '#f07a73';
        statusText.textContent = 'Server disconnected';
    }
}

async function checkExistingData() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/data/files`);
        const data = await response.json();
        const infoBox = document.getElementById('dataStatusBox');
        const infoText = document.getElementById('dataStatusText');
        const processedCache = data.processed_cache || null;
        const hasRawFiles = Boolean(data.success && data.count > 0);

        workflowState.isPreprocessed = Boolean(processedCache);

        if (hasRawFiles) {
            workflowState.hasData = true;
            workflowState.files = data.files;
            infoBox.className = 'info-box success';
            infoText.textContent = processedCache
                ? `Found ${data.count} raw mesh files and an existing processed cache. You can retrain immediately or regenerate preprocessing.`
                : `Found ${data.count} existing mesh files in data/raw. You can proceed directly or add more data.`;
            infoBox.style.display = 'block';
            displayFileList(data.files);
        } else if (processedCache) {
            workflowState.hasData = false;
            workflowState.files = [];
            infoBox.className = 'info-box success';
            infoText.textContent = 'Found an existing processed cache. Training is available even if no raw mesh intake is loaded right now.';
            infoBox.style.display = 'block';
            displayFileList([]);
        } else {
            workflowState.hasData = false;
            workflowState.files = [];
            infoBox.className = 'info-box warning';
            infoText.textContent = 'No mesh files detected. Upload raw data to unlock preprocessing.';
            infoBox.style.display = 'block';
            displayFileList([]);
        }

        refreshDerivedUi();
    } catch (error) {
        console.error('Error checking data:', error);
        showToast('Could not inspect existing raw data.', 'error');
    }
}

function handleDragOver(event) {
    event.preventDefault();
    event.currentTarget.classList.add('dragover');
}

function handleDragLeave(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
}

function handleDrop(event) {
    event.preventDefault();
    event.currentTarget.classList.remove('dragover');
    uploadFiles(Array.from(event.dataTransfer.files));
}

function handleFileSelect(event) {
    uploadFiles(Array.from(event.target.files));
    event.target.value = '';
}

async function uploadFiles(files) {
    if (!files.length) return;

    const progressContainer = document.getElementById('uploadProgress');
    const progressFill = document.getElementById('uploadProgressFill');
    const progressText = document.getElementById('uploadProgressText');
    const progressPercent = document.getElementById('uploadProgressPercent');
    const formData = new FormData();

    files.forEach(file => formData.append('files', file));
    progressContainer.style.display = 'block';
    progressText.textContent = 'Uploading mesh files...';

    let simulatedProgress = 0;
    const progressTimer = window.setInterval(() => {
        simulatedProgress = Math.min(simulatedProgress + 12, 90);
        progressFill.style.width = `${simulatedProgress}%`;
        progressPercent.textContent = `${simulatedProgress}%`;
    }, 180);

    try {
        const response = await fetch(`${API_BASE_URL}/api/upload`, {
            method: 'POST',
            body: formData
        });
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Upload failed.');
        }

        progressFill.style.width = '100%';
        progressPercent.textContent = '100%';

        workflowState.hasData = true;
        workflowState.files = files.map(file => ({
            name: file.name,
            size: file.size,
            modified: new Date().toISOString()
        }));

        document.getElementById('dataStatusBox').className = 'info-box success';
        document.getElementById('dataStatusText').textContent =
            `Uploaded ${data.count} files successfully. Dataset intake is ready for preprocessing.`;
        document.getElementById('dataStatusBox').style.display = 'block';

        displayFileList(workflowState.files);
        workflowState.activeStage = 'preprocess';
        refreshDerivedUi();
        showToast(`Uploaded ${data.count} mesh files.`, 'success');
    } catch (error) {
        console.error('Upload error:', error);
        showToast(error.message || 'Upload failed. Please try again.', 'error');
    } finally {
        window.clearInterval(progressTimer);
        window.setTimeout(() => {
            progressContainer.style.display = 'none';
            progressFill.style.width = '0%';
            progressPercent.textContent = '0%';
        }, 1000);
    }
}

function displayFileList(files) {
    const normalizedFiles = files.map(file => typeof file === 'string' ? { name: file } : file);
    const fileList = document.getElementById('fileList');
    const header = document.getElementById('fileListHeader');
    const countSpan = document.getElementById('fileCount');

    fileList.innerHTML = '';

    if (!normalizedFiles.length) {
        header.style.display = 'none';
        document.getElementById('datasetFileCount').textContent = '0';
        document.getElementById('datasetFormatSummary').textContent = 'None yet';
        document.getElementById('datasetReadiness').textContent = 'Blocked';
        return;
    }

    header.style.display = 'flex';
    countSpan.textContent = `${normalizedFiles.length} files in intake`;

    normalizedFiles.forEach(file => {
        const fileItem = document.createElement('div');
        fileItem.className = 'file-item';
        fileItem.innerHTML = `
            <div class="file-info">
                <div class="file-name">${escapeHtml(file.name || 'unknown')}</div>
            </div>
            <div class="file-size">${file.size ? formatBytes(file.size) : 'Ready'}</div>
        `;
        fileList.appendChild(fileItem);
    });

    document.getElementById('datasetFileCount').textContent = String(normalizedFiles.length);
    document.getElementById('datasetFormatSummary').textContent = summarizeExtensions(normalizedFiles);
    document.getElementById('datasetReadiness').textContent = 'Ready';
}

function toggleFileList() {
    const list = document.getElementById('fileList');
    const btn = document.getElementById('toggleFileListBtn');
    const isCollapsed = list.classList.contains('collapsed');

    list.classList.toggle('collapsed', !isCollapsed);
    list.classList.toggle('expanded', isCollapsed);
    btn.textContent = isCollapsed ? 'Collapse list' : 'Show all';
}

async function preprocessData() {
    if (!workflowState.hasData) {
        showToast('Upload data before preprocessing.', 'error');
        return;
    }

    const btn = document.getElementById('preprocessBtn');
    const progressContainer = document.getElementById('preprocessProgress');
    const progressFill = document.getElementById('preprocessProgressFill');
    const progressText = document.getElementById('preprocessProgressText');

    btn.classList.add('loading');
    btn.disabled = true;
    progressContainer.style.display = 'block';
    progressText.textContent = 'Processing dataset cache...';

    let simulatedProgress = 0;
    const progressTimer = window.setInterval(() => {
        simulatedProgress = Math.min(simulatedProgress + 7, 88);
        progressFill.style.width = `${simulatedProgress}%`;
    }, 220);

    try {
        const params = {
            data: {
                num_points: parseInt(document.getElementById('numPoints').value, 10),
                samples_per_mesh: parseInt(document.getElementById('samplesPerMesh').value, 10),
                normalize_center: document.getElementById('normalizeCenter').value === 'true',
                normalize_scale: document.getElementById('normalize').value === 'true'
            },
            augmentation: {
                rotation_range: parseInt(document.getElementById('rotationRange').value, 10),
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

        if (!data.success) {
            throw new Error(data.error || 'Preprocessing failed.');
        }

        progressFill.style.width = '100%';
        progressText.textContent = 'Processed dataset ready';
        workflowState.isPreprocessed = true;
        workflowState.activeStage = 'configure';
        document.getElementById('preprocessStatusBox').className = 'info-box success';
        document.getElementById('preprocessStatusText').textContent =
            'Processed point-cloud cache generated successfully. You can now configure and launch training.';
        document.getElementById('preprocessStatusBox').style.display = 'block';
        refreshDerivedUi();
        showToast('Preprocessing completed successfully.', 'success');
    } catch (error) {
        console.error('Preprocessing error:', error);
        showToast(error.message || 'Preprocessing failed. Please try again.', 'error');
    } finally {
        window.clearInterval(progressTimer);
        btn.classList.remove('loading');
        btn.disabled = !workflowState.hasData || workflowState.isTraining;
        window.setTimeout(() => {
            progressContainer.style.display = 'none';
            progressFill.style.width = '0%';
        }, 1000);
    }
}

function selectTask(taskType) {
    selectedTask = taskType;
    if (taskType === 'classification' && selectedModel.includes('_ae')) {
        selectedModel = currentConfig?.model?.type || getDefaultModelForTask(taskType);
    }
    if (taskType === 'autoencoder' && !selectedModel.includes('_ae')) {
        selectedModel = getDefaultModelForTask(taskType);
    }
    syncTaskSpecificControls();
    syncSelectionUi();
    refreshDerivedUi();
}

function selectModel(modelType) {
    selectedTask = getTaskForModel(modelType);
    selectedModel = modelType;
    syncTaskSpecificControls();
    syncSelectionUi();
    refreshDerivedUi();
}

function syncSelectionUi() {
    document.querySelectorAll('.task-option').forEach(option => {
        option.classList.toggle('selected', option.dataset.task === selectedTask);
    });

    document.querySelectorAll('.model-option').forEach(option => {
        option.classList.toggle('selected', option.dataset.model === selectedModel);
    });

    updateModelVisibility();
    updateRunSummary();
}

function updateModelVisibility() {
    document.getElementById('classificationModels').style.display =
        selectedTask === 'classification' ? 'grid' : 'none';
    document.getElementById('autoencoderModels').style.display =
        selectedTask === 'autoencoder' ? 'grid' : 'none';
}

async function startTraining() {
    if (!workflowState.isPreprocessed) {
        showToast('Complete preprocessing before training.', 'error');
        return;
    }

    const btn = document.getElementById('trainBtn');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const params = {
            model_type: selectedModel,
            training: {
                batch_size: parseInt(document.getElementById('batchSize').value, 10),
                num_epochs: parseInt(document.getElementById('numEpochs').value, 10),
                learning_rate: parseFloat(document.getElementById('learningRate').value)
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

        if (!data.success) {
            throw new Error(data.error || 'Failed to start training.');
        }

        currentJobId = data.job_id;
        workflowState.isTraining = true;
        workflowState.trainingCompleted = false;
        workflowState.reportGenerated = false;
        workflowState.evaluationResults = null;
        workflowState.trainingStatus = 'running';
        workflowState.runStartedAt = Date.now();
        workflowState.activeStage = 'train';
        workflowState.reportPath = null;

        applyTrainingPayloadToCurrentConfig(params);
        clearEvaluationResults();
        initializeTrainingChart();
        updateRunSummary();
        refreshDerivedUi();
        startPollingTrainingStatus();
        showToast(`Training started with ${formatModelName(data.model_type)}.`, 'success');
    } catch (error) {
        console.error('Training error:', error);
        showToast(error.message || 'Failed to start training.', 'error');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = workflowState.isTraining || !workflowState.isPreprocessed;
    }
}

function applyTrainingPayloadToCurrentConfig(params) {
    if (!currentConfig) return;

    if (params.model_type.includes('_ae')) {
        const aeKey = params.model_type.replace('_ae', '');
        currentConfig.autoencoder = currentConfig.autoencoder || {};
        currentConfig.autoencoder.train = {
            ...(currentConfig.autoencoder.train || {}),
            ...(params.training || {})
        };
        currentConfig.training = {
            ...(currentConfig.training || {}),
            ...(params.training || {})
        };
        currentConfig.autoencoder[aeKey] = {
            ...(currentConfig.autoencoder[aeKey] || {}),
            ...(params.model || {})
        };
        return;
    }

    currentConfig.training = {
        ...(currentConfig.training || {}),
        ...(params.training || {})
    };
    currentConfig.model = {
        ...(currentConfig.model || {}),
        type: params.model_type,
        ...(params.model || {})
    };
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
                    borderColor: '#f07a73',
                    backgroundColor: 'rgba(240, 122, 115, 0.12)',
                    tension: 0.28
                },
                {
                    label: 'Val Loss',
                    data: [],
                    borderColor: '#d7ad5f',
                    backgroundColor: 'rgba(215, 173, 95, 0.12)',
                    tension: 0.28
                },
                {
                    label: 'Train Acc',
                    data: [],
                    borderColor: '#58c79b',
                    backgroundColor: 'rgba(88, 199, 155, 0.12)',
                    tension: 0.28,
                    yAxisID: 'y1'
                },
                {
                    label: 'Val Acc',
                    data: [],
                    borderColor: '#66dcff',
                    backgroundColor: 'rgba(102, 220, 255, 0.12)',
                    tension: 0.28,
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
                    title: {
                        display: true,
                        text: 'Loss'
                    },
                    grid: {
                        color: 'rgba(163, 186, 214, 0.08)'
                    }
                },
                y1: {
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
                    display: false
                }
            }
        }
    });
}

function startPollingTrainingStatus() {
    if (pollingInterval) {
        window.clearInterval(pollingInterval);
    }

    pollTrainingStatus();
    pollingInterval = window.setInterval(pollTrainingStatus, POLL_INTERVAL);
}

async function pollTrainingStatus() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/training/status/${currentJobId}`);
        const data = await response.json();

        if (!data.success) return;

        updateTrainingUI(data);

        if (data.status === 'completed' || data.status === 'failed') {
            window.clearInterval(pollingInterval);
            pollingInterval = null;
            onTrainingComplete(data);
        }
    } catch (error) {
        console.error('Polling error:', error);
    }
}

function updateTrainingUI(data) {
    const isAE = (data.model_type || selectedModel || '').includes('_ae');
    const metrics = data.metrics || {};
    const latestIndex = Math.max((metrics.train_loss || []).length - 1, 0);
    const latestTrainLoss = metrics.train_loss?.[latestIndex];
    const latestValLoss = metrics.val_loss?.[latestIndex];
    const latestTrainAcc = metrics.train_acc?.[latestIndex];
    const latestValAcc = metrics.val_acc?.[latestIndex];

    workflowState.trainingStatus = data.status;
    workflowState.totalEpochs = data.total_epochs || workflowState.totalEpochs;
    workflowState.currentEpoch = data.current_epoch || 0;
    workflowState.latestMetrics = data.metrics || null;
    workflowState.isTraining = data.status === 'running' || data.status === 'queued';
    workflowState.trainingCompleted = data.status === 'completed';

    const statusBadge = document.getElementById('trainingStatus');
    statusBadge.textContent = formatStageStatus(data.status);

    document.getElementById('progressFill').style.width = `${Math.round(data.progress || 0)}%`;
    document.getElementById('progressText').textContent = `Epoch ${data.current_epoch || 0}/${data.total_epochs || 0}`;
    document.getElementById('progressPercent').textContent = `${Math.round(data.progress || 0)}%`;

    document.getElementById('trainLoss').textContent = latestTrainLoss != null ? latestTrainLoss.toFixed(isAE ? 6 : 4) : '-';
    document.getElementById('valLoss').textContent = latestValLoss != null ? latestValLoss.toFixed(isAE ? 6 : 4) : '-';
    document.getElementById('trainAcc').textContent = !isAE && latestTrainAcc != null ? `${(latestTrainAcc * 100).toFixed(2)}%` : '-';
    document.getElementById('valAcc').textContent = !isAE && latestValAcc != null ? `${(latestValAcc * 100).toFixed(2)}%` : '-';

    document.getElementById('bestMetric').textContent = isAE
        ? formatBestLoss(metrics.val_loss || [])
        : formatBestAccuracy(metrics.val_acc || []);

    document.getElementById('elapsedTime').textContent = formatElapsed(data.start_time || workflowState.runStartedAt);

    document.getElementById('trainAccCard').style.display = isAE ? 'none' : 'block';
    document.getElementById('valAccCard').style.display = isAE ? 'none' : 'block';
    document.getElementById('trainAccLabel').style.display = isAE ? 'none' : 'inline-flex';
    document.getElementById('valAccLabel').style.display = isAE ? 'none' : 'inline-flex';

    if (!trainingChart) {
        initializeTrainingChart();
    }

    updateChart(metrics, isAE);
    updateRunSummary();
    refreshDerivedUi();
    fetchTrainingLogs();
}

function updateChart(metrics, isAE) {
    if (!trainingChart) return;

    const epochs = Array.from({ length: (metrics.train_loss || []).length }, (_, index) => index + 1);
    trainingChart.data.labels = epochs;
    trainingChart.data.datasets[0].data = metrics.train_loss || [];
    trainingChart.data.datasets[1].data = metrics.val_loss || [];
    trainingChart.data.datasets[2].data = (metrics.train_acc || []).map(value => value * 100);
    trainingChart.data.datasets[3].data = (metrics.val_acc || []).map(value => value * 100);
    trainingChart.options.scales.y1.display = !isAE;
    trainingChart.setDatasetVisibility(2, !isAE && document.querySelector('[data-dataset="2"]').checked);
    trainingChart.setDatasetVisibility(3, !isAE && document.querySelector('[data-dataset="3"]').checked);
    trainingChart.update('none');
}

function toggleChartDataset(event) {
    if (!trainingChart) return;
    trainingChart.setDatasetVisibility(parseInt(event.target.dataset.dataset, 10), event.target.checked);
    trainingChart.update();
}

async function fetchTrainingLogs() {
    if (!currentJobId) return;

    try {
        const response = await fetch(`${API_BASE_URL}/api/training/logs/${currentJobId}`);
        const data = await response.json();
        if (data.success) {
            updateLogsDisplay(data.logs || []);
        }
    } catch (error) {
        console.error('Log fetch error:', error);
    }
}

function updateLogsDisplay(logs) {
    const logsContainer = document.getElementById('trainingLogs');
    logsContainer.innerHTML = logs.map(log => `<div class="log-entry">${escapeHtml(log)}</div>`).join('');
    logsContainer.scrollTop = logsContainer.scrollHeight;
}

function onTrainingComplete(data) {
    workflowState.isTraining = false;
    workflowState.trainingCompleted = data.status === 'completed';
    workflowState.trainingStatus = data.status;

    if (data.status === 'completed') {
        workflowState.activeStage = 'evaluate';
        document.getElementById('latestCheckpoint').textContent = 'Checkpoint available';
        document.getElementById('latestCheckpointCopy').textContent = 'Generate a report or export the best model artifact.';
        showToast('Training completed successfully.', 'success');
    } else {
        workflowState.activeStage = 'train';
        showToast(data.error || 'Training failed.', 'error');
    }

    refreshDerivedUi();
}

async function evaluateModel() {
    if (!currentJobId || !workflowState.trainingCompleted) {
        showToast('Finish a training run before generating evaluation.', 'error');
        return;
    }

    const btn = document.getElementById('evaluateBtn');
    btn.classList.add('loading');
    btn.disabled = true;

    try {
        const response = await fetch(`${API_BASE_URL}/api/report/generate/${currentJobId}`, {
            method: 'POST'
        });
        const data = await response.json();

        if (!data.success) {
            throw new Error(data.error || 'Evaluation failed.');
        }

        workflowState.reportGenerated = true;
        workflowState.reportPath = data.report_path || null;
        workflowState.evaluationResults = data.results || null;
        renderEvaluationResults(data.results || null, data.model_type || selectedModel);
        workflowState.activeStage = 'export';
        refreshDerivedUi();
        showToast('Evaluation report generated.', 'success');
    } catch (error) {
        console.error('Evaluation error:', error);
        showToast(error.message || 'Evaluation failed.', 'error');
    } finally {
        btn.classList.remove('loading');
        btn.disabled = !workflowState.trainingCompleted;
    }
}

function renderEvaluationResults(results, modelType) {
    const resultsGrid = document.getElementById('resultsGrid');
    const resultsNarrative = document.getElementById('resultsNarrative');
    const summaryText = document.getElementById('summaryText-evaluate');
    const isAE = modelType.includes('_ae');

    if (!results) {
        clearEvaluationResults();
        return;
    }

    if (isAE) {
        const chamferDistance = Number(results.chamfer_distance || 0);
        const reportedAccuracy = Number(results.accuracy || 0) * 100;

        resultsGrid.innerHTML = `
            <article class="metric-card">
                <p class="metric-label">Chamfer distance</p>
                <p class="metric-value">${chamferDistance.toFixed(6)}</p>
            </article>
            <article class="metric-card">
                <p class="metric-label">Reported accuracy</p>
                <p class="metric-value">${reportedAccuracy.toFixed(2)}%</p>
            </article>
            <article class="metric-card">
                <p class="metric-label">Epochs tracked</p>
                <p class="metric-value">${workflowState.currentEpoch || workflowState.totalEpochs || 0}</p>
            </article>
        `;
        resultsNarrative.textContent =
            `The reconstruction report recorded a Chamfer distance of ${chamferDistance.toFixed(6)}. Use this value to compare latent-size or architecture changes across runs.`;
        summaryText.textContent =
            `Autoencoder report ready with Chamfer distance ${chamferDistance.toFixed(6)}.`;
    } else {
        const accuracy = Number(results.accuracy || 0) * 100;
        const precision = Number(results.precision || 0) * 100;
        const recall = Number(results.recall || 0) * 100;
        const f1Score = Number(results.f1_score || 0) * 100;

        resultsGrid.innerHTML = `
            <article class="metric-card">
                <p class="metric-label">Accuracy</p>
                <p class="metric-value">${accuracy.toFixed(2)}%</p>
            </article>
            <article class="metric-card">
                <p class="metric-label">Precision</p>
                <p class="metric-value">${precision.toFixed(2)}%</p>
            </article>
            <article class="metric-card">
                <p class="metric-label">Recall</p>
                <p class="metric-value">${recall.toFixed(2)}%</p>
            </article>
            <article class="metric-card">
                <p class="metric-label">F1 score</p>
                <p class="metric-value">${f1Score.toFixed(2)}%</p>
            </article>
        `;
        resultsNarrative.textContent =
            `The evaluation report measured ${accuracy.toFixed(2)}% accuracy with an F1 score of ${f1Score.toFixed(2)}%. Use precision and recall together to judge whether the classifier is balanced across subjects.`;
        summaryText.textContent =
            `Classification report ready with ${accuracy.toFixed(2)}% accuracy and ${f1Score.toFixed(2)}% F1.`;
    }

    document.getElementById('exportStatus').textContent =
        'Checkpoint download is ready now. The JSON evaluation report is also ready for handoff and grading.';
}

function clearEvaluationResults() {
    document.getElementById('resultsGrid').innerHTML = '';
    document.getElementById('resultsNarrative').textContent =
        'Completed runs surface here as insight cards, interpretation notes, and recommended next actions.';
    document.getElementById('summaryText-evaluate').textContent =
        'No evaluation report has been generated.';
    document.getElementById('exportStatus').textContent =
        'The export panel unlocks once a run finishes and a report has been generated.';
}

async function downloadModel() {
    if (!currentJobId || !workflowState.trainingCompleted) {
        showToast('A completed run is required before exporting a model.', 'error');
        return;
    }

    window.open(`${API_BASE_URL}/api/download/model/${currentJobId}`, '_blank');
    showToast('Model download started.', 'success');
}

async function downloadReport() {
    if (!currentJobId || !workflowState.reportGenerated) {
        showToast('Generate an evaluation report before exporting it.', 'error');
        return;
    }

    window.open(`${API_BASE_URL}/api/download/report/${currentJobId}`, '_blank');
    showToast('Report download started.', 'success');
}

async function loadLatestJob() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/training/jobs`);
        const data = await response.json();

        if (!data.success || !Array.isArray(data.jobs) || !data.jobs.length) {
            return;
        }

        applyLatestJob(data.jobs[0]);
    } catch (error) {
        console.error('Job restore failed:', error);
    }
}

function applyLatestJob(job) {
    currentJobId = job.job_id;
    selectedModel = job.model_type || selectedModel;
    selectedTask = getTaskForModel(selectedModel);

    if (job.config) {
        currentConfig = job.config;
    }

    syncTaskSpecificControls();
    syncSelectionUi();

    workflowState.trainingStatus = job.status || 'idle';
    workflowState.totalEpochs = job.total_epochs || 0;
    workflowState.currentEpoch = job.current_epoch || 0;
    workflowState.latestMetrics = job.metrics || null;
    workflowState.evaluationResults = job.evaluation_results || null;
    workflowState.isTraining = job.status === 'running' || job.status === 'queued';
    workflowState.trainingCompleted = job.status === 'completed';
    workflowState.reportGenerated = Boolean(job.report_path);
    workflowState.reportPath = job.report_path || null;
    workflowState.runStartedAt = job.start_time || null;

    if (job.metrics) {
        updateTrainingUI(job);
    } else {
        refreshDerivedUi();
    }

    if (workflowState.evaluationResults) {
        renderEvaluationResults(workflowState.evaluationResults, selectedModel);
    }

    if (workflowState.isTraining) {
        startPollingTrainingStatus();
    }
}

async function refreshWorkspace() {
    await Promise.all([checkServerHealth(), checkExistingData(), loadLatestJob()]);
    if (currentJobId && workflowState.isTraining && !pollingInterval) {
        startPollingTrainingStatus();
    }
    showToast('Workspace refreshed.', 'info');
}

function refreshDerivedUi() {
    updateRunSummary();
    updateReadinessCopy();
    updateIntroPanels();
    updateStageUI();
    renderContextPanel();
}

function updateReadinessCopy() {
    const readiness = document.getElementById('runReadiness');
    const numPoints = parseInt(document.getElementById('numPoints').value, 10);
    const samplesPerMesh = parseInt(document.getElementById('samplesPerMesh').value, 10);
    const batchSize = parseInt(document.getElementById('batchSize').value, 10);
    const lr = parseFloat(document.getElementById('learningRate').value);

    readiness.textContent =
        `Sampling ${numPoints} points per mesh across ${samplesPerMesh} augmented samples. Batch size ${batchSize} with learning rate ${lr} is positioned as a stable first-pass run.`;
}

function updateIntroPanels() {
    document.getElementById('introMode').textContent = humanizeStage(workflowState.activeStage);
    document.getElementById('introCopy').textContent = getStageNarrative(workflowState.activeStage).description;

    document.getElementById('pipelineHealth').textContent = workflowState.isTraining
        ? 'Training in progress'
        : workflowState.trainingCompleted
            ? 'Run completed'
            : workflowState.isPreprocessed
                ? 'Ready for training'
                : workflowState.hasData
                    ? 'Ready for preprocessing'
                    : 'Waiting for data';

    document.getElementById('pipelineHealthCopy').textContent = workflowState.isTraining
        ? `Live metrics are streaming from epoch ${workflowState.currentEpoch}/${workflowState.totalEpochs || '?'}.`
        : workflowState.trainingCompleted
            ? 'Evaluation and export stages are now available.'
            : workflowState.isPreprocessed
                ? 'Configuration is open and the train action is unlocked.'
                : workflowState.hasData
                    ? 'The raw dataset is present, but no processed cache exists yet.'
                    : 'Upload mesh files to start the notebook flow.';
}

function updateRunSummary() {
    document.getElementById('runMetaTask').textContent = selectedTask === 'classification' ? 'Classification' : 'Autoencoder';
    document.getElementById('runMetaModel').textContent = formatModelName(selectedModel);
    document.getElementById('runMetaEpochs').textContent = document.getElementById('numEpochs').value;
    document.getElementById('runMetaBatch').textContent = document.getElementById('batchSize').value;
    document.getElementById('runMetaLr').textContent = document.getElementById('learningRate').value;
    document.getElementById('runMetaJob').textContent = currentJobId || 'Pending';
}

function resetViewToNextStage() {
    workflowState.activeStage = getNextSuggestedStage();
    refreshDerivedUi();
}

function getNextSuggestedStage() {
    if (workflowState.isTraining) return 'train';
    if (workflowState.reportGenerated) return 'export';
    if (workflowState.trainingCompleted) return 'evaluate';
    if (workflowState.isPreprocessed) return 'configure';
    if (workflowState.hasData) return 'preprocess';
    return 'dataset';
}

function setActiveStage(stageKey) {
    if (!canOpenStage(stageKey)) {
        showToast(`${humanizeStage(stageKey)} is still locked by the current workflow state.`, 'info');
        return;
    }

    if (workflowState.isTraining && stageKey !== 'train') {
        showToast('Training is running. Stay in the control center until the run ends.', 'info');
        return;
    }

    workflowState.activeStage = stageKey;
    updateStageUI();
    renderContextPanel();
    updateIntroPanels();
    document.querySelector(`[data-stage="${stageKey}"]`)?.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function updateStageUI() {
    const workspaceShell = document.getElementById('workspaceShell');
    workspaceShell.classList.toggle('training-mode', workflowState.isTraining);

    STAGE_ORDER.forEach(stageKey => {
        const stageEl = document.querySelector(`[data-stage="${stageKey}"]`);
        const railEl = document.querySelector(`[data-stage-target="${stageKey}"]`);
        const openButton = document.querySelector(`[data-stage-open="${stageKey}"]`);
        const status = getStageStatus(stageKey);
        const active = workflowState.activeStage === stageKey;

        stageEl.classList.toggle('is-active', active);
        stageEl.classList.toggle('is-complete', status === 'complete');
        stageEl.classList.toggle('is-locked', status === 'locked');

        railEl.classList.toggle('is-active', active);
        railEl.classList.toggle('is-locked', status === 'locked');
        railEl.querySelector('.rail-step-status').textContent = formatStageStatus(status);

        document.getElementById(`stageStatus-${stageKey}`).textContent = formatStageStatus(status);
        openButton.disabled = !canOpenStage(stageKey);
    });

    document.getElementById('runStatusBadge').textContent = workflowState.isTraining
        ? `Running ${workflowState.currentEpoch}/${workflowState.totalEpochs || '?'}` 
        : workflowState.trainingCompleted
            ? 'Run completed'
            : currentJobId
                ? 'Run prepared'
                : 'No active run';

    document.getElementById('preprocessBtn').disabled = !workflowState.hasData || workflowState.isTraining;
    document.getElementById('trainBtn').disabled = !workflowState.isPreprocessed || workflowState.isTraining;
    document.getElementById('evaluateBtn').disabled = !workflowState.trainingCompleted;
    document.getElementById('downloadModelBtn').disabled = !workflowState.trainingCompleted;
    document.getElementById('downloadReportBtn').disabled = !workflowState.reportGenerated;
    document.getElementById('viewCurrentRunBtn').disabled = !currentJobId;

    document.getElementById('summaryText-dataset').textContent = workflowState.files.length
        ? `${workflowState.files.length} mesh files detected and ready for preprocessing.`
        : 'No mesh files loaded yet.';
    document.getElementById('summaryText-preprocess').textContent = workflowState.isPreprocessed
        ? `${document.getElementById('numPoints').value} points per sample, ${document.getElementById('samplesPerMesh').value} samples per mesh.`
        : 'No preprocessing profile has been applied yet.';
    document.getElementById('summaryText-configure').textContent =
        `${formatModelName(selectedModel)} with batch ${document.getElementById('batchSize').value}, ${document.getElementById('numEpochs').value} epochs, lr ${document.getElementById('learningRate').value}.`;
    document.getElementById('summaryText-train').textContent = currentJobId
        ? `${formatStageStatus(workflowState.trainingStatus)}. Current epoch ${workflowState.currentEpoch}/${workflowState.totalEpochs || '?'}.`
        : 'No training run has started.';
    document.getElementById('summaryText-export').textContent = workflowState.reportGenerated
        ? 'Checkpoint and report are both ready for export.'
        : workflowState.trainingCompleted
            ? 'Checkpoint available. Generate a report to complete the export bundle.'
            : 'Artifacts become available after training and report generation.';

    updateRailSummaries();
}

function updateRailSummaries() {
    document.getElementById('railSummary-dataset').textContent = workflowState.files.length
        ? `${workflowState.files.length} files loaded`
        : 'Awaiting mesh intake.';
    document.getElementById('railSummary-preprocess').textContent = workflowState.isPreprocessed
        ? 'Processed cache ready.'
        : 'Sampling and normalization profile.';
    document.getElementById('railSummary-configure').textContent =
        `${formatModelName(selectedModel)} selected`;
    document.getElementById('railSummary-train').textContent = currentJobId
        ? `${formatStageStatus(workflowState.trainingStatus)}`
        : 'Live run control and monitoring.';
    document.getElementById('railSummary-evaluate').textContent = workflowState.reportGenerated
        ? 'Report generated.'
        : 'Generate findings and inspect metrics.';
    document.getElementById('railSummary-export').textContent = workflowState.reportGenerated
        ? 'Bundle ready for handoff.'
        : 'Bundle checkpoint and report.';
}

function getStageStatus(stageKey) {
    switch (stageKey) {
        case 'dataset':
            return workflowState.hasData ? 'complete' : 'active';
        case 'preprocess':
            if (!workflowState.hasData) return 'locked';
            if (workflowState.isPreprocessed) return 'complete';
            return workflowState.activeStage === 'preprocess' ? 'active' : 'ready';
        case 'configure':
            if (!workflowState.isPreprocessed) return 'locked';
            if (currentJobId) return 'complete';
            return workflowState.activeStage === 'configure' ? 'active' : 'ready';
        case 'train':
            if (!workflowState.isPreprocessed) return 'locked';
            if (workflowState.isTraining) return 'running';
            if (workflowState.trainingCompleted) return 'complete';
            return workflowState.activeStage === 'train' ? 'active' : 'ready';
        case 'evaluate':
            if (!workflowState.trainingCompleted) return 'locked';
            if (workflowState.reportGenerated) return 'complete';
            return workflowState.activeStage === 'evaluate' ? 'active' : 'ready';
        case 'export':
            if (!workflowState.trainingCompleted) return 'locked';
            return workflowState.activeStage === 'export' ? 'active' : 'ready';
        default:
            return 'locked';
    }
}

function canOpenStage(stageKey) {
    return getStageStatus(stageKey) !== 'locked';
}

function renderContextPanel() {
    const narrative = getStageNarrative(workflowState.activeStage);
    document.getElementById('activeStageTitle').textContent = narrative.title;
    document.getElementById('activeStageDescription').textContent = narrative.description;
    document.getElementById('contextDetails').innerHTML = narrative.notes.map(note => `
        <div class="context-note">
            <p class="mini-label">${escapeHtml(note.label)}</p>
            <p>${escapeHtml(note.body)}</p>
        </div>
    `).join('');
}

function getStageNarrative(stageKey) {
    switch (stageKey) {
        case 'dataset':
            return {
                title: 'Dataset Intake',
                description: 'Bring raw meshes into the workspace and verify the notebook has enough input to begin preprocessing.',
                notes: [
                    { label: 'What to do now', body: 'Upload at least one valid mesh file or review the existing raw intake.' },
                    { label: 'Why it matters', body: 'This stage establishes the entire pipeline and determines whether preprocessing can start.' }
                ]
            };
        case 'preprocess':
            return {
                title: 'Preprocess',
                description: 'Shape the point-cloud cache before training by controlling sampling density, normalization, and augmentation.',
                notes: [
                    { label: 'High impact knobs', body: 'Number of points affects memory and speed. Samples per mesh affects preprocessing time and data breadth.' },
                    { label: 'Recommended pattern', body: 'Center clouds when comparing geometry-driven models and add scale normalization only when subject size variance becomes a problem.' }
                ]
            };
        case 'configure':
            return {
                title: 'Configure Experiment',
                description: 'Finalize the run profile for the next experiment by choosing the task, model family, and optimizer settings.',
                notes: [
                    { label: 'Classification vs autoencoder', body: 'Classification optimizes accuracy, while autoencoder runs emphasize reconstruction loss.' },
                    { label: 'Readiness', body: 'Once the model and optimizer are set, the notebook can pivot into live run monitoring.' }
                ]
            };
        case 'train':
            return {
                title: 'Training Control Center',
                description: 'The notebook is now in a live monitoring state with curves, logs, and compact run metadata pinned for quick reading.',
                notes: [
                    { label: 'What to watch', body: 'Look for widening train/validation gaps, unstable losses, or plateaus near the end of the run.' },
                    { label: 'Control rule', body: 'While training is running, earlier stages stay visible as summaries but the control center remains the focal view.' }
                ]
            };
        case 'evaluate':
            return {
                title: 'Evaluate Findings',
                description: 'Turn backend evaluation metrics into a concise result summary with metric cards and interpretation guidance.',
                notes: [
                    { label: 'Expected outcome', body: 'Generate a report once training completes to unlock the full export bundle.' },
                    { label: 'Use this stage for', body: 'Comparing best validation behavior, reviewing final losses, and deciding the next experiment.' }
                ]
            };
        case 'export':
            return {
                title: 'Export Bundle',
                description: 'Collect the best checkpoint and evaluation report in a clean handoff panel.',
                notes: [
                    { label: 'Checkpoint', body: 'Use the trained model artifact for reproducibility or downstream benchmarking.' },
                    { label: 'Report', body: 'The generated report packages the run for grading, sharing, and experiment comparison.' }
                ]
            };
        default:
            return {
                title: 'Notebook',
                description: 'Follow the workflow from raw data to export-ready results.',
                notes: []
            };
    }
}

function showLoading(message) {
    showToast(message, 'info');
}

function hideLoading() {
}

function showSuccess(message) {
    showToast(message, 'success');
}

function showError(message) {
    showToast(message, 'error');
}

function showToast(message, type = 'info') {
    const region = document.getElementById('toastRegion');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    region.appendChild(toast);

    window.setTimeout(() => {
        toast.remove();
    }, type === 'error' ? 4800 : 2800);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function humanizeStage(stageKey) {
    return stageKey.charAt(0).toUpperCase() + stageKey.slice(1);
}

function formatStageStatus(status) {
    if (!status) return 'idle';
    return status.replace('_', ' ');
}

function formatModelName(model) {
    const modelNames = {
        mlp: 'MLP Baseline',
        cnn1d: '1D-CNN',
        pointnet: 'PointNet Tiny',
        mlp_ae: 'MLP Autoencoder',
        pointnet_ae: 'PointNet Autoencoder'
    };

    return modelNames[model] || model;
}

function summarizeExtensions(files) {
    const counts = files.reduce((acc, file) => {
        const name = file.name || file;
        const ext = name.includes('.') ? name.split('.').pop().toUpperCase() : 'UNKNOWN';
        acc[ext] = (acc[ext] || 0) + 1;
        return acc;
    }, {});

    return Object.entries(counts)
        .slice(0, 3)
        .map(([ext, count]) => `${ext} x${count}`)
        .join(', ');
}

function formatBytes(bytes) {
    if (!bytes && bytes !== 0) return 'Ready';
    if (bytes === 0) return '0 B';
    const units = ['B', 'KB', 'MB', 'GB'];
    const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const value = bytes / (1024 ** unitIndex);
    return `${value.toFixed(value >= 10 || unitIndex === 0 ? 0 : 1)} ${units[unitIndex]}`;
}

function formatElapsed(start) {
    if (!start) return '-';
    const startTime = typeof start === 'number' ? start : new Date(start).getTime();
    if (Number.isNaN(startTime)) return '-';
    const deltaSeconds = Math.max(0, Math.floor((Date.now() - startTime) / 1000));
    const minutes = Math.floor(deltaSeconds / 60);
    const seconds = deltaSeconds % 60;
    if (minutes >= 60) {
        const hours = Math.floor(minutes / 60);
        return `${hours}h ${minutes % 60}m`;
    }
    return `${minutes}m ${String(seconds).padStart(2, '0')}s`;
}

function formatBestAccuracy(values) {
    if (!values.length) return '-';
    return `${(Math.max(...values) * 100).toFixed(2)}%`;
}

function formatBestLoss(values) {
    if (!values.length) return '-';
    return Math.min(...values).toFixed(6);
}
