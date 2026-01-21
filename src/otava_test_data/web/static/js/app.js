/**
 * Otava Test Data Visualizer - Frontend Application
 * With Otava change point detection integration
 */

// State
let currentChart = null;
let miniCharts = [];
let generators = {};

// DOM Elements - Data Generation
const generatorSelect = document.getElementById('generator-select');
const lengthInput = document.getElementById('length-input');
const lengthValue = document.getElementById('length-value');
const seedInput = document.getElementById('seed-input');
const dynamicParams = document.getElementById('dynamic-params');

// DOM Elements - Otava Controls
const runOtavaCheckbox = document.getElementById('run-otava-checkbox');
const windowLenInput = document.getElementById('window-len-input');
const maxPvalueInput = document.getElementById('max-pvalue-input');
const toleranceInput = document.getElementById('tolerance-input');

// DOM Elements - Actions
const generateBtn = document.getElementById('generate-btn');
const showAllBtn = document.getElementById('show-all-btn');

// DOM Elements - Info Display
const generatorTitle = document.getElementById('generator-title');
const generatorDescription = document.getElementById('generator-description');
const changePointInfo = document.getElementById('change-point-info');
const mainChartCanvas = document.getElementById('main-chart');

// DOM Elements - Stats
const statsSection = document.getElementById('stats');
const statLength = document.getElementById('stat-length');
const statMean = document.getElementById('stat-mean');
const statStd = document.getElementById('stat-std');
const statCpTruth = document.getElementById('stat-cp-truth');
const statCpDetected = document.getElementById('stat-cp-detected');

// DOM Elements - Accuracy Metrics
const accuracyMetrics = document.getElementById('accuracy-metrics');
const metricPrecision = document.getElementById('metric-precision');
const metricRecall = document.getElementById('metric-recall');
const metricF1 = document.getElementById('metric-f1');
const metricTp = document.getElementById('metric-tp');
const metricFp = document.getElementById('metric-fp');
const metricFn = document.getElementById('metric-fn');

// DOM Elements - Tables
const cpDetail = document.getElementById('change-points-detail');
const truthTableBody = document.getElementById('truth-table-body');
const detectedTableBody = document.getElementById('detected-table-body');

// DOM Elements - Multi-chart
const multiChartContainer = document.getElementById('multi-chart-container');
const chartGrid = document.getElementById('chart-grid');
const summaryStats = document.getElementById('summary-stats');

// Initialize
document.addEventListener('DOMContentLoaded', async () => {
    await loadGenerators();
    setupEventListeners();
    updateGeneratorInfo();
    await generateData();
});

// Load generator metadata
async function loadGenerators() {
    try {
        const response = await fetch('/api/generators');
        generators = await response.json();
    } catch (error) {
        console.error('Failed to load generators:', error);
    }
}

// Setup event listeners
function setupEventListeners() {
    generatorSelect.addEventListener('change', () => {
        updateGeneratorInfo();
        updateDynamicParams();
        generateData();
    });

    lengthInput.addEventListener('input', () => {
        lengthValue.textContent = lengthInput.value;
    });

    lengthInput.addEventListener('change', generateData);
    seedInput.addEventListener('change', generateData);
    generateBtn.addEventListener('click', generateData);
    showAllBtn.addEventListener('click', showAllPatterns);

    // Otava controls
    runOtavaCheckbox.addEventListener('change', generateData);
    windowLenInput.addEventListener('change', generateData);
    maxPvalueInput.addEventListener('change', generateData);
    toleranceInput.addEventListener('change', generateData);
}

// Update generator info display
function updateGeneratorInfo() {
    const name = generatorSelect.value;
    const info = generators[name];

    if (info) {
        generatorTitle.textContent = info.name;
        generatorDescription.textContent = info.description;

        if (info.has_change_points) {
            changePointInfo.classList.remove('hidden');
        } else {
            changePointInfo.classList.add('hidden');
        }
    }
}

// Update dynamic parameter inputs
function updateDynamicParams() {
    const name = generatorSelect.value;
    const info = generators[name];

    dynamicParams.innerHTML = '';

    if (info && info.params) {
        for (const [paramName, paramInfo] of Object.entries(info.params)) {
            const div = document.createElement('div');
            div.className = 'param-group';

            const label = document.createElement('label');
            label.textContent = formatParamName(paramName);
            label.htmlFor = `param-${paramName}`;

            const input = document.createElement('input');
            input.type = 'number';
            input.id = `param-${paramName}`;
            input.name = paramName;
            input.value = paramInfo.default;
            input.min = paramInfo.min;
            input.max = paramInfo.max;
            input.step = paramInfo.step || 1;
            input.addEventListener('change', generateData);

            div.appendChild(label);
            div.appendChild(input);
            dynamicParams.appendChild(div);
        }
    }
}

// Format parameter name for display
function formatParamName(name) {
    return name
        .replace(/_/g, ' ')
        .replace(/\b\w/g, c => c.toUpperCase());
}

// Generate data and update chart
async function generateData() {
    const name = generatorSelect.value;
    const length = lengthInput.value;
    const seed = seedInput.value;
    const runOtava = runOtavaCheckbox.checked;

    // Build query params
    const params = new URLSearchParams({
        length,
        seed,
        run_otava: runOtava,
        window_len: windowLenInput.value,
        max_pvalue: maxPvalueInput.value,
        tolerance: toleranceInput.value,
    });

    // Add dynamic params
    const paramInputs = dynamicParams.querySelectorAll('input');
    paramInputs.forEach(input => {
        params.append(input.name, input.value);
    });

    try {
        document.body.classList.add('loading');

        const response = await fetch(`/api/generate/${name}?${params}`);
        const data = await response.json();

        if (data.error) {
            alert(`Error: ${data.error}`);
            return;
        }

        updateChart(data);
        updateStats(data);
        updateAccuracyMetrics(data);
        updateComparisonTables(data);

        // Hide multi-chart view when generating single
        multiChartContainer.classList.add('hidden');
        document.querySelector('.chart-container').classList.remove('hidden');
        document.querySelector('.chart-legend').classList.remove('hidden');
        statsSection.classList.remove('hidden');
        accuracyMetrics.classList.remove('hidden');
        cpDetail.classList.remove('hidden');

    } catch (error) {
        console.error('Failed to generate data:', error);
    } finally {
        document.body.classList.remove('loading');
    }
}

// Update the main chart with ground truth and Otava detected points
function updateChart(data) {
    const ctx = mainChartCanvas.getContext('2d');

    // Destroy existing chart
    if (currentChart) {
        currentChart.destroy();
    }

    // Prepare data
    const labels = data.data.map((_, i) => i);
    const values = data.data;

    // Get change point indices
    const groundTruthIndices = data.ground_truth?.indices || data.change_point_indices || [];
    const detectedIndices = data.otava?.detected_indices || [];

    // Determine matched pairs for coloring
    const matchedPairs = data.accuracy?.matched_pairs || [];
    const matchedDetected = new Set(matchedPairs.map(p => p.detected));
    const matchedTruth = new Set(matchedPairs.map(p => p.ground_truth));

    // Create point styles
    const pointBackgroundColors = values.map((_, i) => {
        if (groundTruthIndices.includes(i) && detectedIndices.includes(i)) {
            return '#10b981'; // Both - green (matched)
        } else if (groundTruthIndices.includes(i)) {
            if (matchedTruth.has(i)) {
                return '#10b981'; // Ground truth matched
            }
            return '#10b981'; // Ground truth (will show as missed in table)
        } else if (detectedIndices.includes(i)) {
            if (matchedDetected.has(i)) {
                return '#3b82f6'; // Detected and matched - blue
            }
            return '#ef4444'; // False positive - red
        }
        return 'transparent';
    });

    const pointBorderColors = values.map((_, i) => {
        if (groundTruthIndices.includes(i)) {
            return '#059669'; // Green border for ground truth
        } else if (detectedIndices.includes(i)) {
            if (matchedDetected.has(i)) {
                return '#2563eb'; // Blue border for matched
            }
            return '#dc2626'; // Red border for false positive
        }
        return 'transparent';
    });

    const pointRadii = values.map((_, i) => {
        if (groundTruthIndices.includes(i) || detectedIndices.includes(i)) {
            return 8;
        }
        return 0;
    });

    const pointStyles = values.map((_, i) => {
        if (groundTruthIndices.includes(i) && !detectedIndices.includes(i)) {
            return 'triangle'; // Ground truth only
        } else if (detectedIndices.includes(i) && !groundTruthIndices.includes(i)) {
            return 'rect'; // Detected only
        }
        return 'circle'; // Both or neither
    });

    currentChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [{
                label: data.generator,
                data: values,
                borderColor: '#94a3b8',
                backgroundColor: 'rgba(148, 163, 184, 0.1)',
                borderWidth: 1.5,
                fill: true,
                tension: 0,
                pointBackgroundColor: pointBackgroundColors,
                pointBorderColor: pointBorderColors,
                pointBorderWidth: 2,
                pointRadius: pointRadii,
                pointHoverRadius: 10,
                pointStyle: pointStyles,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    callbacks: {
                        label: function(context) {
                            const idx = context.dataIndex;
                            let label = `Value: ${context.parsed.y.toFixed(2)}`;

                            if (groundTruthIndices.includes(idx)) {
                                const cp = data.ground_truth?.change_points?.find(cp => cp.index === idx);
                                if (cp) {
                                    label += ` | Ground Truth: ${cp.type}`;
                                }
                            }

                            if (detectedIndices.includes(idx)) {
                                const detected = data.otava?.detected_change_points?.find(d => d.index === idx);
                                if (detected) {
                                    label += ` | Otava: p=${detected.pvalue.toExponential(2)}`;
                                }
                            }

                            return label;
                        }
                    }
                }
            },
            scales: {
                x: {
                    title: {
                        display: true,
                        text: 'Index',
                    },
                    grid: {
                        display: false,
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: 'Value',
                    },
                    grid: {
                        color: 'rgba(0, 0, 0, 0.05)',
                    }
                }
            },
            interaction: {
                intersect: false,
                mode: 'index',
            }
        }
    });
}

// Update statistics display
function updateStats(data) {
    const values = data.data;
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    const std = Math.sqrt(
        values.reduce((acc, val) => acc + Math.pow(val - mean, 2), 0) / values.length
    );

    statLength.textContent = values.length;
    statMean.textContent = mean.toFixed(2);
    statStd.textContent = std.toFixed(2);
    statCpTruth.textContent = data.ground_truth?.count ?? data.change_points?.length ?? 0;
    statCpDetected.textContent = data.otava?.count ?? '-';
}

// Update accuracy metrics display
function updateAccuracyMetrics(data) {
    if (!data.accuracy) {
        metricPrecision.textContent = '-';
        metricRecall.textContent = '-';
        metricF1.textContent = '-';
        metricTp.textContent = '-';
        metricFp.textContent = '-';
        metricFn.textContent = '-';
        return;
    }

    const acc = data.accuracy;
    metricPrecision.textContent = (acc.precision * 100).toFixed(0) + '%';
    metricRecall.textContent = (acc.recall * 100).toFixed(0) + '%';
    metricF1.textContent = (acc.f1_score * 100).toFixed(0) + '%';
    metricTp.textContent = acc.true_positives;
    metricFp.textContent = acc.false_positives;
    metricFn.textContent = acc.false_negatives;
}

// Update comparison tables
function updateComparisonTables(data) {
    truthTableBody.innerHTML = '';
    detectedTableBody.innerHTML = '';

    const groundTruth = data.ground_truth?.change_points || data.change_points || [];
    const detected = data.otava?.detected_change_points || [];
    const matchedPairs = data.accuracy?.matched_pairs || [];

    const matchedTruthIndices = new Set(matchedPairs.map(p => p.ground_truth));
    const matchedDetectedIndices = new Set(matchedPairs.map(p => p.detected));

    // Ground truth table
    if (groundTruth.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="4" class="empty-message">No ground truth change points</td>';
        truthTableBody.appendChild(row);
    } else {
        groundTruth.forEach(cp => {
            const matched = matchedTruthIndices.has(cp.index);
            const matchInfo = matchedPairs.find(p => p.ground_truth === cp.index);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${cp.index}</strong></td>
                <td>${cp.type}</td>
                <td>${cp.description || '-'}</td>
                <td class="${matched ? 'status-matched' : 'status-missed'}">
                    ${matched ? `Yes (at ${matchInfo.detected})` : 'No'}
                </td>
            `;
            truthTableBody.appendChild(row);
        });
    }

    // Detected table
    if (detected.length === 0) {
        const row = document.createElement('tr');
        row.innerHTML = '<td colspan="5" class="empty-message">No change points detected by Otava</td>';
        detectedTableBody.appendChild(row);
    } else {
        detected.forEach(cp => {
            const isTP = matchedDetectedIndices.has(cp.index);
            const row = document.createElement('tr');
            row.innerHTML = `
                <td><strong>${cp.index}</strong></td>
                <td>${cp.mean_before.toFixed(2)}</td>
                <td>${cp.mean_after.toFixed(2)}</td>
                <td>${cp.pvalue.toExponential(2)}</td>
                <td class="${isTP ? 'status-tp' : 'status-fp'}">
                    ${isTP ? 'True Positive' : 'False Positive'}
                </td>
            `;
            detectedTableBody.appendChild(row);
        });
    }
}

// Show all patterns with Otava comparison
async function showAllPatterns() {
    const length = lengthInput.value;
    const seed = seedInput.value;
    const windowLen = windowLenInput.value;
    const maxPvalue = maxPvalueInput.value;
    const tolerance = toleranceInput.value;

    try {
        document.body.classList.add('loading');

        // Fetch all generators with Otava analysis
        const results = {};
        let totalTP = 0, totalFP = 0, totalFN = 0;
        let patternsWithCP = 0;

        for (const name of Object.keys(generators)) {
            const params = new URLSearchParams({
                length,
                seed,
                window_len: windowLen,
                max_pvalue: maxPvalue,
                tolerance,
            });

            const response = await fetch(`/api/analyze/${name}?${params}`);
            const data = await response.json();

            if (!data.error) {
                results[name] = data;

                if (data.accuracy) {
                    totalTP += data.accuracy.true_positives;
                    totalFP += data.accuracy.false_positives;
                    totalFN += data.accuracy.false_negatives;
                    if (data.ground_truth?.count > 0) {
                        patternsWithCP++;
                    }
                }
            }
        }

        // Hide single chart view
        document.querySelector('.chart-container').classList.add('hidden');
        document.querySelector('.chart-legend').classList.add('hidden');
        statsSection.classList.add('hidden');
        accuracyMetrics.classList.add('hidden');
        cpDetail.classList.add('hidden');

        // Show multi-chart view
        multiChartContainer.classList.remove('hidden');

        // Update summary stats
        const overallPrecision = totalTP / (totalTP + totalFP) || 0;
        const overallRecall = totalTP / (totalTP + totalFN) || 0;
        const overallF1 = 2 * overallPrecision * overallRecall / (overallPrecision + overallRecall) || 0;

        summaryStats.innerHTML = `
            <div class="summary-stat">
                <h4>Patterns</h4>
                <span>${Object.keys(results).length}</span>
            </div>
            <div class="summary-stat">
                <h4>With CPs</h4>
                <span>${patternsWithCP}</span>
            </div>
            <div class="summary-stat">
                <h4>Total TP</h4>
                <span>${totalTP}</span>
            </div>
            <div class="summary-stat">
                <h4>Total FP</h4>
                <span>${totalFP}</span>
            </div>
            <div class="summary-stat">
                <h4>Total FN</h4>
                <span>${totalFN}</span>
            </div>
            <div class="summary-stat">
                <h4>Overall Precision</h4>
                <span>${(overallPrecision * 100).toFixed(0)}%</span>
            </div>
            <div class="summary-stat">
                <h4>Overall Recall</h4>
                <span>${(overallRecall * 100).toFixed(0)}%</span>
            </div>
            <div class="summary-stat">
                <h4>Overall F1</h4>
                <span>${(overallF1 * 100).toFixed(0)}%</span>
            </div>
        `;

        // Clear existing mini charts
        chartGrid.innerHTML = '';
        miniCharts.forEach(chart => chart.destroy());
        miniCharts = [];

        // Create mini charts for each generator
        for (const [name, data] of Object.entries(results)) {
            if (data.error) continue;

            const info = generators[name];

            const div = document.createElement('div');
            div.className = 'mini-chart';

            const title = document.createElement('h4');
            title.textContent = info ? info.name : name;

            const desc = document.createElement('p');
            desc.textContent = info ? info.description : '';

            const canvas = document.createElement('canvas');

            // Accuracy indicator
            const accuracyDiv = document.createElement('div');
            accuracyDiv.className = 'accuracy-indicator';

            if (data.accuracy && data.ground_truth?.count > 0) {
                const f1 = data.accuracy.f1_score;
                const colorClass = f1 >= 0.8 ? 'accuracy-good' : f1 >= 0.5 ? 'accuracy-medium' : 'accuracy-poor';
                accuracyDiv.innerHTML = `
                    <span>Truth: ${data.ground_truth.count}</span>
                    <span>Detected: ${data.otava?.count || 0}</span>
                    <span class="${colorClass}">F1: ${(f1 * 100).toFixed(0)}%</span>
                `;
            } else {
                accuracyDiv.innerHTML = `
                    <span>No CPs</span>
                    <span>Detected: ${data.otava?.count || 0}</span>
                    <span>${data.otava?.count > 0 ? 'FPs' : 'OK'}</span>
                `;
            }

            div.appendChild(title);
            div.appendChild(desc);
            div.appendChild(canvas);
            div.appendChild(accuracyDiv);
            chartGrid.appendChild(div);

            // Create mini chart
            const ctx = canvas.getContext('2d');
            const groundTruthIndices = data.ground_truth?.indices || [];
            const detectedIndices = data.otava?.detected_indices || [];
            const matchedPairs = data.accuracy?.matched_pairs || [];
            const matchedDetected = new Set(matchedPairs.map(p => p.detected));

            const pointBackgroundColors = data.data.map((_, i) => {
                if (groundTruthIndices.includes(i)) {
                    return '#10b981';
                } else if (detectedIndices.includes(i)) {
                    return matchedDetected.has(i) ? '#3b82f6' : '#ef4444';
                }
                return 'transparent';
            });

            const pointRadii = data.data.map((_, i) =>
                groundTruthIndices.includes(i) || detectedIndices.includes(i) ? 5 : 0
            );

            const chart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.data.map((_, i) => i),
                    datasets: [{
                        data: data.data,
                        borderColor: '#94a3b8',
                        backgroundColor: 'rgba(148, 163, 184, 0.1)',
                        borderWidth: 1,
                        fill: true,
                        tension: 0,
                        pointBackgroundColor: pointBackgroundColors,
                        pointBorderColor: pointBackgroundColors,
                        pointRadius: pointRadii,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false },
                    },
                    scales: {
                        x: { display: false },
                        y: { display: true, grid: { display: false } }
                    },
                    interaction: {
                        intersect: false,
                        mode: 'index',
                    }
                }
            });

            miniCharts.push(chart);

            // Click to view in main chart
            div.style.cursor = 'pointer';
            div.addEventListener('click', () => {
                generatorSelect.value = name;
                updateGeneratorInfo();
                updateDynamicParams();
                generateData();
                window.scrollTo({ top: 0, behavior: 'smooth' });
            });
        }

    } catch (error) {
        console.error('Failed to load all patterns:', error);
    } finally {
        document.body.classList.remove('loading');
    }
}

// Initialize dynamic params on load
updateDynamicParams();
