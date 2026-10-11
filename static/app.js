/**
 * DeepTemp: Microclimate Temperature Client Controller
 * Focuses 100% on verifiable ground-truth temperature forecasting and diagnostics.
 */

let currentCity = "islamabad";
let currentTargetDate = "2026-10-12"; // Default tomorrow
let mainChartInstance = null;

document.addEventListener("DOMContentLoaded", () => {
    initCitySelector();
    initDateControls();
    loadForecastAndInspection(currentCity, currentTargetDate);
});

// Regional City Selector
function initCitySelector() {
    const cityButtons = document.querySelectorAll("#city-selector .city-pill");
    cityButtons.forEach(btn => {
        btn.addEventListener("click", () => {
            cityButtons.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            currentCity = btn.getAttribute("data-city");
            loadForecastAndInspection(currentCity, currentTargetDate);
        });
    });
}

// Date Controls & Presets
function initDateControls() {
    const dateInput = document.getElementById("input-target-date");
    dateInput.addEventListener("change", (e) => {
        currentTargetDate = e.target.value;
        clearPresetActives();
        loadForecastAndInspection(currentCity, currentTargetDate);
    });

    const presets = document.querySelectorAll(".quick-jump-presets .preset-btn");
    presets.forEach(btn => {
        btn.addEventListener("click", () => {
            clearPresetActives();
            btn.classList.add("active");

            if (btn.id === "preset-random") {
                // Random historical date between 2000-06-01 and 2023-11-30
                const start = new Date(2000, 5, 1).getTime();
                const end = new Date(2023, 10, 30).getTime();
                const rand = new Date(start + Math.random() * (end - start));
                currentTargetDate = rand.toISOString().slice(0, 10);
            } else {
                currentTargetDate = btn.getAttribute("data-date");
            }

            dateInput.value = currentTargetDate;
            loadForecastAndInspection(currentCity, currentTargetDate);
        });
    });
}

function clearPresetActives() {
    document.querySelectorAll(".quick-jump-presets .preset-btn").forEach(b => b.classList.remove("active"));
}

// Fetch Unified Forecast and Inspection Data
async function loadForecastAndInspection(city, dateStr) {
    try {
        const response = await fetch(`/api/forecast-and-inspect?city=${city}&date=${dateStr}`);
        if (!response.ok) throw new Error("Forecast API error");
        const data = await response.json();

        updateDashboardUI(data);
    } catch (err) {
        console.error("Dashboard error:", err);
    }
}

// Update DOM with Clean Typography and Diagnostic Data
function updateDashboardUI(data) {
    const isHistorical = data.is_historical;
    const dateBadge = document.getElementById("date-mode-badge");

    // Header Mode Badge
    if (isHistorical) {
        dateBadge.textContent = "HISTORICAL COMPARATIVE ANALYSIS";
        dateBadge.className = "date-mode-tag historical";
        document.getElementById("chip-target-date").textContent = `Verified: ${data.target_date}`;
    } else {
        dateBadge.textContent = data.mode === "today" ? "LIVE TODAY FORECAST" : "FUTURE PREDICTION HORIZON";
        dateBadge.className = "date-mode-tag";
        document.getElementById("chip-target-date").textContent = `Target: ${data.target_date}`;
    }

    // Chart Title
    const stationName = data.city_meta ? data.city_meta.station_name : data.city;
    document.getElementById("chart-panel-title").textContent = isHistorical
        ? `30-Day Sequence Prior to ${data.target_date} • ${stationName}`
        : `Live 30-Day Sequence & Forward Forecast • ${stationName}`;

    // 1. TEMPERATURE SECTION
    document.getElementById("temp-pred-val").innerHTML = `${data.temperature.predicted}<span class="unit">°C</span>`;

    const ratingBadge = document.getElementById("temp-rating-badge");
    const accuracyVerdictBadge = document.getElementById("accuracy-verdict-badge");
    const statDeviation = document.getElementById("stat-deviation");
    const statDevDesc = document.getElementById("stat-deviation-desc");
    const statCiRange = document.getElementById("stat-ci-range");
    const statTestRmse = document.getElementById("stat-test-rmse");
    const devBar = document.getElementById("deviation-bar");
    const devRatioLabel = document.getElementById("deviation-ratio-label");

    statTestRmse.textContent = `±${data.temperature.model_rmse}°C`;

    if (isHistorical) {
        // Temperature Confidence
        document.getElementById("temp-confidence-text").textContent = `95% Confidence: [${data.temperature.confidence_lower}°C to ${data.temperature.confidence_upper}°C]`;

        // Actual Recorded Thermometer
        document.getElementById("label-actual-temp").textContent = "Actual Recorded Thermometer";
        document.getElementById("temp-actual-val").innerHTML = `${data.temperature.actual}<span class="unit">°C</span>`;
        const sign = data.temperature.error >= 0 ? "+" : "";
        document.getElementById("temp-delta-text").textContent = `Absolute Deviation: ${sign}${data.temperature.error}°C`;

        // Rating Badge
        ratingBadge.textContent = data.temperature.accuracy_rating;
        ratingBadge.className = `rating-tag ${data.temperature.rating_color}`;

        accuracyVerdictBadge.textContent = data.temperature.accuracy_rating;
        accuracyVerdictBadge.className = `rating-tag ${data.temperature.rating_color}`;

        statDeviation.textContent = `${sign}${data.temperature.error}°C`;
        statDevDesc.textContent = `Absolute Error: ${data.temperature.abs_error}°C`;
        statCiRange.textContent = `[${data.temperature.confidence_lower}°C to ${data.temperature.confidence_upper}°C]`;

        // Visual Deviation Gauge
        const sigmaRatio = Math.min(100, Math.round((data.temperature.abs_error / data.temperature.model_rmse) * 100));
        devBar.style.width = `${sigmaRatio}%`;
        devBar.style.backgroundColor = data.temperature.abs_error <= 1.0 ? "var(--accent-emerald)" : (data.temperature.abs_error <= 2.0 ? "var(--accent-amber)" : "var(--accent-rose)");
        devRatioLabel.textContent = `${data.temperature.abs_error}°C (${sigmaRatio}% of 1σ bound)`;

        // API Indicator Badge
        const apiBadge = document.getElementById("api-source-badge");
        if (apiBadge && data.source_api) {
            apiBadge.innerHTML = `<span class="api-pulse-dot verified"></span> ${data.source_api.badge_text}`;
            apiBadge.className = "api-source-badge verified";
            apiBadge.title = `${data.source_api.provider} (${data.source_api.type})`;
        }

    } else {
        // Future / Today
        document.getElementById("temp-confidence-text").textContent = `95% Confidence: [${data.temperature.confidence_lower}°C to ${data.temperature.confidence_upper}°C]`;

        document.getElementById("label-actual-temp").textContent = "Actual Thermometer Recording";
        document.getElementById("temp-actual-val").innerHTML = `--<span class="unit">°C</span>`;
        document.getElementById("temp-delta-text").textContent = data.temperature.status_label || "Observation Pending (Future Date)";

        ratingBadge.textContent = `LIVE INFERENCE (±${data.temperature.model_rmse}°C RMSE)`;
        ratingBadge.className = "rating-tag";

        accuracyVerdictBadge.textContent = "FUTURE HORIZON";
        accuracyVerdictBadge.className = "rating-tag";

        statDeviation.textContent = "Pending";
        statDevDesc.textContent = "Ground-Truth After Target Date";
        statCiRange.textContent = `[${data.temperature.confidence_lower}°C to ${data.temperature.confidence_upper}°C]`;

        devBar.style.width = "0%";
        devRatioLabel.textContent = "Pending Ground-Truth Recording";

        // API Indicator Badge
        const apiBadge = document.getElementById("api-source-badge");
        if (apiBadge && data.source_api) {
            apiBadge.innerHTML = `<span class="api-pulse-dot live"></span> ${data.source_api.badge_text}`;
            apiBadge.className = "api-source-badge live";
            apiBadge.title = `${data.source_api.provider} (${data.source_api.type})`;
        }
    }

    // 2. 30-DAY SEQUENCE INPUT PROFILE
    if (data.sequence_stats) {
        document.getElementById("station-meta-badge").textContent = `${data.city_meta.station_name} • ${data.city_meta.elevation}`;
        document.getElementById("seq-mean").textContent = `${data.sequence_stats.mean}°C`;
        document.getElementById("seq-range").textContent = `${data.sequence_stats.min}°C to ${data.sequence_stats.max}°C`;
        document.getElementById("seq-trend").textContent = data.sequence_stats.trend;
        document.getElementById("seq-volatility").textContent = `±${data.sequence_stats.std_dev}°C`;
    }

    // Render Sequence Chart
    renderSequenceChart(data);
}

// Render 30-Day Sequence & Forward Trajectory Chart
function renderSequenceChart(data) {
    const ctx = document.getElementById("mainSequenceChart").getContext("2d");
    if (mainChartInstance) mainChartInstance.destroy();

    const isHistorical = data.is_historical;
    const trailDates = data.trail_dates;
    const trailTemps = data.trail_temps;

    let allLabels = [];
    let datasets = [];

    if (isHistorical) {
        // Historical: 30 days + Target Date point
        allLabels = [...trailDates, data.target_date];
        const observedLine = [...trailTemps, null];

        const predPoint = new Array(trailDates.length).fill(null);
        predPoint.push(data.temperature.predicted);

        const actualPoint = new Array(trailDates.length).fill(null);
        actualPoint.push(data.temperature.actual);

        datasets = [
            {
                label: "Past 30 Days Observed (°C)",
                data: observedLine,
                borderColor: "#38bdf8",
                backgroundColor: "rgba(56, 189, 248, 0.08)",
                borderWidth: 2,
                pointRadius: 2.5,
                pointBackgroundColor: "#38bdf8",
                tension: 0.2,
                fill: true
            },
            {
                label: "LSTM Predicted (°C)",
                data: predPoint,
                borderColor: "#f59e0b",
                backgroundColor: "#f59e0b",
                borderWidth: 0,
                pointRadius: 7,
                pointHoverRadius: 9,
                showLine: false
            },
            {
                label: "Actual Recorded Thermometer (°C)",
                data: actualPoint,
                borderColor: "#10b981",
                backgroundColor: "#10b981",
                borderWidth: 0,
                pointRadius: 7,
                pointHoverRadius: 9,
                showLine: false
            }
        ];

        document.getElementById("legend-actual-item").style.display = "flex";

    } else {
        // Future / Today: 30 days + Forward rollout curve
        const futureDates = data.forecast_trajectory_dates;
        const futureTemps = data.forecast_trajectory_temps;

        allLabels = [...trailDates, ...futureDates];

        // 30 days observed line
        const observedLine = [...trailTemps];
        for (let i = 0; i < futureDates.length; i++) observedLine.push(null);

        // Forward forecast line (connects from last observed day to future)
        const forecastLine = new Array(trailDates.length - 1).fill(null);
        forecastLine.push(trailTemps[trailTemps.length - 1]); // anchor to last observed
        for (let t of futureTemps) forecastLine.push(t);

        datasets = [
            {
                label: "Past 30 Days Observed (°C)",
                data: observedLine,
                borderColor: "#38bdf8",
                backgroundColor: "rgba(56, 189, 248, 0.08)",
                borderWidth: 2,
                pointRadius: 2.5,
                pointBackgroundColor: "#38bdf8",
                tension: 0.2,
                fill: true
            },
            {
                label: "Forward Forecast Trajectory (°C)",
                data: forecastLine,
                borderColor: "#f59e0b",
                backgroundColor: "rgba(245, 158, 11, 0.05)",
                borderWidth: 2.5,
                borderDash: [5, 5],
                pointRadius: 4,
                pointBackgroundColor: "#f59e0b",
                tension: 0.2,
                fill: false
            }
        ];

        document.getElementById("legend-actual-item").style.display = "none";
    }

    mainChartInstance = new Chart(ctx, {
        type: "line",
        data: {
            labels: allLabels,
            datasets: datasets
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: "index",
                intersect: false
            },
            plugins: {
                legend: {
                    display: false // Using custom HTML legend strip
                },
                tooltip: {
                    backgroundColor: "rgba(11, 17, 32, 0.95)",
                    titleColor: "#f8fafc",
                    bodyColor: "#94a3b8",
                    borderColor: "rgba(255, 255, 255, 0.1)",
                    borderWidth: 1,
                    padding: 10,
                    boxPadding: 4
                }
            },
            scales: {
                x: {
                    grid: {
                        color: "rgba(255, 255, 255, 0.04)"
                    },
                    ticks: {
                        color: "#64748b",
                        maxTicksLimit: 10,
                        font: { size: 10 }
                    }
                },
                y: {
                    grid: {
                        color: "rgba(255, 255, 255, 0.04)"
                    },
                    ticks: {
                        color: "#64748b",
                        font: { size: 11 },
                        callback: function(v) { return v + "°C"; }
                    }
                }
            }
        }
    });
}
