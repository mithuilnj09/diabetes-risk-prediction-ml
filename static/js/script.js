/**
 * script.js
 * High-Performance, Lightweight Frontend Logic for Diabetes Risk Prediction.
 * Optimized: Zero duplicate code, AbortController request-cancellation,
 * Chart.js canvas lifecycle management, and lightweight transitions.
 */

document.addEventListener("DOMContentLoaded", () => {
  // Page-specific initialization: only run what exists
  if (document.getElementById("predictionForm")) {
    initFormDualControls();
    initSteppers();
    initDemoDataButtons();
    initResetDefaultsButton();
    initInPlacePredictionSubmission();
    initLiveRiskHUD();
  }

  if (document.getElementById("animatedProbabilityCounter")) {
    initAnimatedCounters();
  }

  if (document.querySelector(".dashboard-tabs-bar")) {
    initDashboardTabs();
  }

  if (document.getElementById("classDistChart")) {
    initDashboardCharts();
  }
});

/**
 * Synchronize Range Sliders & Number Inputs with Zero-Lag
 */
function initFormDualControls() {
  const form = document.getElementById("predictionForm");
  if (!form) return;

  const numberInputs = form.querySelectorAll("input.form-input-control[type='number']");
  numberInputs.forEach((numInput) => {
    const slider = document.getElementById(`slider_${numInput.name}`);
    if (slider) {
      slider.addEventListener("input", (e) => {
        numInput.value = e.target.value;
        validateSingleInput(numInput);
        triggerLiveRiskEvaluation();
      });

      numInput.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        if (!isNaN(val)) {
          slider.value = val;
        }
        validateSingleInput(numInput);
        triggerLiveRiskEvaluation();
      });
    } else {
      numInput.addEventListener("input", () => {
        validateSingleInput(numInput);
        triggerLiveRiskEvaluation();
      });
    }
  });
}

/**
 * Micro-Stepper Controls ([−] and [+])
 */
function initSteppers() {
  const steppers = document.querySelectorAll(".btn-stepper");
  steppers.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.dataset.target;
      const stepDir = parseFloat(btn.dataset.stepDir || 1);
      const stepVal = parseFloat(btn.dataset.step || 1);
      const input = document.getElementById(targetId);

      if (input) {
        let curVal = parseFloat(input.value);
        if (isNaN(curVal)) curVal = 0;

        const min = parseFloat(input.min);
        const max = parseFloat(input.max);
        let newVal = curVal + stepDir * stepVal;

        if (stepVal < 1) {
          newVal = Math.round(newVal * 100) / 100;
        } else {
          newVal = Math.round(newVal);
        }

        if (!isNaN(min) && newVal < min) newVal = min;
        if (!isNaN(max) && newVal > max) newVal = max;

        input.value = newVal;
        const slider = document.getElementById(`slider_${input.name}`);
        if (slider) slider.value = newVal;

        validateSingleInput(input);
        triggerLiveRiskEvaluation();
      }
    });
  });
}

function validateSingleInput(input) {
  const valStr = input.value.trim();
  const min = parseFloat(input.getAttribute("min"));
  const max = parseFloat(input.getAttribute("max"));
  const val = parseFloat(valStr);

  let errorMsg = "";

  if (valStr === "") {
    errorMsg = "Required.";
  } else if (isNaN(val)) {
    errorMsg = "Enter valid number.";
  } else if (!isNaN(min) && val < min) {
    errorMsg = `Min: ${min}`;
  } else if (!isNaN(max) && val > max) {
    errorMsg = `Max: ${max}`;
  }

  const group = input.closest(".form-group");
  const helper = group ? group.querySelector(".form-helper") : null;

  if (errorMsg) {
    input.classList.add("is-invalid");
    input.style.borderColor = "#f43f5e";
    if (helper) {
      helper.dataset.originalText = helper.dataset.originalText || helper.textContent;
      helper.textContent = errorMsg;
      helper.style.color = "#f43f5e";
    }
    return false;
  } else {
    input.classList.remove("is-invalid");
    input.style.borderColor = "";
    if (helper && helper.dataset.originalText) {
      helper.textContent = helper.dataset.originalText;
      helper.style.color = "";
    }
    return true;
  }
}

/**
 * Real-Time Live Risk Evaluation HUD with Request Cancellation (AbortController)
 */
let debounceRiskTimer = null;
let activeRiskAbortController = null;

function triggerLiveRiskEvaluation() {
  clearTimeout(debounceRiskTimer);
  debounceRiskTimer = setTimeout(computeLiveRisk, 150);
}

function initLiveRiskHUD() {
  computeLiveRisk();
}

async function computeLiveRisk() {
  const form = document.getElementById("predictionForm");
  const hudGauge = document.getElementById("hudLiveGauge");
  const hudLabel = document.getElementById("hudLiveLabel");
  const hudProb = document.getElementById("hudLiveProb");
  if (!form || !hudGauge || !hudLabel) return;

  const formData = new FormData(form);
  const payload = {};
  let allValid = true;

  for (let [k, v] of formData.entries()) {
    const num = parseFloat(v);
    if (isNaN(num)) {
      allValid = false;
      break;
    }
    payload[k] = num;
  }

  if (!allValid) return;

  // Cancel any prior in-flight request so we don't queue or process stale responses
  if (activeRiskAbortController) {
    activeRiskAbortController.abort();
  }
  activeRiskAbortController = new AbortController();

  try {
    const res = await fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: activeRiskAbortController.signal,
    });
    if (!res.ok) return;

    const data = await res.json();
    if (data.success) {
      const prob = data.probability;
      hudProb.textContent = `${prob.toFixed(1)}%`;

      if (data.prediction === "High") {
        hudGauge.className = "live-hud-gauge gauge-high";
        hudGauge.textContent = "High";
        hudLabel.innerHTML = `Predicted: <strong style="color: #fb7185;">Elevated Risk</strong>`;
      } else {
        hudGauge.className = "live-hud-gauge gauge-low";
        hudGauge.textContent = "Low";
        hudLabel.innerHTML = `Predicted: <strong style="color: #34d399;">Optimal Low Risk</strong>`;
      }
    }
  } catch (e) {
    // Gracefully ignore aborted requests
    if (e.name !== "AbortError") {
      // ignore network hiccup during fast typing
    }
  }
}

/**
 * Instant In-Place Prediction Submission (ZERO PAGE RELOAD)
 */
function initInPlacePredictionSubmission() {
  const form = document.getElementById("predictionForm");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();

    const numberInputs = form.querySelectorAll("input.form-input-control[type='number']");
    let isValid = true;
    numberInputs.forEach((input) => {
      if (!validateSingleInput(input)) isValid = false;
    });

    if (!isValid) {
      const firstError = form.querySelector(".is-invalid");
      if (firstError) {
        firstError.scrollIntoView({ behavior: "smooth", block: "center" });
        firstError.focus();
      }
      return;
    }

    const submitBtn = document.getElementById("submitPredictBtn");
    const originalBtnContent = submitBtn.innerHTML;

    submitBtn.disabled = true;
    submitBtn.innerHTML = `
      <svg style="animation: spin 0.6s linear infinite; margin-right: 8px;" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <circle cx="12" cy="12" r="10" stroke-opacity="0.25"></circle>
        <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
      </svg>
      Evaluating Risk...
    `;

    const formData = new FormData(form);
    const payload = { _save_history: true };
    for (let [k, v] of formData.entries()) {
      payload[k] = parseFloat(v);
    }

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Server error during prediction");
      const data = await res.json();

      if (data.success) {
        renderInlineResult(data);
      }
    } catch (err) {
      showToast("Error executing prediction: " + err.message);
    } finally {
      submitBtn.disabled = false;
      submitBtn.innerHTML = originalBtnContent;
    }
  });
}

function renderInlineResult(data) {
  const card = document.getElementById("inlineResultCard");
  const wrapper = document.getElementById("inlineResultWrapper");
  const heading = document.getElementById("inlineResultHeading");
  const badge = document.getElementById("inlineResultBadge");
  const desc = document.getElementById("inlineResultDesc");
  const probCounter = document.getElementById("inlineProbCounter");
  const probFill = document.getElementById("inlineProbFill");
  const factorsList = document.getElementById("inlineFactorsList");
  const btnCopy = document.getElementById("btnCopyInlineReport");

  if (!card) return;

  const isHigh = data.prediction === "High";

  wrapper.className = isHigh ? "result-card risk-high" : "result-card risk-low";
  heading.textContent = `Diabetes Risk: ${data.prediction}`;
  badge.textContent = isHigh ? "Elevated Risk Profile" : "Optimal Clinical Profile";

  desc.textContent = isHigh
    ? "The machine learning pipeline indicates an elevated statistical probability of developing diabetes based on current biomarkers."
    : "The patient biomarkers fall within optimal statistical bounds for diabetes under the trained classification model.";

  // Factors List
  factorsList.innerHTML = "";
  if (data.risk_factors && data.risk_factors.length > 0) {
    const fragment = document.createDocumentFragment();
    data.risk_factors.forEach((factor) => {
      const li = document.createElement("li");
      li.textContent = factor;
      fragment.appendChild(li);
    });
    factorsList.appendChild(fragment);
  } else {
    const li = document.createElement("li");
    li.textContent = "Key biomarkers (glucose, BMI, and diastolic pressure) are within normal reference ranges.";
    factorsList.appendChild(li);
  }

  // Display card and smooth scroll
  card.style.display = "block";
  card.scrollIntoView({ behavior: "smooth", block: "start" });

  // Lightweight animated counter & bar
  animateNumericValue(probCounter, data.probability, 400);
  probFill.style.width = `${data.probability}%`;

  if (btnCopy) {
    btnCopy.onclick = () => {
      const summaryText = `DIABETES RISK PREDICTION REPORT\n-------------------------------\nRisk Level: ${data.prediction}\nProbability: ${data.probability}%\nChampion Model: ${data.model}\nTimestamp: ${data.timestamp}\n\nNotice: For educational purposes only. Consult a healthcare professional.`;
      navigator.clipboard.writeText(summaryText).then(() => {
        showToast("Clinical prediction report copied to clipboard!");
      });
    };
  }

  showToast(`Prediction: Diabetes Risk is ${data.prediction} (${data.probability}%)`);
}

function animateNumericValue(element, targetNum, duration = 400) {
  if (!element) return;
  const startTime = performance.now();
  function step(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const ease = 1 - Math.pow(1 - progress, 2);
    const currentVal = (targetNum * ease).toFixed(1);
    element.textContent = `${currentVal}%`;

    if (progress < 1) {
      requestAnimationFrame(step);
    } else {
      element.textContent = `${targetNum.toFixed(1)}%`;
    }
  }
  requestAnimationFrame(step);
}

/**
 * Quick Clinical Presets
 */
function initDemoDataButtons() {
  const btnHealthy = document.getElementById("btnFillHealthy");
  const btnBorderline = document.getElementById("btnFillBorderline");
  const btnHighRisk = document.getElementById("btnFillHighRisk");

  const healthyPreset = {
    Glucose: 85,
    BloodPressure: 66,
    SkinThickness: 23,
    Insulin: 94,
    BMI: 22.5,
    DiabetesPedigreeFunction: 0.25,
    Age: 25,
  };

  const borderlinePreset = {
    Glucose: 138,
    BloodPressure: 78,
    SkinThickness: 30,
    Insulin: 145,
    BMI: 29.2,
    DiabetesPedigreeFunction: 0.48,
    Age: 42,
  };

  const highRiskPreset = {
    Glucose: 172,
    BloodPressure: 88,
    SkinThickness: 38,
    Insulin: 215,
    BMI: 36.8,
    DiabetesPedigreeFunction: 0.88,
    Age: 54,
  };

  if (btnHealthy) {
    btnHealthy.addEventListener("click", () => {
      fillSampleValues(healthyPreset);
      showToast("Loaded Low-Risk Clinical Preset");
    });
  }
  if (btnBorderline) {
    btnBorderline.addEventListener("click", () => {
      fillSampleValues(borderlinePreset);
      showToast("Loaded Borderline Clinical Preset");
    });
  }
  if (btnHighRisk) {
    btnHighRisk.addEventListener("click", () => {
      fillSampleValues(highRiskPreset);
      showToast("Loaded High-Risk Clinical Preset");
    });
  }
}

function initResetDefaultsButton() {
  const btn = document.getElementById("btnResetDefaults");
  if (!btn) return;

  btn.addEventListener("click", () => {
    const form = document.getElementById("predictionForm");
    if (!form) return;

    form.reset();
    const inputs = form.querySelectorAll("input.form-input-control[type='number']");
    inputs.forEach((input) => {
      const slider = document.getElementById(`slider_${input.name}`);
      if (slider) slider.value = input.value;
      validateSingleInput(input);
    });

    const card = document.getElementById("inlineResultCard");
    if (card) card.style.display = "none";

    computeLiveRisk();
    showToast("Reset all biomarkers to clinical baseline defaults");
  });
}

function fillSampleValues(sampleData) {
  for (const [key, value] of Object.entries(sampleData)) {
    const input = document.querySelector(`input[name="${key}"]`);
    const slider = document.getElementById(`slider_${key}`);
    if (input) {
      input.value = value;
      if (slider) slider.value = value;
      validateSingleInput(input);
    }
  }
  computeLiveRisk();
}

/**
 * Animated Probability Counter on Standalone Result Page
 */
function initAnimatedCounters() {
  const probFill = document.querySelector(".prob-meter-fill");
  const counterEl = document.getElementById("animatedProbabilityCounter");

  if (counterEl) {
    const targetNum = parseFloat(counterEl.dataset.targetValue || 0);
    animateNumericValue(counterEl, targetNum, 500);
  }

  if (probFill) {
    const targetWidth = probFill.dataset.targetWidth || probFill.style.width;
    probFill.style.width = targetWidth;
  }
}

/**
 * Dashboard Tabs
 */
function initDashboardTabs() {
  const tabBtns = document.querySelectorAll(".tab-btn");
  if (!tabBtns.length) return;

  tabBtns.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.dataset.tab;

      tabBtns.forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");

      document.querySelectorAll(".tab-pane").forEach((pane) => {
        pane.classList.remove("active");
      });

      const activePane = document.getElementById(targetId);
      if (activePane) {
        activePane.classList.add("active");
      }
    });
  });
}

/**
 * Optimized Chart.js Initializer (Lifecycle managed, zero duplicate memory leaks)
 */
function initDashboardCharts() {
  const chartConfigEl = document.getElementById("chartConfigData");
  if (!chartConfigEl || typeof Chart === "undefined") return;

  let configData = {};
  try {
    configData = JSON.parse(chartConfigEl.textContent);
  } catch (err) {
    return;
  }

  // Lightweight global chart defaults
  Chart.defaults.color = "#94a3b8";
  Chart.defaults.font.family = "'Plus Jakarta Sans', sans-serif";
  Chart.defaults.font.size = 12;
  Chart.defaults.animation = {
    duration: 300,
    easing: "easeOutQuad",
  };

  // Safely destroy any existing chart instance on a canvas before creating
  function safeCreateChart(canvasId, config) {
    const el = document.getElementById(canvasId);
    if (!el) return null;
    const existing = Chart.getChart(el);
    if (existing) {
      existing.destroy();
    }
    return new Chart(el, config);
  }

  // 1. Class Distribution Doughnut Chart
  if (configData.class_distribution) {
    safeCreateChart("classDistChart", {
      type: "doughnut",
      data: {
        labels: configData.class_distribution.labels,
        datasets: [
          {
            data: configData.class_distribution.counts,
            backgroundColor: ["#0ea5e9", "#f43f5e"],
            borderColor: "#0f172a",
            borderWidth: 3,
            hoverOffset: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom", labels: { boxWidth: 12, padding: 14 } },
          tooltip: {
            callbacks: {
              label: (item) => {
                const total = item.dataset.data.reduce((a, b) => a + b, 0);
                const pct = ((item.raw / total) * 100).toFixed(1);
                return ` ${item.label}: ${item.raw} (${pct}%)`;
              },
            },
          },
        },
        cutout: "70%",
      },
    });
  }

  // 2. Model Performance Benchmark Bar Chart
  if (configData.model_comparison) {
    const models = configData.model_comparison.models;
    safeCreateChart("modelCompChart", {
      type: "bar",
      data: {
        labels: models,
        datasets: [
          {
            label: "Accuracy",
            data: configData.model_comparison.accuracy.map((v) => (v * 100).toFixed(1)),
            backgroundColor: "#0ea5e9",
            borderRadius: 6,
          },
          {
            label: "ROC-AUC",
            data: configData.model_comparison.roc_auc.map((v) => (v * 100).toFixed(1)),
            backgroundColor: "#8b5cf6",
            borderRadius: 6,
          },
          {
            label: "Recall",
            data: configData.model_comparison.recall.map((v) => (v * 100).toFixed(1)),
            backgroundColor: "#10b981",
            borderRadius: 6,
          },
          {
            label: "F1-Score",
            data: configData.model_comparison.f1_score.map((v) => (v * 100).toFixed(1)),
            backgroundColor: "#f59e0b",
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: {
            beginAtZero: true,
            max: 100,
            ticks: { callback: (v) => v + "%" },
            grid: { color: "rgba(255, 255, 255, 0.05)" },
          },
          x: { grid: { display: false } },
        },
        plugins: {
          legend: { position: "top", labels: { boxWidth: 10, padding: 12 } },
          tooltip: {
            callbacks: {
              label: (item) => ` ${item.dataset.label}: ${item.raw}%`,
            },
          },
        },
      },
    });
  }

  // 3. Feature Importance Horizontal Bar Chart
  if (configData.feature_importance) {
    safeCreateChart("featureImpChart", {
      type: "bar",
      data: {
        labels: configData.feature_importance.features,
        datasets: [
          {
            label: "Importance Score",
            data: configData.feature_importance.scores,
            backgroundColor: "rgba(6, 182, 212, 0.85)",
            borderColor: "#06b6d4",
            borderWidth: 1,
            borderRadius: 6,
          },
        ],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: {
            beginAtZero: true,
            grid: { color: "rgba(255, 255, 255, 0.05)" },
          },
          y: { grid: { display: false } },
        },
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (item) => ` Weight: ${(item.raw * 100).toFixed(2)}%`,
            },
          },
        },
      },
    });
  }
}

/**
 * Toast Notification Utility
 */
function showToast(msg) {
  let toast = document.getElementById("floatingToast");
  if (!toast) {
    toast = document.createElement("div");
    toast.id = "floatingToast";
    toast.style.cssText = `
      position: fixed;
      bottom: 2rem;
      right: 2rem;
      z-index: 9999;
      background: rgba(15, 23, 42, 0.96);
      border: 1px solid rgba(14, 165, 233, 0.45);
      color: #fff;
      padding: 0.75rem 1.25rem;
      border-radius: 10px;
      font-size: 0.88rem;
      font-weight: 600;
      box-shadow: 0 8px 24px rgba(0,0,0,0.4);
      transition: transform 0.2s ease, opacity 0.2s ease;
      transform: translateY(20px);
      opacity: 0;
    `;
    document.body.appendChild(toast);
  }

  toast.textContent = msg;
  toast.style.transform = "translateY(0)";
  toast.style.opacity = "1";

  setTimeout(() => {
    toast.style.transform = "translateY(20px)";
    toast.style.opacity = "0";
  }, 2200);
}

// Inline minimal keyframe for spin animation
const spinStyle = document.createElement("style");
spinStyle.innerHTML = `@keyframes spin { 100% { transform: rotate(360deg); } }`;
document.head.appendChild(spinStyle);
