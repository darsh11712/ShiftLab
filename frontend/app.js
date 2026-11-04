/* ShiftLab frontend — vanilla JS, no build step. */
"use strict";

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const state = {
  algorithms: [],
  datasets: [],
  experiments: [],
  algoDetail: new Map(),
  datasetDetail: new Map(),
  selectedAlgo: null,
  selectedDataset: null,
  selectedExperiment: null,
};

/* ------------------------------------------------------------------ */
/* Utilities                                                           */
/* ------------------------------------------------------------------ */

function esc(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function fmt(value, digits = 4) {
  if (value === null || value === undefined) return "—";
  if (Array.isArray(value)) return value.map((v) => fmt(v, digits)).join(", ");
  if (typeof value === "object") return JSON.stringify(value);
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return String(value);
    if (Number.isInteger(value)) return value.toLocaleString();
    return value.toFixed(digits);
  }
  return String(value);
}

function fmtTime(seconds) {
  if (seconds === null || seconds === undefined) return "—";
  const s = Number(seconds);
  if (!Number.isFinite(s)) return "—";
  return s < 1 ? `${(s * 1000).toFixed(0)} ms` : `${s.toFixed(2)} s`;
}

let toastTimer = null;
function toast(message, type = "") {
  const node = $("#toast");
  node.textContent = message;
  node.className = `toast ${type}`.trim();
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => node.classList.add("hidden"), 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const text = await response.text();
  let data = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    data = text;
  }
  if (!response.ok) {
    const detail = data && data.detail ? data.detail : text || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

function metricCards(metrics) {
  const entries = Object.entries(metrics || {});
  if (!entries.length) return '<p class="muted">No metrics returned.</p>';
  return `<div class="metric-grid">${entries.map(([key, value]) => `
    <div class="metric"><div class="k">${esc(key)}</div><div class="v">${esc(fmt(value))}</div></div>
  `).join("")}</div>`;
}

function jsonBlock(data) {
  return `<pre>${esc(JSON.stringify(data, null, 2))}</pre>`;
}

function fillSelect(select, items, { valueKey = "name", labelKey = "name", placeholder = null } = {}) {
  const previous = select.value;
  select.innerHTML = "";
  if (placeholder !== null) {
    const opt = document.createElement("option");
    opt.value = "";
    opt.textContent = placeholder;
    select.appendChild(opt);
  }
  const valueOf = (item) =>
    item !== null && typeof item === "object" ? item[valueKey] : item;
  for (const item of items) {
    const opt = document.createElement("option");
    opt.value = valueOf(item);
    opt.textContent = typeof labelKey === "function"
      ? labelKey(item)
      : (item !== null && typeof item === "object" ? item[labelKey] : item);
    select.appendChild(opt);
  }
  if (previous && items.some((item) => String(valueOf(item)) === previous)) {
    select.value = previous;
  }
}

function fillDatasetSelect(select) {
  const previous = select.value;
  select.innerHTML = "";
  const groups = [
    ["Built-in", state.datasets.filter((d) => d.source !== "upload")],
    ["Uploaded", state.datasets.filter((d) => d.source === "upload")],
  ];
  for (const [label, datasets] of groups) {
    if (!datasets.length) continue;
    const group = document.createElement("optgroup");
    group.label = label;
    for (const ds of datasets) {
      const opt = document.createElement("option");
      opt.value = ds.name;
      opt.textContent = `${ds.name} · ${ds.task_type}`;
      group.appendChild(opt);
    }
    select.appendChild(group);
  }
  if (previous && state.datasets.some((d) => d.name === previous)) select.value = previous;
}

const UNSUPERVISED_TASKS = new Set(["clustering", "dimensionality_reduction", "anomaly_detection"]);

function algorithmsForTask(taskType) {
  return state.algorithms.filter((algo) => algo.task_type === taskType);
}

function algorithmsForTraining(taskType) {
  // Supervised algorithms must match the dataset task; unsupervised-family
  // algorithms can be applied to any feature matrix.
  return state.algorithms.filter(
    (algo) => algo.task_type === taskType || UNSUPERVISED_TASKS.has(algo.task_type),
  );
}

/* ------------------------------------------------------------------ */
/* Navigation                                                          */
/* ------------------------------------------------------------------ */

const TAB_META = {
  overview: ["Overview", "Real algorithms, real fits, real metrics."],
  algorithms: ["Algorithms", "Browse the full registry and inspect metadata."],
  datasets: ["Datasets", "Built-in data and your uploaded CSVs."],
  train: ["Train & Predict", "Fit any compatible algorithm and score it on held-out data."],
  experiments: ["Experiments", "Every saved run, with its real metrics."],
  arena: ["Arena", "Compare algorithms on an identical split."],
  reliability: ["Reliability Lab", "Detect and quantify distribution shift."],
};

function showTab(name) {
  $$(".nav-item").forEach((btn) => btn.classList.toggle("active", btn.dataset.tab === name));
  $$(".tab-panel").forEach((panel) => panel.classList.toggle("active", panel.id === `tab-${name}`));
  const [title, subtitle] = TAB_META[name] || ["ShiftLab", ""];
  $("#page-title").textContent = title;
  $("#page-subtitle").textContent = subtitle;
  if (name === "experiments") loadExperiments();
}

/* ------------------------------------------------------------------ */
/* Overview                                                            */
/* ------------------------------------------------------------------ */

function renderOverview() {
  const families = {};
  for (const algo of state.algorithms) {
    families[algo.family] = (families[algo.family] || 0) + 1;
  }
  const maxCount = Math.max(1, ...Object.values(families));

  $("#stat-grid").innerHTML = `
    <div class="stat"><div class="value">${state.algorithms.length}</div><div class="label">Algorithms</div></div>
    <div class="stat"><div class="value">${state.datasets.length}</div><div class="label">Datasets</div></div>
    <div class="stat"><div class="value">${Object.keys(families).length}</div><div class="label">Families</div></div>
    <div class="stat"><div class="value">${state.experiments.length}</div><div class="label">Experiments</div></div>
  `;

  $("#family-breakdown").innerHTML = Object.entries(families)
    .sort((a, b) => b[1] - a[1])
    .map(([family, count]) => `
      <div class="breakdown-row">
        <span class="name">${esc(family)}</span>
        <span class="bar"><i style="width:${(count / maxCount) * 100}%"></i></span>
        <span>${count}</span>
      </div>
    `).join("");

  const recent = state.experiments.slice(0, 6);
  $("#recent-experiments").className = recent.length ? "list" : "list muted-empty";
  if (!recent.length) {
    $("#recent-experiments").textContent = "No saved experiments yet.";
    return;
  }
  $("#recent-experiments").innerHTML = recent.map((exp) => `
    <div class="item" data-exp="${esc(exp.id)}">
      <div class="title">
        <span>${esc(exp.algorithm_slug)}</span>
        <span class="badge">${esc(exp.id)}</span>
      </div>
      <div class="meta">${esc(exp.dataset_name)} · ${esc(fmtTime(exp.train_time_s))} · ${esc(Object.entries(exp.metrics || {}).map(([k, v]) => `${k}=${fmt(v)}`).join(" · "))}</div>
    </div>
  `).join("");
  $$("#recent-experiments .item").forEach((node) => {
    node.addEventListener("click", () => {
      showTab("experiments");
      loadExperiment(node.dataset.exp);
    });
  });
}

/* ------------------------------------------------------------------ */
/* Algorithms                                                          */
/* ------------------------------------------------------------------ */

function renderAlgoFilters() {
  fillSelect($("#algo-family"), [...new Set(state.algorithms.map((a) => a.family))].sort(), { placeholder: "All families" });
  fillSelect($("#algo-task"), [...new Set(state.algorithms.map((a) => a.task_type))].sort(), { placeholder: "All tasks" });
}

function renderAlgoList() {
  const q = $("#algo-search").value.trim().toLowerCase();
  const family = $("#algo-family").value;
  const task = $("#algo-task").value;

  const filtered = state.algorithms.filter((algo) => {
    if (family && algo.family !== family) return false;
    if (task && algo.task_type !== task) return false;
    if (!q) return true;
    return (
      algo.name.toLowerCase().includes(q) ||
      algo.slug.toLowerCase().includes(q) ||
      algo.family.toLowerCase().includes(q) ||
      algo.task_type.toLowerCase().includes(q) ||
      (algo.tags || []).some((tag) => tag.toLowerCase().includes(q))
    );
  });

  $("#algo-count").textContent = `${filtered.length} of ${state.algorithms.length}`;
  const list = $("#algo-list");
  list.className = "list scroll";
  if (!filtered.length) {
    list.innerHTML = '<p class="muted-empty">No matching algorithms.</p>';
    return;
  }
  list.innerHTML = filtered.map((algo) => `
    <div class="item ${state.selectedAlgo === algo.slug ? "selected" : ""}" data-slug="${esc(algo.slug)}">
      <div class="title"><span>${esc(algo.name)}</span><span class="badge">${esc(algo.family)}</span></div>
      <div class="meta">${esc(algo.task_type)}${algo.sklearn_compatible ? "" : " · custom NumPy"}</div>
      <div>${(algo.tags || []).slice(0, 6).map((tag) => `<span class="tag">${esc(tag)}</span>`).join("")}</div>
    </div>
  `).join("");
  $$("#algo-list .item").forEach((node) => {
    node.addEventListener("click", () => loadAlgorithm(node.dataset.slug));
  });
}

async function loadAlgorithm(slug) {
  state.selectedAlgo = slug;
  renderAlgoList();
  const container = $("#algo-detail");
  container.innerHTML = '<p class="muted-empty">Loading…</p>';
  try {
    let detail = state.algoDetail.get(slug);
    if (!detail) {
      detail = await api(`/api/algorithms/${encodeURIComponent(slug)}`);
      state.algoDetail.set(slug, detail);
    }
    const hp = detail.hyperparameters || [];
    container.innerHTML = `
      <h3>${esc(detail.name)}</h3>
      <p class="muted">${esc(detail.description || "")}</p>
      <div class="chip-row">
        <span class="chip">${esc(detail.family)}</span>
        <span class="chip">${esc(detail.task_type)}</span>
        <span class="chip">${detail.sklearn_compatible ? "sklearn-compatible" : "custom NumPy"}</span>
      </div>
      <h4>Math intuition</h4>
      <p class="muted">${esc(detail.math_intuition || "—")}</p>
      <h4>Hyperparameters</h4>
      ${hp.length ? `<table><thead><tr><th>Name</th><th>Type</th><th>Default</th></tr></thead><tbody>${hp.map((p) => `
        <tr><td>${esc(p.name)}</td><td>${esc(p.type || "—")}</td><td>${esc(JSON.stringify(p.default))}</td></tr>
      `).join("")}</tbody></table>` : '<p class="muted">No tunable hyperparameters.</p>'}
      <h4>Strengths</h4>
      <ul>${(detail.strengths || []).map((s) => `<li>${esc(s)}</li>`).join("") || "<li>—</li>"}</ul>
      <h4>Weaknesses</h4>
      <ul>${(detail.weaknesses || []).map((s) => `<li>${esc(s)}</li>`).join("") || "<li>—</li>"}</ul>
      <h4>Use cases</h4>
      <ul>${(detail.use_cases || []).map((s) => `<li>${esc(s)}</li>`).join("") || "<li>—</li>"}</ul>
      <h4>Complexity</h4>
      <p class="muted">Train: ${esc(detail.complexity_train)}<br />Predict: ${esc(detail.complexity_predict)}</p>
      <p class="muted"><strong>Slug:</strong> ${esc(slug)}</p>
    `;
  } catch (error) {
    container.innerHTML = `<p class="muted-empty">Failed to load: ${esc(error.message)}</p>`;
  }
}

/* ------------------------------------------------------------------ */
/* Datasets                                                            */
/* ------------------------------------------------------------------ */

function renderDatasetList() {
  const list = $("#dataset-list");
  list.className = "list scroll";
  list.innerHTML = state.datasets.map((ds) => `
    <div class="item ${state.selectedDataset === ds.name ? "selected" : ""}" data-ds="${esc(ds.name)}">
      <div class="title">
        <span>${esc(ds.name)}</span>
        <span class="badge ${ds.source === "upload" ? "upload" : "builtin"}">${esc(ds.source || "builtin")}</span>
      </div>
      <div class="meta">${esc(ds.task_type)} · ${esc(ds.n_samples ?? "?")} samples · ${esc(ds.n_features ?? "?")} features</div>
      <div class="meta">${esc(ds.description || "")}</div>
    </div>
  `).join("") || '<p class="muted-empty">No datasets.</p>';
  $$("#dataset-list .item").forEach((node) => {
    node.addEventListener("click", () => loadDataset(node.dataset.ds));
  });
}

async function loadDataset(name) {
  state.selectedDataset = name;
  renderDatasetList();
  const container = $("#dataset-detail");
  container.innerHTML = '<p class="muted-empty">Loading…</p>';
  try {
    const detail = await api(`/api/datasets/${encodeURIComponent(name)}`);
    container.className = "list";
    container.innerHTML = `
      <div class="item" style="cursor:default">
        <div class="title"><span>${esc(detail.name)}</span><span class="badge">${esc(detail.task_type)}</span></div>
        <div class="meta">${esc(detail.description || "")}</div>
        <div class="meta">${esc(detail.n_samples)} samples · ${esc(detail.n_features)} features · target: ${esc(detail.target_name || "—")}</div>
        ${detail.n_classes ? `<div class="meta">${esc(detail.n_classes)} classes: ${esc((detail.class_names || []).join(", "))}</div>` : ""}
      </div>
      <div>
        <h4>Feature names</h4>
        <p class="muted">${esc((detail.feature_names || []).join(", "))}</p>
      </div>
    `;
    state.datasetDetail.set(name, detail);
  } catch (error) {
    container.innerHTML = `<p class="muted-empty">Failed to load: ${esc(error.message)}</p>`;
  }
}

async function uploadDataset(event) {
  event.preventDefault();
  const file = $("#upload-file").files[0];
  if (!file) return;
  const form = new FormData();
  form.append("file", file);
  const target = $("#upload-target").value.trim();
  const task = $("#upload-task").value;
  const description = $("#upload-description").value.trim();
  if (target) form.append("target_column", target);
  if (task) form.append("task_type", task);
  if (description) form.append("description", description);

  const button = event.target.querySelector("button[type=submit]");
  button.disabled = true;
  button.textContent = "Uploading…";
  try {
    const result = await api("/api/datasets/upload", { method: "POST", body: form });
    $("#upload-result").classList.remove("hidden");
    $("#upload-result").innerHTML = `
      <p class="muted"><strong>Registered:</strong> ${esc(result.name)} · ${esc(result.task_type)} · target ${esc(result.target_column)}</p>
      ${jsonBlock(result)}
    `;
    toast(`Uploaded ${result.name}`, "ok");
    await loadDatasets();
    refreshDatasetSelects();
  } catch (error) {
    toast(error.message, "error");
  } finally {
    button.disabled = false;
    button.textContent = "Upload & register";
  }
}

/* ------------------------------------------------------------------ */
/* Train                                                               */
/* ------------------------------------------------------------------ */

function currentTrainDataset() {
  return state.datasets.find((ds) => ds.name === $("#train-dataset").value) || null;
}

function refreshTrainAlgorithms() {
  const ds = currentTrainDataset();
  const algorithms = ds ? algorithmsForTraining(ds.task_type) : state.algorithms;
  fillSelect($("#train-algorithm"), algorithms, { valueKey: "slug", labelKey: (a) => `${a.name} · ${a.task_type}` });
  refreshHyperparams();
}

async function refreshHyperparams() {
  const slug = $("#train-algorithm").value;
  const container = $("#hyperparam-fields");
  container.innerHTML = "";
  if (!slug) return;
  try {
    let detail = state.algoDetail.get(slug);
    if (!detail) {
      detail = await api(`/api/algorithms/${encodeURIComponent(slug)}`);
      state.algoDetail.set(slug, detail);
    }
    const schema = detail.hyperparameters || [];
    if (!schema.length) {
      container.innerHTML = '<p class="muted">This algorithm has no tunable hyperparameters.</p>';
      return;
    }
    container.innerHTML = schema.map((p) => {
      const id = `hp-${p.name}`;
      const label = esc(p.name);
      if (p.type === "select") {
        return `<label>${label}
          <select id="${id}" class="input" data-hp="${esc(p.name)}" data-type="select">
            ${(p.options || []).map((opt) => `<option value="${esc(opt)}" ${String(opt) === String(p.default) ? "selected" : ""}>${esc(opt)}</option>`).join("")}
          </select>
        </label>`;
      }
      if (p.type === "bool") {
        return `<label class="checkbox">${label}
          <input id="${id}" type="checkbox" data-hp="${esc(p.name)}" data-type="bool" ${p.default ? "checked" : ""} />
        </label>`;
      }
      if (p.type && p.type.startsWith("list")) {
        const value = Array.isArray(p.default) ? p.default.join(", ") : "";
        return `<label>${label} (comma-separated)
          <input id="${id}" class="input" data-hp="${esc(p.name)}" data-type="list" value="${esc(value)}" />
        </label>`;
      }
      const type = p.type === "int" ? "number" : "number";
      const step = p.type === "int" ? "1" : "any";
      const min = p.min !== undefined && p.min !== null ? `min="${esc(p.min)}"` : "";
      const max = p.max !== undefined && p.max !== null ? `max="${esc(p.max)}"` : "";
      const value = p.default === null || p.default === undefined ? "" : p.default;
      const placeholder = p.default === null || p.default === undefined ? 'placeholder="auto"' : "";
      return `<label>${label}
        <input id="${id}" class="input" type="${type}" step="${step}" ${min} ${max} ${placeholder}
          data-hp="${esc(p.name)}" data-type="${esc(p.type || "float")}" value="${esc(value)}" />
      </label>`;
    }).join("");
  } catch (error) {
    container.innerHTML = `<p class="muted-empty">Could not load schema: ${esc(error.message)}</p>`;
  }
}

function collectHyperparams() {
  const result = {};
  $$("#hyperparam-fields [data-hp]").forEach((input) => {
    const name = input.dataset.hp;
    const type = input.dataset.type;
    if (type === "bool") {
      result[name] = input.checked;
      return;
    }
    const raw = input.value.trim();
    if (raw === "") return;
    if (type === "int") result[name] = parseInt(raw, 10);
    else if (type === "float") result[name] = parseFloat(raw);
    else if (type === "list") result[name] = raw.split(",").map((v) => parseInt(v.trim(), 10)).filter((v) => Number.isFinite(v));
    else result[name] = raw;
  });
  return result;
}

async function trainModel() {
  const dataset = $("#train-dataset").value;
  const slug = $("#train-algorithm").value;
  if (!dataset || !slug) {
    toast("Pick a dataset and algorithm first.", "error");
    return;
  }
  const button = $("#train-btn");
  const status = $("#train-status");
  button.disabled = true;
  status.textContent = "Training…";
  const payload = {
    algorithm_slug: slug,
    dataset_name: dataset,
    test_size: parseFloat($("#train-test-size").value),
    random_state: parseInt($("#train-seed").value, 10) || 42,
    hyperparams: collectHyperparams(),
    scaler: $("#train-scaler").value,
    imputer: $("#train-imputer").value,
    save: $("#train-save").checked,
  };
  try {
    const result = await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    $("#train-result").classList.remove("hidden");
    $("#train-result").innerHTML = `
      <h3>${esc(result.algorithm_slug)}</h3>
      <p class="muted">${esc(result.dataset_name)} · ${esc(result.dataset_task_type || "")} · train ${fmtTime(result.train_time_s)}</p>
      ${metricCards(result.metrics)}
      ${result.experiment_id ? `<p class="muted">Saved experiment: <strong>${esc(result.experiment_id)}</strong></p>` : ""}
      ${jsonBlock(result)}
    `;
    status.textContent = "Done.";
    toast("Training complete.", "ok");
    if (result.experiment_id) await loadExperiments();
  } catch (error) {
    status.textContent = "";
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

/* ------------------------------------------------------------------ */
/* Experiments                                                         */
/* ------------------------------------------------------------------ */

async function loadExperiments() {
  try {
    state.experiments = await api("/api/experiments");
    $("#experiments-count").textContent = `${state.experiments.length} saved`;
    renderExperimentList();
    renderOverview();
  } catch (error) {
    toast(error.message, "error");
  }
}

function renderExperimentList() {
  const list = $("#experiment-list");
  list.className = "list scroll";
  if (!state.experiments.length) {
    list.innerHTML = '<p class="muted-empty">No experiments saved yet.</p>';
    return;
  }
  list.innerHTML = state.experiments.map((exp) => `
    <div class="item ${state.selectedExperiment === exp.id ? "selected" : ""}" data-exp="${esc(exp.id)}">
      <div class="title"><span>${esc(exp.algorithm_slug)}</span><span class="badge">${esc(exp.id)}</span></div>
      <div class="meta">${esc(exp.dataset_name)} · ${esc(fmtTime(exp.train_time_s))}</div>
      <div class="meta">${esc(Object.entries(exp.metrics || {}).map(([k, v]) => `${k}=${fmt(v)}`).join(" · "))}</div>
    </div>
  `).join("");
  $$("#experiment-list .item").forEach((node) => {
    node.addEventListener("click", () => loadExperiment(node.dataset.exp));
  });
}

async function loadExperiment(id) {
  state.selectedExperiment = id;
  renderExperimentList();
  const container = $("#experiment-detail");
  container.innerHTML = '<p class="muted-empty">Loading…</p>';
  try {
    const exp = await api(`/api/experiments/${encodeURIComponent(id)}`);
    container.innerHTML = `
      <h3>${esc(exp.algorithm_slug)}</h3>
      <p class="muted">${esc(exp.dataset_name)} · ${esc(exp.created_at)} · train ${fmtTime(exp.train_time_s)} · inference ${fmtTime(exp.infer_time_s)}</p>
      ${metricCards(exp.metrics)}
      <h4>Hyperparameters</h4>
      ${jsonBlock(exp.hyperparams)}
      ${jsonBlock(exp)}
    `;
  } catch (error) {
    container.innerHTML = `<p class="muted-empty">Failed to load: ${esc(error.message)}</p>`;
  }
}

/* ------------------------------------------------------------------ */
/* Arena                                                               */
/* ------------------------------------------------------------------ */

function refreshArenaAlgorithms() {
  const name = $("#arena-dataset").value;
  const ds = state.datasets.find((d) => d.name === name);
  const container = $("#arena-algo-picker");
  if (!ds) {
    container.innerHTML = '<p class="muted-empty">Pick a dataset.</p>';
    return;
  }
  if (!["classification", "regression"].includes(ds.task_type)) {
    container.innerHTML = '<p class="muted-empty">Arena supports classification and regression datasets.</p>';
    return;
  }
  const algorithms = algorithmsForTask(ds.task_type);
  container.innerHTML = algorithms.map((algo) => `
    <label>
      <input type="checkbox" value="${esc(algo.slug)}" />
      <span>${esc(algo.name)}<small>${esc(algo.family)}</small></span>
    </label>
  `).join("") || '<p class="muted-empty">No compatible algorithms.</p>';
}

async function runArena() {
  const dataset = $("#arena-dataset").value;
  const selected = $$("#arena-algo-picker input:checked").map((input) => input.value);
  if (!dataset || !selected.length) {
    toast("Pick a dataset and at least one algorithm.", "error");
    return;
  }
  const status = $("#arena-status");
  const button = $("#arena-run");
  button.disabled = true;
  status.textContent = `Training ${selected.length} models…`;
  try {
    const result = await api("/api/arena/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        algorithm_slugs: selected,
        dataset_name: dataset,
        test_size: parseFloat($("#arena-test-size").value),
        save: $("#arena-save").checked,
      }),
    });
    $("#arena-result").classList.remove("hidden");
    $("#arena-result").innerHTML = `
      <p class="muted"><strong>${esc(result.dataset_name)}</strong> · ${esc(result.task_type)} · primary metric: <strong>${esc(result.primary_metric)}</strong> · winner: <strong>${esc(result.winner || "—")}</strong></p>
      <table>
        <thead><tr><th>#</th><th>Algorithm</th><th>${esc(result.primary_metric)}</th><th>Train time</th><th>Result</th></tr></thead>
        <tbody>
          ${result.results.map((row) => `
            <tr>
              <td>${esc(row.rank ?? "—")}</td>
              <td>${esc(row.name || row.algorithm_slug)}</td>
              <td class="num">${row.metrics && result.primary_metric in row.metrics ? esc(fmt(row.metrics[result.primary_metric])) : "—"}</td>
              <td>${esc(fmtTime(row.train_time_s))}</td>
              <td>${row.error ? `<span style="color:var(--bad)">${esc(row.error)}</span>` : esc(Object.entries(row.metrics || {}).map(([k, v]) => `${k}=${fmt(v)}`).join(" · "))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
      ${jsonBlock(result)}
    `;
    status.textContent = "Done.";
    toast("Comparison complete.", "ok");
    if ($("#arena-save").checked) await loadExperiments();
  } catch (error) {
    status.textContent = "";
    toast(error.message, "error");
  } finally {
    button.disabled = false;
  }
}

async function loadLeaderboard() {
  const status = $("#arena-status");
  status.textContent = "Loading leaderboard…";
  try {
    const result = await api("/api/arena/leaderboard?limit=100");
    $("#arena-result").classList.remove("hidden");
    if (!result.entries.length) {
      $("#arena-result").innerHTML = '<p class="muted">No saved experiments with rankable metrics yet.</p>';
      status.textContent = "";
      return;
    }
    $("#arena-result").innerHTML = `
      <p class="muted">${result.n_entries} rankable experiments</p>
      <table>
        <thead><tr><th>#</th><th>Algorithm</th><th>Dataset</th><th>Metric</th><th>Value</th><th>Train time</th></tr></thead>
        <tbody>
          ${result.entries.map((row) => `
            <tr>
              <td>${esc(row.rank)}</td>
              <td>${esc(row.name)}</td>
              <td>${esc(row.dataset_name)}</td>
              <td>${esc(row.metric)}</td>
              <td class="num">${esc(fmt(row.value))}</td>
              <td>${esc(fmtTime(row.train_time_s))}</td>
            </tr>
          `).join("")}
        </tbody>
      </table>
    `;
    status.textContent = "";
  } catch (error) {
    status.textContent = "";
    toast(error.message, "error");
  }
}

/* ------------------------------------------------------------------ */
/* Reliability                                                         */
/* ------------------------------------------------------------------ */

function refreshReliabilityAlgorithms() {
  const name = $("#rel-dataset").value;
  const ds = state.datasets.find((d) => d.name === name);
  const algorithms = ds && ["classification", "regression"].includes(ds.task_type)
    ? algorithmsForTask(ds.task_type)
    : state.algorithms.filter((a) => ["classification", "regression"].includes(a.task_type));
  fillSelect($("#rel-algorithm"), algorithms, { valueKey: "slug", labelKey: (a) => `${a.name} (${a.task_type})` });
}

async function runReliability(kind) {
  const dataset = $("#rel-dataset").value;
  if (!dataset) {
    toast("Pick a dataset.", "error");
    return;
  }
  const base = {
    dataset_name: dataset,
    test_size: parseFloat($("#rel-test-size").value),
    random_state: 42,
  };
  let path = "";
  let payload = base;
  if (kind === "covariate") {
    path = "/api/reliability/covariate-shift";
  } else if (kind === "label") {
    path = "/api/reliability/label-shift";
  } else if (kind === "weights") {
    path = "/api/reliability/importance-weights";
    payload = { ...base, method: $("#rel-method").value };
  } else if (kind === "sweep") {
    const magnitudes = $("#rel-magnitudes").value
      .split(",")
      .map((v) => parseFloat(v.trim()))
      .filter((v) => Number.isFinite(v) && v >= 0)
      .sort((a, b) => a - b);
    path = "/api/reliability/performance-under-shift";
    payload = {
      ...base,
      algorithm_slug: $("#rel-algorithm").value,
      shift_type: $("#rel-shift-type").value,
      shift_magnitudes: magnitudes,
    };
  }

  const container = $("#rel-result");
  container.classList.remove("hidden");
  container.innerHTML = '<p class="muted">Computing…</p>';
  try {
    const result = await api(path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    container.innerHTML = renderReliability(kind, result);
    toast("Analysis complete.", "ok");
  } catch (error) {
    container.innerHTML = `<p class="muted-empty">Failed: ${esc(error.message)}</p>`;
    toast(error.message, "error");
  }
}

function renderReliability(kind, result) {
  if (kind === "covariate") {
    const mmd = result.mmd || {};
    const test = result["classifier_two_sample_test"] || {};
    return `
      <h3>Covariate shift</h3>
      <p class="muted">${esc(result.dataset_name)} · ${esc(result.n_train)} train / ${esc(result.n_test)} test</p>
      <div class="metric-grid">
        <div class="metric"><div class="k">MMD²</div><div class="v">${esc(fmt(mmd.mmd2))}</div></div>
        <div class="metric"><div class="k">Severity</div><div class="v">${esc(mmd.shift_severity)}</div></div>
        <div class="metric"><div class="k">Domain AUC</div><div class="v">${esc(fmt(test.auc_mean))}</div></div>
        <div class="metric"><div class="k">Shift detected</div><div class="v">${esc(test.shift_detected)}</div></div>
      </div>
      ${jsonBlock(result)}
    `;
  }
  if (kind === "label") {
    return `
      <h3>Label shift</h3>
      <div class="metric-grid">
        <div class="metric"><div class="k">TV distance</div><div class="v">${esc(fmt(result.tv_distance))}</div></div>
        <div class="metric"><div class="k">χ² p-value</div><div class="v">${esc(fmt(result.chi2_pvalue))}</div></div>
        <div class="metric"><div class="k">Shift detected</div><div class="v">${esc(result.shift_detected)}</div></div>
      </div>
      ${jsonBlock(result)}
    `;
  }
  if (kind === "weights") {
    return `
      <h3>Importance weights</h3>
      <div class="metric-grid">
        <div class="metric"><div class="k">Method</div><div class="v">${esc(result.method)}</div></div>
        <div class="metric"><div class="k">ESS</div><div class="v">${esc(fmt(result.effective_sample_size, 1))}</div></div>
        <div class="metric"><div class="k">ESS ratio</div><div class="v">${esc(fmt(result.ess_ratio))}</div></div>
        <div class="metric"><div class="k">Clip threshold</div><div class="v">${esc(fmt(result.clip_threshold))}</div></div>
      </div>
      <p class="muted">${esc(result.description || "")}</p>
      ${jsonBlock({ ...result, weights: `[${result.weights.length} weights]` })}
    `;
  }
  const rows = (result.results || []).map((row) => `
    <tr>
      <td class="num">${esc(fmt(row.shift_magnitude))}</td>
      <td class="num">${esc(fmt(row.primary_metric))}</td>
      <td class="num">${esc(fmt(row.mmd2))}</td>
      <td>${esc(Object.entries(row.metrics || {}).map(([k, v]) => `${k}=${fmt(v)}`).join(" · "))}</td>
    </tr>
  `).join("");
  return `
    <h3>Performance under ${esc(result.shift_type)} shift</h3>
    <p class="muted">${esc(result.algorithm_slug)} on ${esc(result.dataset_name)} · degradation ${esc(fmt(result.degradation))}</p>
    <table>
      <thead><tr><th>Magnitude</th><th>${esc(result.results?.[0]?.metric_name || "metric")}</th><th>MMD²</th><th>All metrics</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
    ${jsonBlock(result)}
  `;
}

/* ------------------------------------------------------------------ */
/* Data loading / bootstrap                                            */
/* ------------------------------------------------------------------ */

async function loadAlgorithms() {
  state.algorithms = await api("/api/algorithms");
  renderAlgoFilters();
  renderAlgoList();
  renderOverview();
}

async function loadDatasets() {
  state.datasets = await api("/api/datasets");
  renderDatasetList();
  renderOverview();
}

function refreshDatasetSelects() {
  ["#train-dataset", "#arena-dataset", "#rel-dataset"].forEach((sel) => fillDatasetSelect($(sel)));
  refreshTrainAlgorithms();
  refreshArenaAlgorithms();
  refreshReliabilityAlgorithms();
}

async function checkHealth() {
  try {
    const health = await api("/api/health");
    $("#health-dot").className = "dot ok";
    $("#health-text").textContent = `API v${health.version} online`;
  } catch {
    $("#health-dot").className = "dot bad";
    $("#health-text").textContent = "API offline";
  }
}

async function refreshAll() {
  const button = $("#refresh-btn");
  button.disabled = true;
  try {
    await checkHealth();
    const results = await Promise.allSettled([loadAlgorithms(), loadDatasets(), loadExperiments()]);
    const failed = results.filter((r) => r.status === "rejected");
    refreshDatasetSelects();
    if (failed.length) toast(failed[0].reason.message, "error");
  } finally {
    button.disabled = false;
  }
}

function bindEvents() {
  $$(".nav-item").forEach((btn) => btn.addEventListener("click", () => showTab(btn.dataset.tab)));
  $("#refresh-btn").addEventListener("click", refreshAll);

  $("#algo-search").addEventListener("input", renderAlgoList);
  $("#algo-family").addEventListener("change", renderAlgoList);
  $("#algo-task").addEventListener("change", renderAlgoList);

  $("#upload-form").addEventListener("submit", uploadDataset);

  $("#train-dataset").addEventListener("change", refreshTrainAlgorithms);
  $("#train-algorithm").addEventListener("change", refreshHyperparams);
  $("#train-btn").addEventListener("click", trainModel);

  $("#experiments-refresh").addEventListener("click", loadExperiments);

  $("#arena-dataset").addEventListener("change", refreshArenaAlgorithms);
  $("#arena-run").addEventListener("click", runArena);
  $("#arena-leaderboard").addEventListener("click", loadLeaderboard);

  $("#rel-dataset").addEventListener("change", refreshReliabilityAlgorithms);
  $$("[data-rel]").forEach((btn) => btn.addEventListener("click", () => runReliability(btn.dataset.rel)));
}

bindEvents();
refreshAll();
