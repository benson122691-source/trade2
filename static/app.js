const form = document.getElementById("simulation-form");
const submitButton = document.getElementById("submit-button");
const statusMessage = document.getElementById("status-message");
const errorMessage = document.getElementById("error-message");
const results = document.getElementById("results");
const metricGrid = document.getElementById("metric-grid");
const forecastRows = document.getElementById("forecast-rows");
const warningList = document.getElementById("warning-list");
const historyChart = document.getElementById("history-chart");
const chartLegend = document.getElementById("chart-legend");
const numberFormat = new Intl.NumberFormat("zh-TW", { maximumFractionDigits: 0 });

function money(value) {
  return "NT$ " + numberFormat.format(value);
}

function percent(value) {
  if (value === null || value === undefined || !Number.isFinite(value)) {
    return "無法計算";
  }
  const digits = (value >= 0 ? "+" : "") + (value * 100).toFixed(1) + "%";
  return digits;
}

function makeSvg(tag, attributes, text) {
  const node = document.createElementNS("http://www.w3.org/2000/svg", tag);
  Object.entries(attributes || {}).forEach(([key, value]) => {
    node.setAttribute(key, String(value));
  });
  if (text !== undefined) node.textContent = text;
  return node;
}

function addCell(row, text, className) {
  const cell = document.createElement("td");
  if (className) cell.className = className;
  cell.textContent = text;
  row.appendChild(cell);
  return cell;
}

function renderMetrics(items) {
  metricGrid.replaceChildren();
  items.forEach((item) => {
    const card = document.createElement("article");
    card.className = "metric-card";
    card.style.setProperty("--asset-color", item.color);

    const title = document.createElement("p");
    title.className = "metric-name";
    title.textContent = item.name;

    const value = document.createElement("p");
    value.className = "metric-value";
    value.textContent = money(item.final_value);

    const profit = document.createElement("p");
    profit.className = "metric-profit " + (item.profit >= 0 ? "positive" : "negative");
    profit.textContent = "損益 " + (item.profit >= 0 ? "+" : "") + money(item.profit);

    const details = document.createElement("div");
    details.className = "metric-details";
    const values = [
      ["年化報酬", percent(item.annual_return)],
      ["年化波動", percent(item.annual_volatility)],
      ["最大回撤", percent(item.max_drawdown)],
      ["實際投入", money(item.total_invested)],
    ];
    values.forEach(([label, text]) => {
      const group = document.createElement("div");
      const small = document.createElement("span");
      small.textContent = label;
      const strong = document.createElement("strong");
      strong.textContent = text;
      group.append(small, strong);
      details.appendChild(group);
    });

    card.append(title, value, profit, details);
    metricGrid.appendChild(card);
  });
}

function renderHistoryChart(histories, metrics) {
  historyChart.replaceChildren();
  chartLegend.replaceChildren();
  if (!histories.length) return;

  const width = 1000;
  const height = 420;
  const left = 92;
  const right = 28;
  const top = 18;
  const bottom = 46;
  const plotWidth = width - left - right;
  const plotHeight = height - top - bottom;
  const timestamps = histories.map((row) => Date.parse(row.date));
  const minDate = Math.min(...timestamps);
  const maxDate = Math.max(...timestamps);
  const maxValue = Math.max(...histories.map((row) => row.value));
  const yMax = Math.max(1, maxValue * 1.08);

  for (let i = 0; i <= 4; i += 1) {
    const y = top + (plotHeight * i) / 4;
    const amount = yMax * (1 - i / 4);
    historyChart.appendChild(
      makeSvg("line", {
        x1: left,
        y1: y,
        x2: width - right,
        y2: y,
        class: "chart-grid-line",
      })
    );
    historyChart.appendChild(
      makeSvg(
        "text",
        { x: left - 12, y: y + 4, "text-anchor": "end", class: "chart-axis" },
        numberFormat.format(amount)
      )
    );
  }

  for (let i = 0; i <= 4; i += 1) {
    const x = left + (plotWidth * i) / 4;
    const timestamp = minDate + ((maxDate - minDate) * i) / 4;
    const label = new Date(timestamp).toLocaleDateString("zh-TW", {
      year: "numeric",
      month: "2-digit",
    });
    historyChart.appendChild(
      makeSvg("text", {
        x: x,
        y: height - 12,
        "text-anchor": i === 0 ? "start" : i === 4 ? "end" : "middle",
        class: "chart-axis",
      }, label)
    );
  }

  metrics.forEach((metric) => {
    const rows = histories
      .filter((item) => item.ticker === metric.ticker)
      .sort((a, b) => a.date.localeCompare(b.date));
    if (!rows.length) return;
    const points = rows.map((item) => {
      const x = left + ((Date.parse(item.date) - minDate) / (maxDate - minDate || 1)) * plotWidth;
      const y = top + (1 - item.value / yMax) * plotHeight;
      return x.toFixed(1) + "," + y.toFixed(1);
    });
    const path = makeSvg("path", {
      d: "M " + points.join(" L "),
      class: "chart-series",
      stroke: metric.color,
    });
    const title = makeSvg("title", {}, metric.name + "，期末 " + money(rows[rows.length - 1].value));
    path.appendChild(title);
    historyChart.appendChild(path);

    const legend = document.createElement("span");
    legend.className = "legend-item";
    const swatch = document.createElement("span");
    swatch.className = "legend-swatch";
    swatch.style.backgroundColor = metric.color;
    const label = document.createElement("span");
    label.textContent = metric.name;
    legend.append(swatch, label);
    chartLegend.appendChild(legend);
  });
}

function renderForecasts(items) {
  forecastRows.replaceChildren();
  items.forEach((item) => {
    const row = document.createElement("tr");
    addCell(row, item.name);
    addCell(row, item.model, "model-name");
    addCell(row, percent(item.prediction));

    const scenario = document.createElement("td");
    scenario.className = "scenario-cell";
    scenario.textContent = percent(item.lower) + " 至 " + percent(item.upper);
    const amount = document.createElement("small");
    amount.textContent = money(item.lower_value) + " 至 " + money(item.upper_value);
    scenario.appendChild(amount);
    row.appendChild(scenario);

    addCell(row, percent(item.mae));
    addCell(row, percent(item.zero_baseline_mae));
    forecastRows.appendChild(row);
  });
}

function renderWarnings(items) {
  warningList.replaceChildren();
  items.forEach((message) => {
    const paragraph = document.createElement("p");
    paragraph.textContent = message;
    warningList.appendChild(paragraph);
  });
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  errorMessage.hidden = true;
  errorMessage.textContent = "";
  statusMessage.hidden = false;
  statusMessage.lastElementChild.textContent = "正在下載行情並計算，資料量較大時可能需要一些時間。";
  submitButton.disabled = true;
  submitButton.textContent = "正在計算…";

  const payload = {
    start_date: form.elements.start_date.value,
    end_date: form.elements.end_date.value,
    principal: Number(form.elements.principal.value),
    method: form.elements.method.value,
    horizon: Number(form.elements.horizon.value),
    assets: Array.from(form.querySelectorAll('input[name="assets"]:checked')).map(
      (input) => input.value
    ),
  };

  try {
    const response = await fetch("/api/simulate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || "模擬失敗，請檢查輸入條件。");

    renderMetrics(data.metrics);
    renderHistoryChart(data.histories, data.metrics);
    renderForecasts(data.forecasts);
    renderWarnings(data.warnings);
    document.getElementById("period-caption").textContent =
      payload.start_date + " 至 " + payload.end_date +
      (data.method === "monthly" ? " · 定期定額" : " · 一次投入");
    results.hidden = false;
    statusMessage.lastElementChild.textContent = "模擬完成。各標的分別以完整預算計算。";
  } catch (error) {
    errorMessage.textContent = error.message || "模擬失敗，請確認網路連線後再試。";
    errorMessage.hidden = false;
  } finally {
    submitButton.disabled = false;
    submitButton.textContent = "開始模擬";
  }
});
