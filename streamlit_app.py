"""
CreditStream - live credit default prediction
Streamlit host + custom HTML/CSS/JavaScript dashboard (st.components.v2).

Run locally:   streamlit run streamlit_app.py
"""
import math
import random
import threading
import time
from collections import Counter, deque
from datetime import datetime

import pandas as pd
import streamlit as st
from faker import Faker
from river import forest

FIELDS = [
    "timestamp", "customer_id", "name", "age", "annual_income", "employment_years",
    "credit_history_years", "credit_score", "credit_limit", "current_balance",
    "utilization_rate", "debt_to_income", "open_accounts", "recent_inquiries_6m",
    "payment_ratio", "missed_payments_6m", "default_next_month",
]
NON_FEATURES = {"timestamp", "customer_id", "name", "default_next_month"}


# --------------------------------------------------------------------------
# 1. Data generator: a small, realistic model of a credit-card portfolio
#
# How the customer is built (each step depends on the ones before it):
#   latent "financial discipline" z  ->  income, utilization, payment habits,
#   debt load, late payments, inquiries  ->  credit limit and balance  ->
#   credit score (FICO-style weights)  ->  default (logistic risk model).
# --------------------------------------------------------------------------
INTERCEPT = -3.7  # tuned so ~11% of customers default (real portfolios are
                  # 2-5%; a higher rate gives the live metrics enough positives)


def _poisson(lam: float) -> int:
    limit, k, p = math.exp(-lam), 0, 1.0
    while True:
        p *= random.random()
        if p <= limit:
            return k
        k += 1


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def default_logit(c: dict) -> float:
    """Log-odds of default: every term below is a well-known risk driver."""
    return (
        INTERCEPT
        + 0.75 * c["missed_payments_6m"]                       # each late payment hurts
        + 3.2 * (c["utilization_rate"] - 0.30)                 # maxed-out cards
        + 3.5 * (c["debt_to_income"] - 0.30)                   # debt burden
        - 0.010 * (c["credit_score"] - 680)                    # bureau score
        - 1.3 * (c["payment_ratio"] - 0.50)                    # pays full balance vs minimum
        + 0.22 * c["recent_inquiries_6m"]                      # credit-seeking behaviour
        - 0.05 * min(c["employment_years"], 10)                # job stability
        - 0.35 * math.log(c["annual_income"] / 50000)          # income cushion
        + (0.6 if c["age"] < 26 else 0.0)                      # thin, young files
        + (1.4 if c["utilization_rate"] > 0.80 and c["missed_payments_6m"] >= 1 else 0.0)  # stress combo
    )


def generate_live_customer_profile(fake: Faker) -> dict:
    z = random.gauss(0, 1)  # hidden financial discipline (higher = safer)

    # Demographics and income: income rises with age until ~60, then falls.
    age = int(_clip(random.gauss(42, 12), 21, 75))
    life = 0.55 + 0.45 * min(1.0, (age - 21) / 14)
    if age > 62:
        life *= max(0.6, 1 - 0.02 * (age - 62))
    income = int(_clip(50000 * life * math.exp(random.gauss(0, 0.5) + 0.12 * z), 15000, 400000) // 500 * 500)
    employment_years = round(min(age - 21, random.expovariate(1 / 6)), 1)

    # Credit file: history length follows age; more income and history -> more accounts.
    history = round(_clip((age - 18) * random.uniform(0.35, 1.0), 0.5, age - 18), 1)
    open_accounts = int(_clip(1 + _poisson(1.0 + 0.12 * history + income / 100000), 1, 15))

    # Utilization: disciplined and higher-income people use less of their limit.
    mean_util = _clip(0.28 - 0.11 * z - 0.04 * math.log(income / 50000) + (0.05 if age < 30 else 0), 0.03, 0.92)
    util = round(_clip(random.betavariate(mean_util * 5, (1 - mean_util) * 5), 0.01, 0.99), 2)

    # Payment behaviour: "transactors" pay in full; "revolvers" pay near the minimum.
    p_full = _sigmoid(0.2 + 0.9 * z - 3.0 * (util - 0.30))
    if random.random() < p_full:
        ratio = random.uniform(0.95, 1.0)
    else:
        ratio = min(0.9, random.betavariate(1.6, 5.0) * (1 + 0.25 * max(z, -1)))
    payment_ratio = round(ratio, 2)

    # Debt-to-income: higher for low earners, heavy card users and weaker borrowers.
    dti = random.betavariate(2.5, 5.5) * (1 - 0.12 * z) * (1.25 if income < 35000 else 1.0) + 0.12 * util
    dti = round(_clip(dti, 0.02, 0.90), 2)

    # Late payments (0-6 in 6 months) and credit applications.
    lam = (0.14 * math.exp(-0.85 * z) * (1 + 2.0 * max(0, util - 0.5))
           * (1 + 1.5 * max(0, dti - 0.4)) * (1.8 - 1.3 * payment_ratio))
    missed = min(6, _poisson(lam))
    inquiries = min(8, _poisson(0.45 * math.exp(-0.35 * z) * (1.4 if age < 30 else 1.0)))

    # The issuer sets the limit from income and perceived quality; balance follows utilization.
    factor = _clip(0.24 + 0.05 * z, 0.08, 0.5) * random.uniform(0.8, 1.2)
    limit = max(500, int(round(income * factor / 500) * 500))
    balance = int(limit * util)

    # Credit score, FICO-style weights: payment history 35%, utilization 30%,
    # history length 15%, new credit 10%, account mix 10%.
    pay = _clip(1 - 0.30 * missed + 0.05 * (payment_ratio - 0.5), 0, 1)
    util_pts = 1 - min(1.0, (util / 0.9) ** 0.9)
    blend = (0.35 * pay + 0.30 * util_pts + 0.15 * min(history / 20, 1)
             + 0.10 * max(0, 1 - 0.18 * inquiries) + 0.10 * min(open_accounts / 8, 1))
    credit_score = int(_clip(300 + 550 * blend + random.gauss(0, 25), 300, 850))

    customer = {
        "age": age, "annual_income": income, "employment_years": employment_years,
        "credit_history_years": history, "credit_score": credit_score,
        "credit_limit": limit, "current_balance": balance, "utilization_rate": util,
        "debt_to_income": dti, "open_accounts": open_accounts,
        "recent_inquiries_6m": inquiries, "payment_ratio": payment_ratio,
        "missed_payments_6m": missed,
    }
    # Outcome: logistic risk plus hidden factors we never observe (job loss, illness...).
    p_default = _sigmoid(default_logit(customer) + random.gauss(0, 0.5))
    customer["default_next_month"] = 1 if random.random() < p_default else 0

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "customer_id": fake.uuid4()[:8],
        "name": fake.name(),
        **customer,
    }


def to_features(rec: dict) -> dict:
    return {k: float(v) for k, v in rec.items() if k not in NON_FEATURES}


# --------------------------------------------------------------------------
# 2. Streaming engine: runs in a background thread, shared by all sessions
# --------------------------------------------------------------------------
AUC_BINS = 1000


class StreamEngine:
    def __init__(self):
        self.lock = threading.Lock()
        self.fake = Faker()
        self.running = True
        self.delay = 0.3  # seconds per record
        self._init_state()
        threading.Thread(target=self._loop, daemon=True).start()

    def _init_state(self):
        self.model = forest.ARFClassifier(
            n_models=25, grace_period=50, delta=0.1, tau=0.05, seed=42
        )
        self.cm = Counter()  # keys: (actual, predicted)
        # Streaming ROC-AUC: histograms of predicted risk for defaulters (pos)
        # and non-defaulters (neg). Constant memory, covers every record so far.
        self.auc_pos = [0] * AUC_BINS
        self.auc_neg = [0] * AUC_BINS
        self.n = 0
        self.feed = deque(maxlen=14)
        self.history = deque(maxlen=300)
        self.records = deque(maxlen=5000)

    def reset(self):
        with self.lock:
            self._init_state()

    # -- metrics from the confusion matrix -----------------------------------
    def _metrics(self):
        tp, fp = self.cm[(1, 1)], self.cm[(0, 1)]
        fn, tn = self.cm[(1, 0)], self.cm[(0, 0)]
        total = tp + fp + fn + tn
        acc = (tp + tn) / total if total else 0.0
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        return dict(accuracy=acc, precision=prec, recall=rec, f1=f1,
                    auc=self._auc(), tp=tp, fp=fp, fn=fn, tn=tn)

    def _auc(self):
        """P(a random defaulter is ranked riskier than a random non-defaulter)."""
        pos, neg = sum(self.auc_pos), sum(self.auc_neg)
        if pos == 0 or neg == 0:
            return None  # undefined until both outcomes have been seen
        below, wins = 0, 0.0
        for p, n in zip(self.auc_pos, self.auc_neg):
            wins += p * (below + 0.5 * n)  # ties count as half
            below += n
        return wins / (pos * neg)

    def _loop(self):
        while True:
            if not self.running:
                time.sleep(0.3)
                continue
            rec = generate_live_customer_profile(self.fake)
            x, y = to_features(rec), rec["default_next_month"]
            with self.lock:
                proba = self.model.predict_proba_one(x)
                risk = float(proba.get(1, 0.0)) if proba else None
                pred = (1 if risk >= 0.5 else 0) if risk is not None else None
                if pred is not None:  # test-then-train (prequential)
                    self.cm[(y, pred)] += 1
                    b = min(AUC_BINS - 1, int(risk * AUC_BINS))
                    (self.auc_pos if y == 1 else self.auc_neg)[b] += 1
                self.model.learn_one(x, y)
                self.n += 1
                self.records.append(rec)
                self.feed.appendleft({
                    "time": datetime.now().strftime("%H:%M:%S"),
                    "id": rec["customer_id"], "name": rec["name"],
                    "score": rec["credit_score"], "util": rec["utilization_rate"],
                    "missed": rec["missed_payments_6m"],
                    "risk": risk, "pred": pred, "actual": y,
                })
                if self.n % 5 == 0 and pred is not None:
                    m = self._metrics()
                    self.history.append({"n": self.n, "acc": m["accuracy"], "f1": m["f1"], "auc": m["auc"]})
            time.sleep(self.delay)

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "running": self.running,
                "n": self.n,
                "metrics": self._metrics(),
                "history": list(self.history),
                "feed": list(self.feed),
                "default_rate": (
                    sum(r["default_next_month"] for r in self.records) / len(self.records)
                    if self.records else 0.0
                ),
            }

    def score(self, x: dict):
        with self.lock:
            proba = self.model.predict_proba_one(x)
        return float(proba.get(1, 0.0)) if proba else None

    def dataframe(self) -> pd.DataFrame:
        with self.lock:
            return pd.DataFrame(list(self.records), columns=FIELDS)


@st.cache_resource
def get_engine() -> StreamEngine:
    return StreamEngine()


# --------------------------------------------------------------------------
# 3. Front end: HTML + CSS + JavaScript.
#    Mounted ONCE by Streamlit; every refresh only calls the JS function with
#    new data, which updates the existing DOM in place (no reload, no blink).
# --------------------------------------------------------------------------
DASH_HTML = """
<div class="wrap">
  <header>
    <div>
      <h1>Credit<span>Stream</span></h1>
      <div class="sub">Live default prediction with an online Adaptive Random Forest</div>
    </div>
    <div id="status" class="pill"><span class="dot"></span><span id="statusText">LIVE</span></div>
  </header>

  <section class="kpis">
    <div class="card kpi"><div class="label">Accuracy</div><div class="value" id="acc">-</div><div class="hint">overall correct</div></div>
    <div class="card kpi"><div class="label">Precision</div><div class="value" id="prec">-</div><div class="hint">alarms that were real</div></div>
    <div class="card kpi"><div class="label">Recall</div><div class="value" id="rec">-</div><div class="hint">defaults caught</div></div>
    <div class="card kpi"><div class="label">F1 score</div><div class="value" id="f1">-</div><div class="hint">balanced</div></div>
    <div class="card kpi"><div class="label">ROC AUC</div><div class="value" id="auc">-</div><div class="hint">ranking quality</div></div>
    <div class="card kpi"><div class="label">Records</div><div class="value" id="n">0</div><div class="hint" id="dr">warming up</div></div>
  </section>

  <section class="row">
    <div class="card">
      <h2>Performance over time</h2>
      <canvas id="chart"></canvas>
      <div class="legend">
        <span><i style="background:#6c8cff"></i>Accuracy</span>
        <span><i style="background:#fbbf24"></i>F1 score</span>
        <span><i style="background:#34d399"></i>ROC AUC</span>
      </div>
    </div>
    <div class="card">
      <h2>Confusion matrix</h2>
      <div class="cm">
        <div></div><div class="h">Predicted safe</div><div class="h">Predicted default</div>
        <div class="h">Actually<br>safe</div>
        <div class="cell tn"><b id="tn">0</b><small>True negatives</small></div>
        <div class="cell fp"><b id="fp">0</b><small>False alarms</small></div>
        <div class="h">Actually<br>default</div>
        <div class="cell fn"><b id="fn">0</b><small>Missed defaults</small></div>
        <div class="cell tp"><b id="tp">0</b><small>Caught defaults</small></div>
      </div>
    </div>
  </section>

  <section class="card">
    <h2>Live customer feed</h2>
    <div style="overflow-x:auto">
      <table>
        <thead><tr>
          <th>Time</th><th>Customer</th><th>Score</th><th>Util.</th><th>Missed</th>
          <th>Default risk</th><th>Prediction</th><th>Actual</th><th></th>
        </tr></thead>
        <tbody id="feed"></tbody>
      </table>
    </div>
  </section>
</div>
"""

DASH_CSS = """
:host {
  display: block;
  --bg: #0b1020; --card: #131a2e; --line: #1f2a47; --text: #e6e9f2;
  --muted: #8b94ad; --accent: #6c8cff; --good: #34d399; --bad: #f87171; --warn: #fbbf24;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
.wrap { background: var(--bg); color: var(--text); border-radius: 16px; padding: 18px;
        font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 16px; }
h1 { font-size: 22px; font-weight: 700; letter-spacing: .2px; }
h1 span { color: var(--accent); }
.sub { color: var(--muted); font-size: 13px; margin-top: 2px; }
.pill { display: inline-flex; align-items: center; gap: 8px; padding: 6px 12px;
        border-radius: 999px; font-size: 12px; font-weight: 600;
        background: var(--card); border: 1px solid var(--line); }
.dot { width: 8px; height: 8px; border-radius: 50%; background: var(--good); animation: pulse 1.4s infinite; }
.paused .dot { background: var(--warn); animation: none; }
@keyframes pulse { 0% { box-shadow: 0 0 0 0 rgba(52,211,153,.6); }
                   100% { box-shadow: 0 0 0 10px rgba(52,211,153,0); } }

.kpis { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; margin-bottom: 12px; }
.card { background: var(--card); border: 1px solid var(--line); border-radius: 14px; padding: 14px 16px; }
.kpi .label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .6px; }
.kpi .value { font-size: 26px; font-weight: 700; margin-top: 6px; }
.kpi .hint { color: var(--muted); font-size: 11px; margin-top: 2px; }

.row { display: grid; grid-template-columns: 2fr 1fr; gap: 12px; margin-bottom: 12px; }
.card h2 { font-size: 13px; color: var(--muted); font-weight: 600; text-transform: uppercase;
           letter-spacing: .6px; margin-bottom: 10px; }
canvas { width: 100%; height: 250px; display: block; }
.legend { display: flex; gap: 14px; font-size: 12px; color: var(--muted); margin-top: 6px; }
.legend i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 6px; }

.cm { display: grid; grid-template-columns: auto 1fr 1fr; gap: 6px; align-items: stretch; font-size: 12px; }
.cm .h { color: var(--muted); text-align: center; align-self: center; padding: 4px; }
.cm .cell { border-radius: 10px; padding: 16px 8px; text-align: center; }
.cm .cell b { display: block; font-size: 24px; }
.cm .cell small { color: var(--muted); }
.tn, .tp { background: rgba(52,211,153,.14); border: 1px solid rgba(52,211,153,.35); }
.fp, .fn { background: rgba(248,113,113,.12); border: 1px solid rgba(248,113,113,.35); }

table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; color: var(--muted); font-weight: 600; font-size: 11px;
     text-transform: uppercase; letter-spacing: .5px; padding: 6px 8px; }
td { padding: 8px; border-top: 1px solid var(--line); white-space: nowrap; }
.bar { width: 90px; height: 6px; background: var(--line); border-radius: 4px; display: inline-block;
       vertical-align: middle; margin-right: 8px; overflow: hidden; }
.bar span { display: block; height: 100%; border-radius: 4px; }
.badge { padding: 3px 9px; border-radius: 999px; font-size: 11px; font-weight: 600; }
.b-high { background: rgba(248,113,113,.16); color: var(--bad); }
.b-low  { background: rgba(52,211,153,.16); color: var(--good); }
.b-na   { background: var(--line); color: var(--muted); }
.ok { color: var(--good); } .no { color: var(--bad); }
.id { color: var(--muted); }
.empty { color: var(--muted); text-align: center; padding: 24px; font-size: 13px; }

@media (max-width: 760px) {
  .kpis { grid-template-columns: repeat(2, 1fr); }
  .row { grid-template-columns: 1fr; }
  td:nth-child(2), th:nth-child(2) { display: none; }
}
"""

DASH_JS = r"""
let LAST = null;     // most recent {root, data}, used to redraw on window resize
let WIRED = false;   // make sure the resize listener is added only once

const pct = v => (v * 100).toFixed(1) + "%";
const esc = s => String(s).replace(/[&<>"']/g,
  c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));

function setText(root, id, text) {
  const el = root.querySelector("#" + id);
  if (el && el.textContent !== text) el.textContent = text;
}

function drawChart(root, D, retry = true) {
  const c = root.querySelector("#chart");
  if (!c) return;
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (!w) { if (retry) requestAnimationFrame(() => drawChart(root, D, false)); return; }
  c.width = w * dpr; c.height = h * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr);

  const padL = 38, padR = 10, padT = 10, padB = 22;
  const iw = w - padL - padR, ih = h - padT - padB;
  g.font = "11px sans-serif"; g.textBaseline = "middle";
  for (let i = 0; i <= 4; i++) {
    const y = padT + ih * (i / 4);
    g.strokeStyle = "#1f2a47"; g.lineWidth = 1;
    g.beginPath(); g.moveTo(padL, y); g.lineTo(w - padR, y); g.stroke();
    g.fillStyle = "#8b94ad"; g.textAlign = "right";
    g.fillText((100 - i * 25) + "%", padL - 6, y);
  }

  const H = D.history;
  if (H.length < 2) {
    g.fillStyle = "#8b94ad"; g.textAlign = "center";
    g.fillText("Collecting data...", w / 2, h / 2);
    return;
  }
  const x0 = H[0].n, x1 = H[H.length - 1].n;
  const X = n => padL + iw * ((n - x0) / Math.max(1, x1 - x0));
  const Y = v => padT + ih * (1 - v);
  const line = (key, color) => {
    g.strokeStyle = color; g.lineWidth = 2; g.lineJoin = "round";
    g.beginPath();
    let started = false;
    H.forEach(p => {
      if (p[key] === null || p[key] === undefined) return;  // metric not defined yet
      started ? g.lineTo(X(p.n), Y(p[key])) : g.moveTo(X(p.n), Y(p[key]));
      started = true;
    });
    g.stroke();
  };
  line("acc", "#6c8cff");
  line("f1", "#fbbf24");
  line("auc", "#34d399");
  g.fillStyle = "#8b94ad";
  g.textAlign = "left";  g.fillText(String(x0), padL, h - 8);
  g.textAlign = "right"; g.fillText(x1 + " records", w - padR, h - 8);
}

function feedRow(r) {
  const risk = r.risk;
  const color = risk === null ? "#8b94ad" : risk >= 0.5 ? "#f87171" : risk >= 0.25 ? "#fbbf24" : "#34d399";
  const riskCell = risk === null
    ? '<span class="badge b-na">learning</span>'
    : '<span class="bar"><span style="width:' + Math.round(risk * 100) + '%;background:' + color + '"></span></span>'
      + Math.round(risk * 100) + "%";
  const predCell = r.pred === null ? '<span class="badge b-na">-</span>'
    : r.pred === 1 ? '<span class="badge b-high">High risk</span>' : '<span class="badge b-low">Low risk</span>';
  const actCell = r.actual === 1 ? '<span class="badge b-high">Defaulted</span>' : '<span class="badge b-low">Paid</span>';
  const mark = r.pred === null ? "" : (r.pred === r.actual ? '<span class="ok">&#10003;</span>' : '<span class="no">&#10007;</span>');
  return "<tr><td>" + esc(r.time) + "</td><td>" + esc(r.name) + ' <span class="id">#' + esc(r.id) + "</span></td>"
    + "<td>" + r.score + "</td><td>" + Math.round(r.util * 100) + "%</td><td>" + r.missed + "</td>"
    + "<td>" + riskCell + "</td><td>" + predCell + "</td><td>" + actCell + "</td><td>" + mark + "</td></tr>";
}

// Streamlit calls this on mount AND every time new data arrives.
export default function (component) {
  const { data: D, parentElement: root } = component;
  if (!D) return;
  LAST = { root, data: D };

  const status = root.querySelector("#status");
  if (status) status.classList.toggle("paused", !D.running);
  setText(root, "statusText", D.running ? "LIVE" : "PAUSED");

  const m = D.metrics;
  const scored = m.tp + m.fp + m.fn + m.tn > 0;
  setText(root, "acc",  scored ? pct(m.accuracy)  : "-");
  setText(root, "prec", scored ? pct(m.precision) : "-");
  setText(root, "rec",  scored ? pct(m.recall)    : "-");
  setText(root, "f1",   scored ? pct(m.f1)        : "-");
  setText(root, "auc",  m.auc === null || m.auc === undefined ? "-" : m.auc.toFixed(3));
  setText(root, "n",    D.n.toLocaleString());
  setText(root, "dr",   D.n ? pct(D.default_rate) + " defaulted" : "warming up");
  ["tn", "fp", "fn", "tp"].forEach(k => setText(root, k, m[k].toLocaleString()));

  const body = root.querySelector("#feed");
  if (body) {
    body.innerHTML = D.feed.length
      ? D.feed.map(feedRow).join("")
      : '<tr><td colspan="9" class="empty">Waiting for the first records...</td></tr>';
  }

  drawChart(root, D);

  if (!WIRED) {
    WIRED = true;
    window.addEventListener("resize", () => LAST && drawChart(LAST.root, LAST.data));
  }
}
"""

dashboard_component = st.components.v2.component(
    "creditstream_dashboard", html=DASH_HTML, css=DASH_CSS, js=DASH_JS
)


# --------------------------------------------------------------------------
# 4. Streamlit page
# --------------------------------------------------------------------------
st.set_page_config(page_title="CreditStream", page_icon="💳", layout="wide")
st.markdown(
    "<style>.block-container{padding-top:1.5rem;} #MainMenu,footer{visibility:hidden;}</style>",
    unsafe_allow_html=True,
)

engine = get_engine()

with st.sidebar:
    st.header("Controls")
    # Fixed defaults + keys: the widgets keep their state while you interact.
    paused = st.toggle("Pause stream", key="paused")
    engine.running = not paused
    engine.delay = st.slider("Seconds per record", 0.05, 2.0, 0.3, 0.05, key="speed")

    c1, c2 = st.columns(2)
    if c1.button("Reset model", use_container_width=True):
        engine.reset()
    c2.download_button(
        "Download CSV",
        data=lambda: engine.dataframe().to_csv(index=False),  # built when clicked
        file_name="live_credit_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.divider()
    st.header("Score an applicant")
    st.caption("Updates instantly as you move any slider.")
    age = st.slider("Age", 18, 90, 35)
    income = st.slider("Annual income ($)", 10_000, 300_000, 60_000, step=1_000)
    credit_score = st.slider("Credit score", 300, 850, 680)
    util = st.slider("Utilization rate", 0.0, 1.0, 0.30, 0.01)
    missed = st.select_slider("Missed payments (6m)", options=[0, 1, 2, 3, 4, 5, 6], value=0)
    with st.expander("More details"):
        employment_years = st.slider("Years at current job", 0.0, 40.0, 5.0, 0.5)
        history = st.slider("Credit history (years)", 0.5, 50.0, 10.0, 0.5)
        dti = st.slider("Debt-to-income ratio", 0.0, 0.9, 0.35, 0.01)
        open_accounts = st.slider("Open accounts", 1, 15, 4)
        inquiries = st.slider("Credit inquiries (6m)", 0, 8, 0)
        payment_ratio = st.slider("Share of balance paid each month", 0.0, 1.0, 0.60, 0.01)

    limit = int(income * 0.25)  # assumption: credit limit = 25% of income
    risk = engine.score({
        "age": age, "annual_income": income, "employment_years": employment_years,
        "credit_history_years": history, "credit_score": credit_score,
        "credit_limit": limit, "current_balance": int(limit * util),
        "utilization_rate": util, "debt_to_income": dti, "open_accounts": open_accounts,
        "recent_inquiries_6m": inquiries, "payment_ratio": payment_ratio,
        "missed_payments_6m": missed,
    })
    if risk is None:
        st.info("The model is still warming up. Give it a few seconds.")
    else:
        st.metric("Predicted default risk", f"{risk:.0%}")
        st.progress(min(max(risk, 0.0), 1.0))
        (st.error if risk >= 0.5 else st.success)(
            "High risk" if risk >= 0.5 else "Low risk"
        )
        st.caption("Assumes a credit limit of 25% of income.")


@st.fragment(run_every=1)
def dashboard():
    # Same key every time -> the component is updated in place, not recreated.
    dashboard_component(data=engine.snapshot(), key="dashboard")


dashboard()