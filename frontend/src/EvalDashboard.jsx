import { useState, useEffect } from "react";

const API_BASE = "http://localhost:8000/api";

const COLORS = {
  sky: "#38bdf8",
  indigo: "#818cf8",
  teal: "#2dd4bf",
  green: "#4ade80",
  yellow: "#fbbf24",
  red: "#f87171",
  muted: "#64748b",
};

function scoreColor(val, max = 1) {
  const r = val / max;
  if (r >= 0.7) return COLORS.green;
  if (r >= 0.4) return COLORS.yellow;
  return COLORS.red;
}

function RadialGauge({ value, max = 1, label, color }) {
  const pct = Math.min(value / max, 1);
  const r = 38, cx = 50, cy = 50;
  const circumference = 2 * Math.PI * r;
  const dash = pct * circumference * 0.75;
  const gap = circumference * 0.75 - dash;
  const rotation = 135;

  return (
    <div className="gauge-wrap">
      <svg viewBox="0 0 100 100" className="gauge-svg">
        {/* Track */}
        <circle cx={cx} cy={cy} r={r} fill="none" stroke="rgba(255,255,255,0.05)"
          strokeWidth="8" strokeDasharray={`${circumference*0.75} ${circumference*0.25}`}
          strokeDashoffset={0} strokeLinecap="round"
          transform={`rotate(${rotation} ${cx} ${cy})`}/>
        {/* Fill */}
        <circle cx={cx} cy={cy} r={r} fill="none" stroke={color}
          strokeWidth="8" strokeDasharray={`${dash} ${gap + circumference*0.25}`}
          strokeDashoffset={0} strokeLinecap="round"
          transform={`rotate(${rotation} ${cx} ${cy})`}
          style={{ filter: `drop-shadow(0 0 6px ${color}60)`, transition: "stroke-dasharray 0.8s ease" }}/>
        <text x={cx} y={cy - 4} textAnchor="middle" fill="white" fontSize="16" fontWeight="700" fontFamily="Outfit">
          {(value * 100).toFixed(0)}
        </text>
        <text x={cx} y={cy + 10} textAnchor="middle" fill={COLORS.muted} fontSize="7" fontFamily="DM Mono">
          / 100
        </text>
      </svg>
      <p className="gauge-label">{label}</p>
    </div>
  );
}

function BarRow({ label, value, max, color, unit = "" }) {
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="bar-row">
      <span className="bar-label">{label}</span>
      <div className="bar-track">
        <div className="bar-fill" style={{ width: `${pct}%`, background: color, boxShadow: `0 0 8px ${color}60` }} />
      </div>
      <span className="bar-value" style={{ color }}>{typeof value === "number" ? value.toLocaleString() : value}{unit}</span>
    </div>
  );
}

function MetricCard({ title, value, sub, color, icon }) {
  return (
    <div className="metric-card">
      <div className="metric-icon" style={{ color }}>{icon}</div>
      <div>
        <p className="metric-value" style={{ color }}>{value}</p>
        <p className="metric-title">{title}</p>
        {sub && <p className="metric-sub">{sub}</p>}
      </div>
    </div>
  );
}

function QueryRow({ result, idx }) {
  const [open, setOpen] = useState(false);
  return (
    <>
      <tr className={`table-row ${open ? "expanded" : ""}`} onClick={() => setOpen(v => !v)}>
        <td className="td-idx">{idx + 1}</td>
        <td className="td-query">{result.query}</td>
        <td className="td-score" style={{ color: scoreColor(result.faithfulness) }}>
          {(result.faithfulness * 100).toFixed(1)}%
        </td>
        <td className="td-score" style={{ color: scoreColor(result.context_relevance) }}>
          {(result.context_relevance * 100).toFixed(1)}%
        </td>
        <td className="td-score" style={{ color: scoreColor(1 - result.latency_total_ms / 30000) }}>
          {result.latency_total_ms > 1000
            ? `${(result.latency_total_ms / 1000).toFixed(1)}s`
            : `${result.latency_total_ms.toFixed(0)}ms`}
        </td>
        <td className="td-score" style={{ color: COLORS.indigo }}>{result.tokens_in}</td>
        <td className="td-expand">{open ? "▲" : "▼"}</td>
      </tr>
      {open && (
        <tr className="table-detail">
          <td colSpan={7}>
            <div className="detail-box">
              <div className="detail-latency">
                <BarRow label="Retrieval" value={result.latency_retrieval_ms} max={Math.max(result.latency_total_ms, 1)} color={COLORS.sky} unit="ms" />
                <BarRow label="Rerank" value={result.latency_rerank_ms} max={Math.max(result.latency_total_ms, 1)} color={COLORS.indigo} unit="ms" />
                <BarRow label="LLM" value={result.latency_llm_ms} max={Math.max(result.latency_total_ms, 1)} color={COLORS.teal} unit="ms" />
              </div>
              <div className="detail-preview">
                <p className="detail-preview-label">Answer Preview</p>
                <p className="detail-preview-text">{result.answer_preview}…</p>
              </div>
            </div>
          </td>
        </tr>
      )}
    </>
  );
}

export default function EvalDashboard({ onClose }) {
  const [results, setResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [running, setRunning] = useState(false);
  const [docs, setDocs] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState("");
  const [customQueries, setCustomQueries] = useState("");
  const [error, setError] = useState(null);

  useEffect(() => {
    loadResults();
    fetchDocs();
  }, []);

  const loadResults = async () => {
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/eval/results`);
      if (res.ok) { const data = await res.json(); setResults(data); }
    } catch { /* use empty */ }
    finally { setLoading(false); }
  };

  const fetchDocs = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      const data = await res.json();
      const raw = data.documents || [];
      setDocs(raw);
      if (raw.length > 0) setSelectedDoc(raw[0]);
    } catch { }
  };

  const runEval = async () => {
    if (!selectedDoc) return;
    setRunning(true); setError(null);
    const queries = customQueries.trim()
      ? customQueries.split("\n").map(q => q.trim()).filter(Boolean)
      : null;
    try {
      const res = await fetch(`${API_BASE}/eval/run`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ doc_id: selectedDoc, queries }),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.detail || "Eval failed"); }
      const data = await res.json();
      setResults(data);
    } catch(e) { setError(e.message); }
    finally { setRunning(false); }
  };

  // Compute averages
  const avg = results.length > 0 ? {
    faithfulness: results.reduce((s, r) => s + r.faithfulness, 0) / results.length,
    relevance: results.reduce((s, r) => s + r.context_relevance, 0) / results.length,
    latency: results.reduce((s, r) => s + r.latency_total_ms, 0) / results.length,
    tokens: results.reduce((s, r) => s + r.tokens_in, 0) / results.length,
  } : null;

  const maxLatency = results.length > 0 ? Math.max(...results.map(r => r.latency_total_ms)) : 1;

  return (
    <div className="eval-overlay">
      <div className="eval-panel">
        {/* Header */}
        <div className="eval-header">
          <div>
            <h2 className="eval-title">Eval Dashboard</h2>
            <p className="eval-sub">Faithfulness · Relevance · Latency · Token Usage</p>
          </div>
          <button className="eval-close" onClick={onClose}>✕</button>
        </div>

        <div className="eval-body">
          {/* Run eval section */}
          <div className="eval-run-section">
            <div className="eval-run-controls">
              <select className="eval-select" value={selectedDoc} onChange={e => setSelectedDoc(e.target.value)}>
                {docs.map(d => <option key={d} value={d}>{d}</option>)}
              </select>
              <button className={`eval-run-btn ${running ? "running" : ""}`} onClick={runEval} disabled={running || !selectedDoc}>
                {running ? <><span className="eval-spinner"/>Running…</> : "▶ Run Eval"}
              </button>
            </div>
            <textarea className="eval-queries-input" placeholder={"Custom queries (one per line)\nLeave empty to use default queries"} value={customQueries} onChange={e => setCustomQueries(e.target.value)} rows={3}/>
            {error && <p className="eval-error">⚠ {error}</p>}
          </div>

          {loading ? (
            <div className="eval-loading"><span className="eval-spinner large"/><p>Loading results…</p></div>
          ) : results.length === 0 ? (
            <div className="eval-empty"><p>No eval results yet. Run an eval above!</p></div>
          ) : (
            <>
              {/* Gauges */}
              {avg && (
                <div className="eval-gauges">
                  <RadialGauge value={avg.faithfulness} label="Faithfulness" color={scoreColor(avg.faithfulness)} />
                  <RadialGauge value={avg.relevance} label="Relevance" color={scoreColor(avg.relevance)} />
                  <RadialGauge value={Math.max(0, 1 - avg.latency / 30000)} label="Speed Score" color={COLORS.teal} />
                </div>
              )}

              {/* Summary cards */}
              {avg && (
                <div className="eval-cards">
                  <MetricCard title="Avg Faithfulness" value={`${(avg.faithfulness*100).toFixed(1)}%`} sub="Answer ↔ Context overlap" color={scoreColor(avg.faithfulness)} icon="🎯"/>
                  <MetricCard title="Avg Relevance" value={`${(avg.relevance*100).toFixed(1)}%`} sub="Query ↔ Chunks overlap" color={scoreColor(avg.relevance)} icon="🔍"/>
                  <MetricCard title="Avg Latency" value={avg.latency > 1000 ? `${(avg.latency/1000).toFixed(1)}s` : `${avg.latency.toFixed(0)}ms`} sub="Total pipeline time" color={COLORS.sky} icon="⚡"/>
                  <MetricCard title="Avg Tokens In" value={avg.tokens.toFixed(0)} sub="Per query" color={COLORS.indigo} icon="🔢"/>
                </div>
              )}

              {/* Latency breakdown */}
              <div className="eval-section">
                <h3 className="eval-section-title">Latency Breakdown (avg per stage)</h3>
                <div className="eval-latency-bars">
                  <BarRow label="Retrieval" value={Math.round(results.reduce((s,r)=>s+r.latency_retrieval_ms,0)/results.length)} max={avg.latency} color={COLORS.sky} unit="ms"/>
                  <BarRow label="Rerank" value={Math.round(results.reduce((s,r)=>s+r.latency_rerank_ms,0)/results.length)} max={avg.latency} color={COLORS.indigo} unit="ms"/>
                  <BarRow label="LLM" value={Math.round(results.reduce((s,r)=>s+r.latency_llm_ms,0)/results.length)} max={avg.latency} color={COLORS.teal} unit="ms"/>
                </div>
              </div>

              {/* Per-query table */}
              <div className="eval-section">
                <h3 className="eval-section-title">Per-Query Results <span className="eval-badge">{results.length} queries</span></h3>
                <div className="eval-table-wrap">
                  <table className="eval-table">
                    <thead>
                      <tr>
                        <th>#</th><th>Query</th><th>Faithfulness</th><th>Relevance</th><th>Latency</th><th>Tokens In</th><th/>
                      </tr>
                    </thead>
                    <tbody>
                      {results.map((r, i) => <QueryRow key={i} result={r} idx={i}/>)}
                    </tbody>
                  </table>
                </div>
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}