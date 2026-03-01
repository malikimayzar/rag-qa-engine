import "./App.css";
import { jsPDF } from "jspdf";
import EvalDashboard from "./EvalDashboard";
import { useState, useRef, useEffect, useCallback } from "react";

const API_BASE = import.meta.env.VITE_API_URL || "/api";
const HISTORY_KEY = "rag_chat_history";

// ─── Icons ────────────────────────────────────────────────────────────────────
const Icon = {
  upload: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/></svg>,
  send: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><line x1="22" y1="2" x2="11" y2="13"/><polygon points="22 2 15 22 11 13 2 9 22 2"/></svg>,
  doc: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/></svg>,
  check: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><polyline points="20 6 9 17 4 12"/></svg>,
  trash: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14H6L5 6"/><path d="M10 11v6"/><path d="M14 11v6"/><path d="M9 6V4h6v2"/></svg>,
  bot: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><rect x="3" y="11" width="18" height="10" rx="2"/><circle cx="12" cy="5" r="2"/><path d="M12 7v4"/><line x1="8" y1="15" x2="8" y2="15"/><line x1="16" y1="15" x2="16" y2="15"/></svg>,
  user: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"/><circle cx="12" cy="7" r="4"/></svg>,
  link: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><path d="M10 13a5 5 0 0 0 7.54.54l3-3a5 5 0 0 0-7.07-7.07l-1.72 1.71"/><path d="M14 11a5 5 0 0 0-7.54-.54l-3 3a5 5 0 0 0 7.07 7.07l1.71-1.71"/></svg>,
  close: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/></svg>,
  history: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/></svg>,
  globe: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><circle cx="12" cy="12" r="10"/><line x1="2" y1="12" x2="22" y2="12"/><path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z"/></svg>,
  download: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.5} strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>,
  plus: <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>,
};

function parseSources(raw) {
  if (!raw) return [];
  if (Array.isArray(raw)) return raw;
  return raw.split("\n").filter(Boolean).map((line) => {
    const match = line.match(/- (.+), halaman (\d+)/);
    if (match) return { doc_id: match[1].trim(), page: match[2] };
    return { doc_id: line.replace(/^- /, "").trim() };
  });
}

function genId() { return Date.now().toString(36) + Math.random().toString(36).slice(2); }
function formatTime(ts) { return new Date(ts).toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit" }); }

function ParticleBg() {
  const particles = useRef(Array.from({ length: 18 }, () => ({ left: `${Math.random()*100}%`, top: `${Math.random()*100}%`, delay: `${Math.random()*8}s`, duration: `${6+Math.random()*10}s`, size: `${2+Math.random()*3}px`, opacity: 0.15+Math.random()*0.2 }))).current;
  return <div className="particle-bg" aria-hidden>{particles.map((p,i) => <div key={i} className="particle" style={{left:p.left,top:p.top,animationDelay:p.delay,animationDuration:p.duration,width:p.size,height:p.size,opacity:p.opacity}}/>)}</div>;
}

function TypingDots() { return <span className="typing-dots"><span/><span/><span/></span>; }

function CitationBadge({ source }) {
  return (
    <span className="citation-badge">
      <span className="citation-icon">{Icon.link}</span>
      <span>{source.doc_id?.split("/").pop()?.slice(0, 30)}</span>
      {source.page && <span className="citation-page">p.{source.page}</span>}
    </span>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`message-row ${isUser ? "user" : "bot"}`}>
      <div className="avatar">{isUser ? Icon.user : Icon.bot}</div>
      <div className="bubble-wrap">
        <div className={`bubble ${isUser ? "bubble-user" : "bubble-bot"}`}>
          {msg.loading ? <TypingDots /> : <p>{msg.content}</p>}
          {msg.streaming && !msg.loading && <span className="cursor-blink"/>}
        </div>
        {msg.sources && msg.sources.length > 0 && (
          <div className="citations">
            <span className="citations-label">Sources</span>
            {msg.sources.map((s,i) => <CitationBadge key={i} source={s}/>)}
          </div>
        )}
        <span className="msg-time">{formatTime(msg.ts)}</span>
      </div>
    </div>
  );
}

function UploadZone({ onUploadSuccess }) {
  const [dragging, setDragging] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [status, setStatus] = useState(null);
  const [statusMsg, setStatusMsg] = useState("");
  const inputRef = useRef();
  const doUpload = async (file) => {
    if (!file || file.type !== "application/pdf") { setStatus("err"); setStatusMsg("Only PDF files accepted."); return; }
    setUploading(true); setStatus(null);
    const form = new FormData(); form.append("file", file);
    try {
      const res = await fetch(`${API_BASE}/upload`, { method: "POST", body: form });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Upload failed");
      setStatus("ok"); setStatusMsg(`"${file.name}" uploaded!`);
      onUploadSuccess();
    } catch(e) { setStatus("err"); setStatusMsg(e.message); }
    finally { setUploading(false); setTimeout(() => setStatus(null), 3500); }
  };
  const onDrop = useCallback((e) => { e.preventDefault(); setDragging(false); doUpload(e.dataTransfer.files[0]); }, []);
  return (
    <div className={`upload-zone ${dragging?"drag-over":""} ${uploading?"uploading":""}`}
      onDragOver={(e)=>{e.preventDefault();setDragging(true);}} onDragLeave={()=>setDragging(false)} onDrop={onDrop}
      onClick={()=>!uploading&&inputRef.current.click()}>
      <input ref={inputRef} type="file" accept=".pdf" hidden onChange={(e)=>doUpload(e.target.files[0])}/>
      <div className="upload-icon">{Icon.upload}</div>
      <p className="upload-title">{uploading?"Processing…":"Drop PDF here"}</p>
      <p className="upload-sub">{uploading?<span className="upload-spinner"/>:"or click to browse"}</p>
      {status && <div className={`upload-toast ${status}`}>{status==="ok"?Icon.check:Icon.close}<span>{statusMsg}</span></div>}
    </div>
  );
}

function DocList({ docs, selected, onSelect, onRefresh, onDelete }) {
  const [confirmDelete, setConfirmDelete] = useState(null);
  const handleDelete = async (docId) => {
    if (confirmDelete === docId) { await onDelete(docId); setConfirmDelete(null); }
    else { setConfirmDelete(docId); setTimeout(() => setConfirmDelete(null), 3000); }
  };
  return (
    <div className="doc-list">
      <div className="doc-list-header"><span>Documents</span><button className="refresh-btn" onClick={onRefresh} title="Refresh">↻</button></div>
      <div className={`doc-item ${selected===null?"active":""}`} onClick={()=>onSelect(null)}>
        <span className="doc-item-icon">{Icon.globe}</span>
        <span className="doc-item-name">All documents</span>
        {selected===null&&<span className="doc-item-check">{Icon.check}</span>}
      </div>
      {docs.length===0?<p className="doc-empty">No documents indexed yet.</p>:(
        <ul>{docs.map((doc)=>(
          <li key={doc.doc_id} className={`doc-item ${selected===doc.doc_id?"active":""}`}>
            <span className="doc-item-icon" onClick={()=>onSelect(doc.doc_id===selected?null:doc.doc_id)}>{Icon.doc}</span>
            <span className="doc-item-name" onClick={()=>onSelect(doc.doc_id===selected?null:doc.doc_id)}>{doc.filename||doc.doc_id}</span>
            <button className={`delete-btn ${confirmDelete===doc.doc_id?"confirm":""}`}
              onClick={(e)=>{e.stopPropagation();handleDelete(doc.doc_id);}} title={confirmDelete===doc.doc_id?"Click again to confirm":"Delete"}>
              {confirmDelete===doc.doc_id?"?":Icon.trash}
            </button>
            {selected===doc.doc_id&&<span className="doc-item-check">{Icon.check}</span>}
          </li>
        ))}</ul>
      )}
    </div>
  );
}

function HistoryPanel({ sessions, activeId, onSelect, onDelete, onNew, onClose }) {
  return (
    <div className="history-panel">
      <div className="history-header">
        <span>Chat History</span>
        <div style={{display:"flex",gap:6}}>
          <button className="icon-btn" onClick={onNew} title="New chat">{Icon.plus}</button>
          <button className="icon-btn" onClick={onClose} title="Close">{Icon.close}</button>
        </div>
      </div>
      {sessions.length===0?<p className="doc-empty">No history yet.</p>:(
        <ul className="history-list">{sessions.map((s)=>(
          <li key={s.id} className={`history-item ${s.id===activeId?"active":""}`}>
            <div className="history-item-content" onClick={()=>onSelect(s.id)}>
              <p className="history-item-title">{s.title||"Untitled"}</p>
              <p className="history-item-meta">{new Date(s.ts).toLocaleDateString("id-ID")} · {s.messages.length} msgs</p>
            </div>
            <button className="delete-btn" onClick={()=>onDelete(s.id)}>{Icon.trash}</button>
          </li>
        ))}</ul>
      )}
    </div>
  );
}


function exportToPDF(messages) {
  if (!messages.length) return;
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  const pw = doc.internal.pageSize.getWidth();
  const ph = doc.internal.pageSize.getHeight();
  const margin = 16;
  const maxW = pw - margin * 2;
  let y = margin;

  const addPage = () => { doc.addPage(); y = margin; };
  const checkY = (h) => { if (y + h > ph - margin) addPage(); };

  // Header
  doc.setFillColor(8, 12, 20);
  doc.rect(0, 0, pw, 14, "F");
  doc.setTextColor(56, 189, 248);
  doc.setFontSize(11); doc.setFont("helvetica", "bold");
  doc.text("RAG QA Engine — Chat Export", margin, 9);
  doc.setTextColor(100, 116, 139);
  doc.setFontSize(7); doc.setFont("helvetica", "normal");
  doc.text(new Date().toLocaleString("id-ID"), pw - margin, 9, { align: "right" });
  y = 20;

  messages.forEach((msg) => {
    if (msg.loading || msg.streaming) return;
    const isUser = msg.role === "user";

    // Role label
    checkY(8);
    doc.setFontSize(7); doc.setFont("helvetica", "bold");
    doc.setTextColor(isUser ? 56 : 129, isUser ? 189 : 140, isUser ? 248 : 248);
    doc.text(isUser ? "YOU" : "ASSISTANT", margin, y);
    doc.setTextColor(100, 116, 139);
    doc.text(new Date(msg.ts).toLocaleTimeString("id-ID", { hour: "2-digit", minute: "2-digit" }), pw - margin, y, { align: "right" });
    y += 5;

    // Bubble background
    const lines = doc.splitTextToSize(msg.content || "", maxW - 6);
    const bh = lines.length * 5 + 6;
    checkY(bh);
    doc.setFillColor(isUser ? 17 : 19, isUser ? 28 : 25, isUser ? 46 : 46);
    doc.roundedRect(margin, y, maxW, bh, 2, 2, "F");
    doc.setTextColor(226, 232, 240);
    doc.setFontSize(8.5); doc.setFont("helvetica", "normal");
    doc.text(lines, margin + 3, y + 5);
    y += bh + 3;

    // Sources
    if (msg.sources && msg.sources.length > 0) {
      checkY(6);
      doc.setFontSize(6.5); doc.setFont("helvetica", "italic");
      doc.setTextColor(45, 212, 191);
      const srcText = msg.sources.map(s => `${s.doc_id}${s.page ? ` p.${s.page}` : ""}`).join("  ·  ");
      doc.text("Sources: " + srcText, margin, y);
      y += 6;
    }
    y += 4;
  });

  // Footer
  doc.setFillColor(8, 12, 20);
  doc.rect(0, ph - 8, pw, 8, "F");
  doc.setTextColor(100, 116, 139);
  doc.setFontSize(6); doc.setFont("helvetica", "normal");
  doc.text("Generated by RAG QA Engine", pw / 2, ph - 3, { align: "center" });

  doc.save(`rag-chat-${Date.now()}.pdf`);
}

export default function App() {
  const [docs, setDocs] = useState([]);
  const [selectedDoc, setSelectedDoc] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [health, setHealth] = useState(null);
  const [showHistory, setShowHistory] = useState(false);
  const [showEval, setShowEval] = useState(false);
  const [sessions, setSessions] = useState(() => { try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; } catch { return []; } });
  const [activeSessionId, setActiveSessionId] = useState(null);
  const chatEndRef = useRef();
  const textareaRef = useRef();

  useEffect(() => { localStorage.setItem(HISTORY_KEY, JSON.stringify(sessions)); }, [sessions]);

  const fetchDocs = async () => {
    try {
      const res = await fetch(`${API_BASE}/documents`);
      const data = await res.json();
      const raw = data.documents || [];
      setDocs(raw.map((d) => typeof d === "string" ? { doc_id: d, filename: d } : d));
    } catch { setDocs([]); }
  };

  const checkHealth = async () => {
    try {
      const res = await fetch(`${API_BASE}/health`);
      const data = await res.json();
      setHealth(data.status === "ok" || data.status === "healthy" ? "online" : "degraded");
    } catch { setHealth("offline"); }
  };

  useEffect(() => { fetchDocs(); checkHealth(); }, []);
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const saveToSession = (msgs) => {
    if (msgs.length === 0) return;
    const title = msgs.find((m) => m.role === "user")?.content?.slice(0, 40) || "Chat";
    if (activeSessionId) {
      setSessions((prev) => prev.map((s) => s.id === activeSessionId ? { ...s, messages: msgs, title } : s));
    } else {
      const newId = genId();
      setActiveSessionId(newId);
      setSessions((prev) => [{ id: newId, title, messages: msgs, ts: Date.now() }, ...prev]);
    }
  };

  const newChat = () => { if (messages.length > 0) saveToSession(messages); setMessages([]); setActiveSessionId(null); };

  const loadSession = (id) => {
    if (messages.length > 0 && activeSessionId !== id) saveToSession(messages);
    const s = sessions.find((s) => s.id === id);
    if (s) { setMessages(s.messages); setActiveSessionId(id); }
    setShowHistory(false);
  };

  const deleteSession = (id) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (activeSessionId === id) { setMessages([]); setActiveSessionId(null); }
  };

  const deleteDoc = async (docId) => {
    try {
      await fetch(`${API_BASE}/documents/${encodeURIComponent(docId)}`, { method: "DELETE" });
      if (selectedDoc === docId) setSelectedDoc(null);
      fetchDocs();
    } catch(e) { console.error(e); }
  };

  const onInputChange = (e) => {
    setInput(e.target.value);
    e.target.style.height = "auto";
    e.target.style.height = Math.min(e.target.scrollHeight, 140) + "px";
  };

  const sendQuery = async () => {
    const q = input.trim();
    if (!q || loading) return;
    const userMsg = { id: genId(), role: "user", content: q, ts: Date.now() };
    const botId = genId();
    const botMsg = { id: botId, role: "bot", content: "", loading: true, streaming: false, sources: [], ts: Date.now() };
    const newMsgs = [...messages, userMsg, botMsg];
    setMessages(newMsgs);
    setInput("");
    if (textareaRef.current) textareaRef.current.style.height = "auto";
    setLoading(true);
    try {
      const res = await fetch(`${API_BASE}/query`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query: q, doc_id: selectedDoc || undefined, stream: true }),
      });
      if (!res.ok) throw new Error("Query failed");
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "", fullContent = "", sources = [];
      setMessages((prev) => prev.map((m) => m.id === botId ? { ...m, loading: false, streaming: true } : m));
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n"); buffer = lines.pop();
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") break;
          try {
            const evt = JSON.parse(raw);
            if (evt.type === "token") { fullContent += evt.content; setMessages((prev) => prev.map((m) => m.id === botId ? { ...m, content: fullContent } : m)); }
            else if (evt.type === "sources") { sources = parseSources(evt.content); }
          } catch { /* ignore */ }
        }
      }
      const finalMsgs = newMsgs.map((m) => m.id === botId ? { ...m, loading: false, streaming: false, content: fullContent, sources } : m);
      setMessages(finalMsgs);
      saveToSession(finalMsgs);
    } catch(e) {
      setMessages((prev) => prev.map((m) => m.id === botId ? { ...m, loading: false, streaming: false, content: "⚠ Connection error. Is the backend running?" } : m));
    } finally { setLoading(false); }
  };

  const onKeyDown = (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); sendQuery(); } };

  return (
    <div className="app">
      <ParticleBg/>
      <header className="header">
        <div className="header-brand">
          <div className="header-logo">
            <svg viewBox="0 0 32 32" fill="none">
              <circle cx="16" cy="16" r="14" stroke="url(#lg)" strokeWidth="1.5"/>
              <path d="M10 16h12M16 10l6 6-6 6" stroke="url(#lg)" strokeWidth="1.5" strokeLinecap="round"/>
              <defs><linearGradient id="lg" x1="0" y1="0" x2="32" y2="32" gradientUnits="userSpaceOnUse"><stop stopColor="#38bdf8"/><stop offset="1" stopColor="#818cf8"/></linearGradient></defs>
            </svg>
          </div>
          <div>
            <h1 className="header-title">RAG QA Engine</h1>
            <p className="header-sub">Hybrid Retrieval · Groq LLM · Rust BM25</p>
          </div>
        </div>
        <div className="header-right">
          <button className="icon-btn header-btn" onClick={()=>setShowHistory((v)=>!v)} title="Chat history">{Icon.history}</button>
          {messages.length > 0 && <button className="icon-btn header-btn" onClick={()=>exportToPDF(messages)} title="Export chat to PDF">{Icon.download}</button>}
          <button className="icon-btn header-btn eval-header-btn" onClick={()=>setShowEval(true)} title="Eval Dashboard">📊</button>
          <button className="icon-btn header-btn" onClick={newChat} title="New chat">{Icon.plus}</button>
          <div className={`health-dot ${health}`} title={`Backend: ${health||"checking…"}`}><span/>{health||"…"}</div>
        </div>
      </header>
      <main className="main">
        <aside className="sidebar">
          <UploadZone onUploadSuccess={fetchDocs}/>
          <DocList docs={docs} selected={selectedDoc} onSelect={setSelectedDoc} onRefresh={fetchDocs} onDelete={deleteDoc}/>
          {selectedDoc&&<div className="scope-pill"><span>Scoped:</span><strong>{selectedDoc}</strong><button onClick={()=>setSelectedDoc(null)}>{Icon.close}</button></div>}
        </aside>
        <section className="chat-section">
          {showHistory&&(
            <div className="history-overlay">
              <HistoryPanel sessions={sessions} activeId={activeSessionId} onSelect={loadSession} onDelete={deleteSession} onNew={()=>{newChat();setShowHistory(false);}} onClose={()=>setShowHistory(false)}/>
            </div>
          )}
          <div className="chat-messages">
            {messages.length===0&&(
              <div className="chat-empty">
                <div className="chat-empty-icon">{Icon.bot}</div>
                <h2>Ask anything about your documents</h2>
                <p>Upload a PDF on the left, then start chatting.</p>
                <div className="suggestions">
                  {["Apa isi dokumen ini?","Berikan ringkasan dokumen.","Apa poin penting dari dokumen ini?"].map((s)=>(
                    <button key={s} className="suggestion-chip" onClick={()=>{setInput(s);textareaRef.current?.focus();}}>{s}</button>
                  ))}
                </div>
              </div>
            )}
            {messages.map((msg)=><MessageBubble key={msg.id} msg={msg}/>)}
            <div ref={chatEndRef}/>
          {showEval && <EvalDashboard onClose={()=>setShowEval(false)}/>}
          </div>
          <div className="input-bar">
            {messages.length>0&&<button className="clear-btn" onClick={newChat} title="New chat">{Icon.plus}</button>}
            <div className="input-wrap">
              <textarea ref={textareaRef} className="chat-input" rows={1}
                placeholder={selectedDoc?`Ask about "${selectedDoc}"…`:"Ask across all documents… (Enter to send)"}
                value={input} onChange={onInputChange} onKeyDown={onKeyDown} disabled={loading}/>
              <button className={`send-btn ${loading?"loading":""}`} onClick={sendQuery} disabled={loading||!input.trim()}>
                {loading?<span className="send-spinner"/>:Icon.send}
              </button>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}