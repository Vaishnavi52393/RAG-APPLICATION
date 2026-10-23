import React, { useState, useEffect, useRef } from 'react';
import {
  Sparkles,
  Bot,
  User,
  Upload,
  FileText,
  Database,
  Sliders,
  RefreshCw,
  Send,
  CheckCircle,
  AlertTriangle,
  Code,
  Cloud,
  Layers,
  ChevronDown,
  ChevronRight,
  Copy,
  Check,
  Download,
  Search,
  Cpu,
  ShieldCheck,
  BookOpen,
  Terminal,
  ExternalLink,
  Trash2,
} from 'lucide-react';

interface Citation {
  source: string;
  chunk_id: string;
  similarity_score?: number;
  excerpt: string;
}

interface Telemetry {
  retrievalMs: number;
  generationMs: number;
  model: string;
  chunksInspected: number;
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  grounded?: boolean;
  citations?: Citation[];
  telemetry?: Telemetry;
}

interface VectorStats {
  total_chunks: number;
  total_documents: number;
  sources: string[];
  embedding_model: string;
  generation_model: string;
  embedding_dimensions: number;
}

interface ChunkItem {
  id: string;
  source: string;
  page: number;
  section: number;
  text: string;
  charCount: number;
  hasEmbedding: boolean;
  embeddingPreview?: number[];
}

interface PythonFile {
  content: string;
  language: string;
  description: string;
  sizeBytes: number;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'vector_db' | 'python_code' | 'cloud_run'>('chat');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [stats, setStats] = useState<VectorStats>({
    total_chunks: 0,
    total_documents: 0,
    sources: [],
    embedding_model: 'gemini-embedding-2-preview',
    generation_model: 'gemini-3.8-flash',
    embedding_dimensions: 3072,
  });

  // Hyperparameters
  const [chunkSize, setChunkSize] = useState<number>(500);
  const [chunkOverlap, setChunkOverlap] = useState<number>(100);
  const [topK, setTopK] = useState<number>(4);
  const [minSimilarity, setMinSimilarity] = useState<number>(0.25);

  // Ingestion state
  const [isIngesting, setIsIngesting] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<string | null>(null);

  // Chunks tab state
  const [chunks, setChunks] = useState<ChunkItem[]>([]);
  const [chunkFilter, setChunkFilter] = useState('');
  const [querySimText, setQuerySimText] = useState('');
  const [simResults, setSimResults] = useState<any[]>([]);
  const [isSimulating, setIsSimulating] = useState(false);

  // Python files tab state
  const [pythonFiles, setPythonFiles] = useState<Record<string, PythonFile>>({});
  const [selectedFile, setSelectedFile] = useState<string>('app.py');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);

  // Expanded citations state
  const [expandedCitations, setExpandedCitations] = useState<Record<string, boolean>>({});

  const chatEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Initial fetch
  useEffect(() => {
    fetchStats();
    fetchPythonFiles();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  const fetchStats = async () => {
    try {
      const res = await fetch('/api/rag/stats');
      const data = await res.json();
      setStats(data);
    } catch (err) {
      console.error('Error fetching stats:', err);
    }
  };

  const fetchChunks = async () => {
    try {
      const res = await fetch('/api/rag/chunks');
      const data = await res.json();
      setChunks(data);
    } catch (err) {
      console.error('Error fetching chunks:', err);
    }
  };

  const fetchPythonFiles = async () => {
    try {
      const res = await fetch('/api/python-files');
      const data = await res.json();
      setPythonFiles(data);
    } catch (err) {
      console.error('Error fetching python files:', err);
    }
  };

  const handleSendMessage = async (queryText?: string) => {
    const textToSend = queryText || inputValue;
    if (!textToSend.trim() || isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: textToSend,
    };

    setMessages((prev) => [...prev, userMessage]);
    if (!queryText) setInputValue('');
    setIsLoading(true);

    try {
      const res = await fetch('/api/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: textToSend,
          topK,
          minSimilarity,
          temperature: 0.2,
        }),
      });

      const data = await res.json();

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: data.answer,
        grounded: data.grounded,
        citations: data.citations,
        telemetry: data.telemetry,
      };

      setMessages((prev) => [...prev, assistantMessage]);
      // Auto-expand citation for newly received message
      setExpandedCitations((prev) => ({ ...prev, [assistantMessage.id]: true }));
      fetchStats();
    } catch (err: any) {
      const errMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: `Error executing RAG query: ${err.message || 'Unknown network error'}`,
        grounded: false,
      };
      setMessages((prev) => [...prev, errMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (!files || files.length === 0) return;

    setIsIngesting(true);
    setUploadStatus(`Processing ${files.length} document(s)...`);

    for (let i = 0; i < files.length; i++) {
      const file = files[i];
      const reader = new FileReader();

      await new Promise<void>((resolve) => {
        reader.onload = async () => {
          try {
            const base64Data = (reader.result as string).split(',')[1];
            const res = await fetch('/api/rag/ingest', {
              method: 'POST',
              headers: { 'Content-Type': 'application/json' },
              body: JSON.stringify({
                filename: file.name,
                base64: base64Data,
                chunkSize,
                chunkOverlap,
              }),
            });
            const result = await res.json();
            if (result.success) {
              setUploadStatus(`Indexed ${file.name} (${result.indexedChunks} chunks)`);
            } else {
              setUploadStatus(`Error indexing ${file.name}: ${result.error}`);
            }
          } catch (err: any) {
            setUploadStatus(`Failed to upload ${file.name}: ${err.message}`);
          }
          resolve();
        };
        reader.readAsDataURL(file);
      });
    }

    setIsIngesting(false);
    fetchStats();
    if (activeTab === 'vector_db') fetchChunks();
    setTimeout(() => setUploadStatus(null), 4000);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const handleLoadSamples = async () => {
    setIsIngesting(true);
    setUploadStatus('Loading pre-configured enterprise sample documents...');
    try {
      const res = await fetch('/api/rag/load-samples', { method: 'POST' });
      const data = await res.json();
      if (data.success) {
        setUploadStatus('Sample documents successfully indexed into vector store!');
        setStats(data.stats);
        if (activeTab === 'vector_db') fetchChunks();
      }
    } catch (err: any) {
      setUploadStatus(`Error loading samples: ${err.message}`);
    } finally {
      setIsIngesting(false);
      setTimeout(() => setUploadStatus(null), 4000);
    }
  };

  const handleClearStore = async () => {
    if (!window.confirm('Are you sure you want to clear the vector database? All indexed chunks will be deleted.')) {
      return;
    }
    try {
      const res = await fetch('/api/rag/clear', { method: 'POST' });
      const data = await res.json();
      setStats(data.stats);
      setMessages([]);
      setChunks([]);
    } catch (err) {
      console.error('Clear error:', err);
    }
  };

  const handleRunSimulator = async () => {
    if (!querySimText.trim()) return;
    setIsSimulating(true);
    try {
      const res = await fetch('/api/rag/query', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          question: querySimText,
          topK: 6,
          minSimilarity: 0.0,
        }),
      });
      const data = await res.json();
      setSimResults(data.retrieved_chunks || []);
    } catch (err) {
      console.error('Simulator error:', err);
    } finally {
      setIsSimulating(false);
    }
  };

  const copyToClipboard = (text: string, key: string) => {
    navigator.clipboard.writeText(text);
    setCopiedKey(key);
    setTimeout(() => setCopiedKey(null), 2000);
  };

  const downloadFile = (filename: string, content: string) => {
    const blob = new Blob([content], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-slate-900 text-slate-100 font-sans">
      {/* LEFT SIDEBAR: Ingestion & Hyperparameters */}
      <aside className="w-80 flex-shrink-0 bg-slate-950 border-r border-slate-800 flex flex-col justify-between">
        <div className="p-4 border-b border-slate-800">
          <div className="flex items-center gap-2 mb-2">
            <div className="p-2 bg-blue-600/20 text-blue-400 rounded-lg border border-blue-500/30">
              <Cloud className="w-5 h-5 text-blue-400" />
            </div>
            <div>
              <h1 className="text-sm font-bold tracking-tight text-white leading-tight">
                Cloud-Based RAG
              </h1>
              <span className="text-[11px] text-blue-400 font-medium">Document Assistant</span>
            </div>
          </div>
          <div className="flex items-center gap-1.5 px-2.5 py-1 bg-emerald-950/60 border border-emerald-500/30 rounded-md text-[11px] text-emerald-300">
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span>
            <span>Gemini 3.8 Flash & Embeddings Ready</span>
          </div>
        </div>

        {/* Scrollable controls */}
        <div className="flex-1 overflow-y-auto p-4 space-y-5 text-xs">
          {/* Document Ingestion Card */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                <Upload className="w-3.5 h-3.5 text-blue-400" /> Document Ingestion
              </span>
              <span className="text-[10px] text-slate-400">PDF, DOCX, TXT</span>
            </div>

            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileUpload}
              multiple
              accept=".pdf,.docx,.txt,.md,.csv,.json"
              className="hidden"
            />

            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={isIngesting}
              className="w-full flex items-center justify-center gap-2 py-2 px-3 bg-blue-600 hover:bg-blue-500 disabled:bg-slate-800 text-white font-medium rounded-md transition shadow-sm border border-blue-500/50"
            >
              <Upload className="w-3.5 h-3.5" />
              <span>{isIngesting ? 'Processing Documents...' : 'Upload Local Documents'}</span>
            </button>

            <button
              onClick={handleLoadSamples}
              disabled={isIngesting}
              className="w-full flex items-center justify-center gap-2 py-1.5 px-3 bg-slate-800 hover:bg-slate-700 disabled:opacity-50 text-slate-200 rounded-md transition border border-slate-700"
            >
              <BookOpen className="w-3.5 h-3.5 text-amber-400" />
              <span>Load 3 Enterprise Sample Docs</span>
            </button>

            {uploadStatus && (
              <div className="p-2 bg-slate-800/80 rounded border border-slate-700 text-[11px] text-slate-300 animate-fade-in">
                {uploadStatus}
              </div>
            )}
          </div>

          {/* Hyperparameters Card */}
          <div className="p-3 bg-slate-900/90 rounded-lg border border-slate-800 space-y-3">
            <span className="font-semibold text-slate-300 flex items-center gap-1.5">
              <Sliders className="w-3.5 h-3.5 text-purple-400" /> RAG Hyperparameters
            </span>

            {/* Chunk Size */}
            <div className="space-y-1">
              <div className="flex justify-between text-slate-400">
                <span>Chunk Size</span>
                <span className="font-mono text-slate-200">{chunkSize} chars</span>
              </div>
              <input
                type="range"
                min="200"
                max="1500"
                step="50"
                value={chunkSize}
                onChange={(e) => setChunkSize(Number(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer h-1 bg-slate-700 rounded-lg"
              />
            </div>

            {/* Chunk Overlap */}
            <div className="space-y-1">
              <div className="flex justify-between text-slate-400">
                <span>Chunk Overlap</span>
                <span className="font-mono text-slate-200">{chunkOverlap} chars</span>
              </div>
              <input
                type="range"
                min="0"
                max="300"
                step="25"
                value={chunkOverlap}
                onChange={(e) => setChunkOverlap(Number(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer h-1 bg-slate-700 rounded-lg"
              />
            </div>

            {/* Top-K Chunks */}
            <div className="space-y-1">
              <div className="flex justify-between text-slate-400">
                <span>Top-K Retrieved</span>
                <span className="font-mono text-slate-200">{topK} chunks</span>
              </div>
              <input
                type="range"
                min="1"
                max="10"
                step="1"
                value={topK}
                onChange={(e) => setTopK(Number(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer h-1 bg-slate-700 rounded-lg"
              />
            </div>

            {/* Min Similarity */}
            <div className="space-y-1">
              <div className="flex justify-between text-slate-400">
                <span>Similarity Threshold</span>
                <span className="font-mono text-slate-200">{(minSimilarity * 100).toFixed(0)}%</span>
              </div>
              <input
                type="range"
                min="0.0"
                max="0.8"
                step="0.05"
                value={minSimilarity}
                onChange={(e) => setMinSimilarity(Number(e.target.value))}
                className="w-full accent-blue-500 cursor-pointer h-1 bg-slate-700 rounded-lg"
              />
            </div>
          </div>

          {/* Vector Store Health */}
          <div className="p-3 bg-slate-900/90 rounded-lg border border-slate-800 space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-semibold text-slate-300 flex items-center gap-1.5">
                <Database className="w-3.5 h-3.5 text-emerald-400" /> Vector Database
              </span>
              <button
                onClick={fetchStats}
                title="Refresh stats"
                className="text-slate-400 hover:text-white transition"
              >
                <RefreshCw className="w-3 h-3" />
              </button>
            </div>

            <div className="grid grid-cols-2 gap-2 text-center">
              <div className="bg-slate-800/60 p-2 rounded border border-slate-700/60">
                <div className="text-lg font-bold text-white font-mono">{stats.total_chunks}</div>
                <div className="text-[10px] text-slate-400">Indexed Chunks</div>
              </div>
              <div className="bg-slate-800/60 p-2 rounded border border-slate-700/60">
                <div className="text-lg font-bold text-white font-mono">{stats.total_documents}</div>
                <div className="text-[10px] text-slate-400">Documents</div>
              </div>
            </div>

            <div className="text-[10px] text-slate-400 space-y-1 pt-1 border-t border-slate-800">
              <div className="flex justify-between">
                <span>Dimensions:</span>
                <span className="font-mono text-slate-300">{stats.embedding_dimensions}D</span>
              </div>
              <div className="flex justify-between">
                <span>Metric:</span>
                <span className="font-mono text-slate-300">Cosine Distance</span>
              </div>
            </div>

            {stats.sources.length > 0 && (
              <div className="pt-1">
                <div className="text-[10px] text-slate-400 mb-1 font-medium">Active Sources:</div>
                <div className="max-h-24 overflow-y-auto space-y-1 pr-1">
                  {stats.sources.map((s, idx) => (
                    <div
                      key={idx}
                      className="text-[10px] text-slate-300 truncate bg-slate-800/40 px-1.5 py-0.5 rounded border border-slate-700/40 flex items-center gap-1"
                    >
                      <FileText className="w-2.5 h-2.5 text-blue-400 shrink-0" />
                      <span className="truncate">{s}</span>
                    </div>
                  ))}
                </div>
              </div>
            )}

            <button
              onClick={handleClearStore}
              className="w-full flex items-center justify-center gap-1.5 py-1 px-2 mt-2 bg-red-950/40 hover:bg-red-900/50 text-red-300 text-[11px] rounded border border-red-800/50 transition"
            >
              <Trash2 className="w-3 h-3" />
              <span>Reset Vector Store</span>
            </button>
          </div>
        </div>

        {/* Footer info */}
        <div className="p-3 border-t border-slate-800 bg-slate-950/80 text-[10px] text-slate-500 flex items-center justify-between">
          <span>Enterprise GenAI RAG</span>
          <span className="text-slate-400 font-mono">v4.2.0</span>
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <div className="flex-1 flex flex-col h-full bg-slate-900 overflow-hidden">
        {/* TOP NAVIGATION BAR */}
        <header className="h-14 border-b border-slate-800 bg-slate-950/80 backdrop-blur px-6 flex items-center justify-between shrink-0">
          <div className="flex items-center gap-4">
            <nav className="flex space-x-1">
              <button
                onClick={() => setActiveTab('chat')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition ${
                  activeTab === 'chat'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
              >
                <Bot className="w-4 h-4" />
                <span>Interactive Assistant</span>
              </button>

              <button
                onClick={() => {
                  setActiveTab('vector_db');
                  fetchChunks();
                }}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition ${
                  activeTab === 'vector_db'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
              >
                <Database className="w-4 h-4" />
                <span>Vector DB & Chunks ({stats.total_chunks})</span>
              </button>

              <button
                onClick={() => setActiveTab('python_code')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition ${
                  activeTab === 'python_code'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
              >
                <Code className="w-4 h-4" />
                <span>Python Codebase ({Object.keys(pythonFiles).length} files)</span>
              </button>

              <button
                onClick={() => setActiveTab('cloud_run')}
                className={`flex items-center gap-2 px-3 py-1.5 rounded-md text-xs font-medium transition ${
                  activeTab === 'cloud_run'
                    ? 'bg-blue-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-slate-200 hover:bg-slate-800'
                }`}
              >
                <Cloud className="w-4 h-4" />
                <span>Cloud Run Architecture</span>
              </button>
            </nav>
          </div>

          <div className="flex items-center gap-2 text-xs text-slate-400">
            <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded bg-slate-800 border border-slate-700 font-mono text-[11px]">
              <Cpu className="w-3 h-3 text-cyan-400" />
              gemini-3.8-flash
            </span>
          </div>
        </header>

        {/* TAB 1: INTERACTIVE RAG CHAT */}
        {activeTab === 'chat' && (
          <div className="flex-1 flex flex-col h-full overflow-hidden">
            {/* Messages Scroll Area */}
            <div className="flex-1 overflow-y-auto p-6 space-y-6">
              {messages.length === 0 ? (
                <div className="max-w-3xl mx-auto py-10 space-y-6 text-center">
                  <div className="w-16 h-16 mx-auto rounded-2xl bg-gradient-to-tr from-blue-600 to-indigo-500 flex items-center justify-center shadow-lg shadow-blue-500/20">
                    <Sparkles className="w-8 h-8 text-white" />
                  </div>
                  <div className="space-y-2">
                    <h2 className="text-2xl font-bold tracking-tight text-white">
                      Cloud-Based RAG Document Assistant
                    </h2>
                    <p className="text-slate-400 text-sm max-w-xl mx-auto">
                      Enterprise question answering with strict grounding against your indexed documents.
                      Claims are cited with chunk IDs and verified similarity scores. If documents lack the answer,
                      hallucinations are refused.
                    </p>
                  </div>

                  {/* Quick Suggestion Chips */}
                  <div className="pt-4">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                      Try Sample Architectural Queries:
                    </p>
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5 text-left max-w-2xl mx-auto">
                      <button
                        onClick={() =>
                          handleSendMessage(
                            'What are the key architectural specifications and limits for Google Cloud Run?'
                          )
                        }
                        className="p-3 bg-slate-800/80 hover:bg-slate-800 text-slate-200 rounded-lg border border-slate-700/80 transition text-xs flex items-start gap-2.5 group"
                      >
                        <Cloud className="w-4 h-4 text-blue-400 shrink-0 mt-0.5 group-hover:scale-110 transition" />
                        <div>
                          <div className="font-semibold text-white">Cloud Run Specifications</div>
                          <div className="text-[11px] text-slate-400">
                            Concurrency, memory, CPU boost, and timeout limits
                          </div>
                        </div>
                      </button>

                      <button
                        onClick={() =>
                          handleSendMessage(
                            'What is the recommended best practice for chunking tabular and structured data?'
                          )
                        }
                        className="p-3 bg-slate-800/80 hover:bg-slate-800 text-slate-200 rounded-lg border border-slate-700/80 transition text-xs flex items-start gap-2.5 group"
                      >
                        <Layers className="w-4 h-4 text-purple-400 shrink-0 mt-0.5 group-hover:scale-110 transition" />
                        <div>
                          <div className="font-semibold text-white">Tabular Chunking Strategy</div>
                          <div className="text-[11px] text-slate-400">
                            Serialization, Markdown tables, and parent-child pointers
                          </div>
                        </div>
                      </button>

                      <button
                        onClick={() =>
                          handleSendMessage(
                            'What are the data retention requirements for vector indices and chat sessions under the compliance policy?'
                          )
                        }
                        className="p-3 bg-slate-800/80 hover:bg-slate-800 text-slate-200 rounded-lg border border-slate-700/80 transition text-xs flex items-start gap-2.5 group"
                      >
                        <ShieldCheck className="w-4 h-4 text-emerald-400 shrink-0 mt-0.5 group-hover:scale-110 transition" />
                        <div>
                          <div className="font-semibold text-white">Compliance & Retention Rules</div>
                          <div className="text-[11px] text-slate-400">
                            30-day index purges, ephemeral sessions, and audit logging
                          </div>
                        </div>
                      </button>

                      <button
                        onClick={() =>
                          handleSendMessage(
                            'How do you bake a traditional San Francisco sourdough bread?'
                          )
                        }
                        className="p-3 bg-slate-800/80 hover:bg-slate-800 text-slate-200 rounded-lg border border-slate-700/80 transition text-xs flex items-start gap-2.5 group"
                      >
                        <AlertTriangle className="w-4 h-4 text-amber-400 shrink-0 mt-0.5 group-hover:scale-110 transition" />
                        <div>
                          <div className="font-semibold text-white">Out-of-Domain Grounding Test</div>
                          <div className="text-[11px] text-slate-400">
                            Verifies strict refusal when evidence is absent from context
                          </div>
                        </div>
                      </button>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="max-w-4xl mx-auto space-y-6">
                  {messages.map((msg) => (
                    <div
                      key={msg.id}
                      className={`flex gap-4 ${
                        msg.role === 'user' ? 'justify-end' : 'justify-start'
                      }`}
                    >
                      {msg.role === 'assistant' && (
                        <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center shrink-0 text-blue-400">
                          <Bot className="w-5 h-5" />
                        </div>
                      )}

                      <div
                        className={`max-w-2xl rounded-xl p-4 text-sm leading-relaxed ${
                          msg.role === 'user'
                            ? 'bg-blue-600 text-white shadow-md'
                            : 'bg-slate-950/80 border border-slate-800 text-slate-200'
                        }`}
                      >
                        {/* Grounding Badge */}
                        {msg.role === 'assistant' && msg.grounded !== undefined && (
                          <div className="mb-2">
                            {msg.grounded ? (
                              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-emerald-950/80 text-emerald-400 border border-emerald-500/30">
                                <CheckCircle className="w-3 h-3" />
                                Grounding Verified in Context
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-[11px] font-semibold bg-red-950/80 text-red-400 border border-red-500/30">
                                <AlertTriangle className="w-3 h-3" />
                                Refusal Triggered (Negative Constraint)
                              </span>
                            )}
                          </div>
                        )}

                        {/* Message content */}
                        <div className="whitespace-pre-wrap">{msg.content}</div>

                        {/* Expandable Verified Citations */}
                        {msg.citations && msg.citations.length > 0 && (
                          <div className="mt-4 pt-3 border-t border-slate-800">
                            <button
                              onClick={() =>
                                setExpandedCitations((prev) => ({
                                  ...prev,
                                  [msg.id]: !prev[msg.id],
                                }))
                              }
                              className="flex items-center gap-1.5 text-xs font-semibold text-blue-400 hover:text-blue-300 transition"
                            >
                              {expandedCitations[msg.id] ? (
                                <ChevronDown className="w-3.5 h-3.5" />
                              ) : (
                                <ChevronRight className="w-3.5 h-3.5" />
                              )}
                              <span>
                                {msg.citations.length} Verified Source Citation(s)
                              </span>
                            </button>

                            {expandedCitations[msg.id] && (
                              <div className="mt-2.5 space-y-2">
                                {msg.citations.map((cite, cIdx) => (
                                  <div
                                    key={cIdx}
                                    className="p-2.5 bg-slate-900 rounded-lg border border-slate-800 text-xs space-y-1"
                                  >
                                    <div className="flex items-center justify-between text-[11px]">
                                      <div className="flex items-center gap-1 font-semibold text-slate-200">
                                        <FileText className="w-3 h-3 text-blue-400" />
                                        <span>{cite.source}</span>
                                        <span className="text-slate-400 font-mono">
                                          ({cite.chunk_id})
                                        </span>
                                      </div>
                                      {cite.similarity_score !== undefined && (
                                        <span className="px-1.5 py-0.5 bg-blue-950/80 text-blue-300 rounded font-mono text-[10px] border border-blue-800/40">
                                          {(cite.similarity_score * 100).toFixed(1)}% match
                                        </span>
                                      )}
                                    </div>
                                    <p className="text-slate-400 italic text-[11px] leading-normal">
                                      "{cite.excerpt}"
                                    </p>
                                  </div>
                                ))}
                              </div>
                            )}
                          </div>
                        )}

                        {/* Execution Telemetry */}
                        {msg.telemetry && (
                          <div className="mt-3 pt-2 border-t border-slate-800/60 flex items-center justify-between text-[10px] text-slate-500 font-mono">
                            <span>⚡ Retrieval: {msg.telemetry.retrievalMs}ms</span>
                            <span>Model: {msg.telemetry.generationMs}ms</span>
                            <span>{msg.telemetry.model}</span>
                          </div>
                        )}
                      </div>

                      {msg.role === 'user' && (
                        <div className="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center shrink-0 text-slate-300">
                          <User className="w-5 h-5" />
                        </div>
                      )}
                    </div>
                  ))}

                  {isLoading && (
                    <div className="flex gap-4">
                      <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center shrink-0 text-blue-400">
                        <Bot className="w-5 h-5 animate-spin" />
                      </div>
                      <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-xs text-slate-400 space-y-2">
                        <div className="flex items-center gap-2">
                          <span className="w-2 h-2 rounded-full bg-blue-400 animate-ping"></span>
                          <span>Searching ChromaDB semantic index & invoking Gemini 3.8 Flash...</span>
                        </div>
                        <div className="w-48 h-1 bg-slate-800 rounded-full overflow-hidden">
                          <div className="w-full h-full bg-blue-500 animate-pulse"></div>
                        </div>
                      </div>
                    </div>
                  )}

                  <div ref={chatEndRef} />
                </div>
              )}
            </div>

            {/* Bottom Input Box */}
            <div className="p-4 border-t border-slate-800 bg-slate-950/90 shrink-0">
              <div className="max-w-4xl mx-auto flex gap-3">
                <input
                  type="text"
                  value={inputValue}
                  onChange={(e) => setInputValue(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                  placeholder="Ask a question about your uploaded documents (e.g. Cloud Run specs, retention rules)..."
                  className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-4 py-2.5 text-sm text-slate-100 placeholder-slate-500 focus:outline-none focus:border-blue-500 transition"
                  disabled={isLoading}
                />
                <button
                  onClick={() => handleSendMessage()}
                  disabled={isLoading || !inputValue.trim()}
                  className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-40 text-white text-sm font-semibold rounded-lg flex items-center gap-2 transition shadow-md shadow-blue-600/20"
                >
                  <Send className="w-4 h-4" />
                  <span>Ask Assistant</span>
                </button>
              </div>
            </div>
          </div>
        )}

        {/* TAB 2: VECTOR DB & CHUNK INSPECTOR */}
        {activeTab === 'vector_db' && (
          <div className="flex-1 flex flex-col h-full overflow-hidden p-6 space-y-6">
            <div className="flex items-center justify-between">
              <div>
                <h2 className="text-xl font-bold text-white flex items-center gap-2">
                  <Database className="w-5 h-5 text-blue-400" /> Vector Database & Chunk Inspector
                </h2>
                <p className="text-xs text-slate-400">
                  Inspect dense vector chunk embeddings, metadata, and simulate real-time cosine distance retrieval.
                </p>
              </div>
              <button
                onClick={fetchChunks}
                className="px-3 py-1.5 bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 rounded-md border border-slate-700 flex items-center gap-1.5 transition"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Refresh Chunks</span>
              </button>
            </div>

            {/* Semantic Query Simulator Card */}
            <div className="p-4 bg-slate-950/80 rounded-xl border border-slate-800 space-y-3">
              <span className="text-xs font-semibold text-slate-200 flex items-center gap-2">
                <Search className="w-4 h-4 text-purple-400" /> Real-time Semantic Retrieval Simulator
              </span>
              <div className="flex gap-2">
                <input
                  type="text"
                  value={querySimText}
                  onChange={(e) => setQuerySimText(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleRunSimulator()}
                  placeholder="Type a test query to measure raw cosine similarity against indexed chunks..."
                  className="flex-1 bg-slate-900 border border-slate-700 rounded-lg px-3 py-1.5 text-xs text-slate-100 placeholder-slate-500 focus:outline-none focus:border-purple-500"
                />
                <button
                  onClick={handleRunSimulator}
                  disabled={isSimulating || !querySimText.trim()}
                  className="px-4 py-1.5 bg-purple-600 hover:bg-purple-500 disabled:opacity-40 text-white text-xs font-semibold rounded-lg transition"
                >
                  {isSimulating ? 'Computing...' : 'Test Retrieval'}
                </button>
              </div>

              {simResults.length > 0 && (
                <div className="mt-3 space-y-2 max-h-60 overflow-y-auto pr-1">
                  <span className="text-[11px] font-semibold text-slate-400">
                    Top {simResults.length} Nearest Neighbor Chunks:
                  </span>
                  {simResults.map((sim, sIdx) => (
                    <div
                      key={sIdx}
                      className="p-2.5 bg-slate-900 rounded-lg border border-purple-900/30 text-xs space-y-1"
                    >
                      <div className="flex items-center justify-between text-[11px]">
                        <span className="font-semibold text-purple-300 font-mono">
                          #{sIdx + 1} {sim.source} — {sim.chunk_id}
                        </span>
                        <span className="px-2 py-0.5 bg-purple-950 text-purple-300 font-bold font-mono rounded text-[10px]">
                          {(sim.similarity_score * 100).toFixed(2)}% similarity
                        </span>
                      </div>
                      <p className="text-slate-300 text-[11px] leading-relaxed">
                        {sim.text}
                      </p>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Chunks List */}
            <div className="flex-1 flex flex-col overflow-hidden bg-slate-950/80 rounded-xl border border-slate-800">
              <div className="p-3 border-b border-slate-800 flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-300">
                  Indexed Document Chunks ({chunks.length})
                </span>
                <input
                  type="text"
                  value={chunkFilter}
                  onChange={(e) => setChunkFilter(e.target.value)}
                  placeholder="Filter chunks by keyword or filename..."
                  className="bg-slate-900 border border-slate-700 rounded px-2.5 py-1 text-xs text-slate-200 placeholder-slate-500 w-64"
                />
              </div>

              <div className="flex-1 overflow-y-auto p-4 space-y-3">
                {chunks.length === 0 ? (
                  <div className="text-center py-12 text-slate-500 text-xs">
                    No chunks indexed yet. Upload a document or load pre-configured sample documents from the sidebar.
                  </div>
                ) : (
                  chunks
                    .filter(
                      (c) =>
                        c.source.toLowerCase().includes(chunkFilter.toLowerCase()) ||
                        c.text.toLowerCase().includes(chunkFilter.toLowerCase()) ||
                        c.id.toLowerCase().includes(chunkFilter.toLowerCase())
                    )
                    .map((chunk) => (
                      <div
                        key={chunk.id}
                        className="p-3.5 bg-slate-900/90 rounded-lg border border-slate-800/80 hover:border-slate-700 text-xs space-y-2 transition"
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center gap-2">
                            <span className="px-2 py-0.5 bg-blue-950 text-blue-300 font-mono font-bold text-[10px] rounded border border-blue-800/40">
                              {chunk.id}
                            </span>
                            <span className="font-semibold text-slate-200">{chunk.source}</span>
                            <span className="text-slate-500 text-[11px]">
                              Page {chunk.page} • Section {chunk.section}
                            </span>
                          </div>
                          <div className="flex items-center gap-2 text-[10px] text-slate-400 font-mono">
                            <span>{chunk.charCount} chars</span>
                            {chunk.hasEmbedding && (
                              <span className="px-1.5 py-0.5 bg-emerald-950/80 text-emerald-400 rounded border border-emerald-800/40">
                                3072D Vector
                              </span>
                            )}
                          </div>
                        </div>

                        <p className="text-slate-300 leading-relaxed font-mono text-[11px] bg-slate-950 p-2.5 rounded border border-slate-800/50">
                          {chunk.text}
                        </p>

                        {chunk.embeddingPreview && chunk.embeddingPreview.length > 0 && (
                          <div className="text-[10px] text-slate-500 font-mono">
                            Vector Preview: [{chunk.embeddingPreview.map((v) => v.toFixed(4)).join(', ')}, ...]
                          </div>
                        )}
                      </div>
                    ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* TAB 3: PYTHON CODEBASE & ARCHITECTURE */}
        {activeTab === 'python_code' && (
          <div className="flex-1 flex overflow-hidden">
            {/* File Tree Explorer */}
            <div className="w-72 bg-slate-950 border-r border-slate-800 flex flex-col shrink-0">
              <div className="p-3 border-b border-slate-800">
                <span className="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                  <Terminal className="w-3.5 h-3.5 text-blue-400" /> Python Application Files
                </span>
                <span className="text-[10px] text-slate-500">Production RAG Architecture</span>
              </div>

              <div className="flex-1 overflow-y-auto p-2 space-y-1">
                {Object.keys(pythonFiles).map((filename) => {
                  const fileMeta = pythonFiles[filename];
                  const isSelected = selectedFile === filename;
                  return (
                    <button
                      key={filename}
                      onClick={() => setSelectedFile(filename)}
                      className={`w-full text-left p-2 rounded-md text-xs transition flex flex-col ${
                        isSelected
                          ? 'bg-blue-600 text-white font-medium'
                          : 'text-slate-400 hover:text-slate-200 hover:bg-slate-900'
                      }`}
                    >
                      <div className="flex items-center gap-1.5">
                        <FileText className="w-3.5 h-3.5 shrink-0" />
                        <span className="font-mono truncate">{filename}</span>
                      </div>
                      <span
                        className={`text-[10px] truncate mt-0.5 ${
                          isSelected ? 'text-blue-100' : 'text-slate-500'
                        }`}
                      >
                        {fileMeta.description}
                      </span>
                    </button>
                  );
                })}
              </div>
            </div>

            {/* File Viewer */}
            <div className="flex-1 flex flex-col overflow-hidden bg-slate-900">
              {pythonFiles[selectedFile] ? (
                <>
                  <div className="h-12 border-b border-slate-800 px-6 flex items-center justify-between shrink-0 bg-slate-950/60">
                    <div>
                      <span className="font-mono text-sm font-bold text-white">
                        {selectedFile}
                      </span>
                      <span className="text-xs text-slate-400 ml-3">
                        {pythonFiles[selectedFile].description}
                      </span>
                    </div>

                    <div className="flex items-center gap-2">
                      <button
                        onClick={() =>
                          copyToClipboard(
                            pythonFiles[selectedFile].content,
                            selectedFile
                          )
                        }
                        className="px-3 py-1 bg-slate-800 hover:bg-slate-700 text-xs text-slate-200 rounded border border-slate-700 flex items-center gap-1.5 transition"
                      >
                        {copiedKey === selectedFile ? (
                          <>
                            <Check className="w-3.5 h-3.5 text-emerald-400" />
                            <span className="text-emerald-400">Copied!</span>
                          </>
                        ) : (
                          <>
                            <Copy className="w-3.5 h-3.5" />
                            <span>Copy Code</span>
                          </>
                        )}
                      </button>

                      <button
                        onClick={() =>
                          downloadFile(
                            selectedFile,
                            pythonFiles[selectedFile].content
                          )
                        }
                        className="px-3 py-1 bg-blue-600 hover:bg-blue-500 text-xs text-white rounded font-medium flex items-center gap-1.5 transition"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>Download</span>
                      </button>
                    </div>
                  </div>

                  <div className="flex-1 overflow-auto p-4 bg-slate-950 font-mono text-xs text-slate-200 leading-relaxed">
                    <pre>
                      <code>{pythonFiles[selectedFile].content}</code>
                    </pre>
                  </div>
                </>
              ) : (
                <div className="p-8 text-center text-slate-500 text-xs">
                  Select a file from the explorer on the left.
                </div>
              )}
            </div>
          </div>
        )}

        {/* TAB 4: CLOUD RUN DEPLOYMENT ARCHITECTURE */}
        {activeTab === 'cloud_run' && (
          <div className="flex-1 overflow-y-auto p-8 max-w-4xl mx-auto space-y-6">
            <div>
              <h2 className="text-2xl font-bold text-white flex items-center gap-2.5">
                <Cloud className="w-6 h-6 text-blue-400" /> Google Cloud Run Architecture & Deployment Guide
              </h2>
              <p className="text-sm text-slate-400 mt-1">
                Enterprise reference blueprint for deploying the containerized Streamlit RAG application
                with Secret Manager and zero-trust IAM.
              </p>
            </div>

            {/* Architecture Card */}
            <div className="p-5 bg-slate-950 rounded-xl border border-slate-800 space-y-4">
              <h3 className="text-sm font-semibold text-white flex items-center gap-2">
                <ShieldCheck className="w-4 h-4 text-emerald-400" /> Security & Cloud Architecture Specifications
              </h3>
              <div className="grid grid-cols-1 md:grid-cols-3 gap-3 text-xs">
                <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                  <div className="font-semibold text-blue-400 mb-1">Serverless Compute</div>
                  <div className="text-slate-300">Google Cloud Run with 2 vCPUs, 2 GiB RAM, and Startup CPU Boost for rapid container cold-starts.</div>
                </div>
                <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                  <div className="font-semibold text-emerald-400 mb-1">Secret Management</div>
                  <div className="text-slate-300">GEMINI_API_KEY injected directly from Google Cloud Secret Manager at container runtime.</div>
                </div>
                <div className="p-3 bg-slate-900 rounded-lg border border-slate-800">
                  <div className="font-semibold text-purple-400 mb-1">Vector Indexing</div>
                  <div className="text-slate-300">ChromaDB persistent volume or Vertex AI Vector Search with asymmetric Gemini embeddings.</div>
                </div>
              </div>
            </div>

            {/* Single Command Deploy Box */}
            <div className="p-5 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                  <Terminal className="w-4 h-4 text-amber-400" /> Step 1: Deploy to Google Cloud Run
                </span>
                <button
                  onClick={() =>
                    copyToClipboard(
                      `gcloud run deploy cloud-rag-assistant \\\n  --source . \\\n  --region us-central1 \\\n  --platform managed \\\n  --allow-unauthenticated \\\n  --memory 2Gi \\\n  --cpu 2 \\\n  --timeout 300 \\\n  --set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest \\\n  --set-env-vars STREAMLIT_SERVER_PORT=8080,STREAMLIT_SERVER_ADDRESS=0.0.0.0`,
                      'deploy_cmd'
                    )
                  }
                  className="px-2.5 py-1 bg-slate-800 hover:bg-slate-700 text-[11px] text-slate-300 rounded border border-slate-700 flex items-center gap-1 transition"
                >
                  {copiedKey === 'deploy_cmd' ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                  <span>{copiedKey === 'deploy_cmd' ? 'Copied' : 'Copy Command'}</span>
                </button>
              </div>

              <div className="p-3 bg-slate-900 rounded-lg font-mono text-xs text-blue-300 leading-relaxed overflow-x-auto">
                <span className="text-slate-500"># Single-command build and deploy directly to Cloud Run</span>
                <br />
                gcloud run deploy cloud-rag-assistant \
                <br />
                &nbsp;&nbsp;--source . \
                <br />
                &nbsp;&nbsp;--region us-central1 \
                <br />
                &nbsp;&nbsp;--platform managed \
                <br />
                &nbsp;&nbsp;--allow-unauthenticated \
                <br />
                &nbsp;&nbsp;--memory 2Gi \
                <br />
                &nbsp;&nbsp;--cpu 2 \
                <br />
                &nbsp;&nbsp;--timeout 300 \
                <br />
                &nbsp;&nbsp;--set-secrets GEMINI_API_KEY=GEMINI_API_KEY:latest \
                <br />
                &nbsp;&nbsp;--set-env-vars STREAMLIT_SERVER_PORT=8080,STREAMLIT_SERVER_ADDRESS=0.0.0.0
              </div>
            </div>

            {/* Testing Guide */}
            <div className="p-5 bg-slate-950 rounded-xl border border-slate-800 space-y-3">
              <span className="text-xs font-semibold text-slate-200 flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4 text-emerald-400" /> Step 2: Run Verification Test Suite
              </span>
              <p className="text-xs text-slate-400">
                Execute unit tests verifying document chunking, PDF/DOCX parsing, prompt formatting, and refusal logic:
              </p>
              <div className="p-3 bg-slate-900 rounded-lg font-mono text-xs text-emerald-300">
                pytest tests/test_rag.py -v
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
