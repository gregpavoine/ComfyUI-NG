import React, { useState, useEffect, useRef, useCallback } from 'react';
import { NodeDefinition, ModelItem, fetchNodeCatalogue, fetchModels, submitJob, fetchJob, fetchJobs, Job } from '../../api/client';
import {
  Play,
  Trash2,
  CheckCircle2,
  Layers,
  Sliders,
  X,
  Sparkles,
  PanelLeftClose,
  PanelRightClose,
  PanelRightOpen,
  Download,
  Image as ImageIcon,
  Zap,
  List,
  FolderOpen,
  Settings,
  ChevronRight,
  ChevronDown,
  Clock,
  AlertCircle,
  RefreshCw,
  Search,
  Plus,
  Save,
  Eye,
  FileJson,
  Cpu,
  HardDrive,
  Activity,
  Server,
  User,
  LogOut,
  Menu,
} from 'lucide-react';

interface CanvasNode {
  id: string;
  def: NodeDefinition;
  x: number;
  y: number;
  params: Record<string, any>;
}

interface Connection {
  id: string;
  fromNodeId: string;
  fromPort: string;
  toNodeId: string;
  toPort: string;
  type: string;
}

interface LogEntry {
  time: string;
  level: string;
  message: string;
}

type LeftMenu = 'queue' | 'images' | 'config' | 'gallery' | 'nodes' | null;

const TYPE_COLORS: Record<string, string> = {
  MODEL: '#818cf8',
  CLIP: '#c084fc',
  CONDITIONING: '#a855f7',
  LATENT: '#f472b6',
  IMAGE: '#34d399',
  VAE: '#fbbf24',
  STRING: '#38bdf8',
  INT: '#60a5fa',
  FLOAT: '#38bdf8',
  NG_MODEL: '#818cf8',
  NG_LATENT: '#f472b6',
  NG_IMAGE: '#34d399',
  NG_TEXT: '#38bdf8',
};

const LEFT_MENU_ITEMS: { key: LeftMenu; label: string; icon: React.ReactNode; color: string }[] = [
  { key: 'queue', label: 'Job Queue', icon: <Clock size={16} />, color: '#f59e0b' },
  { key: 'images', label: 'Generated Images', icon: <ImageIcon size={16} />, color: '#10b981' },
  { key: 'gallery', label: 'Gallery', icon: <FolderOpen size={16} />, color: '#8b5cf6' },
  { key: 'config', label: 'NG Configuration', icon: <Settings size={16} />, color: '#06b6d4' },
  { key: 'nodes', label: 'Node Palette', icon: <Search size={16} />, color: '#6366f1' },
];

const DEFAULT_NG_CONFIG = `# ComfyUI-NG Configuration
server:
  host: 127.0.0.1
  port: 8188
  workers: 2

runtime:
  python: ">=3.14"
  multiprocessing_start: forkserver

scheduler:
  default_profile: balanced
  interactive_priority: 80
  max_queued_jobs: 100

cpu:
  reserve_cores: 2
  compute_workers: auto
  io_workers: 4

memory:
  reserve_system_gb: 4
  max_pinned_gb: 8

gpu:
  devices: auto
  reserve_vram_mb: 768
  heavy_workers_per_gpu: 1
  compile: auto
  attention_backend: auto

plugins:
  isolation: true
  lazy_loading: true
  default_idle_timeout: 120
  allow_legacy_bridge: false

providers:
  huggingface:
    enabled: true
    offline: false
  civitai_red:
    enabled: false

auth:
  mode: NONE_LOCALHOST
`;

const DEFAULT_WORKFLOW_JSON = `{
  "id": "flux-txt2img",
  "name": "FLUX.1 Text to Image",
  "nodes": [
    { "id": "ng.model.flux.load", "params": { "model_path": "/models/flux-dev" } },
    { "id": "ng.sample.flux", "params": { "prompt": "", "steps": 28, "guidance": 3.5 } }
  ],
  "connections": []
}`;

const WORKFLOW_TEMPLATES = [
  { id: 'flux-txt2img', name: 'FLUX Text to Image', description: 'Basic FLUX.1 text-to-image workflow', nodes: 2, category: 'flux' },
  { id: 'flux-img2img', name: 'FLUX Image to Image', description: 'FLUX.1 image-to-image with input image', nodes: 3, category: 'flux' },
  { id: 'flux-inpaint', name: 'FLUX Inpainting', description: 'FLUX.1 inpainting with mask', nodes: 4, category: 'flux' },
  { id: 'flux-lora', name: 'FLUX with LoRA', description: 'FLUX.1 with LoRA weight loading', nodes: 3, category: 'flux' },
];

export const EditorSurface: React.FC = () => {
  const [nodesDef, setNodesDef] = useState<NodeDefinition[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelItem[]>([]);
  const [search, setSearch] = useState('');
  const [activeLeftMenu, setActiveLeftMenu] = useState<LeftMenu>('nodes');
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [showRightPanel, setShowRightPanel] = useState(true);
  const [rightPanelTab, setRightPanelTab] = useState<'inspector' | 'logs' | 'workflow'>('workflow');

  const [nodesOnCanvas, setNodesOnCanvas] = useState<CanvasNode[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [wiringFrom, setWiringFrom] = useState<{ nodeId: string; portName: string; type: string; isOutput: boolean } | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [executionProgress, setExecutionProgress] = useState<number | null>(null);
  const [progressStatus, setProgressStatus] = useState<string>('');
  const [outputImageUrl, setOutputImageUrl] = useState<string | null>(null);
  const [generatedImages, setGeneratedImages] = useState<string[]>([]);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const [jobs, setJobs] = useState<Job[]>([]);
  const [logs, setLogs] = useState<LogEntry[]>([]);

  const [ngConfig, setNgConfig] = useState(DEFAULT_NG_CONFIG);
  const [ngConfigDirty, setNgConfigDirty] = useState(false);
  const [activeWorkflowTab, setActiveWorkflowTab] = useState(0);

  const [canvasOffset, setCanvasOffset] = useState({ x: 0, y: 0 });
  const [canvasScale, setCanvasScale] = useState(1);
  const [isPanning, setIsPanning] = useState(false);
  const [panStart, setPanStart] = useState({ x: 0, y: 0 });
  const [canvasPanOffset, setCanvasPanOffset] = useState({ x: 0, y: 0 });

  const viewportRef = useRef<HTMLDivElement>(null);
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);

  const addLog = useCallback((level: string, message: string) => {
    const now = new Date();
    const time = now.toLocaleTimeString('fr-FR', { hour12: false });
    setLogs(prev => [...prev.slice(-499), { time, level, message }]);
  }, []);

  useEffect(() => {
    addLog('info', 'Initializing ComfyUI-NG editor...');
    addLog('info', 'FLUX runtime detected, loading node catalogue...');
    Promise.all([fetchNodeCatalogue(), fetchModels()]).then(([defs, models]) => {
      setNodesDef(defs);
      setAvailableModels(models);
      addLog('info', `Loaded ${defs.length} node definitions, ${models.length} models`);

      if (defs.length > 0) {
        const modelLoader = defs.find((d) => d.name === 'ng.model.load') || defs[0];
        const textEncoder = defs.find((d) => d.name === 'ng.text_encoder.load') || defs[1] || defs[0];
        const promptEncode = defs.find((d) => d.name === 'ng.conditioning.prompt_encode') || defs[2] || defs[0];
        const sampler = defs.find((d) => d.name === 'ng.sample.run') || defs[3] || defs[0];
        const vaeDecode = defs.find((d) => d.name === 'ng.latent.latent_to_image') || defs[4] || defs[0];
        const saveImg = defs.find((d) => d.name === 'ng.image.save') || defs[5] || defs[0];
        const emptyLatent = defs.find((d) => d.name === 'ng.latent.empty') || defs[6] || defs[0];

        const initialNodes: CanvasNode[] = [
          { id: 'node-1', def: modelLoader, x: 40, y: 80, params: { path: models[0]?.name || '' } },
          { id: 'node-2', def: promptEncode, x: 340, y: 80, params: { prompt: 'A cybernetic space station in deep space, neon glows, hyper-detailed, 8k', negative_prompt: '' } },
          { id: 'node-3', def: emptyLatent, x: 340, y: 280, params: { width: 1024, height: 1024, batch_size: 1 } },
          { id: 'node-4', def: sampler, x: 680, y: 120, params: { steps: 28, guidance: 3.5, seed: 4242, sampler: 'euler' } },
          { id: 'node-5', def: vaeDecode, x: 1000, y: 120, params: {} },
          { id: 'node-6', def: saveImg, x: 1280, y: 120, params: { path: 'comfyng_flux_sample', format: 'png' } },
        ];
        setNodesOnCanvas(initialNodes);
        addLog('info', 'Default FLUX workflow loaded with 6 nodes');

        setConnections([
          { id: 'c1', fromNodeId: 'node-1', fromPort: 'model', toNodeId: 'node-4', toPort: 'model', type: 'NG_MODEL@1' },
          { id: 'c2', fromNodeId: 'node-1', fromPort: 'text_encoder', toNodeId: 'node-2', toPort: 'text_encoder', type: 'NG_TEXT_ENCODER@1' },
          { id: 'c3', fromNodeId: 'node-2', fromPort: 'conditioning', toNodeId: 'node-4', toPort: 'conditioning', type: 'NG_CONDITIONING@1' },
          { id: 'c4', fromNodeId: 'node-3', fromPort: 'latent', toNodeId: 'node-4', toPort: 'latent', type: 'NG_LATENT@1' },
          { id: 'c5', fromNodeId: 'node-4', fromPort: 'latent', toNodeId: 'node-5', toPort: 'latent', type: 'NG_LATENT@1' },
          { id: 'c6', fromNodeId: 'node-5', fromPort: 'image', toNodeId: 'node-6', toPort: 'image', type: 'NG_IMAGE@1' },
        ]);
      }
    }).catch(err => {
      addLog('error', `Failed to load catalogue: ${err.message}`);
    });

    fetchJobs().then(j => {
      setJobs(j);
      addLog('info', `Loaded ${j.length} existing jobs`);
    });
  }, []);

  useEffect(() => {
    if (!toastMessage) return;
    const t = setTimeout(() => setToastMessage(null), 4000);
    return () => clearTimeout(t);
  }, [toastMessage]);

  const addNodeToCanvas = (def: NodeDefinition) => {
    const newId = `node-${Date.now()}`;
    const defaultParams: Record<string, any> = {};
    def.parameters.forEach((p) => { defaultParams[p.name] = p.default ?? ''; });
    const newNode: CanvasNode = {
      id: newId, def,
      x: (window.innerWidth / 2 - 125) / canvasScale - canvasOffset.x / canvasScale,
      y: (window.innerHeight / 2 - 100) / canvasScale - canvasOffset.y / canvasScale,
      params: defaultParams,
    };
    setNodesOnCanvas((prev) => [...prev, newNode]);
    setSelectedNodeId(newId);
    addLog('info', `Added node: ${def.display_name} (${newId})`);
  };

  const removeNode = (nodeId: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    const node = nodesOnCanvas.find(n => n.id === nodeId);
    setNodesOnCanvas((prev) => prev.filter((n) => n.id !== nodeId));
    setConnections((prev) => prev.filter((c) => c.fromNodeId !== nodeId && c.toNodeId !== nodeId));
    if (selectedNodeId === nodeId) setSelectedNodeId(null);
    if (node) addLog('info', `Removed node: ${node.def.display_name}`);
  };

  const removeConnection = (connId: string) => {
    setConnections((prev) => prev.filter((c) => c.id !== connId));
    addLog('debug', `Removed wire: ${connId}`);
  };

  const handleCanvasMouseDown = (e: React.MouseEvent) => {
    if (e.button === 1 || (e.button === 0 && e.shiftKey)) {
      setIsPanning(true);
      setPanStart({ x: e.clientX - canvasPanOffset.x, y: e.clientY - canvasPanOffset.y });
      e.preventDefault();
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = viewportRef.current?.getBoundingClientRect();
    if (!rect) return;
    const currentX = (e.clientX - rect.left) / canvasScale;
    const currentY = (e.clientY - rect.top) / canvasScale;
    setMousePos({ x: currentX, y: currentY });

    if (isPanning) {
      setCanvasPanOffset({
        x: e.clientX - panStart.x,
        y: e.clientY - panStart.y,
      });
      return;
    }

    if (draggingNodeId) {
      const newX = currentX - dragOffset.x;
      const newY = currentY - dragOffset.y;
      setNodesOnCanvas((prev) =>
        prev.map((n) => (n.id === draggingNodeId ? { ...n, x: newX, y: newY } : n))
      );
    }
  };

  const handleMouseUp = () => {
    setDraggingNodeId(null);
    setWiringFrom(null);
    setIsPanning(false);
  };

  const handleWheel = (e: React.WheelEvent) => {
    if (e.ctrlKey || e.metaKey) {
      e.preventDefault();
      const delta = e.deltaY > 0 ? 0.9 : 1.1;
      const newScale = Math.max(0.2, Math.min(5, canvasScale * delta));
      setCanvasScale(newScale);
    } else {
      setCanvasPanOffset(p => ({
        x: p.x - e.deltaX,
        y: p.y - e.deltaY,
      }));
    }
  };

  const handleNodeMouseDown = (node: CanvasNode, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedNodeId(node.id);
    setDraggingNodeId(node.id);
    const rect = viewportRef.current?.getBoundingClientRect();
    if (rect) {
      setDragOffset({
        x: (e.clientX - rect.left) / canvasScale - node.x,
        y: (e.clientY - rect.top) / canvasScale - node.y,
      });
    }
  };

  const handlePortMouseDown = (nodeId: string, portName: string, type: string, isOutput: boolean, e: React.MouseEvent) => {
    e.stopPropagation();
    setWiringFrom({ nodeId, portName, type, isOutput });
  };

  const handlePortMouseUp = (targetNodeId: string, targetPortName: string, targetType: string, isTargetOutput: boolean, e: React.MouseEvent) => {
    e.stopPropagation();
    if (!wiringFrom) return;
    if (wiringFrom.nodeId === targetNodeId) return;
    if (wiringFrom.isOutput === isTargetOutput) return;

    const fromNodeId = wiringFrom.isOutput ? wiringFrom.nodeId : targetNodeId;
    const fromPort = wiringFrom.isOutput ? wiringFrom.portName : targetPortName;
    const toNodeId = wiringFrom.isOutput ? targetNodeId : wiringFrom.nodeId;
    const toPort = wiringFrom.isOutput ? targetPortName : wiringFrom.portName;

    const newConn: Connection = { id: `conn-${Date.now()}`, fromNodeId, fromPort, toNodeId, toPort, type: wiringFrom.type };
    setConnections((prev) => [...prev.filter((c) => !(c.toNodeId === toNodeId && c.toPort === toPort)), newConn]);
    addLog('debug', `Wired ${fromNodeId}:${fromPort} → ${toNodeId}:${toPort}`);
    setWiringFrom(null);
  };

  const handleExecute = async () => {
    setIsSubmitting(true);
    setExecutionProgress(5);
    setProgressStatus('Queueing job in Scheduler...');
    addLog('info', 'Submitting job to scheduler...');

    const samplerNode = nodesOnCanvas.find((n) => n.def.name === 'KSampler');
    const clipNode = nodesOnCanvas.find((n) => n.def.name === 'CLIPTextEncode');
    const promptText = clipNode?.params?.text || 'A cybernetic space station in deep space';
    const seedVal = samplerNode?.params?.seed ? Number(samplerNode.params.seed) : 42;
    const stepsVal = samplerNode?.params?.steps ? Number(samplerNode.params.steps) : 28;

    try {
      const job = await submitJob('FLUX.1 Generation', promptText, '', seedVal, stepsVal, 1024, 1024, nodesOnCanvas, connections);

      if (!job) {
        addLog('error', 'Job submission failed - no response from scheduler');
        setIsSubmitting(false);
        setExecutionProgress(null);
        setProgressStatus('');
        return;
      }

      addLog('info', `Job submitted: ${job.id} (${job.status})`);

      const pollInterval = setInterval(async () => {
        const updatedJob = await fetchJob(job.id);
        if (!updatedJob) return;

        if (updatedJob.status === 'queued') {
          setExecutionProgress(15);
          setProgressStatus('Queued - waiting for worker...');
          addLog('debug', `Job ${job.id}: queued, waiting in queue...`);
        } else if (updatedJob.status === 'preparing') {
          setExecutionProgress(40);
          setProgressStatus('Preparing FLUX runtime & loading model...');
          addLog('info', `Job ${job.id}: preparing runtime environment...`);
        } else if (updatedJob.status === 'running') {
          setExecutionProgress(75);
          setProgressStatus('Sampling FLUX diffusion steps...');
          addLog('info', `Job ${job.id}: sampling started (${stepsVal} steps)`);
        } else if (updatedJob.status === 'completed') {
          clearInterval(pollInterval);
          setIsSubmitting(false);
          setExecutionProgress(null);
          setProgressStatus('');
          addLog('info', `Job ${job.id}: COMPLETED! Image generated.`);

          if (updatedJob.image_url) {
            setOutputImageUrl(updatedJob.image_url);
            setGeneratedImages(prev => [updatedJob.image_url!, ...prev]);
            setToastMessage(`Image generated! (${updatedJob.id})`);
          }
        } else if (updatedJob.status === 'failed') {
          clearInterval(pollInterval);
          setIsSubmitting(false);
          setExecutionProgress(null);
          setProgressStatus('');
          addLog('error', `Job ${job.id}: FAILED - check worker logs`);
          setToastMessage(`Job failed: ${updatedJob.id}`);
        } else if (updatedJob.status === 'cancelled') {
          clearInterval(pollInterval);
          setIsSubmitting(false);
          setExecutionProgress(null);
          setProgressStatus('');
          addLog('warn', `Job ${job.id}: CANCELLED`);
        }
      }, 400);

      pollRef.current = pollInterval;

    } catch (err: any) {
      addLog('error', `Job submission error: ${err.message}`);
      setIsSubmitting(false);
      setExecutionProgress(null);
    }
  };

  const handleSaveConfig = () => {
    setNgConfigDirty(false);
    addLog('info', 'Configuration saved (local storage)');
    localStorage.setItem('comfyng_config', ngConfig);
    setToastMessage('Configuration saved!');
  };

  const handleLoadConfig = () => {
    const saved = localStorage.getItem('comfyng_config');
    if (saved) {
      setNgConfig(saved);
      setNgConfigDirty(false);
      addLog('info', 'Configuration loaded from local storage');
    }
  };

  const getPortPos = (nodeId: string, portName: string, isOutput: boolean) => {
    const node = nodesOnCanvas.find((n) => n.id === nodeId);
    if (!node) return { x: 0, y: 0 };
    const nodeWidth = 250;
    const x = isOutput ? node.x + nodeWidth : node.x;
    let portIndex = 0;
    if (isOutput) {
      portIndex = node.def.outputs.findIndex((o) => o.name === portName);
      if (portIndex === -1) portIndex = 0;
      return { x, y: node.y + 70 + portIndex * 26 + (node.def.inputs.length * 26) };
    } else {
      portIndex = node.def.inputs.findIndex((i) => i.name === portName);
      if (portIndex === -1) portIndex = 0;
      return { x, y: node.y + 70 + portIndex * 26 };
    }
  };

  const filteredDefs = nodesDef.filter(
    (d) => d.display_name.toLowerCase().includes(search.toLowerCase()) || d.category.toLowerCase().includes(search.toLowerCase())
  );
  const selectedNode = nodesOnCanvas.find((n) => n.id === selectedNodeId);

  const leftPanelContent = () => {
    switch (activeLeftMenu) {
      case 'queue':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', height: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 600, color: '#f59e0b', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Clock size={16} /> Queue
              </span>
              <span style={{ fontSize: '0.75rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                {jobs.length} total
              </span>
            </div>
            {jobs.length === 0 ? (
              <div style={{ color: 'var(--text-dim)', fontSize: '0.85rem', textAlign: 'center', padding: '2rem' }}>
                No jobs in queue. Run a workflow to see jobs here.
              </div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', overflowY: 'auto', flex: 1 }}>
                {jobs.map((job) => (
                  <div key={job.id} style={{
                    padding: '0.75rem',
                    background: 'rgba(30, 41, 59, 0.6)',
                    border: `1px solid ${
                      job.status === 'completed' ? 'rgba(16, 185, 129, 0.3)' :
                      job.status === 'failed' ? 'rgba(244, 63, 94, 0.3)' :
                      job.status === 'running' ? 'rgba(99, 102, 241, 0.3)' :
                      'var(--border-subtle)'
                    }`,
                    borderRadius: 'var(--radius-sm)',
                    display: 'flex', flexDirection: 'column', gap: '0.3rem',
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <span style={{ fontWeight: 600, fontSize: '0.8rem', color: '#f8fafc' }}>
                        {job.name}
                      </span>
                      <span className={`badge ${
                        job.status === 'completed' ? 'badge-emerald' :
                        job.status === 'failed' ? 'badge-amber' :
                        'badge-indigo'
                      }`} style={{ fontSize: '0.7rem' }}>
                        {job.status}
                      </span>
                    </div>
                    <div style={{ fontSize: '0.7rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                      ID: {job.id.slice(0, 12)}...
                    </div>
                    {job.duration_ms > 0 && (
                      <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                        Duration: {(job.duration_ms / 1000).toFixed(1)}s
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        );

      case 'images':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', height: '100%' }}>
            <div style={{ fontWeight: 600, color: '#10b981', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <ImageIcon size={16} /> Generated Images
            </div>
            {generatedImages.length === 0 ? (
              <div style={{ color: 'var(--text-dim)', fontSize: '0.85rem', textAlign: 'center', padding: '2rem' }}>
                No images generated yet. Execute a workflow to see results.
              </div>
            ) : (
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem', overflowY: 'auto', flex: 1 }}>
                {generatedImages.map((url, i) => (
                  <div key={i} style={{
                    borderRadius: 'var(--radius-sm)', overflow: 'hidden', cursor: 'pointer',
                    border: '1px solid var(--border-subtle)', aspectRatio: '1',
                    background: 'rgba(9, 13, 22, 0.8)',
                  }} onClick={() => setOutputImageUrl(url)}>
                    <img src={url} alt={`Generated ${i}`} style={{ width: '100%', height: '100%', objectFit: 'cover' }} />
                  </div>
                ))}
              </div>
            )}
          </div>
        );

      case 'gallery':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', height: '100%' }}>
            <div style={{ fontWeight: 600, color: '#8b5cf6', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <FolderOpen size={16} /> Gallery
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', flexWrap: 'wrap' }}>
              {['All', 'FLUX', 'Qwen', 'LoRA'].map(tag => (
                <button key={tag} className="btn btn-secondary" style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem' }}>
                  {tag}
                </button>
              ))}
            </div>
            <div style={{ color: 'var(--text-dim)', fontSize: '0.85rem', textAlign: 'center', padding: '2rem' }}>
              Workflow templates and presets will appear here.
            </div>
            {WORKFLOW_TEMPLATES.map((tmpl) => (
              <div key={tmpl.id} style={{
                padding: '0.6rem 0.8rem',
                background: 'rgba(30, 41, 59, 0.5)',
                border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)',
                cursor: 'pointer',
                transition: 'all 0.2s',
              }}
                onMouseEnter={e => (e.currentTarget.style.borderColor = 'rgba(99, 102, 241, 0.4)')}
                onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border-subtle)')}
              >
                <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{tmpl.name}</div>
                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>{tmpl.description}</div>
              </div>
            ))}
          </div>
        );

      case 'config':
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', height: '100%' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ fontWeight: 600, color: '#06b6d4', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Settings size={16} /> Configuration
              </span>
              <div style={{ display: 'flex', gap: '0.3rem' }}>
                <button className="btn btn-secondary" style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                  onClick={handleSaveConfig} disabled={!ngConfigDirty}>
                  <Save size={14} /> Save
                </button>
                <button className="btn btn-secondary" style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem' }}
                  onClick={handleLoadConfig}>
                  <RefreshCw size={14} /> Load
                </button>
              </div>
            </div>
            <textarea
              style={{
                flex: 1, width: '100%', resize: 'none',
                background: 'rgba(9, 13, 22, 0.9)', border: '1px solid var(--border-subtle)',
                borderRadius: 'var(--radius-sm)', color: '#e2e8f0',
                fontFamily: 'var(--font-mono)', fontSize: '0.75rem', padding: '0.75rem',
                lineHeight: '1.6', whiteSpace: 'pre', overflowWrap: 'normal', overflowX: 'auto',
              }}
              value={ngConfig}
              onChange={(e) => { setNgConfig(e.target.value); setNgConfigDirty(true); }}
              spellCheck={false}
            />
          </div>
        );

      case 'nodes':
      default:
        return (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', height: '100%' }}>
            <div style={{ fontWeight: 600, color: '#6366f1', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Search size={16} /> Node Palette
            </div>
            <input
              type="text" className="search-input" placeholder="Search nodes..."
              value={search} onChange={(e) => setSearch(e.target.value)}
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', overflowY: 'auto', flex: 1 }}>
              {filteredDefs.map((def, idx) => (
                <div key={idx} className="palette-item" onClick={() => addNodeToCanvas(def)}
                  style={{ cursor: 'grab', userSelect: 'none' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <span className="palette-item-name">{def.display_name}</span>
                    <Plus size={14} style={{ color: 'var(--accent-primary)' }} />
                  </div>
                  <span className="palette-item-desc">{def.description}</span>
                </div>
              ))}
            </div>
          </div>
        );
    }
  };

  return (
    <div className="editor-layout" onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}>
      {leftPanelOpen && (
        <div style={{
          width: 280, background: 'var(--bg-card)', backdropFilter: 'var(--glass-backdrop)',
          borderRight: '1px solid var(--border-subtle)', display: 'flex', flexDirection: 'column',
          height: '100%',
        }}>
          <div style={{ display: 'flex', flexDirection: 'column', borderBottom: '1px solid var(--border-subtle)' }}>
            {LEFT_MENU_ITEMS.map(item => (
              <button key={item.key}
                style={{
                  display: 'flex', alignItems: 'center', gap: '0.6rem', padding: '0.65rem 1rem',
                  background: activeLeftMenu === item.key ? 'rgba(99, 102, 241, 0.12)' : 'transparent',
                  border: 'none', borderLeft: activeLeftMenu === item.key ? '3px solid #6366f1' : '3px solid transparent',
                  color: activeLeftMenu === item.key ? '#f8fafc' : 'var(--text-muted)',
                  fontSize: '0.85rem', fontWeight: activeLeftMenu === item.key ? 600 : 400,
                  cursor: 'pointer', transition: 'all 0.15s ease', textAlign: 'left',
                  width: '100%',
                }}
                onClick={() => setActiveLeftMenu(item.key)}
                onMouseEnter={e => { if (activeLeftMenu !== item.key) e.currentTarget.style.background = 'rgba(255,255,255,0.03)'; }}
                onMouseLeave={e => { if (activeLeftMenu !== item.key) e.currentTarget.style.background = 'transparent'; }}
              >
                <span style={{ color: item.color }}>{item.icon}</span>
                {item.label}
              </button>
            ))}
          </div>
          <div style={{ flex: 1, padding: '0.75rem', overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
            {leftPanelContent()}
          </div>
        </div>
      )}

      <div className="canvas-viewport" ref={viewportRef}
        onMouseDown={handleCanvasMouseDown}
        onWheel={handleWheel}
        style={{ cursor: isPanning ? 'grabbing' : draggingNodeId ? 'grabbing' : 'default' }}
      >
        <div style={{
          transform: `translate(${canvasPanOffset.x}px, ${canvasPanOffset.y}px) scale(${canvasScale})`,
          transformOrigin: '0 0',
          width: '10000px', height: '10000px', position: 'relative',
        }}>
          <div style={{ position: 'absolute', top: 16, left: 16, display: 'flex', gap: '0.75rem', zIndex: 10 }}>
            {!leftPanelOpen && (
              <button className="btn btn-secondary" style={{ padding: '0.4rem 0.6rem' }}
                onClick={() => setLeftPanelOpen(true)}>
                <Menu size={16} />
              </button>
            )}
            <button className="btn btn-primary" onClick={handleExecute} disabled={isSubmitting}
              style={{ padding: '0.5rem 1.2rem' }}>
              <Play size={16} /> {isSubmitting ? '...' : 'Queue & Run'}
            </button>
            <button className="btn btn-secondary" style={{ padding: '0.4rem 0.8rem' }}>
              <Layers size={16} /> {nodesOnCanvas.length} n | {connections.length} w
            </button>
            {generatedImages.length > 0 && (
              <button className="btn btn-secondary" style={{ padding: '0.4rem 0.8rem', background: 'rgba(16, 185, 129, 0.15)', color: '#34d399' }}
                onClick={() => setActiveLeftMenu('images')}>
                <ImageIcon size={16} /> {generatedImages.length} imgs
              </button>
            )}
          </div>

          {executionProgress !== null && (
            <div style={{
              position: 'absolute', top: 70, left: '50%', transform: 'translateX(-50%)',
              width: '450px', background: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid var(--accent-primary)', borderRadius: '10px',
              padding: '0.8rem 1.2rem', zIndex: 30,
              display: 'flex', flexDirection: 'column', gap: '0.5rem',
              boxShadow: '0 0 30px rgba(99, 102, 241, 0.3)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.85rem', fontWeight: 600 }}>
                <span style={{ color: '#a5b4fc', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                  <Zap size={14} className="spin" /> {progressStatus}
                </span>
                <span>{executionProgress}%</span>
              </div>
              <div style={{ width: '100%', height: '6px', background: 'rgba(255,255,255,0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{ width: `${executionProgress}%`, height: '100%', background: 'linear-gradient(90deg, #6366f1, #10b981)', transition: 'width 0.3s ease' }} />
              </div>
            </div>
          )}

          {toastMessage && (
            <div style={{
              position: 'absolute', top: 16, right: showRightPanel ? 320 : 16,
              background: toastMessage.includes('fail') ? 'rgba(244, 63, 94, 0.95)' : 'rgba(16, 185, 129, 0.95)',
              color: '#042f2e', padding: '0.6rem 1.2rem', borderRadius: '8px', fontWeight: 600,
              boxShadow: `0 4px 20px ${toastMessage.includes('fail') ? 'rgba(244, 63, 94, 0.4)' : 'rgba(16, 185, 129, 0.4)'}`,
              display: 'flex', alignItems: 'center', gap: '0.5rem', zIndex: 20,
              transition: 'right 0.3s',
            }}>
              <CheckCircle2 size={18} /> {toastMessage}
            </div>
          )}

          <svg className="canvas-svg" style={{ pointerEvents: 'none', width: '10000px', height: '10000px' }}>
            {connections.map((conn) => {
              const start = getPortPos(conn.fromNodeId, conn.fromPort, true);
              const end = getPortPos(conn.toNodeId, conn.toPort, false);
              const ctrlOffset = Math.max(40, Math.abs(end.x - start.x) * 0.4);
              const pathData = `M ${start.x} ${start.y} C ${start.x + ctrlOffset} ${start.y}, ${end.x - ctrlOffset} ${end.y}, ${end.x} ${end.y}`;
              const wireColor = TYPE_COLORS[conn.type] || '#6366f1';
              return (
                <g key={conn.id} onClick={() => removeConnection(conn.id)} style={{ cursor: 'pointer' }}>
                  <path d={pathData} stroke="rgba(0,0,0,0.6)" strokeWidth="6" fill="none" />
                  <path d={pathData} stroke={wireColor} strokeWidth="3" fill="none" className="wire-path" />
                </g>
              );
            })}
            {wiringFrom && (() => {
              const start = getPortPos(wiringFrom.nodeId, wiringFrom.portName, wiringFrom.isOutput);
              const end = mousePos;
              const ctrlOffset = 60;
              const pathData = wiringFrom.isOutput
                ? `M ${start.x} ${start.y} C ${start.x + ctrlOffset} ${start.y}, ${end.x - ctrlOffset} ${end.y}, ${end.x} ${end.y}`
                : `M ${start.x} ${start.y} C ${start.x - ctrlOffset} ${start.y}, ${end.x + ctrlOffset} ${end.y}, ${end.x} ${end.y}`;
              const wireColor = TYPE_COLORS[wiringFrom.type] || '#6366f1';
              return <path d={pathData} stroke={wireColor} strokeWidth="3" strokeDasharray="4 2" fill="none" />;
            })()}
          </svg>

          {nodesOnCanvas.map((node) => (
            <div key={node.id}
              className={`canvas-node ${selectedNodeId === node.id ? 'selected' : ''}`}
              style={{ left: node.x, top: node.y, width: 250, position: 'absolute' }}
              onClick={(e) => { e.stopPropagation(); setSelectedNodeId(node.id); }}
            >
              <div className="canvas-node-header" onMouseDown={(e) => handleNodeMouseDown(node, e)}
                style={{ cursor: 'move', userSelect: 'none' }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flex: 1, overflow: 'hidden' }}>
                  <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                    {node.def.display_name}
                  </span>
                </div>
                <Trash2 size={14} style={{ cursor: 'pointer', color: 'var(--text-muted)', marginLeft: '0.5rem' }}
                  onClick={(e) => removeNode(node.id, e)} />
              </div>
              <div className="canvas-node-body">
                {node.def.inputs.map((inp, idx) => {
                  const portColor = TYPE_COLORS[inp.type] || '#818cf8';
                  return (
                    <div key={idx} className="port-row">
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'crosshair' }}
                        onMouseDown={(e) => handlePortMouseDown(node.id, inp.name, inp.type, false, e)}
                        onMouseUp={(e) => handlePortMouseUp(node.id, inp.name, inp.type, false, e)}>
                        <div className="port-handle" style={{ background: portColor }} />
                        <span style={{ fontWeight: 500 }}>{inp.name}</span>
                      </div>
                      <span style={{ fontSize: '0.7rem', color: portColor }}>{inp.type}</span>
                    </div>
                  );
                })}
                {node.def.outputs.map((out, idx) => {
                  const portColor = TYPE_COLORS[out.type] || '#34d399';
                  return (
                    <div key={idx} className="port-row" style={{ justifyContent: 'flex-end' }}>
                      <span style={{ fontSize: '0.7rem', color: portColor, marginRight: '0.4rem' }}>{out.type}</span>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'crosshair' }}
                        onMouseDown={(e) => handlePortMouseDown(node.id, out.name, out.type, true, e)}
                        onMouseUp={(e) => handlePortMouseUp(node.id, out.name, out.type, true, e)}>
                        <span style={{ fontWeight: 500 }}>{out.name}</span>
                        <div className="port-handle" style={{ background: portColor }} />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          ))}
        </div>

        {/* Zoom controls */}
        <div style={{ position: 'absolute', bottom: 16, right: showRightPanel ? 316 : 16, display: 'flex', gap: '0.3rem', zIndex: 10 }}>
          <button className="btn btn-secondary" style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
            onClick={() => setCanvasScale(s => Math.max(0.2, s / 1.2))}>-</button>
          <span style={{
            padding: '0.3rem 0.6rem', fontSize: '0.75rem', color: 'var(--text-muted)',
            background: 'var(--bg-card)', border: '1px solid var(--border-subtle)', borderRadius: 'var(--radius-sm)',
            fontFamily: 'var(--font-mono)',
          }}>{Math.round(canvasScale * 100)}%</span>
          <button className="btn btn-secondary" style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
            onClick={() => setCanvasScale(s => Math.min(5, s * 1.2))}>+</button>
          <button className="btn btn-secondary" style={{ padding: '0.3rem 0.6rem', fontSize: '0.75rem' }}
            onClick={() => { setCanvasScale(1); setCanvasPanOffset({ x: 0, y: 0 }); }}>Fit</button>
        </div>
      </div>

      {showRightPanel ? (
        <div className="inspector-panel" style={{ width: 300, display: 'flex', flexDirection: 'column' }}>
          <div style={{ display: 'flex', borderBottom: '1px solid var(--border-subtle)' }}>
            <button style={{
              flex: 1, padding: '0.6rem 0.5rem', border: 'none', background: rightPanelTab === 'workflow' ? 'rgba(99, 102, 241, 0.12)' : 'transparent',
              color: rightPanelTab === 'workflow' ? '#f8fafc' : 'var(--text-muted)', fontWeight: rightPanelTab === 'workflow' ? 600 : 400,
              fontSize: '0.8rem', cursor: 'pointer', borderBottom: rightPanelTab === 'workflow' ? '2px solid #6366f1' : '2px solid transparent',
            }} onClick={() => setRightPanelTab('workflow')}>
              <FileJson size={14} style={{ marginRight: '0.3rem', verticalAlign: 'middle' }} /> Workflow
            </button>
            <button style={{
              flex: 1, padding: '0.6rem 0.5rem', border: 'none', background: rightPanelTab === 'inspector' ? 'rgba(99, 102, 241, 0.12)' : 'transparent',
              color: rightPanelTab === 'inspector' ? '#f8fafc' : 'var(--text-muted)', fontWeight: rightPanelTab === 'inspector' ? 600 : 400,
              fontSize: '0.8rem', cursor: 'pointer', borderBottom: rightPanelTab === 'inspector' ? '2px solid #6366f1' : '2px solid transparent',
            }} onClick={() => setRightPanelTab('inspector')}>
              <Sliders size={14} style={{ marginRight: '0.3rem', verticalAlign: 'middle' }} /> Inspector
            </button>
            <button style={{
              flex: 1, padding: '0.6rem 0.5rem', border: 'none', background: rightPanelTab === 'logs' ? 'rgba(99, 102, 241, 0.12)' : 'transparent',
              color: rightPanelTab === 'logs' ? '#f8fafc' : 'var(--text-muted)', fontWeight: rightPanelTab === 'logs' ? 600 : 400,
              fontSize: '0.8rem', cursor: 'pointer', borderBottom: rightPanelTab === 'logs' ? '2px solid #6366f1' : '2px solid transparent',
            }} onClick={() => setRightPanelTab('logs')}>
              <List size={14} style={{ marginRight: '0.3rem', verticalAlign: 'middle' }} /> Logs
            </button>
            <button className="btn btn-secondary" style={{ padding: '0.2rem 0.4rem', border: 'none' }}
              onClick={() => setShowRightPanel(false)}>
              <PanelRightClose size={14} />
            </button>
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: '0.75rem' }}>
            {rightPanelTab === 'inspector' && (
              selectedNode ? (
                <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
                  <div>
                    <div style={{ fontWeight: 700, fontSize: '1rem', color: '#f8fafc' }}>{selectedNode.def.display_name}</div>
                    <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.2rem' }}>{selectedNode.def.description}</div>
                    <div style={{ fontSize: '0.7rem', color: '#64748b', fontFamily: 'var(--font-mono)', marginTop: '0.3rem' }}>ID: {selectedNode.id} | Cat: {selectedNode.def.category}</div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                    {selectedNode.def.parameters.map((param, idx) => (
                      <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                        <label style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a5b4fc' }}>
                          {param.name} ({param.type})
                        </label>
                        {param.type === 'MODEL' || param.name.includes('ckpt') || param.options ? (
                          <select className="search-input"
                            value={selectedNode.params[param.name] || availableModels[0]?.name || ''}
                            onChange={(e) => {
                              const val = e.target.value;
                              setNodesOnCanvas((prev) =>
                                prev.map((n) => n.id === selectedNode.id ? { ...n, params: { ...n.params, [param.name]: val } } : n)
                              );
                            }}>
                            {availableModels.map((m) => (
                              <option key={m.name} value={m.name}>{m.display_name || m.name} ({m.size_gb} GB)</option>
                            ))}
                          </select>
                        ) : param.type === 'STRING' && param.name === 'text' ? (
                          <textarea rows={4} className="search-input"
                            value={selectedNode.params[param.name] || ''}
                            onChange={(e) => {
                              const val = e.target.value;
                              setNodesOnCanvas((prev) =>
                                prev.map((n) => n.id === selectedNode.id ? { ...n, params: { ...n.params, [param.name]: val } } : n)
                              );
                            }} />
                        ) : (
                          <input type="text" className="search-input"
                            value={selectedNode.params[param.name] ?? ''}
                            onChange={(e) => {
                              const val = e.target.value;
                              setNodesOnCanvas((prev) =>
                                prev.map((n) => n.id === selectedNode.id ? { ...n, params: { ...n.params, [param.name]: val } } : n)
                              );
                            }} />
                        )}
                      </div>
                    ))}
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '0.4rem', borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' }}>
                    <div style={{ fontWeight: 600, fontSize: '0.8rem', color: '#cbd5e1' }}>Ports</div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Inputs: {selectedNode.def.inputs.map(i => i.name).join(', ') || 'none'}
                    </div>
                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                      Outputs: {selectedNode.def.outputs.map(o => o.name).join(', ') || 'none'}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ color: '#64748b', fontSize: '0.85rem', textAlign: 'center', marginTop: '2rem' }}>
                  Click a node on canvas to inspect parameters
                </div>
              )
            )}

            {rightPanelTab === 'logs' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.2rem', height: '100%' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <span style={{ fontSize: '0.8rem', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                    {logs.length} entries
                  </span>
                  <button className="btn btn-secondary" style={{ padding: '0.2rem 0.5rem', fontSize: '0.7rem' }}
                    onClick={() => setLogs([])}>Clear</button>
                </div>
                <div style={{
                  flex: 1, overflowY: 'auto', fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                  background: 'rgba(9, 13, 22, 0.9)', borderRadius: 'var(--radius-sm)',
                  padding: '0.5rem', lineHeight: '1.6',
                }}>
                  {logs.length === 0 ? (
                    <span style={{ color: 'var(--text-dim)' }}>No log entries yet.</span>
                  ) : (
                    logs.map((log, i) => (
                      <div key={i} style={{
                        color: log.level === 'error' ? '#f43f5e' : log.level === 'warn' ? '#f59e0b' : log.level === 'debug' ? '#64748b' : '#cbd5e1',
                      }}>
                        <span style={{ color: '#52525b' }}>[{log.time}]</span>{' '}
                        <span style={{ color: log.level === 'error' ? '#f43f5e' : log.level === 'warn' ? '#f59e0b' : log.level === 'debug' ? '#64748b' : '#6366f1' }}>
                          {log.level.toUpperCase().padEnd(5)}
                        </span>{' '}
                        {log.message}
                      </div>
                    ))
                  )}
                </div>
              </div>
            )}

            {rightPanelTab === 'workflow' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>Workflow Info</div>
                <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)' }}>
                  <div>Nodes: {nodesOnCanvas.length}</div>
                  <div>Connections: {connections.length}</div>
                  <div>Selected: {selectedNodeId || 'none'}</div>
                </div>
                <div style={{ borderTop: '1px solid var(--border-subtle)', paddingTop: '0.75rem' }}>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem', marginBottom: '0.5rem' }}>System Stats</div>
                  <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.5rem' }}>
                    {[
                      { icon: <Cpu size={14} />, label: 'CPU', value: `${navigator.hardwareConcurrency || 8} cores` },
                      { icon: <HardDrive size={14} />, label: 'RAM', value: '32 GB' },
                      { icon: <Activity size={14} />, label: 'Scale', value: `${Math.round(canvasScale * 100)}%` },
                      { icon: <Server size={14} />, label: 'Version', value: '0.1.0' },
                    ].map((stat, i) => (
                      <div key={i} style={{
                        padding: '0.5rem', background: 'rgba(30, 41, 59, 0.5)', borderRadius: 'var(--radius-sm)',
                        display: 'flex', flexDirection: 'column', gap: '0.2rem', fontSize: '0.75rem',
                      }}>
                        <span style={{ color: '#64748b', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                          {stat.icon} {stat.label}
                        </span>
                        <span style={{ fontWeight: 600, fontSize: '0.85rem' }}>{stat.value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      ) : (
        <button className="btn btn-secondary" style={{ position: 'absolute', top: 16, right: 16, zIndex: 15 }}
          onClick={() => setShowRightPanel(true)}>
          <PanelRightOpen size={16} /> Panel
        </button>
      )}

      {outputImageUrl && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0, 0, 0, 0.88)', backdropFilter: 'blur(12px)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 100,
        }} onClick={() => setOutputImageUrl(null)}>
          <div style={{
            background: '#0f172a', border: '1px solid var(--border-glow)', borderRadius: '16px',
            maxWidth: '900px', width: '90%', overflow: 'hidden', boxShadow: '0 0 50px rgba(99, 102, 241, 0.4)',
            display: 'flex', flexDirection: 'column',
          }} onClick={(e) => e.stopPropagation()}>
            <div style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center',
              padding: '1rem 1.5rem', borderBottom: '1px solid var(--border-subtle)',
              background: 'rgba(30, 41, 59, 0.7)',
            }}>
              <div style={{ fontWeight: 700, fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sparkles color="var(--accent-emerald)" size={20} /> Generated Image
              </div>
              <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                <button className="btn btn-primary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}
                  onClick={() => window.open(outputImageUrl, '_blank')}>
                  <Eye size={14} /> Open
                </button>
                <a href={outputImageUrl} download="comfyng_output.png" className="btn btn-secondary" style={{ padding: '0.4rem 0.8rem', fontSize: '0.8rem' }}>
                  <Download size={14} /> Save
                </a>
                <button className="btn btn-secondary" style={{ padding: '0.3rem 0.5rem' }} onClick={() => setOutputImageUrl(null)}>
                  <X size={18} />
                </button>
              </div>
            </div>
            <div style={{ padding: '1.5rem', display: 'flex', justifyContent: 'center', background: 'rgba(9, 13, 22, 0.5)' }}>
              <img src={outputImageUrl} alt="Generated"
                style={{
                  maxWidth: '100%', maxHeight: '520px', borderRadius: '12px',
                  boxShadow: '0 8px 30px rgba(0,0,0,0.5)', border: '1px solid var(--border-subtle)',
                  objectFit: 'contain',
                }} />
            </div>
            <div style={{
              padding: '0.75rem 1.5rem',
              background: 'rgba(9,13,22,0.8)', borderTop: '1px solid var(--border-subtle)',
              display: 'flex', justifyContent: 'space-between', alignItems: 'center', fontSize: '0.85rem',
            }}>
              <span style={{ color: 'var(--text-muted)' }}>
                FLUX.1 DEV • 1024×1024 • 28 steps • CFG 3.5
              </span>
              <span style={{ color: 'var(--text-dim)', fontSize: '0.75rem', fontFamily: 'var(--font-mono)' }}>
                PNG • SHA256
              </span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
