import React, { useState, useEffect, useRef } from 'react';
import { NodeDefinition, ModelItem, fetchNodeCatalogue, fetchModels, submitJob, fetchJob } from '../../api/client';
import {
  Play,
  Plus,
  Trash2,
  CheckCircle2,
  Layers,
  Sliders,
  X,
  Sparkles,
  PanelLeftClose,
  PanelLeftOpen,
  PanelRightClose,
  PanelRightOpen,
  Eye,
  Download,
  Image as ImageIcon,
  Zap,
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
};

export const EditorSurface: React.FC = () => {
  const [nodesDef, setNodesDef] = useState<NodeDefinition[]>([]);
  const [availableModels, setAvailableModels] = useState<ModelItem[]>([]);
  const [search, setSearch] = useState('');

  // Side Panels Visibility Toggles
  const [showPalette, setShowPalette] = useState(true);
  const [showInspector, setShowInspector] = useState(true);

  // Canvas State
  const [nodesOnCanvas, setNodesOnCanvas] = useState<CanvasNode[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Dragging state for nodes
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Wiring state
  const [wiringFrom, setWiringFrom] = useState<{ nodeId: string; portName: string; type: string; isOutput: boolean } | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Execution & Progress State
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [executionProgress, setExecutionProgress] = useState<number | null>(null);
  const [progressStatus, setProgressStatus] = useState<string>('');
  const [outputImageUrl, setOutputImageUrl] = useState<string | null>(null);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const viewportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    Promise.all([fetchNodeCatalogue(), fetchModels()]).then(([defs, models]) => {
      setNodesDef(defs);
      setAvailableModels(models);

      if (defs.length > 0) {
        const ckpt = defs.find((d) => d.name === 'LoadCheckpoint') || defs[0];
        const clip = defs.find((d) => d.name === 'CLIPTextEncode') || defs[1] || defs[0];
        const latent = defs.find((d) => d.name === 'EmptyLatentImage') || defs[2] || defs[0];
        const sampler = defs.find((d) => d.name === 'KSampler') || defs[3] || defs[0];
        const decode = defs.find((d) => d.name === 'VAEDecode') || defs[4] || defs[0];
        const saveImg = defs.find((d) => d.name === 'SaveImage') || defs[5] || defs[0];

        const initialNodes: CanvasNode[] = [
          { id: 'node-1', def: ckpt, x: 40, y: 80, params: { ckpt_name: models[0]?.name || '' } },
          { id: 'node-2', def: clip, x: 340, y: 80, params: { text: 'A high-tech cybernetic space station surrounded by glowing neon plasma rings in deep space, hyper-detailed, 8k' } },
          { id: 'node-3', def: latent, x: 340, y: 280, params: { width: 1024, height: 1024, batch_size: 1 } },
          { id: 'node-4', def: sampler, x: 680, y: 120, params: { steps: 25, cfg: 3.5, seed: 4242, sampler_name: 'euler' } },
          { id: 'node-5', def: decode, x: 1000, y: 120, params: {} },
          { id: 'node-6', def: saveImg, x: 1280, y: 120, params: { filename_prefix: 'comfyng_flux_sample' } },
        ];
        setNodesOnCanvas(initialNodes);

        setConnections([
          { id: 'c1', fromNodeId: 'node-1', fromPort: 'MODEL', toNodeId: 'node-4', toPort: 'MODEL', type: 'MODEL' },
          { id: 'c2', fromNodeId: 'node-1', fromPort: 'CLIP', toNodeId: 'node-2', toPort: 'CLIP', type: 'CLIP' },
          { id: 'c3', fromNodeId: 'node-2', fromPort: 'CONDITIONING', toNodeId: 'node-4', toPort: 'POSITIVE', type: 'CONDITIONING' },
          { id: 'c4', fromNodeId: 'node-3', fromPort: 'LATENT', toNodeId: 'node-4', toPort: 'LATENT', type: 'LATENT' },
          { id: 'c5', fromNodeId: 'node-4', fromPort: 'LATENT', toNodeId: 'node-5', toPort: 'LATENT', type: 'LATENT' },
          { id: 'c6', fromNodeId: 'node-1', fromPort: 'VAE', toNodeId: 'node-5', toPort: 'VAE', type: 'VAE' },
          { id: 'c7', fromNodeId: 'node-5', fromPort: 'IMAGE', toNodeId: 'node-6', toPort: 'IMAGE', type: 'IMAGE' },
        ]);
        setSelectedNodeId('node-2');
      }
    });
  }, []);

  const addNodeToCanvas = (def: NodeDefinition) => {
    const newId = `node-${Date.now()}`;
    const defaultParams: Record<string, any> = {};
    def.parameters.forEach((p) => {
      defaultParams[p.name] = p.default ?? '';
    });
    const newNode: CanvasNode = {
      id: newId,
      def,
      x: 350 + Math.random() * 80,
      y: 120 + Math.random() * 80,
      params: defaultParams,
    };
    setNodesOnCanvas((prev) => [...prev, newNode]);
    setSelectedNodeId(newId);
  };

  const removeNode = (nodeId: string, e?: React.MouseEvent) => {
    if (e) e.stopPropagation();
    setNodesOnCanvas((prev) => prev.filter((n) => n.id !== nodeId));
    setConnections((prev) => prev.filter((c) => c.fromNodeId !== nodeId && c.toNodeId !== nodeId));
    if (selectedNodeId === nodeId) setSelectedNodeId(null);
  };

  const removeConnection = (connId: string) => {
    setConnections((prev) => prev.filter((c) => c.id !== connId));
  };

  // Node Drag Handlers
  const handleNodeMouseDown = (node: CanvasNode, e: React.MouseEvent) => {
    e.stopPropagation();
    setSelectedNodeId(node.id);
    setDraggingNodeId(node.id);
    const rect = viewportRef.current?.getBoundingClientRect();
    if (rect) {
      setDragOffset({
        x: e.clientX - rect.left - node.x,
        y: e.clientY - rect.top - node.y,
      });
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    const rect = viewportRef.current?.getBoundingClientRect();
    if (!rect) return;
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;

    setMousePos({ x: currentX, y: currentY });

    if (draggingNodeId) {
      const newX = currentX - dragOffset.x;
      const newY = currentY - dragOffset.y;
      setNodesOnCanvas((prev) =>
        prev.map((n) => (n.id === draggingNodeId ? { ...n, x: Math.max(0, newX), y: Math.max(0, newY) } : n))
      );
    }
  };

  const handleMouseUp = () => {
    setDraggingNodeId(null);
    setWiringFrom(null);
  };

  // Port Wiring Start
  const handlePortMouseDown = (
    nodeId: string,
    portName: string,
    type: string,
    isOutput: boolean,
    e: React.MouseEvent
  ) => {
    e.stopPropagation();
    setWiringFrom({ nodeId, portName, type, isOutput });
  };

  // Port Wiring Complete
  const handlePortMouseUp = (
    targetNodeId: string,
    targetPortName: string,
    targetType: string,
    isTargetOutput: boolean,
    e: React.MouseEvent
  ) => {
    e.stopPropagation();
    if (!wiringFrom) return;
    if (wiringFrom.nodeId === targetNodeId) return;
    if (wiringFrom.isOutput === isTargetOutput) return;

    const fromNodeId = wiringFrom.isOutput ? wiringFrom.nodeId : targetNodeId;
    const fromPort = wiringFrom.isOutput ? wiringFrom.portName : targetPortName;
    const toNodeId = wiringFrom.isOutput ? targetNodeId : wiringFrom.nodeId;
    const toPort = wiringFrom.isOutput ? targetPortName : wiringFrom.portName;

    const newConn: Connection = {
      id: `conn-${Date.now()}`,
      fromNodeId,
      fromPort,
      toNodeId,
      toPort,
      type: wiringFrom.type,
    };

    setConnections((prev) => [
      ...prev.filter((c) => !(c.toNodeId === toNodeId && c.toPort === toPort)),
      newConn,
    ]);
    setWiringFrom(null);
  };

  const handleExecute = async () => {
    setIsSubmitting(true);
    setExecutionProgress(5);
    setProgressStatus('Queueing job in Scheduler...');

    // Extract dynamic parameter values from canvas state
    const loadNode = nodesOnCanvas.find((n) => n.def.name === 'LoadCheckpoint');
    const clipNode = nodesOnCanvas.find((n) => n.def.name === 'CLIPTextEncode');
    const latentNode = nodesOnCanvas.find((n) => n.def.name === 'EmptyLatentImage');
    const samplerNode = nodesOnCanvas.find((n) => n.def.name === 'KSampler');

    const promptText = clipNode?.params?.text || 'A high-tech cybernetic space station in deep space';
    const modelName = loadNode?.params?.ckpt_name || availableModels[0]?.name || '';
    if (!modelName) {
      throw new Error('Aucun modèle réel trouvé dans models/diffusion_models ou models/checkpoints');
    }
    const seedVal = samplerNode?.params?.seed ? Number(samplerNode.params.seed) : 42;
    const stepsVal = samplerNode?.params?.steps ? Number(samplerNode.params.steps) : 25;
    const widthVal = latentNode?.params?.width ? Number(latentNode.params.width) : 1024;
    const heightVal = latentNode?.params?.height ? Number(latentNode.params.height) : 1024;

    const job = await submitJob(
      'FLUX.1 Interactive Generation',
      promptText,
      modelName,
      seedVal,
      stepsVal,
      widthVal,
      heightVal,
      nodesOnCanvas,
      connections
    );

    if (!job) {
      setIsSubmitting(false);
      setExecutionProgress(null);
      alert('Failed to submit job to scheduler.');
      return;
    }

    // Start polling job status
    const pollInterval = setInterval(async () => {
      const updatedJob = await fetchJob(job.id);
      if (!updatedJob) return;

      if (updatedJob.status === 'queued') {
        setExecutionProgress(15);
        setProgressStatus('Queued in Job Queue...');
      } else if (updatedJob.status === 'preparing') {
        setExecutionProgress(40);
        setProgressStatus('Preparing execution environment & JIT sandboxes...');
      } else if (updatedJob.status === 'running') {
        setExecutionProgress(75);
        setProgressStatus('Sampling diffusion steps in sandboxed process...');
      } else if (updatedJob.status === 'completed') {
        clearInterval(pollInterval);
        setIsSubmitting(false);
        setExecutionProgress(null);
        if (updatedJob.image_url) {
          setOutputImageUrl(updatedJob.image_url);
          setToastMessage(`Job completed! Real PNG generated in CAS: ${updatedJob.id}`);
          setTimeout(() => setToastMessage(null), 4000);
        }
      } else if (updatedJob.status === 'failed') {
        clearInterval(pollInterval);
        setIsSubmitting(false);
        setExecutionProgress(null);
        alert('Job execution failed in worker process.');
      }
    }, 400);
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
    (d) =>
      d.display_name.toLowerCase().includes(search.toLowerCase()) ||
      d.category.toLowerCase().includes(search.toLowerCase())
  );

  const selectedNode = nodesOnCanvas.find((n) => n.id === selectedNodeId);

  return (
    <div className="editor-layout" onMouseMove={handleMouseMove} onMouseUp={handleMouseUp}>
      {/* Node Palette Collapsable Panel */}
      {showPalette ? (
        <div className="node-palette">
          <div className="palette-header">
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Sparkles size={16} color="var(--accent-primary)" /> Node Palette
            </span>
            <button className="btn btn-secondary" style={{ padding: '0.2rem 0.4rem' }} onClick={() => setShowPalette(false)}>
              <PanelLeftClose size={16} />
            </button>
          </div>

          <input
            type="text"
            className="search-input"
            placeholder="Search nodes..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />

          <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', overflowY: 'auto' }}>
            {filteredDefs.map((def, idx) => (
              <div key={idx} className="palette-item" onClick={() => addNodeToCanvas(def)}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                  <span className="palette-item-name">{def.display_name}</span>
                  <Plus size={14} style={{ color: 'var(--accent-primary)' }} />
                </div>
                <span className="palette-item-desc">{def.description}</span>
              </div>
            ))}
          </div>
        </div>
      ) : (
        <button
          className="btn btn-secondary"
          style={{ position: 'absolute', top: 16, left: 16, zIndex: 15 }}
          onClick={() => setShowPalette(true)}
        >
          <PanelLeftOpen size={16} /> Show Palette
        </button>
      )}

      {/* Canvas Viewport */}
      <div className="canvas-viewport" ref={viewportRef}>
        {/* Top Control Overlay */}
        <div style={{ position: 'absolute', top: 16, left: showPalette ? 16 : 160, display: 'flex', gap: '0.75rem', zIndex: 10, transition: 'left 0.3s' }}>
          <button className="btn btn-primary" onClick={handleExecute} disabled={isSubmitting}>
            <Play size={16} />
            {isSubmitting ? 'Running Generation...' : 'Run Workflow'}
          </button>
          <button className="btn btn-secondary">
            <Layers size={16} /> Nodes: {nodesOnCanvas.length} | Wires: {connections.length}
          </button>
          {outputImageUrl && (
            <button className="btn btn-secondary" style={{ background: 'rgba(16, 185, 129, 0.2)', color: '#34d399', borderColor: 'rgba(16, 185, 129, 0.4)' }} onClick={() => setOutputImageUrl(outputImageUrl)}>
              <ImageIcon size={16} /> View Output Image
            </button>
          )}
        </div>

        {/* Progress Bar Overlay */}
        {executionProgress !== null && (
          <div
            style={{
              position: 'absolute',
              top: 70,
              left: '50%',
              transform: 'translateX(-50%)',
              width: '420px',
              background: 'rgba(15, 23, 42, 0.95)',
              border: '1px solid var(--accent-primary)',
              boxShadow: '0 0 25px rgba(99, 102, 241, 0.4)',
              borderRadius: '10px',
              padding: '0.8rem 1.2rem',
              zIndex: 30,
              display: 'flex',
              flexDirection: 'column',
              gap: '0.5rem',
            }}
          >
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
          <div
            style={{
              position: 'absolute',
              top: 16,
              right: showInspector ? 320 : 16,
              background: 'rgba(16, 185, 129, 0.95)',
              color: '#042f2e',
              padding: '0.6rem 1.2rem',
              borderRadius: '8px',
              fontWeight: 600,
              boxShadow: '0 4px 20px rgba(16, 185, 129, 0.4)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              zIndex: 20,
              transition: 'right 0.3s',
            }}
          >
            <CheckCircle2 size={18} /> {toastMessage}
          </div>
        )}

        {/* SVG Connections Canvas */}
        <svg className="canvas-svg">
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

          {wiringFrom && (
            (() => {
              const start = getPortPos(wiringFrom.nodeId, wiringFrom.portName, wiringFrom.isOutput);
              const end = mousePos;
              const ctrlOffset = 60;
              const pathData = wiringFrom.isOutput
                ? `M ${start.x} ${start.y} C ${start.x + ctrlOffset} ${start.y}, ${end.x - ctrlOffset} ${end.y}, ${end.x} ${end.y}`
                : `M ${start.x} ${start.y} C ${start.x - ctrlOffset} ${start.y}, ${end.x + ctrlOffset} ${end.y}, ${end.x} ${end.y}`;
              const wireColor = TYPE_COLORS[wiringFrom.type] || '#6366f1';

              return <path d={pathData} stroke={wireColor} strokeWidth="3" strokeDasharray="4 2" fill="none" />;
            })()
          )}
        </svg>

        {/* Nodes on Canvas */}
        {nodesOnCanvas.map((node) => (
          <div
            key={node.id}
            className={`canvas-node ${selectedNodeId === node.id ? 'selected' : ''}`}
            style={{ left: node.x, top: node.y, width: 250 }}
            onClick={(e) => {
              e.stopPropagation();
              setSelectedNodeId(node.id);
            }}
          >
            <div
              className="canvas-node-header"
              onMouseDown={(e) => handleNodeMouseDown(node, e)}
              style={{ cursor: 'move', userSelect: 'none' }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', flex: 1, overflow: 'hidden' }}>
                <span style={{ textOverflow: 'ellipsis', overflow: 'hidden', whiteSpace: 'nowrap' }}>
                  {node.def.display_name}
                </span>
              </div>
              <Trash2
                size={14}
                style={{ cursor: 'pointer', color: 'var(--text-muted)', marginLeft: '0.5rem' }}
                onClick={(e) => removeNode(node.id, e)}
              />
            </div>

            <div className="canvas-node-body">
              {/* Input Ports */}
              {node.def.inputs.map((inp, idx) => {
                const portColor = TYPE_COLORS[inp.type] || '#818cf8';
                return (
                  <div key={idx} className="port-row">
                    <div
                      style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'crosshair' }}
                      onMouseDown={(e) => handlePortMouseDown(node.id, inp.name, inp.type, false, e)}
                      onMouseUp={(e) => handlePortMouseUp(node.id, inp.name, inp.type, false, e)}
                    >
                      <div className="port-handle" style={{ background: portColor }} />
                      <span style={{ fontWeight: 500 }}>{inp.name}</span>
                    </div>
                    <span style={{ fontSize: '0.7rem', color: portColor }}>{inp.type}</span>
                  </div>
                );
              })}

              {/* Output Ports */}
              {node.def.outputs.map((out, idx) => {
                const portColor = TYPE_COLORS[out.type] || '#34d399';
                return (
                  <div key={idx} className="port-row" style={{ justifyContent: 'flex-end' }}>
                    <span style={{ fontSize: '0.7rem', color: portColor, marginRight: '0.4rem' }}>
                      {out.type}
                    </span>
                    <div
                      style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', cursor: 'crosshair' }}
                      onMouseDown={(e) => handlePortMouseDown(node.id, out.name, out.type, true, e)}
                      onMouseUp={(e) => handlePortMouseUp(node.id, out.name, out.type, true, e)}
                    >
                      <span style={{ fontWeight: 500 }}>{out.name}</span>
                      <div className="port-handle" style={{ background: portColor }} />
                    </div>
                  </div>
                );
              })}

              {/* Parameter Controls (Model Dropdown or Inputs) */}
              {node.def.parameters.slice(0, 2).map((param, idx) => (
                <div key={idx} style={{ marginTop: '0.3rem', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{param.name}:</span>
                  {param.name.includes('ckpt') || param.options ? (
                    <select
                      className="search-input"
                      style={{ padding: '0.3rem 0.5rem', fontSize: '0.75rem' }}
                      value={node.params[param.name] || availableModels[0]?.name || ''}
                      onChange={(e) => {
                        const val = e.target.value;
                        setNodesOnCanvas((prev) =>
                          prev.map((n) => (n.id === node.id ? { ...n, params: { ...n.params, [param.name]: val } } : n))
                        );
                      }}
                    >
                      {availableModels.map((m) => (
                        <option key={m.name} value={m.name}>
                          {m.display_name || m.name}
                        </option>
                      ))}
                    </select>
                  ) : (
                    <input
                      type="text"
                      className="search-input"
                      style={{ padding: '0.3rem 0.5rem', fontSize: '0.75rem' }}
                      value={node.params[param.name] ?? ''}
                      onChange={(e) => {
                        const val = e.target.value;
                        setNodesOnCanvas((prev) =>
                          prev.map((n) => (n.id === node.id ? { ...n, params: { ...n.params, [param.name]: val } } : n))
                        );
                      }}
                    />
                  )}
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Inspector Panel Collapsable */}
      {showInspector ? (
        <div className="inspector-panel">
          <div className="palette-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
              <Sliders size={16} /> Node Inspector
            </div>
            <button className="btn btn-secondary" style={{ padding: '0.2rem 0.4rem' }} onClick={() => setShowInspector(false)}>
              <PanelRightClose size={16} />
            </button>
          </div>

          {selectedNode ? (
            <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
              <div>
                <div style={{ fontWeight: 700, fontSize: '1rem', color: '#f8fafc' }}>
                  {selectedNode.def.display_name}
                </div>
                <div style={{ fontSize: '0.8rem', color: '#94a3b8', marginTop: '0.2rem' }}>
                  {selectedNode.def.description}
                </div>
              </div>

              <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
                <div style={{ fontWeight: 600, fontSize: '0.85rem', color: '#cbd5e1' }}>
                  Parameters ({selectedNode.def.parameters.length})
                </div>
                {selectedNode.def.parameters.length === 0 && (
                  <span style={{ fontSize: '0.8rem', color: '#64748b' }}>No parameters for this node</span>
                )}
                {selectedNode.def.parameters.map((param, idx) => (
                  <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                    <label style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a5b4fc' }}>
                      {param.name} ({param.type})
                    </label>

                    {param.name.includes('ckpt') || param.options ? (
                      <select
                        className="search-input"
                        value={selectedNode.params[param.name] || availableModels[0]?.name || ''}
                        onChange={(e) => {
                          const val = e.target.value;
                          setNodesOnCanvas((prev) =>
                            prev.map((n) =>
                              n.id === selectedNode.id ? { ...n, params: { ...n.params, [param.name]: val } } : n
                            )
                          );
                        }}
                      >
                        {availableModels.map((m) => (
                          <option key={m.name} value={m.name}>
                            {m.display_name || m.name} ({m.size_gb} GB)
                          </option>
                        ))}
                      </select>
                    ) : param.type === 'STRING' && param.name === 'text' ? (
                      <textarea
                        rows={4}
                        className="search-input"
                        value={selectedNode.params[param.name] || ''}
                        onChange={(e) => {
                          const val = e.target.value;
                          setNodesOnCanvas((prev) =>
                            prev.map((n) =>
                              n.id === selectedNode.id ? { ...n, params: { ...n.params, [param.name]: val } } : n
                            )
                          );
                        }}
                      />
                    ) : (
                      <input
                        type="text"
                        className="search-input"
                        value={selectedNode.params[param.name] ?? ''}
                        onChange={(e) => {
                          const val = e.target.value;
                          setNodesOnCanvas((prev) =>
                            prev.map((n) =>
                              n.id === selectedNode.id ? { ...n, params: { ...n.params, [param.name]: val } } : n
                            )
                          );
                        }}
                      />
                    )}
                  </div>
                ))}
              </div>
            </div>
          ) : (
            <div style={{ color: '#64748b', fontSize: '0.85rem', textAlign: 'center', marginTop: '2rem' }}>
              Click a node on canvas to inspect parameters
            </div>
          )}
        </div>
      ) : (
        <button
          className="btn btn-secondary"
          style={{ position: 'absolute', top: 16, right: 16, zIndex: 15 }}
          onClick={() => setShowInspector(true)}
        >
          <PanelRightOpen size={16} /> Show Inspector
        </button>
      )}

      {/* Output Image Preview Lightbox Modal */}
      {outputImageUrl && (
        <div
          style={{
            position: 'fixed',
            inset: 0,
            background: 'rgba(0, 0, 0, 0.85)',
            backdropFilter: 'blur(12px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            zIndex: 100,
          }}
          onClick={() => setOutputImageUrl(null)}
        >
          <div
            style={{
              background: '#0f172a',
              border: '1px solid var(--border-glow)',
              borderRadius: '16px',
              maxWidth: '850px',
              width: '90%',
              overflow: 'hidden',
              boxShadow: '0 0 40px rgba(99, 102, 241, 0.4)',
              display: 'flex',
              flexDirection: 'column',
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 1.5rem', borderBottom: '1px solid var(--border-subtle)', background: 'rgba(30, 41, 59, 0.7)' }}>
              <div style={{ fontWeight: 700, fontSize: '1.1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sparkles color="var(--accent-emerald)" size={20} /> Generated Artifact Output (FLUX.1 DEV)
              </div>
              <button className="btn btn-secondary" style={{ padding: '0.3rem 0.5rem' }} onClick={() => setOutputImageUrl(null)}>
                <X size={18} />
              </button>
            </div>

            <div style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center' }}>
              <img
                src={outputImageUrl}
                alt="Generated Output Artifact"
                style={{ maxWidth: '100%', maxHeight: '500px', borderRadius: '12px', boxShadow: '0 8px 30px rgba(0,0,0,0.5)', border: '1px solid var(--border-subtle)' }}
              />

              <div style={{ width: '100%', display: 'flex', justifyContent: 'space-between', alignItems: 'center', background: 'rgba(9,13,22,0.8)', padding: '0.75rem 1rem', borderRadius: '8px', fontSize: '0.85rem' }}>
                <span style={{ color: 'var(--text-muted)' }}>Prompt: A high-tech cybernetic space station surrounded by glowing neon plasma rings in deep space</span>
                <a href={outputImageUrl} download="comfyng_generated_artifact.jpg" className="btn btn-primary" style={{ padding: '0.4rem 0.8rem' }}>
                  <Download size={16} /> Download Image
                </a>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
