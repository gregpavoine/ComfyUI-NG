import React, { useState, useEffect, useRef } from 'react';
import { NodeDefinition, fetchNodeCatalogue, submitJob } from '../../api/client';
import { Play, Plus, Trash2, CheckCircle2, Layers, Sliders, X, Sparkles } from 'lucide-react';

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
  const [search, setSearch] = useState('');
  const [nodesOnCanvas, setNodesOnCanvas] = useState<CanvasNode[]>([]);
  const [connections, setConnections] = useState<Connection[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);

  // Dragging state for nodes
  const [draggingNodeId, setDraggingNodeId] = useState<string | null>(null);
  const [dragOffset, setDragOffset] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Wiring state
  const [wiringFrom, setWiringFrom] = useState<{ nodeId: string; portName: string; type: string; isOutput: boolean } | null>(null);
  const [mousePos, setMousePos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  const [isSubmitting, setIsSubmitting] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  const viewportRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetchNodeCatalogue().then((defs) => {
      setNodesDef(defs);
      if (defs.length > 0) {
        const ckpt = defs.find((d) => d.name === 'LoadCheckpoint') || defs[0];
        const clip = defs.find((d) => d.name === 'CLIPTextEncode') || defs[1] || defs[0];
        const sampler = defs.find((d) => d.name === 'KSampler') || defs[3] || defs[0];
        const saveImg = defs.find((d) => d.name === 'SaveImage') || defs[5] || defs[0];

        const initialNodes: CanvasNode[] = [
          { id: 'node-1', def: ckpt, x: 50, y: 80, params: { ckpt_name: 'flux1-dev.safetensors' } },
          { id: 'node-2', def: clip, x: 350, y: 80, params: { text: 'A futuristic cybernetic neon station in deep space, 8k resolution' } },
          { id: 'node-3', def: sampler, x: 680, y: 120, params: { steps: 25, cfg: 3.5, seed: 4242 } },
          { id: 'node-4', def: saveImg, x: 990, y: 160, params: { filename_prefix: 'comfyng_output' } },
        ];
        setNodesOnCanvas(initialNodes);

        // Pre-wire sample connections
        setConnections([
          { id: 'c1', fromNodeId: 'node-1', fromPort: 'MODEL', toNodeId: 'node-3', toPort: 'MODEL', type: 'MODEL' },
          { id: 'c2', fromNodeId: 'node-1', fromPort: 'CLIP', toNodeId: 'node-2', toPort: 'CLIP', type: 'CLIP' },
          { id: 'c3', fromNodeId: 'node-2', fromPort: 'CONDITIONING', toNodeId: 'node-3', toPort: 'POSITIVE', type: 'CONDITIONING' },
          { id: 'c4', fromNodeId: 'node-3', fromPort: 'LATENT', toNodeId: 'node-4', toPort: 'IMAGE', type: 'IMAGE' },
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
    if (wiringFrom.nodeId === targetNodeId) return; // Cannot connect to self
    if (wiringFrom.isOutput === isTargetOutput) return; // Must connect output to input

    const fromNodeId = wiringFrom.isOutput ? wiringFrom.nodeId : targetNodeId;
    const fromPort = wiringFrom.isOutput ? wiringFrom.portName : targetPortName;
    const toNodeId = wiringFrom.isOutput ? targetNodeId : wiringFrom.nodeId;
    const toPort = wiringFrom.isOutput ? targetPortName : wiringFrom.portName;
    const connType = wiringFrom.type;

    // Replace or add connection
    const newConn: Connection = {
      id: `conn-${Date.now()}`,
      fromNodeId,
      fromPort,
      toNodeId,
      toPort,
      type: connType,
    };

    setConnections((prev) => [
      ...prev.filter((c) => !(c.toNodeId === toNodeId && c.toPort === toPort)),
      newConn,
    ]);
    setWiringFrom(null);
  };

  const handleExecute = async () => {
    setIsSubmitting(true);
    const clipNode = nodesOnCanvas.find((n) => n.def.name === 'CLIPTextEncode');
    const promptText = clipNode?.params?.text || 'Interactive FLUX.1 generation request';

    const res = await submitJob('FLUX.1 Interactive Generation', promptText);
    setIsSubmitting(false);

    if (res) {
      setToastMessage(`Job queued successfully! Job ID: ${res.id}`);
      setTimeout(() => setToastMessage(null), 4000);
    }
  };

  // Helper to calculate port position
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
      {/* Node Palette */}
      <div className="node-palette">
        <div className="palette-header">
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Sparkles size={16} color="var(--accent-primary)" /> Node Palette
          </span>
          <span className="brand-tag">{nodesDef.length} nodes</span>
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

      {/* Canvas Viewport */}
      <div className="canvas-viewport" ref={viewportRef}>
        {/* Top Control Overlay */}
        <div style={{ position: 'absolute', top: 16, left: 16, display: 'flex', gap: '0.75rem', zIndex: 10 }}>
          <button className="btn btn-primary" onClick={handleExecute} disabled={isSubmitting}>
            <Play size={16} />
            {isSubmitting ? 'Submitting...' : 'Run Workflow'}
          </button>
          <button className="btn btn-secondary">
            <Layers size={16} /> Nodes: {nodesOnCanvas.length} | Wires: {connections.length}
          </button>
        </div>

        {toastMessage && (
          <div
            style={{
              position: 'absolute',
              top: 16,
              right: 16,
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
            }}
          >
            <CheckCircle2 size={18} /> {toastMessage}
          </div>
        )}

        {/* SVG Connections Canvas */}
        <svg className="canvas-svg">
          {/* Active Wires */}
          {connections.map((conn) => {
            const start = getPortPos(conn.fromNodeId, conn.fromPort, true);
            const end = getPortPos(conn.toNodeId, conn.toPort, false);
            const ctrlOffset = Math.max(40, Math.abs(end.x - start.x) * 0.4);
            const pathData = `M ${start.x} ${start.y} C ${start.x + ctrlOffset} ${start.y}, ${end.x - ctrlOffset} ${end.y}, ${end.x} ${end.y}`;
            const wireColor = TYPE_COLORS[conn.type] || '#6366f1';

            return (
              <g key={conn.id} onClick={() => removeConnection(conn.id)} style={{ cursor: 'pointer' }}>
                <path d={pathData} stroke="rgba(0,0,0,0.5)" strokeWidth="6" fill="none" />
                <path
                  d={pathData}
                  stroke={wireColor}
                  strokeWidth="3"
                  fill="none"
                  className="wire-path"
                />
              </g>
            );
          })}

          {/* Wire Currently Being Dragged */}
          {wiringFrom && (
            (() => {
              const start = getPortPos(wiringFrom.nodeId, wiringFrom.portName, wiringFrom.isOutput);
              const end = mousePos;
              const ctrlOffset = 60;
              const pathData = wiringFrom.isOutput
                ? `M ${start.x} ${start.y} C ${start.x + ctrlOffset} ${start.y}, ${end.x - ctrlOffset} ${end.y}, ${end.x} ${end.y}`
                : `M ${start.x} ${start.y} C ${start.x - ctrlOffset} ${start.y}, ${end.x + ctrlOffset} ${end.y}, ${end.x} ${end.y}`;
              const wireColor = TYPE_COLORS[wiringFrom.type] || '#6366f1';

              return (
                <path
                  d={pathData}
                  stroke={wireColor}
                  strokeWidth="3"
                  strokeDasharray="4 2"
                  fill="none"
                />
              );
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

              {/* Inline Parameter Inputs */}
              {node.def.parameters.slice(0, 2).map((param, idx) => (
                <div key={idx} style={{ marginTop: '0.3rem', display: 'flex', flexDirection: 'column', gap: '0.2rem' }}>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-dim)' }}>{param.name}:</span>
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
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Inspector Panel */}
      <div className="inspector-panel">
        <div className="palette-header">
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
            <Sliders size={16} /> Node Inspector
          </div>
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
                  {param.type === 'STRING' && param.name === 'text' ? (
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
            Click a node on canvas to inspect and edit parameters
          </div>
        )}
      </div>
    </div>
  );
};
