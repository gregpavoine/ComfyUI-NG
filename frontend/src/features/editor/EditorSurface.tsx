import React, { useState, useEffect } from 'react';
import { NodeDefinition, fetchNodeCatalogue, submitJob } from '../../api/client';
import { Play, Plus, Search, CheckCircle2, Layers, Cpu, Code2, Sliders } from 'lucide-react';

interface CanvasNode {
  id: string;
  def: NodeDefinition;
  x: number;
  y: number;
  params: Record<string, any>;
}

export const EditorSurface: React.FC = () => {
  const [nodesDef, setNodesDef] = useState<NodeDefinition[]>([]);
  const [search, setSearch] = useState('');
  const [nodesOnCanvas, setNodesOnCanvas] = useState<CanvasNode[]>([]);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [toastMessage, setToastMessage] = useState<string | null>(null);

  useEffect(() => {
    fetchNodeCatalogue().then((defs) => {
      setNodesDef(defs);
      // Pre-populate standard sample workflow on initial render
      if (defs.length > 0) {
        const ckpt = defs.find((d) => d.name === 'LoadCheckpoint') || defs[0];
        const clip = defs.find((d) => d.name === 'CLIPTextEncode') || defs[1] || defs[0];
        const sampler = defs.find((d) => d.name === 'KSampler') || defs[3] || defs[0];
        const saveImg = defs.find((d) => d.name === 'SaveImage') || defs[5] || defs[0];

        setNodesOnCanvas([
          { id: 'node-1', def: ckpt, x: 40, y: 50, params: { ckpt_name: 'flux1-dev.safetensors' } },
          { id: 'node-2', def: clip, x: 320, y: 50, params: { text: 'A futuristic cybernetic station in deep space, cinematic 8k' } },
          { id: 'node-3', def: sampler, x: 600, y: 80, params: { steps: 25, cfg: 3.5, seed: 1337 } },
          { id: 'node-4', def: saveImg, x: 890, y: 120, params: { filename_prefix: 'comfyng_flux_sample' } },
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
      x: 350 + Math.random() * 50,
      y: 100 + Math.random() * 50,
      params: defaultParams,
    };
    setNodesOnCanvas((prev) => [...prev, newNode]);
    setSelectedNodeId(newId);
  };

  const handleExecute = async () => {
    setIsSubmitting(true);
    const clipNode = nodesOnCanvas.find((n) => n.def.name === 'CLIPTextEncode');
    const promptText = clipNode?.params?.text || 'Standard generation request';

    const res = await submitJob('FLUX.1 Interactive Generation', promptText);
    setIsSubmitting(false);

    if (res) {
      setToastMessage(`Job queued successfully! Job ID: ${res.id}`);
      setTimeout(() => setToastMessage(null), 4000);
    }
  };

  const filteredDefs = nodesDef.filter(
    (d) =>
      d.display_name.toLowerCase().includes(search.toLowerCase()) ||
      d.category.toLowerCase().includes(search.toLowerCase())
  );

  const selectedNode = nodesOnCanvas.find((n) => n.id === selectedNodeId);

  return (
    <div className="editor-layout">
      {/* Node Palette */}
      <div className="node-palette">
        <div className="palette-header">
          <span>Node Catalogue</span>
          <span className="brand-tag">{nodesDef.length} official</span>
        </div>

        <div style={{ position: 'relative' }}>
          <input
            type="text"
            className="search-input"
            placeholder="Search nodes or categories..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </div>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.8rem' }}>
          {filteredDefs.map((def, idx) => (
            <div key={idx} className="palette-item" onClick={() => addNodeToCanvas(def)}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                <span className="palette-item-name">{def.display_name}</span>
                <Plus size={14} style={{ opacity: 0.6 }} />
              </div>
              <span className="palette-item-desc">{def.description}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Canvas Viewport */}
      <div className="canvas-viewport">
        {/* Top Control Overlay */}
        <div
          style={{
            position: 'absolute',
            top: 16,
            left: 16,
            display: 'flex',
            gap: '0.75rem',
            zIndex: 10,
          }}
        >
          <button className="btn btn-primary" onClick={handleExecute} disabled={isSubmitting}>
            <Play size={16} />
            {isSubmitting ? 'Submitting Job...' : 'Run Workflow'}
          </button>
          <button className="btn btn-secondary">
            <Layers size={16} /> Nodes: {nodesOnCanvas.length}
          </button>
        </div>

        {toastMessage && (
          <div
            style={{
              position: 'absolute',
              top: 16,
              right: 16,
              background: 'rgba(16, 185, 129, 0.9)',
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

        {/* Connection Wires SVG */}
        <svg className="canvas-svg">
          {nodesOnCanvas.slice(0, -1).map((node, i) => {
            const nextNode = nodesOnCanvas[i + 1];
            const startX = node.x + 240;
            const startY = node.y + 60;
            const endX = nextNode.x;
            const endY = nextNode.y + 60;
            const ctrlX = (startX + endX) / 2;
            return (
              <path
                key={i}
                d={`M ${startX} ${startY} C ${ctrlX} ${startY}, ${ctrlX} ${endY}, ${endX} ${endY}`}
                stroke="#6366f1"
                strokeWidth="3"
                fill="none"
                strokeDasharray="6 2"
              />
            );
          })}
        </svg>

        {/* Nodes on Canvas */}
        {nodesOnCanvas.map((node) => (
          <div
            key={node.id}
            className={`canvas-node ${selectedNodeId === node.id ? 'selected' : ''}`}
            style={{ left: node.x, top: node.y }}
            onClick={() => setSelectedNodeId(node.id)}
          >
            <div className="canvas-node-header">
              <span>{node.def.display_name}</span>
              <span className="badge badge-indigo">{node.def.category}</span>
            </div>
            <div className="canvas-node-body">
              {node.def.inputs.map((inp, idx) => (
                <div key={idx} className="port-row">
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <div className="port-handle" />
                    <span>{inp.name}</span>
                  </div>
                  <span style={{ fontSize: '0.7rem', color: '#818cf8' }}>{inp.type}</span>
                </div>
              ))}
              {node.def.outputs.map((out, idx) => (
                <div key={idx} className="port-row" style={{ justifyContent: 'flex-end' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span style={{ fontSize: '0.7rem', color: '#34d399' }}>{out.type}</span>
                    <span>{out.name}</span>
                    <div className="port-handle" style={{ background: '#10b981' }} />
                  </div>
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
                Parameters
              </div>
              {selectedNode.def.parameters.length === 0 && (
                <span style={{ fontSize: '0.8rem', color: '#64748b' }}>No parameters</span>
              )}
              {selectedNode.def.parameters.map((param, idx) => (
                <div key={idx} style={{ display: 'flex', flexDirection: 'column', gap: '0.3rem' }}>
                  <label style={{ fontSize: '0.75rem', fontWeight: 600, color: '#a5b4fc' }}>
                    {param.name}
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
                            n.id === selectedNode.id
                              ? { ...n, params: { ...n.params, [param.name]: val } }
                              : n
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
                            n.id === selectedNode.id
                              ? { ...n, params: { ...n.params, [param.name]: val } }
                              : n
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
            Select a node on canvas to inspect properties
          </div>
        )}
      </div>
    </div>
  );
};
