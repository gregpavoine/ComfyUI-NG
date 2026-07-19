import React, { useState, useEffect } from 'react';
import { fetchModels, ModelItem } from '../../api/client';
import { Box, HardDrive, ShieldCheck, Tag } from 'lucide-react';

export const ModelsSurface: React.FC = () => {
  const [models, setModels] = useState<ModelItem[]>([]);

  useEffect(() => {
    fetchModels().then(setModels);
  }, []);

  return (
    <div className="surface-container">
      <div className="surface-header">
        <div>
          <h2 className="surface-title">Modern Model Assets Registry</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Content-Addressed Storage (CAS) safetensors models with architecture detection
          </p>
        </div>
      </div>

      <div className="card-grid">
        {models.map((m, idx) => (
          <div key={idx} className="data-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Box size={20} color="var(--accent-cyan)" />
                <span style={{ fontWeight: 700, fontSize: '1.05rem' }}>{m.name}</span>
              </div>
              <span className="badge badge-emerald">
                <ShieldCheck size={12} /> {m.status}
              </span>
            </div>

            <div style={{ display: 'flex', gap: '0.5rem', marginTop: '0.25rem' }}>
              <span className="badge badge-indigo">
                <Tag size={12} /> {m.architecture}
              </span>
              <span className="badge badge-amber">{m.format}</span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-muted)', fontSize: '0.85rem' }}>
              <HardDrive size={16} /> Size: {m.size_gb} GB
            </div>

            <div
              style={{
                fontSize: '0.75rem',
                fontFamily: 'var(--font-mono)',
                color: 'var(--text-dim)',
                background: 'rgba(9, 13, 22, 0.6)',
                padding: '0.5rem',
                borderRadius: '6px',
                wordBreak: 'break-all',
              }}
            >
              Digest: {m.digest}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
