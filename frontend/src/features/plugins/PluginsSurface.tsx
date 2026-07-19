import React, { useState, useEffect } from 'react';
import { fetchPlugins, PluginItem } from '../../api/client';
import { Puzzle, Shield, CheckCircle2, Lock } from 'lucide-react';

export const PluginsSurface: React.FC = () => {
  const [plugins, setPlugins] = useState<PluginItem[]>([]);

  useEffect(() => {
    fetchPlugins().then(setPlugins);
  }, []);

  return (
    <div className="surface-container">
      <div className="surface-header">
        <div>
          <h2 className="surface-title">Isolated JIT Plugins & Sandboxes</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Process-level sandboxing with granular audit hook permissions
          </p>
        </div>
      </div>

      <div className="card-grid">
        {plugins.map((p) => (
          <div key={p.id} className="data-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Puzzle size={20} color="var(--accent-amber)" />
                <div>
                  <div style={{ fontWeight: 700, fontSize: '1.05rem' }}>{p.name}</div>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-dim)' }}>v{p.version}</div>
                </div>
              </div>
              <span className="badge badge-emerald">
                <CheckCircle2 size={12} /> {p.status}
              </span>
            </div>

            <div
              style={{
                display: 'flex',
                flexDirection: 'column',
                gap: '0.4rem',
                marginTop: '0.5rem',
                padding: '0.75rem',
                background: 'rgba(9, 13, 22, 0.6)',
                borderRadius: '8px',
                fontSize: '0.8rem',
              }}
            >
              <div style={{ fontWeight: 600, color: '#a5b4fc', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <Shield size={14} /> Sandbox Policy
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Filesystem Access:</span>
                <span style={{ fontWeight: 600, color: '#34d399' }}>{p.permissions.filesystem}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Network Access:</span>
                <span style={{ color: p.permissions.network ? '#34d399' : '#f43f5e' }}>
                  {p.permissions.network ? 'Allowed' : 'Denied'}
                </span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span>Subprocess Creation:</span>
                <span style={{ color: p.permissions.subprocess ? '#34d399' : '#f43f5e' }}>
                  {p.permissions.subprocess ? 'Allowed' : 'Denied'}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
