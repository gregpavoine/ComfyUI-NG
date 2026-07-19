import React, { useState, useEffect } from 'react';
import { fetchSystemInfo, SystemInfo } from '../../api/client';
import { Activity, Cpu, Server, Database, ExternalLink } from 'lucide-react';

export const SystemSurface: React.FC = () => {
  const [info, setInfo] = useState<SystemInfo | null>(null);

  useEffect(() => {
    fetchSystemInfo().then(setInfo);
  }, []);

  return (
    <div className="surface-container">
      <div className="surface-header">
        <div>
          <h2 className="surface-title">System Health & Diagnostics</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Control plane process, Python 3.14 runtime & worker supervisor
          </p>
        </div>
        <a href="/docs" target="_blank" rel="noreferrer" className="btn btn-secondary">
          <ExternalLink size={16} /> OpenAPI Swagger Docs
        </a>
      </div>

      {info && (
        <div className="card-grid">
          <div className="data-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Server size={20} color="var(--accent-primary)" />
              <span style={{ fontWeight: 700, fontSize: '1.1rem' }}>ComfyUI-NG Control Plane</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Status:</span>
                <span className="badge badge-emerald">{info.status}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Version:</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>v{info.version}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Python:</span>
                <span style={{ fontFamily: 'var(--font-mono)' }}>v{info.python}</span>
              </div>
            </div>
          </div>

          <div className="data-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Cpu size={20} color="var(--accent-cyan)" />
              <span style={{ fontWeight: 700, fontSize: '1.1rem' }}>Multiprocessing & Workers</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Start Method:</span>
                <span className="badge badge-indigo">{info.multiprocessing}</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                <span style={{ color: 'var(--text-muted)' }}>Active Workers:</span>
                <span style={{ fontWeight: 600 }}>{info.active_workers} isolated workers</span>
              </div>
            </div>
          </div>

          <div className="data-card">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Database size={20} color="var(--accent-emerald)" />
              <span style={{ fontWeight: 700, fontSize: '1.1rem' }}>Data Root & Storage</span>
            </div>
            <div style={{ marginTop: '0.5rem', fontSize: '0.8rem', fontFamily: 'var(--font-mono)', wordBreak: 'break-all', background: 'rgba(9, 13, 22, 0.6)', padding: '0.6rem', borderRadius: '6px' }}>
              {info.data_root}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
