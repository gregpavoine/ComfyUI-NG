import React, { useState, useEffect } from 'react';
import { Layers, FileCode, Clock, GitBranch, ArrowUpRight } from 'lucide-react';

interface WorkflowItem {
  id: string;
  name: string;
  description: string;
  nodes_count: number;
  updated_at: string;
}

export const WorkflowsSurface: React.FC = () => {
  const [workflows, setWorkflows] = useState<WorkflowItem[]>([]);

  useEffect(() => {
    fetch('/api/v1/workflows')
      .then((r) => r.json())
      .then((d) => setWorkflows(d.workflows || []))
      .catch(() => {});
  }, []);

  return (
    <div className="surface-container">
      <div className="surface-header">
        <div>
          <h2 className="surface-title">Saved Workflows & Templates</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Version-controlled typed generation DAGs
          </p>
        </div>
        <button className="btn btn-primary">
          <Layers size={16} /> New Blank Workflow
        </button>
      </div>

      <div className="card-grid">
        {workflows.map((wf) => (
          <div key={wf.id} className="data-card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <FileCode size={20} color="var(--accent-primary)" />
                <span style={{ fontWeight: 700, fontSize: '1.05rem' }}>{wf.name}</span>
              </div>
              <span className="badge badge-indigo">
                <GitBranch size={12} /> v1.0
              </span>
            </div>

            <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>{wf.description}</p>

            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                marginTop: '0.5rem',
                paddingTop: '0.75rem',
                borderTop: '1px solid var(--border-subtle)',
                fontSize: '0.8rem',
                color: 'var(--text-dim)',
              }}
            >
              <span>{wf.nodes_count} Nodes</span>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <Clock size={14} /> {wf.updated_at}
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};
