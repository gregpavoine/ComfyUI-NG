import React, { useState, useEffect } from 'react';
import { fetchJobs, Job } from '../../api/client';
import { Play, CheckCircle2, Clock, AlertCircle, RefreshCw } from 'lucide-react';

export const JobsSurface: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([]);

  const loadJobs = () => {
    fetchJobs().then(setJobs);
  };

  useEffect(() => {
    loadJobs();
    const interval = setInterval(loadJobs, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="surface-container">
      <div className="surface-header">
        <div>
          <h2 className="surface-title">Asynchronous Job Queue</h2>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
            Resource-aware scheduler execution progress
          </p>
        </div>
        <button className="btn btn-secondary" onClick={loadJobs}>
          <RefreshCw size={16} /> Refresh Queue
        </button>
      </div>

      <div className="table-wrapper">
        <table>
          <thead>
            <tr>
              <th>Job ID</th>
              <th>Task Name</th>
              <th>Status</th>
              <th>Priority</th>
              <th>Created At</th>
              <th>Duration</th>
              <th>Artefacts</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id}>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{job.id}</td>
                <td style={{ fontWeight: 600 }}>{job.name}</td>
                <td>
                  {job.status === 'completed' && <span className="badge badge-emerald"><CheckCircle2 size={12} /> Completed</span>}
                  {job.status === 'running' && <span className="badge badge-indigo"><RefreshCw size={12} className="spin" /> Running</span>}
                  {job.status === 'queued' && <span className="badge badge-amber"><Clock size={12} /> Queued</span>}
                  {job.status === 'failed' && <span className="badge badge-rose"><AlertCircle size={12} /> Failed</span>}
                </td>
                <td><span className="badge badge-indigo">{job.priority}</span></td>
                <td style={{ color: 'var(--text-muted)' }}>{job.created_at}</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{job.duration_ms} ms</td>
                <td>
                  {job.artefacts.length > 0 ? (
                    <span style={{ color: 'var(--accent-cyan)', fontWeight: 500 }}>
                      {job.artefacts.join(', ')}
                    </span>
                  ) : (
                    <span style={{ color: 'var(--text-dim)' }}>—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
