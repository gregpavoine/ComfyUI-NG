import React, { useState, useEffect } from 'react';
import { fetchJobs, Job } from '../../api/client';
import { CheckCircle2, Clock, AlertCircle, RefreshCw, Eye, Download, X, Sparkles } from 'lucide-react';

export const JobsSurface: React.FC = () => {
  const [jobs, setJobs] = useState<Job[]>([]);
  const [selectedJob, setSelectedJob] = useState<Job | null>(null);

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
            Resource-aware scheduler execution progress and generated artifacts
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
              <th>Artefacts & Output</th>
            </tr>
          </thead>
          <tbody>
            {jobs.map((job) => (
              <tr key={job.id} style={{ cursor: 'pointer' }} onClick={() => setSelectedJob(job)}>
                <td style={{ fontFamily: 'var(--font-mono)', fontWeight: 600 }}>{job.id}</td>
                <td style={{ fontWeight: 600 }}>{job.name}</td>
                <td>
                  {job.status === 'completed' && <span className="badge badge-emerald"><CheckCircle2 size={12} /> Completed</span>}
                  {job.status === 'preparing' && <span className="badge badge-indigo"><RefreshCw size={12} className="spin" /> Preparing</span>}
                  {job.status === 'running' && <span className="badge badge-indigo"><RefreshCw size={12} className="spin" /> Running</span>}
                  {job.status === 'queued' && <span className="badge badge-amber"><Clock size={12} /> Queued</span>}
                  {job.status === 'failed' && <span className="badge badge-rose"><AlertCircle size={12} /> Failed</span>}
                  {job.status === 'cancelled' && <span className="badge badge-rose" style={{ opacity: 0.7 }}><X size={12} /> Cancelled</span>}
                </td>
                <td><span className="badge badge-indigo">{job.priority}</span></td>
                <td style={{ color: 'var(--text-muted)' }}>{job.created_at}</td>
                <td style={{ fontFamily: 'var(--font-mono)' }}>{job.duration_ms} ms</td>
                <td>
                  {job.artefacts.length > 0 ? (
                    <button
                      className="btn btn-secondary"
                      style={{ padding: '0.25rem 0.6rem', fontSize: '0.75rem', color: '#34d399', borderColor: 'rgba(16,185,129,0.3)' }}
                      onClick={(e) => {
                        e.stopPropagation();
                        setSelectedJob(job);
                      }}
                    >
                      <Eye size={14} /> View Image ({job.artefacts[0]})
                    </button>
                  ) : (
                    <span style={{ color: 'var(--text-dim)' }}>—</span>
                  )}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Lightbox Modal for Selected Job Image */}
      {selectedJob && (
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
          onClick={() => setSelectedJob(null)}
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
                <Sparkles color="var(--accent-emerald)" size={20} /> Job Output Preview — {selectedJob.id}
              </div>
              <button className="btn btn-secondary" style={{ padding: '0.3rem 0.5rem' }} onClick={() => setSelectedJob(null)}>
                <X size={18} />
              </button>
            </div>

            <div style={{ padding: '1.5rem', display: 'flex', flexDirection: 'column', gap: '1rem', alignItems: 'center' }}>
              {selectedJob.image_url && (
                <img
                  src={selectedJob.image_url}
                  alt={selectedJob.name}
                  style={{ maxWidth: '100%', maxHeight: '500px', borderRadius: '12px', boxShadow: '0 8px 30px rgba(0,0,0,0.5)', border: '1px solid var(--border-subtle)' }}
                />
              )}

              <div style={{ width: '100%', display: 'flex', flexDirection: 'column', gap: '0.5rem', background: 'rgba(9,13,22,0.8)', padding: '0.8rem 1rem', borderRadius: '8px', fontSize: '0.85rem' }}>
                <div style={{ fontWeight: 600, color: '#a5b4fc' }}>Task: {selectedJob.name}</div>
                <div style={{ color: 'var(--text-muted)' }}>Prompt: {selectedJob.prompt}</div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '0.5rem', alignItems: 'center' }}>
                  <span className="badge badge-emerald">Completed in {selectedJob.duration_ms} ms</span>
                  {selectedJob.image_url && (
                    <a href={selectedJob.image_url} download="comfyng_job_artifact.png" className="btn btn-primary" style={{ padding: '0.4rem 0.8rem' }}>
                      <Download size={16} /> Download PNG Artifact
                    </a>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
