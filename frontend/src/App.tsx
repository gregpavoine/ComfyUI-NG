import React, { useState, useEffect } from 'react';
import { EditorSurface } from './features/editor/EditorSurface';
import { WorkflowsSurface } from './features/workflows/WorkflowsSurface';
import { JobsSurface } from './features/jobs/JobsSurface';
import { ModelsSurface } from './features/models/ModelsSurface';
import { PluginsSurface } from './features/plugins/PluginsSurface';
import { SystemSurface } from './features/system/SystemSurface';
import { fetchSystemInfo, SystemInfo } from './api/client';
import { Zap, Layers, Play, Box, Puzzle, Activity, FileCode } from 'lucide-react';

type Tab = 'editor' | 'workflows' | 'jobs' | 'models' | 'plugins' | 'system';

export const App: React.FC = () => {
  const [activeTab, setActiveTab] = useState<Tab>('editor');
  const [systemInfo, setSystemInfo] = useState<SystemInfo | null>(null);

  useEffect(() => {
    fetchSystemInfo().then(setSystemInfo);
  }, []);

  return (
    <div className="app-shell">
      {/* Top Bar Navigation */}
      <header className="top-bar">
        <div className="brand">
          <div className="brand-icon">⚡</div>
          <span className="brand-title">ComfyUI-NG</span>
          <span className="brand-tag">v0.1.0</span>
        </div>

        <nav className="nav-tabs">
          <button
            className={`nav-tab ${activeTab === 'editor' ? 'active' : ''}`}
            onClick={() => setActiveTab('editor')}
          >
            <Zap size={16} /> Nodal Editor
          </button>
          <button
            className={`nav-tab ${activeTab === 'workflows' ? 'active' : ''}`}
            onClick={() => setActiveTab('workflows')}
          >
            <FileCode size={16} /> Workflows
          </button>
          <button
            className={`nav-tab ${activeTab === 'jobs' ? 'active' : ''}`}
            onClick={() => setActiveTab('jobs')}
          >
            <Play size={16} /> Jobs Queue
          </button>
          <button
            className={`nav-tab ${activeTab === 'models' ? 'active' : ''}`}
            onClick={() => setActiveTab('models')}
          >
            <Box size={16} /> Models
          </button>
          <button
            className={`nav-tab ${activeTab === 'plugins' ? 'active' : ''}`}
            onClick={() => setActiveTab('plugins')}
          >
            <Puzzle size={16} /> Plugins
          </button>
          <button
            className={`nav-tab ${activeTab === 'system' ? 'active' : ''}`}
            onClick={() => setActiveTab('system')}
          >
            <Activity size={16} /> System
          </button>
        </nav>

        <div className="status-pills">
          <div className="status-pill">
            <div className="status-dot" />
            <span>Online ({systemInfo?.python ? `Python ${systemInfo.python}` : 'Py3.14'})</span>
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="content-area">
        {activeTab === 'editor' && <EditorSurface />}
        {activeTab === 'workflows' && <WorkflowsSurface />}
        {activeTab === 'jobs' && <JobsSurface />}
        {activeTab === 'models' && <ModelsSurface />}
        {activeTab === 'plugins' && <PluginsSurface />}
        {activeTab === 'system' && <SystemSurface />}
      </main>
    </div>
  );
};

export default App;
