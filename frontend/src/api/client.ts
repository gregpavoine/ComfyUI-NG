export interface NodeDefinition {
  name: string;
  display_name: string;
  category: string;
  description: string;
  inputs: Array<{ name: string; type: string; required?: boolean }>;
  outputs: Array<{ name: string; type: string }>;
  parameters: Array<{ name: string; type: string; default?: any; description?: string }>;
}

export interface SystemInfo {
  status: string;
  version: string;
  python: string;
  data_root: string;
  multiprocessing: string;
  active_workers: number;
}

export interface Job {
  id: string;
  name: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  priority: number;
  created_at: string;
  duration_ms: number;
  artefacts: string[];
}

export interface ModelItem {
  name: string;
  architecture: string;
  size_gb: number;
  format: string;
  digest: string;
  status: string;
}

export interface PluginItem {
  id: string;
  name: string;
  version: string;
  status: string;
  permissions: {
    filesystem: string;
    network: boolean;
    subprocess: boolean;
  };
}

export async function fetchSystemInfo(): Promise<SystemInfo> {
  try {
    const res = await fetch('/api/v1/system/info');
    if (!res.ok) throw new Error('Failed');
    return await res.json();
  } catch {
    return {
      status: 'ok',
      version: '0.1.0',
      python: '3.14.4',
      data_root: '/home/gp/.local/share/comfyui-ng',
      multiprocessing: 'forkserver',
      active_workers: 2,
    };
  }
}

export async function fetchNodeCatalogue(): Promise<NodeDefinition[]> {
  try {
    const res = await fetch('/api/v1/nodes/catalogue');
    const data = await res.json();
    return data.nodes || [];
  } catch {
    return [];
  }
}

export async function fetchJobs(): Promise<Job[]> {
  try {
    const res = await fetch('/api/v1/jobs');
    const data = await res.json();
    return data.jobs || [];
  } catch {
    return [];
  }
}

export async function submitJob(name: string, prompt: string): Promise<Job | null> {
  try {
    const res = await fetch('/api/v1/jobs/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, prompt }),
    });
    const data = await res.json();
    return data.job || null;
  } catch {
    return null;
  }
}

export async function fetchModels(): Promise<ModelItem[]> {
  try {
    const res = await fetch('/api/v1/models');
    const data = await res.json();
    return data.models || [];
  } catch {
    return [];
  }
}

export async function fetchPlugins(): Promise<PluginItem[]> {
  try {
    const res = await fetch('/api/v1/plugins');
    const data = await res.json();
    return data.plugins || [];
  } catch {
    return [];
  }
}
