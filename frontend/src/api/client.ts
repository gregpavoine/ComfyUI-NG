export interface NodeDefinition {
  name: string;
  display_name: string;
  category: string;
  description: string;
  inputs: Array<{ name: string; type: string; required?: boolean }>;
  outputs: Array<{ name: string; type: string }>;
  parameters: Array<{ name: string; type: string; default?: any; options?: string[]; description?: string }>;
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
  status: 'queued' | 'preparing' | 'running' | 'completed' | 'failed' | 'cancelled';
  priority: number;
  created_at: string;
  duration_ms: number;
  artefacts: string[];
  image_url?: string;
  prompt?: string;
  error?: string;
}

export interface ModelItem {
  name: string;
  display_name?: string;
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
  const res = await fetch('/api/v1/system/info');
  if (!res.ok) throw new Error('Failed');
  return await res.json();
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

export async function submitJob(
  name: string,
  prompt: string,
  modelName?: string,
  seed?: number,
  steps?: number,
  width?: number,
  height?: number,
  nodes?: any[],
  connections?: any[]
): Promise<Job | null> {
  try {
    const res = await fetch('/api/v1/jobs/submit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name,
        prompt,
        model_name: modelName,
        seed: seed !== undefined ? Number(seed) : undefined,
        steps: steps !== undefined ? Number(steps) : undefined,
        width: width !== undefined ? Number(width) : undefined,
        height: height !== undefined ? Number(height) : undefined,
        nodes,
        connections,
      }),
    });
    const data = await res.json();
    return data.job || null;
  } catch {
    return null;
  }
}

export async function fetchJob(jobId: string): Promise<Job | null> {
  try {
    const res = await fetch(`/api/v1/jobs/${jobId}`);
    if (!res.ok) return null;
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
