import axios from 'axios';
import type {
  TaskResponse,
  WorkflowStatusResponse,
  QuestionsResponse,
  RequirementsSummaryResponse,
  ConfigResponse,
  SettingsResponse,
  FileUploadResponse,
  LLMModelsUpdateRequest,
  OpenRouterModelsResponse,
  SessionDetail,
  SessionDeleteReport,
  SessionMessage,
  SessionSummary,
  SessionTask,
} from '../types/api';

const api = axios.create({
  baseURL: '/api/v1',
  timeout: 30000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Workflows API
export const workflowsApi = {
  startPaperToCode: async (
    inputSource: string,
    inputType: 'file' | 'url',
    enableIndexing: boolean = false,
    enableUserInteraction: boolean = true,
    sessionId?: string | null
  ): Promise<TaskResponse> => {
    const response = await api.post<TaskResponse>('/workflows/paper-to-code', {
      input_source: inputSource,
      input_type: inputType,
      enable_indexing: enableIndexing,
      enable_user_interaction: enableUserInteraction,
      session_id: sessionId ?? null,
    });
    return response.data;
  },

  startChatPlanning: async (
    requirements: string,
    enableIndexing: boolean = false,
    enableUserInteraction: boolean = true,
    sessionId?: string | null
  ): Promise<TaskResponse> => {
    const response = await api.post<TaskResponse>('/workflows/chat-planning', {
      requirements,
      enable_indexing: enableIndexing,
      enable_user_interaction: enableUserInteraction,
      session_id: sessionId ?? null,
    });
    return response.data;
  },

  getStatus: async (taskId: string): Promise<WorkflowStatusResponse> => {
    const response = await api.get<WorkflowStatusResponse>(
      `/workflows/status/${taskId}`
    );
    return response.data;
  },

  cancel: async (taskId: string): Promise<void> => {
    await api.post(`/workflows/cancel/${taskId}`);
  },

  getActiveTasks: async (): Promise<{ tasks: Array<{
    task_id: string;
    status: string;
    progress: number;
    message: string;
    started_at: string | null;
  }> }> => {
    const response = await api.get('/workflows/active');
    return response.data;
  },

  getRecentTasks: async (limit: number = 10): Promise<{ tasks: Array<{
    task_id: string;
    status: string;
    progress: number;
    message: string;
    result: Record<string, unknown> | null;
    error: string | null;
    started_at: string | null;
    completed_at: string | null;
  }> }> => {
    const response = await api.get(`/workflows/recent?limit=${limit}`);
    return response.data;
  },

  // User-in-Loop interaction APIs
  respondToInteraction: async (
    taskId: string,
    action: string,
    data: Record<string, unknown> = {},
    skipped: boolean = false
  ): Promise<{ status: string; task_id: string; action: string }> => {
    const response = await api.post(`/workflows/respond/${taskId}`, {
      action,
      data,
      skipped,
    });
    return response.data;
  },

  getInteraction: async (taskId: string): Promise<{
    has_interaction: boolean;
    task_id: string;
    status: string;
    interaction?: {
      type: string;
      title: string;
      description: string;
      data: Record<string, unknown>;
      options: Record<string, string>;
      required: boolean;
    };
  }> => {
    const response = await api.get(`/workflows/interaction/${taskId}`);
    return response.data;
  },
};

// Sessions API
export const sessionsApi = {
  list: async (
    limit: number = 50,
    order: 'recent' | 'created' = 'recent'
  ): Promise<{ sessions: SessionSummary[] }> => {
    const response = await api.get('/sessions', { params: { limit, order } });
    return response.data;
  },

  create: async (title: string = ''): Promise<SessionDetail> => {
    const response = await api.post<SessionDetail>('/sessions', { title });
    return response.data;
  },

  get: async (sessionId: string): Promise<SessionDetail> => {
    const response = await api.get<SessionDetail>(`/sessions/${sessionId}`);
    return response.data;
  },

  delete: async (sessionId: string): Promise<SessionDeleteReport> => {
    const response = await api.delete<SessionDeleteReport>(
      `/sessions/${sessionId}`
    );
    return response.data;
  },

  appendMessage: async (
    sessionId: string,
    role: string,
    content: string
  ): Promise<SessionMessage> => {
    const response = await api.post<SessionMessage>(
      `/sessions/${sessionId}/messages`,
      { role, content }
    );
    return response.data;
  },

  branch: async (
    sessionId: string,
    fromMessageIndex: number,
    title?: string
  ): Promise<SessionDetail> => {
    const response = await api.post<SessionDetail>(
      `/sessions/${sessionId}/branch`,
      {
        from_message_index: fromMessageIndex,
        title,
      }
    );
    return response.data;
  },

  getTasks: async (sessionId: string): Promise<{ tasks: SessionTask[] }> => {
    const response = await api.get(`/sessions/${sessionId}/tasks`);
    return response.data;
  },
};

// Requirements API
export const requirementsApi = {
  generateQuestions: async (
    initialRequirement: string
  ): Promise<QuestionsResponse> => {
    const response = await api.post<QuestionsResponse>('/requirements/questions', {
      initial_requirement: initialRequirement,
    });
    return response.data;
  },

  summarize: async (
    initialRequirement: string,
    userAnswers: Record<string, string>
  ): Promise<RequirementsSummaryResponse> => {
    const response = await api.post<RequirementsSummaryResponse>(
      '/requirements/summarize',
      {
        initial_requirement: initialRequirement,
        user_answers: userAnswers,
      }
    );
    return response.data;
  },

  modify: async (
    currentRequirements: string,
    modificationFeedback: string
  ): Promise<RequirementsSummaryResponse> => {
    const response = await api.put<RequirementsSummaryResponse>(
      '/requirements/modify',
      {
        current_requirements: currentRequirements,
        modification_feedback: modificationFeedback,
      }
    );
    return response.data;
  },
};

// Config API
export const configApi = {
  getSettings: async (): Promise<SettingsResponse> => {
    const response = await api.get<SettingsResponse>('/config/settings');
    return response.data;
  },

  getLLMProviders: async (): Promise<ConfigResponse> => {
    const response = await api.get<ConfigResponse>('/config/llm-providers');
    return response.data;
  },

  setLLMProvider: async (provider: string): Promise<void> => {
    await api.put('/config/llm-provider', { provider });
  },

  getOpenRouterModels: async (
    supportedParameters?: string
  ): Promise<OpenRouterModelsResponse> => {
    const response = await api.get<OpenRouterModelsResponse>(
      '/config/openrouter/models',
      { params: { supported_parameters: supportedParameters } }
    );
    return response.data;
  },

  setLLMModels: async (request: LLMModelsUpdateRequest): Promise<void> => {
    await api.put('/config/llm-models', request);
  },
};

// Research Copilot API (Scout + Librarian)
export interface PaperHit {
  title: string;
  url: string;
  snippet: string;
  published: string;
  arxiv_id: string;
}

export interface DiscoverResponse {
  interest: string;
  query: string;
  papers: PaperHit[];
}

export interface CompileResponse {
  title: string;
  topic: string;
  rel_path: string;
  path: string;
  markdown: string;
  wiki_root: string;
  cached: boolean;
  cached_at: string | null;
}

export interface PrebakedFile {
  path: string;
  size: number;
}

export interface PrebakedResponse {
  task_id: string;
  repo_name: string;
  file_count: number;
  files: PrebakedFile[];
  readme: string;
  summary: string;
  plan: string;
}

export interface PrebakedFileResponse {
  path: string;
  content: string;
  truncated: boolean;
}

export type GraphNodeType = 'paper' | 'method' | 'dataset' | 'metric' | 'concept' | 'task';

export interface GraphNode {
  id: string;
  label: string;
  type: GraphNodeType;
  detail: string;
  url: string;
}

export interface GraphEdge {
  source: string;
  target: string;
  relation: string;
}

export interface GraphSource {
  title: string;
  url: string;
  snippet: string;
}

export interface KnowledgeGraph {
  topic: string;
  seed: string;
  effort: string;
  elapsed: number;
  cached: boolean;
  generated_at: string;
  summary: string;
  nodes: GraphNode[];
  edges: GraphEdge[];
  sources: GraphSource[];
  spend: { total: number; search: number; research: number };
}

export interface Spend {
  total: number;
  search: number;
  research: number;
}

export const copilotApi = {
  discover: async (
    interest: string,
    count: number = 8,
    freshness: string | null = 'year'
  ): Promise<DiscoverResponse> => {
    const response = await api.post<DiscoverResponse>('/copilot/discover', {
      interest,
      count,
      freshness,
      arxiv_only: true,
    });
    return response.data;
  },

  compileWiki: async (
    paperUrl: string,
    topic: string = 'llm-agents',
    interest?: string | null,
    refresh: boolean = false
  ): Promise<CompileResponse> => {
    const response = await api.post<CompileResponse>('/copilot/wiki/compile', {
      paper_url: paperUrl,
      topic,
      interest: interest ?? null,
      refresh,
    }, { timeout: 120000 });
    return response.data;
  },

  getPrebaked: async (): Promise<PrebakedResponse> => {
    const response = await api.get<PrebakedResponse>('/copilot/prebaked');
    return response.data;
  },

  getPrebakedFile: async (path: string): Promise<PrebakedFileResponse> => {
    const response = await api.get<PrebakedFileResponse>('/copilot/prebaked/file', {
      params: { path },
    });
    return response.data;
  },

  getCachedGraph: async (topic?: string): Promise<KnowledgeGraph> => {
    const response = await api.get<KnowledgeGraph>('/copilot/graph', {
      params: topic ? { topic } : undefined,
    });
    return response.data;
  },

  buildGraph: async (
    topic: string,
    seed?: string | null,
    effort: string = 'standard'
  ): Promise<KnowledgeGraph> => {
    const response = await api.post<KnowledgeGraph>('/copilot/graph', {
      topic,
      seed: seed ?? null,
      effort,
    }, { timeout: 300000 });
    return response.data;
  },

  getSpend: async (): Promise<Spend> => {
    const response = await api.get<Spend>('/copilot/spend');
    return response.data;
  },
};

// Files API
export const filesApi = {
  upload: async (file: File): Promise<FileUploadResponse> => {
    const formData = new FormData();
    formData.append('file', file);

    const response = await api.post<FileUploadResponse>('/files/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
    return response.data;
  },

  delete: async (fileId: string): Promise<void> => {
    await api.delete(`/files/delete/${fileId}`);
  },

  getInfo: async (fileId: string): Promise<FileUploadResponse> => {
    const response = await api.get<FileUploadResponse>(`/files/info/${fileId}`);
    return response.data;
  },
};

export default api;
