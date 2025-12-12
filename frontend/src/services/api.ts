import { Cause, Recommendation, SubgraphData } from '../types/knowledgeGraph';
import pipelineApis from './pipelineApi';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// Generic fetch function with error handling
const apiFetch = async (endpoint: string, options: RequestInit = {}) => {
  try {
    const response = await fetch(`${API_BASE_URL}${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });
    
    if (!response.ok) {
      throw new Error(`API request failed with status ${response.status}`);
    }
    
    return await response.json();
  } catch (error) {
    console.error(`Error fetching from ${endpoint}:`, error);
    throw error;
  }
};

// Knowledge Graph API functions
export const knowledgeGraphApi = {
  // Get causes of a defect
  getCausesOfDefect: async (defect: string, minConfidence: number = 0.5): Promise<Cause[]> => {
    const data = await apiFetch(`/api/knowledge-graph/causes/${defect}?min_confidence=${minConfidence}`);
    return data.causes || [];
  },

  // Get intervention recommendations
  getRecommendations: async (defect: string, parameterValues: Record<string, number>): Promise<Recommendation[]> => {
    const data = await apiFetch(`/api/knowledge-graph/recommendations/${defect}`, {
      method: 'POST',
      body: JSON.stringify({ parameter_values: parameterValues }),
    });
    return data.recommendations || [];
  },

  // Get subgraph for visualization
  getSubgraph: async (defect: string, maxDepth: number = 2): Promise<SubgraphData> => {
    const data = await apiFetch(`/api/knowledge-graph/subgraph/${defect}?max_depth=${maxDepth}`);
    return data;
  },
};

// Other API functions that might be needed
export const systemApi = {
  // Get system health status
  getHealthStatus: async () => {
    return await apiFetch('/health');
  },

  // Get model predictions
  getModelPredictions: async () => {
    return await apiFetch('/models/predictions');
  },

  // Get digital twin state
  getDigitalTwinState: async () => {  return await apiFetch('/api/digital-twin/state');
  },
};

// Export pipeline APIs for Phases 5-8
export const { 
  explainability: explainabilityApi,
  metrics: metricsApi,
  features: featuresApi,
  autonomy: autonomyApi,
  pipeline: pipelineApi,
  training: trainingApi
} = pipelineApis;