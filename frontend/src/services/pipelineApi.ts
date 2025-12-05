/**
 * Pipeline API Service - Handles all new backend endpoints for Phases 5-8
 * Explainability, Metrics, Autonomy, Features, and Continual Learning
 */

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

// ==================== PHASE 6: EXPLAINABILITY API ====================

export interface FeatureAttribution {
  feature_name: string;
  importance: number;
  value: number;
}

export interface ExplanationData {
  timestamp: string;
  model_name: string;
  explanations: {
    shap_values?: FeatureAttribution[];
    lime_explanation?: any;
    top_features?: FeatureAttribution[];
  };
}

export const explainabilityApi = {
  // Get explanation for latest prediction
  getPredictionExplanation: async (modelName: string = 'lstm'): Promise<ExplanationData> => {
    return await apiFetch(`/api/explainability/prediction?model_name=${modelName}`);
  },
};

// ==================== PHASE 7: METRICS API ====================

export interface PipelineMetrics {
  pipeline_executions: number;
  successful_predictions: number;
  failed_predictions: number;
  avg_latency_ms: number;
  feature_extraction_time_ms: number;
  prediction_time_ms: number;
  explanation_time_ms: number;
}

export interface MetricsResponse {
  timestamp: string;
  pipeline_metrics: PipelineMetrics;
  system_metrics: any;
}

export const metricsApi = {
  // Get pipeline performance metrics
  getPipelineMetrics: async (): Promise<MetricsResponse> => {
    return await apiFetch('/api/pipeline/metrics');
  },
};

// ==================== FEATURES API ====================

export interface EngineeredFeatures {
  timestamp: string;
  features: {
    // Domain features
    furnace_temperature?: number;
    melt_level?: number;
    belt_speed?: number;
    mold_temp?: number;
    pressure?: number;
    humidity?: number;
    viscosity?: number;
    conveyor_speed?: number;
    annealing_temp?: number;
    quality_score?: number;
    
    // Statistical features
    temperature_mean?: number;
    temperature_std?: number;
    temperature_trend?: number;
    
    // Real-time features
    [key: string]: any;
  };
}

export const featuresApi = {
  // Get latest engineered features
  getLatestFeatures: async (): Promise<EngineeredFeatures> => {
    return await apiFetch('/api/features/latest');
  },
};

// ==================== PHASE 5: AUTONOMY API ====================

export interface AutonomousAction {
  action: string;
  confidence: number;
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH';
  expected_impact: string;
}

export interface AutonomyStatus {
  timestamp: string;
  autonomy_enabled: boolean;
  autonomous_actions_count: number;
  approval_required_count: number;
  safety_checks_enabled: boolean;
  actions_to_execute?: AutonomousAction[];
  actions_requiring_approval?: AutonomousAction[];
}

export const autonomyApi = {
  // Get autonomous action decision status
  getAutonomyStatus: async (): Promise<AutonomyStatus> => {
    return await apiFetch('/api/autonomy/status');
  },
};

// ==================== PIPELINE PROCESSING API ====================

export interface SensorData {
  production_line?: string;
  furnace_temperature?: number;
  belt_speed?: number;
  mold_temp?: number;
  [key: string]: any;
}

export interface PipelineOutput {
  timestamp: string;
  sensor_data: SensorData;
  engineered_features: any;
  predictions: any;
  explanations: any;
  recommendations: any;
  autonomous_actions: any;
  performance_metrics: {
    total_latency_ms: number;
    feature_extraction_ms: number;
    prediction_ms: number;
    explanation_ms: number;
  };
}

export const pipelineApi = {
  // Process sensor data through complete pipeline
  processSensorData: async (sensorData: SensorData): Promise<PipelineOutput> => {
    return await apiFetch('/api/pipeline/process', {
      method: 'POST',
      body: JSON.stringify(sensorData),
    });
  },
};

// ==================== PHASE 8: CONTINUAL LEARNING API ====================

export interface TrainingStatus {
  timestamp: string;
  continual_learning_enabled: boolean;
  experience_buffer_size: number;
  current_task?: number;
  training_progress?: number;
}

export const trainingApi = {
  // Get continual learning training status
  getTrainingStatus: async (): Promise<TrainingStatus> => {
    return await apiFetch('/api/training/status');
  },
};

// ==================== COMBINED API EXPORTS ====================

export const pipelineApis = {
  explainability: explainabilityApi,
  metrics: metricsApi,
  features: featuresApi,
  autonomy: autonomyApi,
  pipeline: pipelineApi,
  training: trainingApi,
};

export default pipelineApis;
