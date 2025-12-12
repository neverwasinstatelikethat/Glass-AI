export interface Evidence {
  type: string;
  parameter: string;
  current_value: number;
  normal_mean: number;
  deviation_sigma: number;
  timestamp: string;
}

export interface Cause {
  cause: string;
  cause_type: string;
  relationship_type: string;
  strength: number;
  confidence: number;
  evidence: Evidence[];
  observations: number;
  last_updated: string;
}

export interface Recommendation {
  parameter: string;
  current_value: number;
  target_value: number;
  unit: string;
  action: string;
  confidence: number;
  strength: number;
  expected_impact: string;
  priority: string;
  ml_enhanced?: boolean;
  ml_confidence?: number;
}

export interface SubgraphNode {
  id: number | string;
  label: string;
  name: string;
  nodeType: 'defect' | 'cause' | 'parameter' | 'recommendation' | 'human_decision' | 'equipment';
  confidence: number;
  properties: Record<string, any>;
}

export interface SubgraphEdge {
  source: number;
  target: number;
  type: string;
  strength: number;
  confidence: number;
}

export interface SubgraphData {
  nodes: SubgraphNode[];
  edges: SubgraphEdge[];
  defect: string;
  timestamp: string;
}