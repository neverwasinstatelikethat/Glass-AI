export interface Cause {
  cause: string;
  cause_type: string;
  relationship_type: string;
  strength: number;
  confidence: number;
  evidence: string[];
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
}

export interface SubgraphNode {
  id: number;
  label: string;
  name: string;
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