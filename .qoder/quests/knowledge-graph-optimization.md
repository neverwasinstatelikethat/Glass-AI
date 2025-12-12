# Knowledge Graph Optimization Design Document

## 1. Executive Summary

This document outlines the design for optimizing and fixing the Knowledge Graph functionality in the glass production predictive analytics system. The current implementation has several issues including non-functional buttons, hardcoded metrics, non-working visualization, and various Neo4j query errors. The goal is to make all functionality ML-driven, fix the errors, and ensure proper integration between frontend and backend components.

## 2. Problem Statement

The current Knowledge Graph implementation suffers from several critical issues:

1. **Non-functional UI elements**: Buttons in the frontend don't work properly
2. **Hardcoded metrics**: Statistics and metrics are hardcoded instead of being dynamically calculated
3. **Broken visualization**: Graph visualization doesn't show actual nodes and relationships
4. **Neo4j query errors**: Multiple syntax errors in Cypher queries causing runtime exceptions
5. **Incomplete integration**: Lack of proper data flow between ML models and Knowledge Graph

## 3. System Architecture Overview

The Knowledge Graph system consists of the following components:

### 3.1 Backend Components
- **EnhancedGlassProductionKnowledgeGraph**: Main Knowledge Graph class with Neo4j backend
- **KnowledgeBaseInitializer**: Initializes domain knowledge in the graph
- **RootCauseAnalyzer**: Analyzes root causes of defects
- **FastAPI Backend**: Exposes REST endpoints for frontend interaction

### 3.2 Frontend Components
- **KnowledgeGraph.tsx**: Main React component for displaying Knowledge Graph
- **KnowledgeGraph Types**: TypeScript interfaces for data structures
- **WebSocket Hooks**: Real-time data streaming hooks

### 3.3 Integration Points
- **UnifiedGlassProductionSystem**: Central system integrator
- **ML Models**: LSTM, GNN, and Vision Transformer models
- **Digital Twin**: Physics simulation engine
- **RL Agent**: Reinforcement learning optimizer

## 4. Key Issues Identified

### 4.1 Neo4j Query Errors
Several Cypher queries in `causal_graph.py` contain syntax errors:
- Invalid use of `WHERE` clause after `MERGE` statement
- Deprecated use of `id()` function
- Incorrect query structure causing runtime exceptions

### 4.2 Missing Data Flow
- Lack of real-time data enrichment from ML models
- No proper connection between sensor data and Knowledge Graph updates
- Missing feedback loop from human decisions to KG

### 4.3 UI/UX Issues
- Visualization shows no actual nodes
- Buttons don't trigger proper backend calls
- Metrics display hardcoded values instead of real data

## 5. Proposed Solution

### 5.1 Fix Neo4j Queries

#### 5.1.1 Correct Relationship Weight Update Query
Replace the erroneous query in `update_relationship_weights` method:
```cypher
MATCH (param:Parameter {name: $param_name})-[:RELATED_TO]->(equip:Equipment)
MATCH (defect:Defect {type: $defect_type})
MERGE (cause:Cause)-[:CAUSES]->(defect)
WHERE (cause)-[:RELATED_TO]->(param) AND (cause)-[:FROM_EQUIPMENT]->(equip)
SET cause.confidence = $new_confidence,
    cause.last_updated = $timestamp
```

With the corrected version:
```cypher
MATCH (param:Parameter {name: $param_name})<-[:RELATED_TO]-(cause:Cause)-[:FROM_EQUIPMENT]->(equip:Equipment)
MATCH (defect:Defect {type: $defect_type})<-[:CAUSES]-(cause)
SET cause.confidence = $new_confidence,
    cause.last_updated = $timestamp
```

#### 5.1.2 Fix Subgraph Export Query
Replace the erroneous query in `export_subgraph` method that causes deprecation warnings:
```cypher
MATCH (d:Defect {type: $defect_type})
RETURN d.type AS type, id(d) AS id
```

With the corrected version that avoids deprecated `id()` function:
```cypher
MATCH (d:Defect {type: $defect_type})
RETURN d.type AS type, toString(id(d)) AS id
```

Also fix the causes query that has similar issues:
```cypher
MATCH (d:Defect {type: $defect_type})<-[:CAUSES]-(cause)
OPTIONAL MATCH (cause)-[:RELATED_TO]->(param:Parameter)
OPTIONAL MATCH (cause)-[:FROM_EQUIPMENT]->(equip:Equipment)
RETURN cause.confidence AS confidence,
       id(cause) AS cause_id,
       param.name AS param_name,
       id(param) AS param_id,
       equip.equipment_id AS equip_id,
       id(equip) AS equip_id_internal
```

With the corrected version:
```cypher
MATCH (d:Defect {type: $defect_type})<-[:CAUSES]-(cause)
OPTIONAL MATCH (cause)-[:RELATED_TO]->(param:Parameter)
OPTIONAL MATCH (cause)-[:FROM_EQUIPMENT]->(equip:Equipment)
RETURN cause.confidence AS confidence,
       toString(id(cause)) AS cause_id,
       param.name AS param_name,
       toString(id(param)) AS param_id,
       equip.equipment_id AS equip_id,
       toString(id(equip)) AS equip_id_internal
```

#### 5.1.2 Replace Deprecated Functions
Replace all uses of `id()` function with explicit ID properties to avoid deprecation warnings.

### 5.2 Implement Real-time Data Enrichment

#### 5.2.1 ML Prediction Enrichment Pipeline
Create a robust pipeline for enriching the Knowledge Graph with ML predictions:

1. **LSTM Defect Predictions Integration**
   - Implement automatic KG updates when LSTM predicts high-probability defects
   - Update causal relationship weights based on prediction confidence
   - Create defect occurrence nodes with timestamp and probability metadata

2. **GNN Anomaly Detection Integration**
   - Identify unusual sensor patterns and correlate with defect occurrences
   - Update parameter-defect relationships based on anomaly detection results
   - Create anomaly nodes linked to relevant parameters and equipment

3. **Vision Transformer Results Integration**
   - Incorporate visual defect detection findings into the KG
   - Link visual defects to physical parameters and equipment
   - Update defect severity scores based on visual analysis
#### 5.2.2 RL Recommendation Integration
Implement proper integration of RL agent recommendations:

1. **Create Recommendation Nodes**
   - Generate recommendation nodes linked to defect nodes with proper metadata
   - Include confidence scores, expected impact, and implementation details
   - Add timestamps and source tracking for audit purposes

2. **Track Recommendation Effectiveness**
   - Monitor applied vs. dismissed recommendations
   - Calculate success rates for different recommendation types
   - Generate reports on recommendation performance

3. **Update Causal Relationships**
   - Modify relationship weights based on recommendation outcomes
   - Implement feedback loops for continuous learning
   - Adjust confidence scores based on real-world results
#### 5.2.3 Human Decision Feedback Loop
Establish a feedback mechanism for human operator decisions:
1. Record applied/dismissed recommendations
2. Update relationship confidences based on outcomes
3. Provide feedback to RL agent for continuous learning

### 5.3 Enhance Frontend Integration

#### 5.3.1 Dynamic Visualization
Implement proper graph visualization showing:
1. Actual nodes from Neo4j query results
2. Real-time updates through WebSocket
3. Interactive node exploration
4. Visual distinction between node types (defects, parameters, equipment, recommendations)

#### 5.3.2 Functional UI Elements
Ensure all buttons and controls work properly:
1. Defect type selection triggers backend API calls
2. Parameter adjustments update recommendations
3. Visualization refresh works with actual data
4. Metrics display real statistics from the system

#### 5.3.3 Real-time Data Updates
Implement WebSocket-based real-time updates:
1. Stream live sensor data to frontend
2. Update visualization as new data arrives
3. Show live defect predictions and recommendations

## 6. Detailed Implementation Plan

### 6.1 Backend Implementation

#### 6.1.1 Knowledge Graph Class Improvements
Modify `EnhancedGlassProductionKnowledgeGraph` class:

1. **Fix Query Syntax Errors**
   - Correct all Cypher queries with proper structure as outlined in section 5.1
   - Replace deprecated functions and add proper error handling for database operations
   - Implement transaction management for related operations

2. **Implement Caching Strategy**
   - Use Redis for frequently accessed data with appropriate TTL values
   - Implement cache invalidation policies for data updates
   - Add cache warming mechanisms for commonly requested queries

3. **Add Transaction Management**
   - Wrap related operations in database transactions to ensure consistency
   - Implement rollback mechanisms for failed operations
   - Ensure data consistency across related updates with proper error handling

4. **Enhance Error Handling**
   - Add comprehensive exception handling for all database operations
   - Implement retry mechanisms for transient database errors
   - Add detailed logging for debugging and monitoring purposes

1. **Fix Query Syntax Errors**
   - Correct all Cypher queries with proper structure
   - Replace deprecated functions
   - Add proper error handling for database operations

2. **Implement Caching Strategy**
   - Use Redis for frequently accessed data
   - Implement cache invalidation policies
   - Add cache warming for commonly requested queries

3. **Add Transaction Management**
   - Wrap related operations in transactions
   - Implement rollback mechanisms for failed operations
   - Ensure data consistency across related updates

#### 6.1.2 API Endpoint Enhancement
Update FastAPI endpoints in `fastapi_backend.py`:

1. **Fix GET Endpoints**
   - `/api/knowledge-graph/causes/{defect}`: Return actual causes from database with proper error handling
   - `/api/knowledge-graph/subgraph/{defect}`: Return real graph data with converted IDs
   - `/api/knowledge-graph/defect-recommendations/{defect}`: Return actual recommendations with proper data structure

2. **Implement POST Endpoints**
   - `/api/knowledge-graph/recommendations/{defect}`: Generate recommendations based on current parameters with validation
   - `/api/knowledge-graph/enrich/human-decision`: Process human feedback with proper data validation and KG updates

3. **Add Error Handling**
   - Implement proper HTTP status codes for different error conditions
   - Add detailed error messages for debugging
   - Include request validation for all endpoints

1. **Fix GET Endpoints**
   - `/api/knowledge-graph/causes/{defect}`: Return actual causes from database with proper pagination and filtering
   - `/api/knowledge-graph/subgraph/{defect}`: Return real graph data with proper node and edge serialization
   - `/api/knowledge-graph/defect-recommendations/{defect}`: Return actual recommendations with confidence scoring

2. **Implement POST Endpoints**
   - `/api/knowledge-graph/recommendations/{defect}`: Generate recommendations based on current parameters
   - `/api/knowledge-graph/enrich/human-decision`: Process human feedback

3. **Add New Endpoints**
   - `/api/knowledge-graph/statistics`: Return real metrics and statistics
   - `/api/knowledge-graph/search`: Enable searching for specific nodes/relationships

### 6.2 Frontend Implementation

#### 6.2.1 Component Refactoring
Refactor `KnowledgeGraph.tsx` component:

1. **Fix Data Fetching Hooks**
   - Update `fetchCauses` to properly handle API response structure
   - Fix `fetchRecommendations` to send correct parameter values
   - Correct `fetchSubgraph` to process actual graph data instead of mock data

2. **Implement Dynamic Graph Visualization**
   - Replace static visualization with D3.js or similar library
   - Map Neo4j nodes and edges to visualization elements
   - Add interactive features like zoom, pan, and node selection

3. **Fix Parameter Value Handling**
   - Correct `handleParameterChange` to properly update state
   - Validate parameter inputs before sending to backend
   - Implement real-time parameter updates through WebSocket

#### 6.2.2 Type Definitions
Update TypeScript interfaces in `knowledgeGraph.ts`:

1. **Enhance Data Structures**
   - Add proper typing for all Knowledge Graph entities
   - Include relationship properties and metadata
   - Define union types for node/edge classifications

2. **Add Validation**
   - Implement runtime type checking
   - Add data validation for API responses
   - Include proper error type definitions

### 6.3 Integration Layer

#### 6.3.1 Unified System Integration
Enhance `UnifiedGlassProductionSystem`:

1. **Connect ML Models to KG**
   - Implement automatic KG updates from model predictions with confidence-weighted relationships
   - Add anomaly detection integration with real-time alerting
   - Connect visual defect detection results with spatial mapping to physical parameters

2. **Implement Feedback Loops**
   - Create bidirectional communication between KG and ML models
   - Implement continuous learning from human decisions
   - Add performance metrics tracking

#### 6.3.2 Real-time Data Streaming
Implement WebSocket broadcasting:

1. **Sensor Data Streaming**
   - Broadcast live sensor readings to connected clients
   - Implement subscription filtering by equipment/parameter
   - Add data aggregation for performance optimization

2. **KG Update Notifications**
   - Notify clients of KG changes in real-time
   - Implement selective update broadcasting
   - Add update batching for efficiency

## 7. Data Model Design

### 7.1 Node Types

#### 7.1.1 Defect Nodes
```cypher
(:Defect {
  type: string,           // crack, bubble, chip, etc.
  severity: string,       // LOW, MEDIUM, HIGH, CRITICAL
  description: string,
  created_at: datetime,
  updated_at: datetime
})
```

#### 7.1.2 Parameter Nodes
```cypher
(:Parameter {
  name: string,           // furnace_temperature, belt_speed, etc.
  category: string,       // thermal, mechanical, process, etc.
  unit: string,           // °C, m/min, MPa, etc.
  min_value: float,
  max_value: float,
  created_at: datetime
})
```

#### 7.1.3 Equipment Nodes
```cypher
(:Equipment {
  equipment_id: string,   // furnace_A, forming_B, etc.
  type: string,           // Furnace, Forming, Annealing, etc.
  zone: string,           // melting, shaping, cooling, etc.
  created_at: datetime
})
```

#### 7.1.4 Cause Nodes
```cypher
(:Cause {
  parameter: string,
  equipment: string,
  defect: string,
  confidence: float,       // 0.0 - 1.0
  description: string,
  mechanism: string,
  created_at: datetime,
  updated_at: datetime
})
```

#### 7.1.5 Recommendation Nodes
```cypher
(:Recommendation {
  recommendation_id: string,
  action: string,
  parameters: map,         // Target parameter values
  confidence: float,
  urgency: string,        // LOW, MEDIUM, HIGH
  expected_impact: string,
  source: string,         // RL_Agent, KnowledgeGraph, etc.
  applied: boolean,
  applied_timestamp: datetime,
  created_at: datetime
})
```

### 7.2 Relationship Types

#### 7.2.1 CAUSES
```cypher
(:Cause)-[:CAUSES]->(:Defect)
```
Relationship representing that a cause leads to a defect with a certain confidence level.

#### 7.2.2 RELATED_TO
```cypher
(:Cause)-[:RELATED_TO]->(:Parameter)
```
Relationship linking a cause to the parameter it involves.

#### 7.2.3 FROM_EQUIPMENT
```cypher
(:Cause)-[:FROM_EQUIPMENT]->(:Equipment)
```
Relationship linking a cause to the equipment involved.

#### 7.2.4 ADDRESSES
```cypher
(:Recommendation)-[:ADDRESSES]->(:Defect)
```
Relationship showing that a recommendation addresses a specific defect.

#### 7.2.5 REGARDING
```cypher
(:HumanDecision)-[:REGARDING]->(:Defect)
```
Relationship connecting human decisions to the defects they address.

## 8. Performance Optimization

### 8.1 Database Indexing
Create appropriate indexes for improved query performance:
```cypher
CREATE INDEX defect_type FOR (d:Defect) ON (d.type)
CREATE INDEX parameter_name FOR (p:Parameter) ON (p.name)
CREATE INDEX equipment_id FOR (e:Equipment) ON (e.equipment_id)
CREATE INDEX cause_confidence FOR (c:Cause) ON (c.confidence)
CREATE INDEX recommendation_applied FOR (r:Recommendation) ON (r.applied)
```

### 8.2 Caching Strategy
Implement multi-level caching:
1. **Redis Caching**: Cache frequently accessed queries for 30 minutes
2. **Application-Level Caching**: Cache computed results in memory
3. **Browser Caching**: Use HTTP caching headers for static data

### 8.3 Query Optimization
Optimize database queries:
1. **Batch Operations**: Combine multiple operations in single transactions
2. **Pagination**: Implement pagination for large result sets
3. **Selective Loading**: Load only required properties for specific views

## 9. Security Considerations

### 9.1 Authentication and Authorization
Implement proper access controls:
1. **API Authentication**: Use JWT tokens for API endpoint protection
2. **Role-Based Access**: Different permissions for operators, engineers, and administrators
3. **Audit Logging**: Log all KG modification operations

### 9.2 Data Protection
Ensure data security:
1. **Encryption**: Encrypt sensitive data at rest and in transit
2. **Input Validation**: Validate all user inputs to prevent injection attacks
3. **Rate Limiting**: Implement rate limiting to prevent abuse

## 10. Testing Strategy

### 10.1 Unit Testing
Implement comprehensive unit tests:

1. **Knowledge Graph Methods**
   - Test all KG class methods with mock databases to verify query correctness
   - Validate error handling for database connection failures
   - Test caching mechanisms with Redis mock

2. **API Endpoints**
   - Test all REST endpoints with various input scenarios including edge cases
   - Validate proper HTTP status codes and error messages
   - Test request validation and parameter sanitization

3. **Frontend Components**
   - Test React components with different data states (loading, success, error)
   - Validate form inputs and user interactions
   - Test responsive design across different screen sizes
### 10.2 Integration Testing
Test component interactions:

1. **Database Integration**
   - Test Neo4j connectivity and query execution with real database
   - Validate data integrity during insert/update operations
   - Test performance with large datasets

2. **ML Model Integration**
   - Test data flow from models to KG with realistic prediction data
   - Validate automatic KG updates based on model outputs
   - Test confidence score propagation through the system

3. **Frontend-Backend Communication**
   - Test API interactions and data serialization between frontend and backend
   - Validate real-time updates through WebSocket connections
   - Test error scenarios and recovery mechanisms

### 10.3 Performance Testing
Validate system performance:

1. **Load Testing**
   - Test with concurrent users and high data volumes
   - Validate system behavior under peak load conditions
   - Measure response times and throughput metrics

2. **Query Performance**
   - Measure and optimize database query times
   - Test with different graph sizes and complexities
   - Validate index usage and query execution plans

3. **Caching Effectiveness**
   - Verify cache hit rates and performance improvements
   - Test cache invalidation and warming mechanisms
   - Validate memory usage and resource consumption

## 11. Monitoring and Observability

### 11.1 Metrics Collection
Implement comprehensive metrics:
1. **Database Performance**: Query execution times, connection pool usage
2. **API Performance**: Response times, error rates, throughput
3. **Cache Efficiency**: Hit rates, eviction rates, memory usage

### 11.2 Logging
Implement structured logging:
1. **Operation Tracking**: Log all KG modification operations
2. **Error Reporting**: Detailed error logs with context information
3. **Performance Logging**: Log slow queries and operations

### 11.3 Alerting
Set up proactive monitoring:
1. **Database Alerts**: Connection failures, slow queries
2. **API Alerts**: High error rates, performance degradation
3. **System Health**: Overall system status and component availability

## 12. Deployment Considerations

### 12.1 Environment Configuration
Support multiple deployment environments:
1. **Development**: Local development with mock services
2. **Staging**: Pre-production testing environment
3. **Production**: Live production environment with full services

### 12.2 Scaling Strategy
Plan for horizontal scaling:
1. **Database Sharding**: Distribute graph data across multiple instances
2. **API Load Balancing**: Distribute requests across multiple backend instances
3. **Frontend CDN**: Serve static assets through content delivery networks

### 12.3 Backup and Recovery
Implement data protection:
1. **Regular Backups**: Automated backup schedules for Neo4j and Redis
2. **Disaster Recovery**: Procedures for restoring from backups
3. **Data Archiving**: Archive old data to reduce database size

## 13. Success Criteria

### 13.1 Functional Requirements
1. All UI buttons and controls function correctly
2. Graph visualization displays actual nodes and relationships
3. Metrics show real-time data instead of hardcoded values
4. All Neo4j query errors are resolved
5. ML-driven functionality works end-to-end

### 13.2 Performance Requirements
1. API response times under 500ms for 95% of requests
2. Graph visualization loads within 2 seconds
3. Real-time updates propagate within 1 second
4. System handles 100 concurrent users without performance degradation

### 13.3 Reliability Requirements
1. System uptime of 99.9%
2. Zero data loss during normal operations
3. Graceful degradation during partial system failures
4. Automatic recovery from transient errors

## 14. Risks and Mitigation Strategies

### 14.1 Technical Risks
1. **Database Performance**: Mitigate with proper indexing and query optimization
2. **Integration Complexity**: Address with comprehensive testing and clear interfaces
3. **Real-time Data Volume**: Handle with streaming architecture and data filtering

### 14.2 Operational Risks
1. **Deployment Failures**: Prevent with blue-green deployments and rollback procedures
2. **Data Corruption**: Protect with regular backups and validation checks
3. **Security Breaches**: Guard with proper authentication and monitoring

## 15. Implementation Timeline

### Phase 1: Critical Fixes (Week 1)
1. Fix Neo4j query syntax errors in `causal_graph.py`
2. Resolve deprecated function warnings
3. Implement proper error handling in Knowledge Graph class
4. Fix API endpoints to return actual data instead of mock responses

### Phase 2: Frontend Integration (Week 2)
1. Refactor `KnowledgeGraph.tsx` component to use real data
2. Implement dynamic graph visualization
3. Fix all UI controls and buttons
4. Add proper loading states and error handling

### Phase 3: ML Integration (Week 3)
1. Implement automatic KG updates from ML model predictions
2. Connect RL agent recommendations to Knowledge Graph
3. Establish human decision feedback loop
4. Add real-time data streaming through WebSocket

### Phase 4: Performance Optimization (Week 4)
1. Implement caching strategy with Redis
2. Optimize database queries and add proper indexing
3. Add monitoring and observability features
4. Conduct comprehensive testing and validation

## 16. Conclusion

This design document provides a comprehensive plan for optimizing and fixing the Knowledge Graph functionality in the glass production predictive analytics system. By addressing the identified issues and implementing the proposed solutions, we will create a robust, ML-driven Knowledge Graph that provides valuable insights for production optimization while maintaining high performance and reliability standards.
