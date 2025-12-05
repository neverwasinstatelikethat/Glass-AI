# System Audit and Refactoring Design

## Objective

Conduct a comprehensive audit of the Glass Production Predictive Analytics System and refactor it to align with task requirements. The current system has critical architectural flaws and incomplete implementations that prevent it from functioning as a production-ready predictive analytics platform.

## Docker Deployment Infrastructure

### Container Architecture

The system is deployed using Docker Compose with the following services running on a custom bridge network `glass_network`:

#### Data Infrastructure Services

**Kafka** (`glass_kafka`)
- Service Name: `kafka`
- Internal Host: `kafka:9092`
- External Port: `9093` (localhost:9093)
- Purpose: Data streaming pipeline
- Dependencies: Zookeeper

**Zookeeper** (`glass_zookeeper`)
- Service Name: `zookeeper`
- Internal Host: `zookeeper:2181`
- External Port: `2181`
- Purpose: Kafka coordination

**InfluxDB** (`glass_influxdb`)
- Service Name: `influxdb`
- Internal Host: `http://influxdb:8086`
- External Port: `8086`
- Credentials: admin/adminpass123
- Organization: `glass_factory`
- Bucket: `sensors`
- Token: `my-super-secret-auth-token`
- Purpose: Time-series sensor data storage

**PostgreSQL** (`glass_postgres`)
- Service Name: `postgres`
- Internal Host: `postgres:5432`
- External Port: `5432`
- Database: `glass_production`
- Credentials: glass_admin/glass_secure_pass
- Purpose: Metadata and quality metrics storage

**Redis** (`glass_redis`)
- Service Name: `redis`
- Internal Host: `redis:6379`
- External Port: `6379`
- Purpose: Caching and Knowledge Graph temporary storage

**Neo4j** (`glass_neo4j`)
- Service Name: `neo4j`
- Internal Bolt: `bolt://neo4j:7687`
- Internal HTTP: `http://neo4j:7474`
- External Ports: `7474` (HTTP), `7687` (Bolt)
- Credentials: neo4j/neo4jpassword
- Purpose: Causal Knowledge Graph storage

**MinIO** (`glass_minio`)
- Service Name: `minio`
- Internal API: `minio:9000`
- Internal Console: `minio:9001`
- External Ports: `9000` (API), `9001` (Console)
- Credentials: minioadmin/minioadmin123
- Purpose: S3-compatible model storage

#### Application Services

**Backend** (`glass_backend`)
- Service Name: `backend`
- Internal Host: `backend:8000`
- External Port: `8000`
- Entry Point: `uvicorn backend.fastapi_backend:app`
- Environment Variables:
  - `KAFKA_BOOTSTRAP_SERVERS=kafka:9092`
  - `INFLUXDB_URL=http://influxdb:8086`
  - `POSTGRES_URL=postgresql://glass_admin:glass_secure_pass@postgres:5432/glass_production`
  - `REDIS_URL=redis://redis:6379`
  - `NEO4J_URI=bolt://neo4j:7687`
- Purpose: FastAPI REST API + WebSocket server

**Frontend** (`glass_frontend`)
- Service Name: `frontend`
- Internal Host: `frontend:3000`
- External Port: `3000`
- Environment Variables:
  - `REACT_APP_API_URL=http://localhost:8000`
  - `REACT_APP_WS_URL=ws://localhost:8000`
- Purpose: React dashboard UI

#### Industrial Protocol Simulators

**MQTT Broker** (`glass_mosquitto`)
- Service Name: `mosquitto`
- Internal Host: `mosquitto:1883`
- External Ports: `1883` (MQTT), `9003` (WebSocket)
- Purpose: MQTT message broker for sensor data

**OPC UA Server** (`glass_opcua_server`)
- Service Name: `opcua_server`
- Internal Host: `opcua_server:4840`
- External Port: `4840`
- Purpose: Simulated OPC UA industrial server

**Modbus Server** (`glass_modbus_server`)
- Service Name: `modbus_server`
- Internal Host: `modbus_server:5020`
- External Port: `5020`
- Purpose: Simulated Modbus TCP device

### Container Networking Rules

**CRITICAL**: All inter-service communication MUST use Docker service names, not `localhost` or IP addresses:

- ✅ Correct: `http://influxdb:8086`
- ❌ Wrong: `http://localhost:8086`
- ✅ Correct: `kafka:9092`
- ❌ Wrong: `localhost:9092`

**External Access** (from host machine):
- Backend API: `http://localhost:8000`
- Frontend UI: `http://localhost:3000`
- InfluxDB: `http://localhost:8086`
- Neo4j Browser: `http://localhost:7474`
- PostgreSQL: `localhost:5432`

### Container Management and Debugging

**Development Workflow**:

After making code changes, containers must be restarted to reflect updates:

```bash
# Stop all containers
docker-compose down

# Rebuild and start containers (use when Dockerfile or dependencies changed)
docker-compose up --build

# Start containers without rebuild (use for code changes only)
docker-compose up

# Start in detached mode (background)
docker-compose up -d
```

**Monitoring Container Logs**:

To view real-time logs and debug issues:

```bash
# View all container logs
docker-compose logs -f

# View specific service logs
docker-compose logs -f backend
docker-compose logs -f frontend
docker-compose logs -f influxdb

# View last 100 lines
docker-compose logs --tail=100 backend

# View logs for specific container
docker logs glass_backend -f
docker logs glass_frontend -f
```

**Common Debug Commands**:

```bash
# Check container status
docker-compose ps

# Inspect container details
docker inspect glass_backend

# Execute command inside running container
docker exec -it glass_backend /bin/bash
docker exec -it glass_backend python -c "import sys; print(sys.version)"

# View container resource usage
docker stats

# Restart specific service
docker-compose restart backend

# Remove all stopped containers and rebuild
docker-compose down -v  # -v removes volumes (use with caution!)
docker-compose up --build
```

**Hot Reload Configuration**:

Currently configured with hot reload for development:

- **Backend**: Uvicorn auto-reload enabled via `--reload` flag
  - Watches: `backend/`, `data_ingestion/`, `industrial_connectors/`
  - Changes trigger automatic restart (no manual restart needed)
  - View reload logs: `docker logs glass_backend -f`

- **Frontend**: React development server with hot module replacement
  - Changes automatically refresh browser
  - Check for errors: `docker logs glass_frontend -f`

**IMPORTANT**: If changes are not reflected:
1. Check if file is in reload watch list
2. Verify no syntax errors in logs: `docker-compose logs -f backend`
3. Restart container if needed: `docker-compose restart backend`
4. For dependency changes: `docker-compose down && docker-compose up --build`

### Access URLs and Endpoints

**Backend API Base URL**: `http://localhost:8000`

**WebSocket Endpoint**: `ws://localhost:8000/ws`

**Frontend Dashboard**: `http://localhost:3000`

**Database Dashboards**:
- InfluxDB UI: `http://localhost:8086`
- Neo4j Browser: `http://localhost:7474`
- MinIO Console: `http://localhost:9001`

## Critical Issues Identified

### 1. Fundamental Data Flow Architecture Flaw

**Problem**: The system assumes connection to real production sensors that do not exist. Industrial connectors are configured to connect to simulated external servers (`opcua_server:4840`, `modbus_server:5020`, camera device "0"), but these simulators provide no realistic production data.

**Current Behavior**:
- System attempts to connect to `opcua_server:4840` (OPC UA simulator) - connects but receives no meaningful data
- System attempts to connect to `modbus_server:5020` (Modbus simulator) - connects but receives no meaningful data
- System attempts to open camera device "0" - fails in containerized environment (no camera device)
- Data ingestion layer produces minimal or no realistic production data
- All downstream ML models, Digital Twin, and RL agents receive no input or meaningless input
- **Container networking issue**: Some code still uses `localhost` instead of service names, causing connection failures

**Required Behavior**:
- Generate realistic synthetic sensor data based on task.md specifications
- Simulate production parameters with realistic variance and anomalies
- Emit data continuously to enable real-time dashboard updates
- Inject controlled defects and parameter drift for ML model training
- **Use Docker service names** for all inter-container communication

### 2. Digital Twin Module is Non-Functional

**Problem**: Digital Twin exists only as frontend visualization without backend logic.

**Missing Components**:
- Shadow Mode: Run parallel simulations to validate changes before applying to real system
- What-If Analyzer: Test parameter changes and predict outcomes
- Real-time synchronization with sensor data
- Physics-based simulation engine integration
- State prediction and validation mechanisms

**Current State**: `digital_twin/physics_simulation.py` contains physics models but they are not integrated into the data pipeline or exposed via API endpoints.

### 3. ML Models Are Untrained Dummy Implementations

**Problem**: Models generated via `generate_models.py` are architectural skeletons with random weights.

**Impact**:
- LSTM Attention model: Cannot predict defects from time series
- Vision Transformer: Cannot detect defects from images
- GNN Sensor Network: Cannot model sensor correlations
- Ensemble Meta-Learner: Combines meaningless outputs

**Required Actions**:
- Implement training pipeline with synthetic data
- Train models to minimum performance thresholds (Accuracy >85%, Recall >90%, Precision >80%)
- Implement continuous learning pipeline for model updates
- Add model versioning and A/B testing infrastructure

### 4. Frontend Dashboard Shows Static/Mock Data

**Problem**: Dashboard metrics do not update because backend does not emit real-time data.

**Issues**:
- KPI cards display hardcoded values
- Prediction graphs do not refresh
- Recommendations panel shows static text
- Alerts/notifications tab marked "Under Development"
- Knowledge Graph visualization not connected to backend
- AR Visualization tab incomplete

### 5. Knowledge Graph Not Integrated

**Problem**: `CausalKnowledgeGraph` class exists but is not populated with domain knowledge or connected to inference pipeline.

**Missing**:
- Causal relationships between parameters and defects
- Root cause analysis queries
- Intervention recommendation logic
- Graph visualization API endpoints

### 6. RL Agent Not Autonomous

**Problem**: RL optimizer exists but does not autonomously apply recommendations.

**Current**: RL agent can generate actions but there is no:
- Action application mechanism
- Safety validation before parameter changes
- Rollback capability if quality degrades
- Feedback loop from applied actions to reward calculation

### 7. Incomplete Frontend Modules

**Missing Implementations**:
- Notifications/Alerts tab (placeholder only)
- Settings/Configuration panel
- AR Visualization (component exists but not functional)
- Model performance monitoring dashboard
- Historical analysis and trend reports

### 8. No Continuous Learning Pipeline

**Problem**: Models are static after initial generation. No infrastructure for:
- Collecting labeled data from production
- Periodic model retraining
- Model performance monitoring and degradation detection
- Automated model deployment and rollback

## Architectural Refactoring Strategy

### Phase 1: Fix Data Foundation

#### 1.1 Replace Industrial Connectors with Synthetic Data Generators

**Component**: `data_ingestion/synthetic_data_generator.py`

**Container Integration**:
- Runs inside `glass_backend` container
- Publishes data to `kafka:9092` (not `localhost:9092`)
- Stores data in `influxdb:8086` (not `localhost:8086`)
- Uses environment variables from docker-compose for service discovery

**Responsibilities**:
- Generate time series sensor data matching task.md specifications
- Simulate furnace parameters: temperature (1400-1600°C), pressure (0-50 kPa), melt level (0-5000 mm)
- Simulate forming parameters: belt speed (0-200 m/min), mold temperature (20-600°C), forming pressure (0-100 MPa)
- Simulate annealing parameters: temperature profile, cooling rate
- Inject realistic anomalies: temperature drift, pressure spikes, speed variations
- Generate defect events correlated with parameter deviations

**Data Emission Patterns**:
- Furnace sensors: 1 reading/minute
- Forming sensors: 1 reading/second  
- Annealing sensors: 1 reading/2 seconds
- MIK-1 defect detection: 1 inspection per unit (every 3-5 seconds at 150 m/min)

**Output Format**:
| Field | Type | Description |
|-------|------|-------------|
| timestamp | ISO8601 | UTC timestamp of reading |
| sensor_id | string | Unique sensor identifier |
| parameter_name | string | Physical parameter name |
| value | float | Measured value |
| unit | string | Measurement unit |
| status | enum | OK, WARNING, ERROR, CRITICAL |
| production_line | string | Line_A, Line_B, Line_C |

#### 1.2 Refactor Data Ingestion Pipeline

**Current Flow** (containerized):
```
opcua_server:4840/modbus_server:5020 → Sensor Aggregator → Data Router → influxdb:8086/kafka:9092
```

**New Flow** (containerized):
```
Synthetic Data Generator (in glass_backend) → Data Router → influxdb:8086 (storage) + WebSocket (real-time) + kafka:9092 (streaming) + ML Feature Store
```

**Container-Specific Changes**:
- Remove dependencies on `opcua_server` and `modbus_server` containers (they provide no value)
- Update all connection strings to use Docker service names:
  - `INFLUXDB_URL=http://influxdb:8086` (already correct in docker-compose)
  - `KAFKA_BOOTSTRAP_SERVERS=kafka:9092` (already correct)
  - `POSTGRES_URL=postgresql://glass_admin:glass_secure_pass@postgres:5432/glass_production` (already correct)
- Ensure `industrial_connectors` code uses environment variables, not hardcoded `localhost`

**Key Changes**:
- Remove dependencies on external OPC UA/Modbus servers
- Replace `industrial_connectors` with internal synthetic generators
- Add WebSocket broadcaster for real-time frontend updates
- Add ML feature extraction and storage for model training

#### 1.3 Implement Real-Time Data Broadcasting

**Component**: `streaming_pipeline/websocket_broadcaster.py`

**Responsibilities**:
- Maintain WebSocket connections with frontend clients
- Broadcast aggregated sensor data every 5-10 seconds
- Broadcast ML predictions every 1-2 minutes
- Broadcast alerts immediately when detected
- Broadcast RL recommendations when generated

**Container Integration**:
- WebSocket server runs in `glass_backend` container on port `8000`
- Frontend connects via `ws://localhost:8000/ws` (external)
- Backend-to-backend communication (if needed) uses `ws://backend:8000/ws`
- Consumes data from `kafka:9092` for streaming updates

**Message Types**:
| Type | Frequency | Payload |
|------|-----------|---------|
| sensor_update | 5-10s | Latest sensor readings with statistics |
| prediction_update | 1-2min | Defect probabilities for next 1h, 4h, 24h |
| alert | immediate | Anomaly detection, threshold violations |
| recommendation | on-change | RL-generated parameter adjustments |
| quality_metrics | 30min | Production quality KPIs |

### Phase 2: Implement Digital Twin Functionality

#### 2.1 Shadow Mode Simulator

**Component**: `simulation/shadow_mode.py` (enhance existing)

**Purpose**: Run parallel simulation alongside real data stream to validate system behavior.

**Workflow**:
```
Real Sensor Data → Digital Twin Model → Predicted State → Compare with Actual State → Validation Metrics
```

**Capabilities**:
- Maintain synchronized state with production line
- Predict next state based on current parameters
- Detect deviations between predicted and actual (indicates model drift or equipment issues)
- Log prediction accuracy for model health monitoring

**State Representation**:
| Component | State Variables |
|-----------|----------------|
| Furnace | temperature_profile (spatial), melt_level, viscosity, gas_composition |
| Forming | belt_speed, mold_temperature, pressure_profile, product_dimensions |
| Annealing | temperature_gradient, stress_distribution, cooling_rate |
| Quality | defect_probabilities, quality_score, production_rate |

#### 2.2 What-If Analyzer

**Component**: `simulation/what_if_analyzer.py` (enhance existing)

**Purpose**: Allow operators to test parameter changes before applying them.

**User Interface Flow**:
```
Operator selects parameter to adjust → Inputs new value → System runs simulation → Shows predicted outcome → Operator confirms or rejects
```

**Analysis Outputs**:
| Metric | Calculation |
|--------|-------------|
| Defect Rate Change | Predicted defects after change vs current baseline |
| Quality Score Impact | Change in overall quality percentage |
| Production Rate Impact | Units per hour change |
| Energy Consumption | Estimated energy cost change |
| Time to Effect | How long until change impacts output |

**Simulation Modes**:
- Single parameter change: Test one adjustment in isolation
- Multi-parameter optimization: Find optimal combination
- Scenario replay: Replay historical events with different parameters
- Stress testing: Find parameter limits before critical failures

#### 2.3 Physics Engine Integration

**Component**: `physics_engine/*` integration with Digital Twin

**Current State**: Physics models exist in isolation:
- `thermal_dynamics.py`: Heat transfer simulation
- `viscosity_model.py`: Glass viscosity calculation
- `stress_analyzer.py`: Internal stress prediction

**Integration Requirements**:
- Feed real sensor data into physics models
- Update physics state at appropriate frequencies
- Expose physics predictions via Digital Twin API
- Use physics constraints to validate RL agent recommendations

**Physics Simulation Frequency**:
| Model | Update Rate | Reason |
|-------|-------------|--------|
| Thermal Dynamics | Every 10s | Heat transfer is relatively slow process |
| Viscosity Model | Every 5s | Viscosity changes with temperature |
| Stress Analyzer | Per product | Calculates stress for each formed unit |

### Phase 3: ML Model Training and Deployment

#### 3.1 Synthetic Training Data Generation

**Component**: `training/synthetic_training_data_generator.py`

**Approach**: Generate labeled datasets with known parameter-defect correlations.

**Training Dataset Structure**:

**Time Series Dataset** (for LSTM):
- Window size: 120 timesteps (2 hours of minute-level data)
- Features: 20 sensor readings (temperature, pressure, speed, etc.)
- Labels: Defect occurrence in next 1h, 4h, 24h (binary + probability)
- Size: 100,000 sequences with 15% positive defect cases

**Image Dataset** (for Vision Transformer):
- Image size: 224x224 RGB
- Synthetic defects: Cracks, bubbles, chips, cloudiness, deformation, stains
- Augmentation: Rotation, brightness, blur to simulate real camera variations
- Size: 50,000 images with bounding boxes and defect type labels

**Graph Dataset** (for GNN):
- Nodes: 20 sensors with spatial/functional relationships
- Edges: Correlations (strong >0.7, medium 0.4-0.7, weak 0.2-0.4)
- Features: Temporal statistics (mean, std, trend, autocorrelation)
- Labels: Sensor failure prediction, anomaly detection
- Size: 10,000 graph snapshots

#### 3.2 Model Training Pipeline

**Component**: `training/model_training_pipeline.py`

**Training Strategy**:

**LSTM Predictor**:
- Architecture: 2 layers, 128 units, 0.2 dropout, attention mechanism
- Loss: Binary cross-entropy + Mean Squared Error for probability
- Optimizer: Adam with learning rate 0.001
- Training: 50 epochs with early stopping (patience=5)
- Validation: Time-based split (last 20% of time series)
- Target Metrics: Recall >90%, Precision >80%, F1 >85%

**Vision Transformer**:
- Architecture: Patch size 16, 6 transformer layers, 384 embedding dimension
- Loss: Multi-task (classification + segmentation)
- Optimizer: AdamW with cosine annealing schedule
- Training: 100 epochs with data augmentation
- Validation: Stratified split (20%)
- Target Metrics: mAP >85% for defect detection, IoU >70% for segmentation

**GNN Sensor Network**:
- Architecture: 3 GAT layers with edge attention, 64 hidden dimension
- Loss: Node classification + edge prediction
- Optimizer: Adam with weight decay
- Training: 30 epochs
- Validation: Random split (20%)
- Target Metrics: Node classification accuracy >88%, Edge prediction AUC >90%

**Ensemble Meta-Learner**:
- Inputs: Outputs from LSTM, ViT, GNN
- Architecture: Stacking with diversity regularization
- Training: Train on validation predictions from base models
- Target Metrics: Combined accuracy >92%, calibration error <5%

#### 3.3 Model Deployment and Versioning

**Component**: `training/model_deployment_manager.py`

**Model Registry**:
- Storage: Local filesystem with MLflow tracking
- Versioning: Semantic versioning (major.minor.patch)
- Metadata: Training date, performance metrics, hyperparameters, dataset version

**Deployment Process**:
```
Train new model → Evaluate on test set → Compare with production model → A/B test (10% traffic) → Monitor metrics for 24h → Full rollback if quality degrades → Full deployment if improved
```

**Rollback Criteria**:
- Prediction accuracy drops >5% relative
- False positive rate increases >10%
- Inference latency exceeds 100ms (p95)
- Model crashes or returns errors

#### 3.4 Continuous Learning Pipeline

**Component**: `training/continuous_learning.py`

**Data Collection**:
- Stream sensor data and defect labels to feature store
- Buffer 24 hours of data before retraining
- Maintain training dataset with sliding window (last 30 days)

**Retraining Schedule**:
- LSTM: Weekly (every Sunday 02:00 UTC)
- Vision Transformer: Bi-weekly (every other Monday 02:00 UTC)
- GNN: Weekly (every Wednesday 02:00 UTC)
- Ensemble: After any base model update

**Training Automation**:
- Triggered by schedule or performance degradation detection
- Automated hyperparameter tuning with Optuna (limited budget)
- Automated validation and comparison with current production model
- Automated deployment if improvement detected

### Phase 4: Knowledge Graph Integration

#### 4.1 Populate Causal Knowledge Base

**Component**: `knowledge_graph/knowledge_base_initializer.py`

**Container Integration**:
- Connects to Neo4j at `bolt://neo4j:7687` (not `localhost:7687`)
- Uses credentials from environment: `NEO4J_USER=neo4j`, `NEO4J_PASSWORD=neo4jpassword`
- Uses Redis at `redis:6379` for caching (not `localhost:6379`)
- Initialization runs on backend startup

**Domain Knowledge to Encode**:

**Parameter → Defect Causation**:
| Parameter | Defect Type | Confidence | Mechanism |
|-----------|-------------|------------|-----------|
| furnace_temperature > 1600°C | crack | 0.85 | Excessive thermal stress during cooling |
| furnace_temperature < 1450°C | cloudiness | 0.78 | Incomplete melting, crystallization |
| belt_speed > 180 m/min | deformation | 0.82 | Insufficient forming time |
| mold_temperature < 280°C | stress_fracture | 0.75 | Rapid cooling creates internal stress |
| pressure_variation > 15 MPa | bubble | 0.88 | Gas entrapment during forming |
| cooling_rate > 7°C/min | crack | 0.90 | Thermal shock exceeds material limits |

**Equipment → Parameter Relationships**:
| Equipment | Controls | Affects | Response Time |
|-----------|----------|---------|---------------|
| furnace_burner | furnace_temperature | melt_viscosity | 15-30 min |
| belt_motor | belt_speed | forming_pressure | 5-10 sec |
| mold_heater | mold_temperature | product_stress | 10-20 min |
| cooling_fan | cooling_rate | annealing_quality | 2-5 min |

**Intervention Strategies**:
| Defect Detected | Recommended Action | Expected Outcome | Implementation Time |
|-----------------|-------------------|------------------|---------------------|
| crack (probability >70%) | Reduce furnace temp by 20-30°C | Defect rate -40% | 20-40 min |
| bubble (count >5/min) | Decrease forming pressure by 10 MPa | Defect rate -60% | 1-2 min |
| deformation (score >0.3) | Reduce belt speed by 15% | Defect rate -50% | 10-30 sec |

#### 4.2 Root Cause Analysis Engine

**Component**: `knowledge_graph/root_cause_analyzer.py`

**Analysis Workflow**:
```
Defect detected → Query knowledge graph for possible causes → Retrieve recent sensor data → Match patterns to known causes → Rank by confidence → Return top 3 root causes with evidence
```

**Ranking Criteria**:
- Causal confidence from knowledge base
- Temporal correlation (parameter deviation occurred before defect)
- Magnitude of deviation (how far outside normal range)
- Historical frequency (how often this cause led to this defect)

**Output Format**:
| Field | Description |
|-------|-------------|
| root_cause_id | Unique identifier |
| parameter_name | Which sensor/parameter is root cause |
| deviation_magnitude | How far from normal (in standard deviations) |
| confidence | Probability this is true cause (0-1) |
| evidence | List of supporting data points |
| recommended_action | Suggested intervention |

#### 4.3 Graph Visualization API

**Component**: `backend/api/knowledge_graph_endpoints.py`

**Endpoints**:

**GET /api/knowledge-graph/subgraph**
- Query Parameters: defect_type, max_depth
- Returns: Nodes and edges for visualization in D3.js format

**GET /api/knowledge-graph/root-cause**
- Query Parameters: defect_type, timestamp, min_confidence
- Returns: Ranked list of possible root causes

**GET /api/knowledge-graph/intervention**
- Query Parameters: defect_type, current_parameters
- Returns: Recommended parameter adjustments with predicted outcomes

### Phase 5: RL Agent Autonomy and Safety

#### 5.1 Safe Exploration Framework

**Component**: `reinforcement_learning/safe_exploration.py`

**Safety Constraints**:
- Parameter bounds: Never exceed physical equipment limits
- Rate limits: Limit speed of parameter changes
- Rollback capability: Revert to previous state if quality degrades
- Human approval: Require operator confirmation for large changes

**Constraint Definitions**:
| Parameter | Min | Max | Max Change Rate | Requires Approval |
|-----------|-----|-----|----------------|-------------------|
| furnace_temperature | 1200°C | 1700°C | ±50°C per hour | if change >100°C |
| belt_speed | 50 m/min | 200 m/min | ±20 m/min per minute | if change >50 m/min |
| mold_temperature | 200°C | 600°C | ±30°C per hour | if change >80°C |
| forming_pressure | 0 MPa | 120 MPa | ±15 MPa per minute | if change >30 MPa |

#### 5.2 Reward Function Design

**Component**: `reinforcement_learning/reward_calculator.py`

**Multi-Objective Reward**:
```
reward = -10 * defect_rate + 0.1 * production_rate - 0.01 * energy_consumption - 100 * safety_violation
```

**Component Breakdown**:
- Defect minimization (highest weight): Penalize each defect heavily
- Production optimization: Reward higher throughput
- Energy efficiency: Minor penalty for energy use
- Safety: Massive penalty for constraint violations

**Reward Shaping**:
- Dense rewards: Provide intermediate rewards for improvement trends
- Delayed rewards: Account for time lag between action and defect (use eligibility traces)
- Exploration bonus: Encourage trying unexplored state-action pairs

#### 5.3 Autonomous Action Application

**Component**: `reinforcement_learning/action_executor.py`

**Action Application Workflow**:
```
RL agent selects action → Safety validation → Simulate outcome in Digital Twin → If safe and beneficial → Apply to production → Monitor quality for 30 min → Rollback if quality degrades → Log outcome for learning
```

**Action Types**:
| Action | Control Interface | Validation |
|--------|------------------|------------|
| Adjust furnace power | Synthetic control signal | Check temperature limits |
| Change belt speed | Synthetic motor control | Check speed limits and product impact |
| Modify mold temperature | Synthetic heater control | Check temperature limits and stress |
| Adjust cooling rate | Synthetic fan control | Check cooling rate limits |

**Rollback Mechanism**:
- Store previous parameter state before change
- Monitor quality metrics for rollback_window (default 30 min)
- If defect_rate increases >20% relative to baseline → automatic rollback
- If operator manually rejects change → immediate rollback
- Log rollback event for RL penalty

### Phase 6: Frontend Dashboard Completion

#### 6.1 Real-Time Metrics Updates

**Component**: `frontend/src/hooks/useWebSocket.ts` (enhance)

**WebSocket Integration**:
- Connect to backend WebSocket endpoint on mount
- Subscribe to multiple channels: sensors, predictions, alerts, recommendations
- Update React state on message receipt
- Handle reconnection on connection loss

**Update Frequencies**:
| Component | Update Rate | Data Source |
|-----------|-------------|-------------|
| KPI Cards | 10 seconds | Aggregated sensor stats |
| Sensor Graphs | 5 seconds | Latest sensor readings |
| Prediction Charts | 2 minutes | ML model outputs |
| Recommendations | On change | RL agent + Knowledge Graph |
| Alerts | Immediate | Anomaly detection |

#### 6.2 Alerts and Notifications Tab

**Component**: `frontend/src/components/AlertsPanel.tsx` (new)

**Alert Categories**:
- CRITICAL: Safety violations, equipment failures, quality catastrophic drop
- HIGH: Defect rate exceeding threshold, sensor anomalies
- MEDIUM: Parameter drift, prediction confidence drop
- LOW: Informational messages, maintenance reminders

**Alert Display**:
| Field | Display |
|-------|---------|
| Severity | Color-coded icon (red, orange, yellow, blue) |
| Timestamp | Relative time (e.g., "5 minutes ago") |
| Message | Brief description |
| Affected Equipment | Production line, specific sensor/equipment |
| Recommended Action | What operator should do |
| Acknowledge Button | Mark alert as read |

**Alert Actions**:
- Acknowledge: Mark as read, remove from active list
- Investigate: Open detailed view with sensor data and root cause analysis
- Apply Recommendation: Execute suggested parameter change
- Dismiss: Ignore alert (requires justification)

#### 6.3 Settings and Configuration Panel

**Component**: `frontend/src/components/SettingsPanel.tsx` (new)

**Configurable Settings**:

**Alert Thresholds**:
- Defect rate threshold (default: >5% triggers HIGH alert)
- Temperature deviation threshold (default: ±40°C triggers MEDIUM alert)
- Quality score threshold (default: <90% triggers HIGH alert)

**Dashboard Preferences**:
- Update frequency (5s, 10s, 30s)
- Default time range for graphs (1h, 4h, 24h, 7d)
- Visible KPI cards (checkboxes for each metric)

**Model Settings**:
- Enable/disable specific models (LSTM, ViT, GNN)
- Model confidence threshold for recommendations (default: 0.7)
- Autonomous action approval mode (manual, semi-auto, auto)

**Notification Settings**:
- Email notifications on/off
- Sound alerts on/off
- Notification severity filter (show CRITICAL + HIGH only)

#### 6.4 AR Visualization Enhancement

**Component**: `frontend/src/components/ARVisualization.tsx` (enhance)

**Current State**: Component skeleton exists but not functional.

**Required Features**:
- 3D model of production line (Three.js)
- Real-time sensor overlay (show values on equipment)
- Defect location markers (highlight where defects detected on product)
- Parameter adjustment controls (interactive sliders)
- What-if simulation results overlay

**AR Overlay Elements**:
| Element | Visualization | Data Source |
|---------|--------------|-------------|
| Furnace temperature | Color-coded heat map on furnace | Real-time sensor |
| Belt speed | Animated conveyor movement | Real-time sensor |
| Defect locations | Red markers on glass product | MIK-1 detections |
| Predicted defects | Yellow markers (future positions) | LSTM predictions |
| Recommended changes | Green highlights on controls | RL agent |

### Phase 7: Backend API Enhancements

#### 7.1 Missing API Endpoints

**Component**: `backend/fastapi_backend.py` (enhance)

**Container Context**:
- Backend runs in `glass_backend` container
- Accessible externally at `http://localhost:8000`
- Internal service name: `backend:8000`
- Mounts project directory at `/app` for hot-reload
- Command: `uvicorn backend.fastapi_backend:app --host 0.0.0.0 --port 8000 --reload`

**New Endpoints**:

**POST /api/digital-twin/what-if**
- Request: Parameter changes to simulate
- Response: Predicted quality impact, defect probabilities, production rate

**GET /api/digital-twin/shadow-state**
- Response: Current shadow mode state, prediction accuracy metrics

**POST /api/rl-agent/apply-recommendation**
- Request: Recommendation ID, approval token
- Response: Application status, expected time to effect

**POST /api/rl-agent/rollback**
- Request: Action ID to rollback
- Response: Rollback status, restored parameter values

**GET /api/models/performance**
- Response: Current model metrics, training history, version info

**POST /api/models/retrain**
- Request: Model name, training config
- Response: Training job ID, estimated completion time

**GET /api/alerts**
- Query: severity_filter, time_range, acknowledged
- Response: List of alerts matching criteria

**POST /api/alerts/{alert_id}/acknowledge**
- Response: Acknowledgment confirmation

#### 7.2 WebSocket Channels

**Component**: `backend/websocket_manager.py` (new)

**Channel Structure**:
- `/ws/sensors`: Real-time sensor data stream
- `/ws/predictions`: ML model prediction updates
- `/ws/alerts`: Alert notifications
- `/ws/recommendations`: RL agent recommendations
- `/ws/quality`: Quality metrics and KPIs

**Container Networking**:
- WebSocket endpoint: `ws://backend:8000/ws` (internal)
- Frontend connects to: `ws://localhost:8000/ws` (external via REACT_APP_WS_URL)
- Backend broadcasts to all connected clients
- No CORS issues due to proper FastAPI WebSocket configuration

**Message Format**:
```
{
  "channel": "sensors",
  "timestamp": "2025-01-15T10:30:00Z",
  "data": { sensor readings },
  "metadata": { production_line, equipment_id }
}
```

### Phase 8: Metrics and KPIs Implementation

#### 8.1 Quality Metrics Calculation

**Component**: `backend/metrics/quality_calculator.py` (new)

**Metrics to Calculate**:

**Defect Metrics**:
- Total defects (count per shift/day/week)
- Defect rate (defects per 1000 units)
- Defect distribution (percentage by type: crack, bubble, chip, etc.)
- First-pass yield (percentage of units passing initial inspection)

**Production Metrics**:
- Production rate (units per hour)
- Line efficiency (actual output / theoretical maximum)
- Downtime (percentage of time line is stopped)
- OEE (Overall Equipment Effectiveness = Availability × Performance × Quality)

**Prediction Metrics**:
- Model accuracy (correct predictions / total predictions)
- Precision (true positives / (true positives + false positives))
- Recall (true positives / (true positives + false negatives))
- F1 Score (harmonic mean of precision and recall)
- Calibration error (how well predicted probabilities match actual frequencies)

#### 8.2 KPI Dashboard Cards

**Component**: `frontend/src/components/KPICards.tsx` (enhance)

**Card Specifications**:

**Total Defects Card**:
- Value: Count from last shift
- Trend: Arrow and percentage change vs previous shift
- Color: Green if decreasing, red if increasing

**Quality Rate Card**:
- Value: Percentage of defect-free units
- Threshold: Green >95%, Yellow 85-95%, Red <85%
- Sparkline: Trend over last 24 hours

**Critical Alerts Card**:
- Value: Count of unacknowledged CRITICAL alerts
- Pulsing animation if count >0
- Click to open alerts panel

**Prediction Accuracy Card**:
- Value: Percentage of correct predictions in last 24h
- Trend: Change vs last week
- Gauge visualization

**Production Rate Card**:
- Value: Units per hour
- Target: Comparison to expected rate
- Bar graph: Current vs target

**Response Time Card**:
- Value: Average time from alert to corrective action
- Target: <10 minutes
- Histogram: Distribution of response times

## Implementation Sequence

### Week 1: Data Foundation
- Day 1-2: Implement synthetic data generator
- Day 3-4: Refactor data ingestion pipeline
- Day 5: Implement WebSocket broadcaster
- Day 6-7: Testing and validation

### Week 2: Digital Twin and Physics
- Day 1-2: Enhance Shadow Mode with state synchronization
- Day 3-4: Implement What-If Analyzer logic
- Day 5: Integrate physics engine
- Day 6-7: API endpoints and testing

### Week 3: ML Training and Deployment
- Day 1-2: Generate synthetic training datasets
- Day 3-5: Train all models to target performance
- Day 6: Implement model deployment pipeline
- Day 7: Set up continuous learning automation

### Week 4: Knowledge Graph and RL
- Day 1-2: Populate knowledge graph with domain knowledge
- Day 3: Implement root cause analysis
- Day 4-5: Safe exploration and autonomous action application
- Day 6-7: Testing and validation

### Week 5: Frontend Completion
- Day 1-2: Real-time metrics and WebSocket integration
- Day 3: Alerts and notifications panel
- Day 4: Settings panel
- Day 5: AR visualization enhancements
- Day 6-7: Integration testing and polish

### Week 6: Backend APIs and Metrics
- Day 1-2: Implement missing API endpoints
- Day 3: WebSocket channel management
- Day 4-5: Metrics calculation and KPI implementation
- Day 6-7: End-to-end testing and documentation

## Success Criteria

### Functional Requirements
- Dashboard metrics update in real-time (every 5-10 seconds)
- Digital Twin shows synchronized state with production
- What-If Analyzer returns predictions within 2 seconds
- ML models achieve target performance (Accuracy >85%, Recall >90%, Precision >80%)
- Knowledge Graph returns root causes with >70% confidence
- RL agent generates and applies recommendations autonomously
- All frontend tabs are functional (no "Under Development" placeholders)
- Alerts are generated and displayed within 1 second of detection

### Performance Requirements
- WebSocket latency <100ms (p95)
- API response time <500ms (p95)
- ML inference time <100ms per prediction
- Digital Twin simulation <2s per What-If scenario
- System handles 100+ concurrent sensor streams

### Quality Requirements
- Zero initialization errors (all components start successfully)
- Data flows from generation → processing → ML → visualization without gaps
- Models retrain weekly without manual intervention
- Autonomous actions improve quality (defect rate reduction >10% within 1 week)
- System uptime >99.5% (excluding planned maintenance)

## Risk Mitigation

### Risk: Synthetic data too unrealistic
- Mitigation: Validate synthetic patterns against task.md specifications, consult domain expert if available

### Risk: Models fail to train to target performance
- Mitigation: Implement iterative training with hyperparameter tuning, use pretrained weights if available

### Risk: RL agent causes production quality degradation
- Mitigation: Enforce strict safety constraints, require human approval for large changes, implement automatic rollback

### Risk: WebSocket connections unstable
- Mitigation: Implement reconnection logic with exponential backoff, message buffering during disconnection

### Risk: Performance degradation with real-time processing
- Mitigation: Profile and optimize hot paths, implement caching, use asynchronous processing where possible
