# Dashboard and Digital Twin Enhancement - ML-Driven Integration Design

## Executive Overview

This design document addresses critical integration gaps between the frontend dashboard, backend ML systems, and digital twin simulation infrastructure. The current implementation suffers from hardcoded data, disconnected visualization components, and missing real-time ML-driven analytics. This enhancement will transform the system into a fully integrated, ML-driven predictive analytics platform.

## Problem Statement

### Current System Issues

| Component | Current State | Required State |
|-----------|--------------|----------------|
| **AdvancedDashboard.tsx** | Hardcoded metrics, static visualizations | Real-time ML predictions, dynamic defect detection |
| **DigitalTwin3D.tsx** | Basic 3D model, no What-If analysis | Shadow mode, What-If scenarios, physics simulation |
| **KnowledgeGraph.tsx** | Static hardcoded causes/recommendations | ML-driven root cause analysis, Neo4j integration |
| **WebSocket Integration** | Connection errors (403), limited data flow | Continuous defect alerts, parameter streaming |
| **Notifications Tab** | Not implemented | Critical alerts, ML-driven recommendations |
| **Analytics Tab** | Not implemented | Interactive graphs, statistical analysis |
| **Model Explainability** | Placeholder data | SHAP/LIME analysis, feature attribution |

### Key Requirements

1. **Real-Time ML Integration**: All dashboard metrics must derive from live ML model predictions
2. **Synthetic Production Simulation**: Realistic generation of defects, temperature fluctuations, pressure variations
3. **WebSocket Broadcasting**: Seamless data streaming from backend to frontend without 403 errors
4. **Interactive Digital Twin**: Shadow mode testing and What-If analysis capabilities
5. **ML-Driven Recommendations**: Knowledge graph powered by causal inference
6. **Complete Notification System**: Alert classification, prioritization, and action tracking
7. **Advanced Analytics Dashboard**: Statistical visualizations beyond surface-level metrics

## System Architecture Enhancement

### Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Production Environment                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ Furnace  │  │ Forming  │  │ Annealing│  │ Quality  │       │
│  │ Sensors  │  │ Sensors  │  │ Sensors  │  │ Camera   │       │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘       │
└───────┼─────────────┼──────────────┼─────────────┼─────────────┘
        │             │              │             │
        └─────────────┴──────────────┴─────────────┘
                      │
        ┌─────────────▼─────────────────────┐
        │ Synthetic Data Generator          │
        │ - Temperature simulation          │
        │ - Defect injection (probabilistic)│
        │ - Parameter variation engine      │
        └─────────────┬─────────────────────┘
                      │
        ┌─────────────▼─────────────────────┐
        │ WebSocket Broadcaster             │
        │ - Real-time streaming             │
        │ - Alert aggregation               │
        │ - Defect notification pipeline    │
        └─────────────┬─────────────────────┘
                      │
        ┌─────────────▼─────────────────────┐
        │ ML Inference Engine               │
        │ ┌───────────────────────────────┐ │
        │ │ LSTM Predictor                │ │
        │ │ - 1h/4h/24h predictions       │ │
        │ │ - Defect probability          │ │
        │ └───────────────────────────────┘ │
        │ ┌───────────────────────────────┐ │
        │ │ GNN Sensor Network            │ │
        │ │ - Anomaly detection           │ │
        │ │ - Causal relationships        │ │
        │ └───────────────────────────────┘ │
        │ ┌───────────────────────────────┐ │
        │ │ Vision Transformer            │ │
        │ │ - Visual defect classification│ │
        │ └───────────────────────────────┘ │
        └─────────────┬─────────────────────┘
                      │
        ┌─────────────▼─────────────────────┐
        │ Knowledge Graph Engine            │
        │ - Neo4j causal inference          │
        │ - Root cause analysis             │
        │ - Recommendation generation       │
        └─────────────┬─────────────────────┘
                      │
        ┌─────────────▼─────────────────────┐
        │ Digital Twin Simulation           │
        │ ┌───────────────────────────────┐ │
        │ │ Shadow Mode                   │ │
        │ │ - Real vs Predicted tracking  │ │
        │ └───────────────────────────────┘ │
        │ ┌───────────────────────────────┐ │
        │ │ What-If Analyzer              │ │
        │ │ - Parameter optimization      │ │
        │ │ - Risk assessment             │ │
        │ └───────────────────────────────┘ │
        └─────────────┬─────────────────────┘
                      │
        ┌─────────────▼─────────────────────┐
        │ Frontend Dashboard                │
        │ ┌───────────────────────────────┐ │
        │ │ Real-Time Metrics Panel       │ │
        │ │ Defect Distribution (live)    │ │
        │ │ AI Recommendations            │ │
        │ │ Model Explainability (SHAP)   │ │
        │ │ Pipeline Metrics Monitor      │ │
        │ │ Notifications Center          │ │
        │ │ Advanced Analytics Tab        │ │
        │ │ Digital Twin 3D Viewer        │ │
        │ │ Knowledge Graph Visualizer    │ │
        │ └───────────────────────────────┘ │
        └───────────────────────────────────┘
```

## Component Design Specifications

### 1. WebSocket Broadcasting Enhancement

#### Purpose
Establish reliable real-time data streaming from backend to frontend with comprehensive defect alerting and parameter monitoring.

#### Current Issues
- 403 Forbidden errors on WebSocket connection
- Missing CORS configuration for WebSocket endpoint
- No defect-to-notification pipeline
- Limited parameter streaming

#### Enhanced WebSocket Architecture

**Backend Modifications** (`streaming_pipeline/websocket_broadcaster.py`)

Add the following capabilities to the WebSocketBroadcaster:

| Feature | Description | Implementation Strategy |
|---------|-------------|-------------------------|
| **Defect Alert Pipeline** | Broadcast defects detected by ML models in real-time | Subscribe to ML inference results, filter critical defects (confidence > 0.7), emit structured alert messages |
| **Parameter Streaming** | Continuous broadcast of furnace temperature, belt speed, mold temperature, pressure | Poll production simulator every 2 seconds, calculate trends, emit parameter update messages |
| **Synthetic Defect Generation** | Inject realistic defect occurrences based on production state | Use probabilistic model: high temperature increases crack probability, low pressure increases bubble probability |
| **Temperature Fluctuation Simulation** | Simulate realistic furnace temperature variations | Apply Gaussian noise with mean drift based on time-of-day patterns |
| **Connection Health Management** | Handle 403 errors, implement reconnection logic | Add authentication token validation, CORS whitelist configuration, heartbeat mechanism |

**Message Payload Structure**

```
Defect Alert Message:
{
  "type": "defect_alert",
  "timestamp": ISO 8601 string,
  "defect_type": "crack" | "bubble" | "chip" | "stain" | "cloudiness" | "deformation",
  "severity": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
  "location": { "line": string, "position_mm": number },
  "confidence": number (0.0-1.0),
  "ml_source": "LSTM" | "GNN" | "VisionTransformer",
  "recommended_actions": string[]
}

Parameter Update Message:
{
  "type": "parameter_update",
  "timestamp": ISO 8601 string,
  "furnace": {
    "temperature": number,
    "temperature_trend": "rising" | "falling" | "stable",
    "pressure": number,
    "melt_level": number
  },
  "forming": {
    "belt_speed": number,
    "mold_temperature": number,
    "forming_pressure": number
  },
  "quality_metrics": {
    "current_quality_rate": number,
    "defect_count_hourly": number,
    "production_rate": number
  }
}

Model Prediction Message:
{
  "type": "ml_prediction",
  "timestamp": ISO 8601 string,
  "model": "LSTM" | "GNN" | "Ensemble",
  "predictions": {
    "1h_defect_probability": { "crack": number, "bubble": number, ... },
    "4h_defect_probability": { "crack": number, "bubble": number, ... },
    "quality_score_forecast": number
  },
  "confidence_intervals": {
    "lower_bound": number,
    "upper_bound": number
  }
}
```

**CORS and Authentication Configuration**

Modify WebSocket endpoint in `fastapi_backend.py`:

| Configuration | Value | Rationale |
|---------------|-------|----------|
| **Allowed Origins** | `["http://localhost:3000", "http://frontend:3000"]` | React development server and Docker container |
| **WebSocket Path** | `/ws/realtime` | Dedicated real-time data stream |
| **Authentication** | Optional Bearer token in query params | Enable secure production deployment |
| **Heartbeat Interval** | 30 seconds | Detect dead connections |
| **Max Connections** | 100 | Prevent resource exhaustion |

### 2. Synthetic Data Generator Enhancement

#### Purpose
Create realistic production line simulation with correlated parameter variations and defect generation.

#### Defect Generation Model

**Probabilistic Defect Model**

| Defect Type | Base Rate | Temperature Factor | Pressure Factor | Speed Factor |
|-------------|-----------|-------------------|----------------|-------------|
| **Crack** | 0.05 | +0.02 per 10°C above 1580°C | +0.01 per 5kPa below 45kPa | +0.015 per 10 m/min above 170 m/min |
| **Bubble** | 0.08 | +0.025 per 10°C below 1450°C | -0.01 per 5kPa above optimal | -0.005 per 10 m/min below optimal |
| **Chip** | 0.03 | +0.01 per 10°C variance from 1550°C | +0.02 per 10kPa above 55kPa | +0.02 per 10 m/min above 180 m/min |
| **Stain** | 0.04 | +0.015 per 10°C above 1600°C | N/A | -0.01 per 10 m/min below 140 m/min |
| **Cloudiness** | 0.06 | +0.02 per 10°C variance from optimal | +0.01 per 5kPa variance | N/A |
| **Deformation** | 0.02 | +0.03 per 10°C above 1620°C | +0.025 per 10kPa above 60kPa | +0.03 per 10 m/min above 185 m/min |

**Parameter Variation Patterns**

```
Furnace Temperature Simulation:
- Base: 1550°C
- Diurnal pattern: ±15°C sinusoidal variation over 24h
- Random drift: Gaussian noise (σ=5°C)
- Anomaly events: 1% probability per hour of +40°C spike lasting 10-30 minutes

Belt Speed Simulation:
- Base: 150 m/min
- Production load factor: Varies 140-170 m/min based on demand
- Acceleration/deceleration: Maximum 5 m/min per minute
- Maintenance stops: 0.5% probability per hour, duration 15-45 minutes

Mold Temperature Simulation:
- Base: 320°C
- Correlation with furnace: 0.6 (lags by 15 minutes)
- Cooling system efficiency: Degrades 0.1% per hour, resets after maintenance

Pressure Simulation:
- Base: 50 kPa
- Correlation with belt speed: 0.4
- Valve oscillation: ±3 kPa with 5-minute period
```

### 3. Dashboard Component Enhancements

#### 3.1 AdvancedDashboard.tsx ML Integration

**Real-Time Metrics Panel**

Replace hardcoded values with WebSocket-driven data:

| Metric | Data Source | Update Frequency | Visualization |
|--------|-------------|------------------|---------------|
| **Quality Rate** | `quality_metrics.current_quality_rate` from WebSocket | 5 seconds | Animated gauge with trend arrow |
| **Defect Count** | `quality_metrics.defect_count_hourly` from WebSocket | 5 seconds | Counter with 24h sparkline |
| **Furnace Temperature** | `furnace.temperature` from WebSocket | 2 seconds | Thermometer with color gradient |
| **Belt Speed** | `forming.belt_speed` from WebSocket | 2 seconds | Speed gauge with optimal range indicator |

**Defect Distribution Chart**

Replace static data with aggregated defect alerts:

```
Data Aggregation Strategy:
1. Maintain rolling window buffer (last 1 hour of defect alerts)
2. Group by defect_type
3. Count occurrences
4. Calculate percentage distribution
5. Update chart every 10 seconds

Visualization:
- Radial bar chart (current implementation maintained)
- Color mapping from severity: CRITICAL=red, HIGH=orange, MEDIUM=yellow, LOW=green
- Tooltip shows: defect count, percentage, severity distribution
```

**AI Recommendations Panel**

Integrate with Knowledge Graph recommendations:

```
Data Flow:
1. Frontend polls `/api/recommendations` endpoint every 30 seconds
2. Backend queries Knowledge Graph for intervention recommendations
3. Filter by priority (HIGH, MEDIUM, LOW)
4. Sort by confidence score
5. Display top 5 recommendations

Recommendation Card Structure:
- Priority badge (color-coded)
- Action description from Knowledge Graph
- Confidence score as percentage
- Expected impact (from impact analysis)
- "Apply" button (triggers What-If analysis)
- "Dismiss" button (feedback to ML system)
```

**Model Explainability Integration**

Connect to `/api/explainability/feature-importance` endpoint:

```
Display Components:
1. Feature Importance Bar Chart
   - Top 10 features from SHAP analysis
   - Color-coded by feature category (thermal, mechanical, chemical)

2. SHAP Force Plot
   - Interactive visualization showing how features push prediction
   - Display for most recent critical defect prediction

3. Partial Dependence Plots
   - Show relationship between furnace temperature and defect probability
   - Interactive slider to explore parameter space
```

**Pipeline Metrics Monitor**

Connect to `/api/metrics/pipeline-performance` endpoint:

```
Metrics to Display:
- Model Inference Latency (LSTM, GNN, ViT)
- Prediction Accuracy (last 100 predictions)
- Data Pipeline Throughput (messages/second)
- Model Drift Indicator (data distribution shift)
- Retraining Status (last training timestamp, next scheduled)

Visualization:
- Real-time line charts for latency metrics
- Accuracy gauge with threshold indicator
- Status badges for pipeline health
```

#### 3.2 Notifications Tab Implementation

**New Component**: `NotificationsCenter.tsx`

**Notification Categories**

| Category | Trigger Condition | Priority | Auto-Dismiss |
|----------|------------------|----------|-------------|
| **Critical Defect Alert** | Defect confidence > 0.85 AND severity = CRITICAL | CRITICAL | No |
| **Parameter Anomaly** | Temperature > 1600°C OR Pressure < 40kPa | HIGH | After 5 minutes if resolved |
| **Model Prediction Warning** | 1h defect probability > 0.6 | MEDIUM | After 15 minutes |
| **Maintenance Reminder** | Equipment runtime > threshold | LOW | After acknowledgment |
| **System Health** | ML model error OR database connection loss | CRITICAL | No |

**Notification Data Structure**

```
Notification Object:
{
  "id": UUID string,
  "timestamp": ISO 8601 string,
  "category": enum,
  "priority": "CRITICAL" | "HIGH" | "MEDIUM" | "LOW",
  "title": string (brief description),
  "message": string (detailed message),
  "source": "ML_MODEL" | "SENSOR" | "SYSTEM",
  "actions": [
    {
      "label": string,
      "action": string (API endpoint or function),
      "params": object
    }
  ],
  "acknowledged": boolean,
  "resolved": boolean,
  "resolution_notes": optional string
}
```

**Notification Display Features**

1. **Real-Time Feed**: Scrollable list with newest on top
2. **Filtering**: By priority, category, date range
3. **Search**: Full-text search in message content
4. **Action Buttons**: Quick actions (e.g., "Adjust Temperature", "View Details")
5. **Grouping**: Collapse similar notifications
6. **Sound Alerts**: Optional audio notification for CRITICAL priority
7. **Badge Counter**: Unacknowledged count in navigation tab

#### 3.3 Analytics Tab Implementation

**New Component**: `AnalyticsDashboard.tsx`

**Section 1: Statistical Overview**

| Chart Type | Data Source | Purpose |
|------------|-------------|----------|
| **Defect Trend Analysis** | Historical defect data (24h, 7d, 30d) | Identify patterns and seasonal variations |
| **Quality Score Histogram** | Quality metrics over time | Distribution analysis |
| **Parameter Correlation Matrix** | Sensor readings with defect rates | Discover hidden relationships |
| **Production Efficiency Curve** | Units produced vs quality rate | Optimize throughput |

**Section 2: Predictive Analytics Visualization**

```
Forecasting Charts:
1. Multi-Horizon Prediction Timeline
   - X-axis: Time (now to +24h)
   - Y-axis: Defect probability
   - Multiple lines: 1h, 4h, 24h predictions
   - Confidence bands (shaded areas)

2. What-If Scenario Comparison
   - Side-by-side comparison of baseline vs proposed parameters
   - Expected quality impact
   - Risk assessment visualization

3. Model Performance Dashboard
   - Confusion matrix for classification models
   - Regression error plots
   - Calibration curves
```

**Section 3: Root Cause Analysis Explorer**

```
Interactive Features:
1. Defect Type Selector (dropdown)
2. Time Range Filter (date picker)
3. Causal Path Visualization
   - Sankey diagram showing parameter → defect flows
   - Node size represents impact magnitude
4. Contributing Factors Table
   - Ranked list of parameters
   - SHAP values
   - Confidence intervals
```

### 4. Digital Twin 3D Enhancements

#### 4.1 Shadow Mode Integration

**Purpose**: Compare real production outcomes with Digital Twin predictions to validate model accuracy.

**Implementation Strategy for DigitalTwin3D.tsx**

```
Shadow Mode Visualization:

Split-Screen View:
- Left Panel: Real Production Line (current implementation)
- Right Panel: Predicted Production Line (shadow mode simulation)

Real-Time Comparison Metrics:
1. Parameter Deviation Indicators
   - Furnace Temperature: Real vs Predicted difference
   - Belt Speed: Real vs Predicted difference
   - Defect Count: Real vs Predicted difference
   
2. Prediction Accuracy Gauge
   - Calculate running accuracy over last 1 hour
   - Display as percentage with color coding

3. Divergence Alerts
   - Trigger warning when Real vs Predicted deviation > threshold
   - Example: "Shadow Mode Alert: Actual temperature 25°C higher than predicted"
```

**Data Flow for Shadow Mode**

```
Backend Integration:
1. Frontend requests shadow mode activation via `/api/digital-twin/shadow-mode/start`
2. Backend ShadowModeSimulator begins parallel simulation
3. Every 5 seconds:
   - Frontend receives real sensor data via WebSocket
   - Frontend receives predicted state via `/api/digital-twin/shadow-mode/state`
4. Frontend calculates and visualizes deviations
5. Backend tracks prediction errors for model retraining feedback
```

**Visual Indicators**

| Component | Real Production | Shadow Mode Prediction | Deviation Display |
|-----------|----------------|------------------------|-------------------|
| **Furnace Glow** | Intensity based on real temperature | Intensity based on predicted temperature | Overlay badge showing ΔT |
| **Belt Movement** | Speed matches real belt speed | Speed matches predicted belt speed | Speedometer comparison |
| **Defect Particles** | Show actual defects detected | Show predicted defects | Color difference (red=real, blue=predicted) |

#### 4.2 What-If Analysis Interface

**Purpose**: Allow operators to test parameter changes before applying them to production.

**UI Components to Add to DigitalTwin3D.tsx**

```
What-If Control Panel:

Location: Overlay on 3D view (collapsible sidebar)

Parameter Adjustment Sliders:
1. Furnace Temperature
   - Range: 1400°C - 1700°C
   - Current value indicator
   - Optimal range highlight (1450-1550°C)

2. Belt Speed
   - Range: 100 - 200 m/min
   - Current value indicator
   - Optimal range highlight (130-170 m/min)

3. Mold Temperature
   - Range: 250°C - 400°C
   - Current value indicator
   - Optimal range highlight (300-340°C)

4. Forming Pressure
   - Range: 30 - 80 MPa
   - Current value indicator
   - Optimal range highlight (40-60 MPa)
```

**Scenario Analysis Display**

```
When operator adjusts parameters:

1. Frontend sends request to `/api/digital-twin/what-if/analyze`
   Payload: { parameter_changes: { furnace_temperature: 1570, ... } }

2. Backend WhatIfAnalyzer simulates scenario
   - Validates parameter safety
   - Runs physics simulation
   - Predicts defect rates
   - Calculates quality impact

3. Frontend displays impact analysis:
   ┌─────────────────────────────────────────┐
   │ Impact Analysis Results                 │
   ├─────────────────────────────────────────┤
   │ Quality Score Change: +3.2%             │
   │ Defect Rate Change: -12.5%              │
   │ Production Rate Change: -2.1%           │
   │ Energy Consumption: +5.3%               │
   │ Time to Effect: 18 minutes              │
   │ Risk Level: MEDIUM                      │
   ├─────────────────────────────────────────┤
   │ Warnings:                               │
   │ ⚠️ Moving outside optimal range        │
   │                                         │
   │ Recommendations:                        │
   │ ✅ Expected defect reduction           │
   │ 💡 Monitor temperature closely          │
   ├─────────────────────────────────────────┤
   │ [Apply Changes] [Cancel] [Save Scenario]│
   └─────────────────────────────────────────┘
```

**3D Visualization Updates for What-If**

```
When What-If scenario is active:

1. Furnace Appearance:
   - Predicted temperature changes glow intensity
   - Label updates to show "Scenario: 1570°C" in different color

2. Belt Speed Animation:
   - Belt movement speed adjusts to predicted speed
   - Ghosted overlay shows current vs predicted

3. Defect Particle Forecast:
   - Show predicted defect distribution
   - Use different color/shape for "future" defects
   - Animate transition from current to predicted state

4. Quality Heatmap Overlay:
   - Add color-coded overlay on 3D model
   - Green = improved quality
   - Red = degraded quality
   - Based on What-If analysis results
```

### 5. Knowledge Graph Visualization Enhancement

#### Current Issues in KnowledgeGraph.tsx

1. **Hardcoded Causes**: Static array of causes not connected to Neo4j
2. **Hardcoded Recommendations**: Not ML-driven
3. **Non-functional Buttons**: Buttons don't trigger actual API calls
4. **Static Graph Visualization**: Placeholder nodes not representing real causal relationships

#### Enhanced Knowledge Graph Integration

**Data Connection Strategy**

```
API Endpoints to Integrate:

1. GET /knowledge-graph/causes/{defect_type}
   - Retrieves ML-driven causes from Neo4j
   - Filters by min_confidence parameter
   - Returns: cause name, confidence, strength, observations, evidence

2. POST /knowledge-graph/recommendations/{defect_type}
   - Sends current parameter values
   - Retrieves intervention recommendations
   - Returns: parameter adjustments, expected impact, priority

3. GET /knowledge-graph/subgraph/{defect_type}
   - Retrieves causal subgraph for visualization
   - Returns: nodes (defects, causes, parameters), edges (relationships)
   - Max depth: 2 (defect → cause → parameter)
```

**Interactive Features to Add**

| Feature | Description | Implementation |
|---------|-------------|--------------|
| **Node Click Handler** | Click on graph node to see details | Modal popup with node properties, related edges, historical data |
| **Edge Interaction** | Hover over edges to see relationship strength | Tooltip showing confidence, number of observations |
| **Filter Controls** | Adjust confidence threshold in real-time | Slider that re-queries API and updates graph |
| **Export Graph** | Download graph as image or JSON | Button triggers client-side generation |
| **Historical Playback** | Replay how graph evolved over time | Time slider showing confidence changes |

**Graph Layout Algorithm**

```
Replace current static positioning with force-directed layout:

1. Central Node: Selected defect type (largest node)
2. First Ring: Direct causes (size proportional to confidence)
3. Second Ring: Contributing parameters (size proportional to impact)
4. Edge Thickness: Represents relationship strength
5. Edge Color: Gradient based on confidence (red=low, green=high)

Layout Physics:
- Repulsion force: Prevents node overlap
- Attraction force: Pulls connected nodes together
- Centering force: Keeps graph centered in viewport
```

**Cause Details Panel Enhancement**

```
When cause is selected, display:

┌─────────────────────────────────────────────────────┐
│ Cause: High Furnace Temperature                     │
├─────────────────────────────────────────────────────┤
│ Confidence: 92%                                     │
│ Strength: 0.85                                      │
│ Observations: 145 instances                         │
│                                                     │
│ Evidence:                                           │
│ • Temperature exceeded 1600°C in 87% of cases       │
│ • Rapid heating rate observed (>5°C/min)            │
│ • Correlation coefficient: 0.78                     │
│                                                     │
│ Contributing Parameters:                            │
│ • Furnace Power Setting (Impact: 0.9)               │
│ • Burner Zone 2 Temperature (Impact: 0.7)           │
│ • Ambient Temperature (Impact: 0.3)                 │
│                                                     │
│ Historical Trend:                                   │
│ [Mini line chart showing confidence over time]      │
│                                                     │
│ [View Full Analysis] [Mark as Reviewed]             │
└─────────────────────────────────────────────────────┘
```

### 6. Backend API Enhancements

#### New/Modified Endpoints Required

**Digital Twin Endpoints** (`fastapi_backend.py`)

```
Endpoint Group: /api/digital-twin

GET /api/digital-twin/state
Description: Returns current digital twin state (furnace temp, belt speed, defects, quality score)
Response:
{
  "timestamp": ISO string,
  "furnace": { "temperature": number, "pressure": number, "melt_level": number },
  "forming": { "belt_speed": number, "mold_temp": number, "pressure": number },
  "quality_score": number,
  "defects": { "crack": number, "bubble": number, ... },
  "production_rate": number
}

POST /api/digital-twin/shadow-mode/start
Description: Activates shadow mode simulation
Request Body: { "prediction_window_seconds": number }
Response: { "status": "active", "session_id": UUID }

GET /api/digital-twin/shadow-mode/state
Description: Returns shadow mode prediction vs actual comparison
Response:
{
  "real_state": { ...current production state... },
  "predicted_state": { ...shadow mode prediction... },
  "deviations": {
    "temperature_deviation": number,
    "defect_count_deviation": number,
    "quality_deviation": number
  },
  "prediction_accuracy": number
}

POST /api/digital-twin/what-if/analyze
Description: Analyzes impact of parameter changes
Request Body:
{
  "parameter_changes": {
    "furnace_temperature": number,
    "belt_speed": number,
    "mold_temperature": number
  },
  "current_state": optional object
}
Response:
{
  "impact_analysis": {
    "defect_rate_change": number,
    "quality_score_impact": number,
    "production_rate_impact": number,
    "energy_consumption_change": number,
    "time_to_effect_minutes": number,
    "risk_level": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
    "warnings": string[],
    "recommendations": string[]
  },
  "predicted_state": { ...modified state after changes... }
}
```

**Knowledge Graph Endpoints** (already exist, ensure integration)

```
GET /knowledge-graph/causes/{defect_type}
Query Params: min_confidence (default 0.5)

POST /knowledge-graph/recommendations/{defect_type}
Request Body: { "parameter_values": { "furnace_temperature": 1580, ... } }

GET /knowledge-graph/subgraph/{defect_type}
Query Params: max_depth (default 2)
```

**Notification Endpoints** (new)

```
Endpoint Group: /api/notifications

GET /api/notifications/active
Description: Returns all unacknowledged notifications
Query Params: priority (optional filter), category (optional filter)
Response:
{
  "notifications": [
    {
      "id": UUID,
      "timestamp": ISO string,
      "priority": string,
      "category": string,
      "title": string,
      "message": string,
      "source": string,
      "actions": array,
      "acknowledged": boolean
    }
  ],
  "unacknowledged_count": number
}

POST /api/notifications/{notification_id}/acknowledge
Description: Marks notification as acknowledged
Response: { "status": "acknowledged" }

DELETE /api/notifications/{notification_id}
Description: Dismisses notification
Response: { "status": "dismissed" }
```

**Analytics Endpoints** (new)

```
Endpoint Group: /api/analytics

GET /api/analytics/defect-trends
Query Params: timerange (24h, 7d, 30d), grouping (hourly, daily)
Description: Returns defect occurrence trends over time

GET /api/analytics/parameter-correlations
Description: Returns correlation matrix between parameters and defects

GET /api/analytics/production-efficiency
Query Params: timerange
Description: Returns production rate vs quality rate data points
```

### 7. WebSocket Message Flow Integration

#### Frontend WebSocket Handler Updates

**Modify `useDashboardData.ts` and create `useWebSocketStream.ts`**

```
WebSocket Connection Strategy:

1. Connection Initialization:
   - URL: ws://localhost:8000/ws/realtime
   - Add authentication token in connection request
   - Set reconnection logic (exponential backoff)

2. Message Type Handlers:

   onMessage("defect_alert"):
     - Add to notifications state
     - Update defect distribution chart
     - Play sound alert if priority = CRITICAL
     - Increment defect count metric

   onMessage("parameter_update"):
     - Update real-time metrics panel
     - Update performance trend chart
     - Recalculate quality rate

   onMessage("ml_prediction"):
     - Update AI recommendations panel
     - Update forecast charts in Analytics tab
     - Update model confidence indicators

   onMessage("system_health"):
     - Update system status indicator
     - Show warning banner if degraded

3. State Management:
   - Use React Context for WebSocket data
   - Implement message queue for buffering
   - Deduplicate messages based on timestamp
```

#### Backend WebSocket Broadcasting Logic

**Modify `streaming_pipeline/websocket_broadcaster.py`**

```
Broadcast Strategy:

1. Defect Detection Pipeline:
   - Subscribe to ML inference results
   - Filter defects with confidence > 0.7
   - Classify severity based on defect type and confidence
   - Broadcast defect_alert message

2. Parameter Monitoring Loop (every 2 seconds):
   - Poll production simulator for current state
   - Calculate parameter trends (rising/falling/stable)
   - Broadcast parameter_update message

3. ML Prediction Broadcast (every 30 seconds):
   - Trigger LSTM inference
   - Combine predictions from multiple models
   - Broadcast ml_prediction message

4. Alert Aggregation:
   - Maintain alert queue
   - Aggregate similar alerts (e.g., multiple temperature warnings)
   - Broadcast consolidated alerts
```

### 8. Implementation Phases

#### Phase 1: WebSocket & Data Streaming (Priority: CRITICAL)

**Deliverables**:
1. Fix WebSocket 403 errors in `fastapi_backend.py`
2. Enhance `WebSocketBroadcaster` with defect alert pipeline
3. Implement parameter streaming with realistic fluctuations
4. Add defect generation to `synthetic_data_generator.py`

**Acceptance Criteria**:
- WebSocket connects without errors
- Defect alerts appear in console
- Parameters update every 2 seconds
- Frontend receives all message types

#### Phase 2: Dashboard ML Integration (Priority: HIGH)

**Deliverables**:
1. Update `AdvancedDashboard.tsx` to consume WebSocket data
2. Replace hardcoded metrics with real-time values
3. Implement defect distribution aggregation
4. Connect AI recommendations to Knowledge Graph API

**Acceptance Criteria**:
- All metrics update in real-time
- Defect distribution reflects actual alerts
- Recommendations come from backend API
- Charts animate smoothly

#### Phase 3: Notifications Center (Priority: HIGH)

**Deliverables**:
1. Create `NotificationsCenter.tsx` component
2. Implement notification backend endpoints
3. Add navigation tab for Notifications
4. Implement sound alerts for CRITICAL priority

**Acceptance Criteria**:
- Notifications display in real-time
- Filtering and search work correctly
- Action buttons trigger appropriate responses
- Badge counter shows unacknowledged count

#### Phase 4: Digital Twin Enhancements (Priority: MEDIUM)

**Deliverables**:
1. Integrate Shadow Mode into `DigitalTwin3D.tsx`
2. Implement What-If analysis control panel
3. Add split-screen comparison view
4. Create `/api/digital-twin/*` endpoints

**Acceptance Criteria**:
- Shadow mode shows real vs predicted comparison
- What-If sliders trigger impact analysis
- 3D visualization updates based on scenarios
- Risk warnings display correctly

#### Phase 5: Knowledge Graph Functionality (Priority: MEDIUM)

**Deliverables**:
1. Connect `KnowledgeGraph.tsx` to Neo4j APIs
2. Implement interactive node/edge handlers
3. Add force-directed graph layout
4. Create detailed cause analysis panel

**Acceptance Criteria**:
- Graph displays real causal relationships
- Nodes and edges are clickable
- Confidence filter updates graph dynamically
- Recommendations update based on parameters

#### Phase 6: Advanced Analytics Tab (Priority: LOW)

**Deliverables**:
1. Create `AnalyticsDashboard.tsx` component
2. Implement statistical visualizations
3. Add multi-horizon prediction charts
4. Create root cause analysis explorer

**Acceptance Criteria**:
- All chart types render correctly
- Time range filters work
- Correlation matrix displays
- Predictions show confidence bands

#### Phase 7: Model Explainability Integration (Priority: LOW)

**Deliverables**:
1. Integrate SHAP analysis into dashboard
2. Add feature importance visualization
3. Implement partial dependence plots
4. Create model performance metrics display

**Acceptance Criteria**:
- Feature importance updates with predictions
- SHAP force plot displays correctly
- Metrics reflect actual model performance

### 9. Testing Strategy

#### Integration Testing Scenarios

**Scenario 1: Defect Detection Flow**
```
Test Steps:
1. Synthetic generator creates high-temperature event (1620°C)
2. Defect probability increases for "crack" type
3. ML model (LSTM) predicts crack with 0.85 confidence
4. WebSocket broadcasts defect_alert message
5. Frontend notification appears
6. Defect distribution chart updates
7. Knowledge Graph shows "High Temperature" as cause
8. Recommendations suggest temperature reduction

Expected Result: End-to-end flow completes in <5 seconds
```

**Scenario 2: What-If Analysis**
```
Test Steps:
1. Operator opens Digital Twin 3D view
2. Activates What-If mode
3. Adjusts furnace temperature from 1550°C to 1520°C
4. Clicks "Analyze Impact"
5. Backend simulates scenario
6. Impact analysis displays:
   - Quality improvement: +2.5%
   - Defect reduction: -8%
   - Risk level: LOW
7. Operator clicks "Apply Changes"
8. System schedules parameter adjustment

Expected Result: Analysis completes in <3 seconds, results are accurate
```

**Scenario 3: Shadow Mode Validation**
```
Test Steps:
1. Activate shadow mode via API
2. Run production for 1 hour
3. Compare real vs predicted states every 5 minutes
4. Calculate prediction accuracy
5. Display deviations in split-screen view

Expected Result: Prediction accuracy > 80%, deviations highlighted when >threshold
```

#### Performance Requirements

| Metric | Target | Measurement Method |
|--------|--------|-----------------|
| **WebSocket Latency** | <100ms | Time from backend broadcast to frontend receipt |
| **Dashboard Render Time** | <2 seconds | Time to initial chart render |
| **API Response Time** | <500ms | 95th percentile for all endpoints |
| **ML Inference Time** | <200ms | LSTM + GNN combined |
| **What-If Analysis Time** | <3 seconds | Full scenario simulation |
| **Graph Rendering** | <1 second | Knowledge graph force layout |

### 10. Data Schemas

#### Frontend State Management

```
Dashboard State Schema:
{
  "realTimeMetrics": {
    "furnaceTemperature": number,
    "furnacePressure": number,
    "meltLevel": number,
    "beltSpeed": number,
    "moldTemperature": number,
    "qualityRate": number,
    "defectCountHourly": number,
    "productionRate": number,
    "lastUpdate": ISO string
  },
  "defectDistribution": [
    { "type": string, "count": number, "percentage": number, "severity": string }
  ],
  "aiRecommendations": [
    {
      "id": UUID,
      "action": string,
      "priority": string,
      "confidence": number,
      "expectedImpact": string,
      "source": "LSTM" | "GNN" | "KnowledgeGraph"
    }
  ],
  "notifications": [
    { ...notification object... }
  ],
  "modelExplainability": {
    "featureImportance": [ { "feature": string, "importance": number } ],
    "shapValues": { ...SHAP analysis data... },
    "lastUpdate": ISO string
  },
  "pipelineMetrics": {
    "lstmLatency": number,
    "gnnLatency": number,
    "vitLatency": number,
    "throughput": number,
    "accuracy": number,
    "drift": number
  }
}
```

### 11. Security Considerations

| Concern | Mitigation Strategy |
|---------|--------------------|
| **WebSocket Authentication** | Implement token-based auth in connection handshake |
| **API Rate Limiting** | 100 requests/minute per client IP |
| **Input Validation** | Validate all parameter ranges on backend |
| **CORS Configuration** | Whitelist only frontend origin |
| **Sensitive Data** | Never log production parameters, use anonymization |

### 12. Configuration Management

**Environment Variables Required**

```
Frontend (.env):
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000/ws/realtime
REACT_APP_ENABLE_SOUND_ALERTS=true
REACT_APP_NOTIFICATION_RETENTION_HOURS=24

Backend (environment or .env):
WEBSOCKET_ALLOWED_ORIGINS=["http://localhost:3000","http://frontend:3000"]
DEFECT_CONFIDENCE_THRESHOLD=0.7
SHADOW_MODE_PREDICTION_WINDOW=300
NOTIFICATION_RETENTION_DAYS=7
ML_MODEL_PATHS_LSTM=./models/lstm_predictor/lstm_model.onnx
ML_MODEL_PATHS_GNN=./models/gnn_sensor_network/gnn_model.onnx
```

## Success Criteria

### Functional Requirements

- [ ] WebSocket connection establishes without 403 errors
- [ ] Defect alerts appear in real-time on dashboard
- [ ] All metrics update dynamically (no hardcoded values)
- [ ] Notifications tab displays and categorizes alerts
- [ ] Analytics tab provides statistical insights
- [ ] Digital Twin supports Shadow Mode and What-If analysis
- [ ] Knowledge Graph displays ML-driven causes and recommendations
- [ ] Model explainability shows SHAP analysis
- [ ] Pipeline metrics reflect actual system performance

### Non-Functional Requirements

- [ ] Dashboard loads in <2 seconds
- [ ] WebSocket latency <100ms
- [ ] API response times <500ms (95th percentile)
- [ ] No memory leaks after 24h continuous operation
- [ ] Supports 100 concurrent WebSocket connections
- [ ] ML inference completes in <200ms
- [ ] What-If analysis in <3 seconds

## Appendix A: Detailed File Analysis and Modification Plan

### Critical Issues Found in Current Implementation

#### 1. `frontend/src/components/AdvancedDashboard.tsx` (Lines 1-1553)

**Hardcoded Data Locations:**

| Line Range | Issue | Current Implementation | Required Change |
|------------|-------|------------------------|------------------|
| 88-94 | Defect distribution hardcoded | Static array with fixed values | Replace with `defectDistribution` from WebSocket aggregation |
| 96-104 | Performance data hardcoded | Static time series data | Replace with real-time data from `parameter_update` messages |
| 106-139 | Real-time metrics hardcoded | Fixed sensor values | Connect to WebSocket `furnace`, `forming` objects |
| 148-160 | AI recommendations hardcoded | Static recommendation text | Replace with `/api/recommendations` endpoint response |

**Specific Code Modifications Required:**

```
Line 264-265 (useDashboardData hook usage):
CURRENT:
const { data, loading, error } = useDashboardData();

REQUIRED:
const { data, loading, error } = useDashboardData();
const { wsData, isConnected } = useWebSocketStream(); // New hook

// Merge WebSocket data with API data
const mergedData = React.useMemo(() => {
  if (!data || !wsData) return data;
  return {
    ...data,
    realTimeMetrics: wsData.parameters || data.realTimeMetrics,
    defectDistribution: wsData.defectAggregation || data.defectDistribution,
    aiRecommendations: wsData.recommendations || data.aiRecommendations
  };
}, [data, wsData]);
```

**Performance Trend Chart Enhancement (Lines 571-668):**

The `PerformanceTrendChart` component uses static data. Needs dynamic update:

```
Add state management:
const [performanceHistory, setPerformanceHistory] = useState<PerformanceDataPoint[]>([]);

WebSocket message handler:
if (message.type === 'parameter_update') {
  const newPoint = {
    time: new Date(message.timestamp).toLocaleTimeString(),
    quality: message.quality_metrics.current_quality_rate,
    defects: message.quality_metrics.defect_count_hourly
  };
  setPerformanceHistory(prev => [...prev.slice(-20), newPoint]); // Keep last 20 points
}
```

#### 2. `frontend/src/hooks/useDashboardData.ts` (Lines 1-325)

**Critical WebSocket Issues:**

| Line | Issue | Fix Required |
|------|-------|-------------|
| 271 | WebSocket URL construction may fail | Add validation: `const wsUrl = API_BASE_URL ? API_BASE_URL.replace('http', 'ws') + '/ws/realtime' : 'ws://localhost:8000/ws/realtime';` |
| 279-301 | Limited message type handling | Add handlers for `defect_alert`, `ml_prediction`, `system_health` |
| 293 | Only updates quality metrics | Need to update all dashboard sections |

**Required Message Handlers:**

```
Line 279: Expand onMessage handler:

ws.onmessage = (event) => {
  try {
    const message = JSON.parse(event.data);
    
    switch(message.type) {
      case 'defect_alert':
        handleDefectAlert(message);
        break;
      case 'parameter_update':
        handleParameterUpdate(message);
        break;
      case 'ml_prediction':
        handleMLPrediction(message);
        break;
      case 'system_health':
        handleSystemHealth(message);
        break;
    }
  } catch (err) {
    console.error('Error parsing WebSocket message:', err);
  }
};

function handleDefectAlert(message: any) {
  setData(prevData => {
    if (!prevData) return prevData;
    
    // Add to defect distribution
    const updatedDistribution = [...prevData.defectDistribution];
    const defectIndex = updatedDistribution.findIndex(d => d.name === message.defect_type);
    if (defectIndex >= 0) {
      updatedDistribution[defectIndex].value += 1;
    }
    
    // Increment defect count
    return {
      ...prevData,
      defectDistribution: updatedDistribution,
      kpiData: {
        ...prevData.kpiData,
        defectCount: prevData.kpiData.defectCount + 1
      }
    };
  });
  
  // Trigger notification
  addNotification({
    id: message.timestamp,
    type: 'defect',
    severity: message.severity,
    message: `${message.defect_type} detected at ${message.location.line}`,
    timestamp: message.timestamp
  });
}
```

#### 3. `frontend/src/components/DigitalTwin3D.tsx` (Lines 1-374)

**Missing Functionality:**

| Feature | Current State | Required Implementation |
|---------|--------------|-------------------------|
| Shadow Mode | Not implemented | Add split-screen view with real vs predicted comparison |
| What-If Analysis | Not implemented | Add parameter adjustment panel with impact analysis |
| Dynamic Updates | Polls every 5 seconds | Should use WebSocket for real-time updates |
| Defect Visualization | Basic particles | Need severity-based coloring and count accuracy |

**Line 277-305: Inadequate Data Fetching**

Replace polling with WebSocket subscription:

```
REMOVE: Lines 277-305 (useEffect with polling)

ADD: WebSocket integration

import { useWebSocketStream } from '../hooks/useWebSocketStream';

const DigitalTwin3D = () => {
  const { wsData, isConnected } = useWebSocketStream();
  const [shadowModeActive, setShadowModeActive] = useState(false);
  const [whatIfMode, setWhatIfMode] = useState(false);
  const [whatIfParams, setWhatIfParams] = useState<any>(null);
  
  // Update from WebSocket
  useEffect(() => {
    if (wsData?.parameters) {
      setFurnaceTemp(wsData.parameters.furnace.temperature);
      setMoldTemp(wsData.parameters.forming.mold_temperature);
      setBeltSpeed(wsData.parameters.forming.belt_speed);
    }
    if (wsData?.defectAggregation) {
      const defectMap = wsData.defectAggregation.reduce((acc, d) => {
        acc[d.type] = d.count / 100; // Normalize
        return acc;
      }, {});
      setDefects(defectMap);
    }
  }, [wsData]);
```

**Add Shadow Mode UI (after line 370):**

```
{shadowModeActive && (
  <Paper
    sx={{
      position: 'absolute',
      top: 16,
      left: 16,
      p: 2,
      minWidth: 300,
      bgcolor: 'rgba(0,0,0,0.8)',
      borderRadius: 2
    }}
  >
    <Typography variant="h6" color="primary">Shadow Mode Active</Typography>
    <Divider sx={{ my: 1 }} />
    <Box sx={{ display: 'flex', flexDirection: 'column', gap: 1 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="body2">Real Temperature:</Typography>
        <Typography variant="body2" fontWeight="bold">{furnaceTemp}°C</Typography>
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="body2">Predicted Temperature:</Typography>
        <Typography variant="body2" fontWeight="bold" color="info.main">
          {shadowModeData?.predicted_state?.furnace?.temperature || 'N/A'}°C
        </Typography>
      </Box>
      <Box sx={{ display: 'flex', justifyContent: 'space-between' }}>
        <Typography variant="body2">Deviation:</Typography>
        <Typography 
          variant="body2" 
          fontWeight="bold"
          color={Math.abs(shadowModeData?.deviations?.temperature_deviation || 0) > 20 ? 'error' : 'success.main'}
        >
          {shadowModeData?.deviations?.temperature_deviation?.toFixed(1) || 0}°C
        </Typography>
      </Box>
      <LinearProgress 
        variant="determinate" 
        value={shadowModeData?.prediction_accuracy || 0} 
        sx={{ mt: 2 }}
      />
      <Typography variant="caption" align="center">
        Prediction Accuracy: {shadowModeData?.prediction_accuracy?.toFixed(1) || 0}%
      </Typography>
    </Box>
  </Paper>
)}
```

#### 4. `frontend/src/components/KnowledgeGraph.tsx` (Lines 1-1431)

**Hardcoded Data Issues:**

| Line Range | Problem | Solution |
|------------|---------|----------|
| 332-357 | Hardcoded causes array | Fetch from `/knowledge-graph/causes/${defectType}` |
| 359-382 | Hardcoded recommendations | Fetch from `/knowledge-graph/recommendations/${defectType}` |
| 384-402 | Static subgraph nodes | Fetch from `/knowledge-graph/subgraph/${defectType}` |
| 408-449 | Functions don't call APIs | Implement actual API calls |

**Line 408-421: Replace Mock API Call**

```
REMOVE:
const fetchCauses = async () => {
  setLoading(true);
  setError(null);
  try {
    await new Promise(resolve => setTimeout(resolve, 1000));
  } catch (err) {
    setError('Ошибка при загрузке причин дефекта');
  } finally {
    setLoading(false);
  }
};

REPLACE WITH:
const fetchCauses = async () => {
  setLoading(true);
  setError(null);
  try {
    const response = await fetch(
      `${API_BASE_URL}/knowledge-graph/causes/${defectType}?min_confidence=${minConfidence}`
    );
    if (!response.ok) throw new Error('Failed to fetch causes');
    const data = await response.json();
    
    setCauses(data.causes.map((cause: any) => ({
      cause: cause.cause || cause.parameter,
      confidence: cause.confidence,
      strength: cause.strength,
      observations: cause.observations || 0,
      evidence: cause.evidence || [],
      cause_type: cause.cause_type || 'UNKNOWN'
    })));
  } catch (err) {
    setError('Ошибка при загрузке причин дефекта');
    console.error(err);
  } finally {
    setLoading(false);
  }
};
```

**Line 423-435: Implement Real Recommendations Fetch**

```
const fetchRecommendations = async () => {
  setLoading(true);
  setError(null);
  try {
    const response = await fetch(
      `${API_BASE_URL}/knowledge-graph/recommendations/${defectType}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ parameter_values: parameterValues })
      }
    );
    if (!response.ok) throw new Error('Failed to fetch recommendations');
    const data = await response.json();
    
    setRecommendations(data.recommendations.map((rec: any) => ({
      parameter: rec.parameter,
      current_value: rec.current_value,
      target_value: rec.target_value,
      unit: rec.unit,
      action: rec.action,
      confidence: rec.confidence,
      strength: rec.strength,
      priority: rec.priority,
      expected_impact: rec.expected_impact
    })));
  } catch (err) {
    setError('Ошибка при загрузке рекомендаций');
    console.error(err);
  } finally {
    setLoading(false);
  }
};
```

**Line 437-449: Implement Subgraph Fetch**

```
const fetchSubgraph = async () => {
  setLoading(true);
  setError(null);
  try {
    const response = await fetch(
      `${API_BASE_URL}/knowledge-graph/subgraph/${defectType}?max_depth=2`
    );
    if (!response.ok) throw new Error('Failed to fetch subgraph');
    const data = await response.json();
    
    setSubgraph({
      defect: data.defect,
      nodes: data.nodes || [],
      edges: data.edges || []
    });
  } catch (err) {
    setError('Ошибка при загрузке графа знаний');
    console.error(err);
  } finally {
    setLoading(false);
  }
};
```

#### 5. `backend/fastapi_backend.py` (Lines 1-1451)

**WebSocket 403 Error Root Cause:**

| Line | Issue | Fix |
|------|-------|-----|
| 316-322 | CORS allows all origins but WebSocket not configured | Add WebSocket-specific CORS handling |
| Missing | No WebSocket connection validation | Add connection acceptance logic |
| Missing | No `/ws/realtime` endpoint | Implement WebSocket endpoint |

**Add After Line 322:**

```
# WebSocket endpoint for real-time data streaming
@app.websocket("/ws/realtime")
async def websocket_realtime(websocket: WebSocket):
    """
    WebSocket endpoint for real-time production data streaming
    Broadcasts: defect alerts, parameter updates, ML predictions
    """
    # Accept connection (fixes 403 error)
    await app_state.ws_manager.connect(websocket, {"type": "realtime_client"})
    
    # Send initial connection confirmation
    await websocket.send_json({
        "type": "connection_established",
        "timestamp": datetime.utcnow().isoformat(),
        "message": "Connected to real-time data stream"
    })
    
    try:
        while True:
            # Keep connection alive and wait for client messages
            data = await websocket.receive_text()
            
            # Echo back or handle client requests
            if data == "ping":
                await websocket.send_json({
                    "type": "pong",
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        app_state.ws_manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        app_state.ws_manager.disconnect(websocket)
```

**Add Missing Digital Twin Endpoints (after line 400):**

```
@app.get("/api/digital-twin/state")
async def get_digital_twin_state():
    """
    Get current digital twin state
    Returns furnace parameters, forming parameters, quality metrics, defects
    """
    try:
        # Get state from system integrator's digital twin
        if (hasattr(app_state.system_integrator, 'system_integrator') and
            app_state.system_integrator.system_integrator and
            hasattr(app_state.system_integrator.system_integrator, 'digital_twin')):
            
            digital_twin = app_state.system_integrator.system_integrator.digital_twin
            state = digital_twin.get_current_state()
            
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "data": state
            }
        else:
            # Return synthetic data if digital twin not available
            return {
                "status": "success",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "furnace": {
                        "temperature": 1520 + np.random.normal(0, 10),
                        "pressure": 15 + np.random.normal(0, 2),
                        "melt_level": 2500 + np.random.normal(0, 50)
                    },
                    "forming": {
                        "belt_speed": 150 + np.random.normal(0, 5),
                        "mold_temp": 320 + np.random.normal(0, 5),
                        "pressure": 50 + np.random.normal(0, 3)
                    },
                    "quality_score": 0.85 + np.random.normal(0, 0.05),
                    "defects": {
                        "crack": np.random.uniform(0, 0.2),
                        "bubble": np.random.uniform(0, 0.15),
                        "chip": np.random.uniform(0, 0.1)
                    }
                }
            }
    except Exception as e:
        logger.error(f"Error getting digital twin state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/digital-twin/shadow-mode/start")
async def start_shadow_mode(prediction_window: int = 300):
    """
    Start shadow mode simulation
    Runs parallel prediction to compare with real production
    """
    try:
        if app_state.shadow_mode:
            session_id = await app_state.shadow_mode.start_shadow_mode(
                prediction_window_seconds=prediction_window
            )
            return {
                "status": "active",
                "session_id": session_id,
                "prediction_window": prediction_window
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="Shadow mode not available (Digital Twin not initialized)"
            )
    except Exception as e:
        logger.error(f"Error starting shadow mode: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/digital-twin/shadow-mode/state")
async def get_shadow_mode_state():
    """
    Get shadow mode comparison: real vs predicted
    """
    try:
        if app_state.shadow_mode:
            comparison = await app_state.shadow_mode.get_comparison()
            return comparison
        else:
            raise HTTPException(status_code=503, detail="Shadow mode not active")
    except Exception as e:
        logger.error(f"Error getting shadow mode state: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/digital-twin/what-if/analyze")
async def analyze_what_if_scenario(parameter_changes: Dict[str, float]):
    """
    Analyze impact of parameter changes using What-If analyzer
    """
    try:
        if app_state.what_if_analyzer:
            # Convert parameter names to ParameterType enum
            from simulation.what_if_analyzer import ParameterType
            
            typed_changes = {}
            for param_name, value in parameter_changes.items():
                if param_name == "furnace_temperature":
                    typed_changes[ParameterType.FURNACE_TEMPERATURE] = value
                elif param_name == "belt_speed":
                    typed_changes[ParameterType.BELT_SPEED] = value
                elif param_name == "mold_temperature":
                    typed_changes[ParameterType.MOLD_TEMPERATURE] = value
            
            # Analyze impact
            impact = app_state.what_if_analyzer.analyze_multi_parameter_optimization(
                typed_changes
            )
            
            return {
                "status": "success",
                "impact_analysis": {
                    "defect_rate_change": impact.defect_rate_change,
                    "quality_score_impact": impact.quality_score_impact,
                    "production_rate_impact": impact.production_rate_impact,
                    "energy_consumption_change": impact.energy_consumption_change,
                    "time_to_effect_minutes": impact.time_to_effect_minutes,
                    "risk_level": impact.risk_level,
                    "warnings": impact.warnings,
                    "recommendations": impact.recommendations
                }
            }
        else:
            raise HTTPException(
                status_code=503,
                detail="What-If analyzer not available"
            )
    except Exception as e:
        logger.error(f"Error analyzing what-if scenario: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/notifications/active")
async def get_active_notifications(
    priority: Optional[str] = None,
    category: Optional[str] = None
):
    """
    Get active notifications with optional filtering
    """
    try:
        # Get notifications from cache or database
        notifications = app_state.cache.get("notifications", [])
        
        # Filter by priority
        if priority:
            notifications = [n for n in notifications if n.get("priority") == priority]
        
        # Filter by category
        if category:
            notifications = [n for n in notifications if n.get("category") == category]
        
        # Get unacknowledged count
        unacknowledged_count = len([n for n in notifications if not n.get("acknowledged", False)])
        
        return {
            "notifications": notifications,
            "unacknowledged_count": unacknowledged_count,
            "total_count": len(notifications)
        }
    except Exception as e:
        logger.error(f"Error getting notifications: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/notifications/{notification_id}/acknowledge")
async def acknowledge_notification(notification_id: str):
    """
    Mark notification as acknowledged
    """
    try:
        notifications = app_state.cache.get("notifications", [])
        
        for notification in notifications:
            if notification.get("id") == notification_id:
                notification["acknowledged"] = True
                notification["acknowledged_at"] = datetime.utcnow().isoformat()
                break
        
        app_state.cache["notifications"] = notifications
        
        return {"status": "acknowledged", "notification_id": notification_id}
    except Exception as e:
        logger.error(f"Error acknowledging notification: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

#### 6. `streaming_pipeline/websocket_broadcaster.py`

**Missing Defect Alert Pipeline:**

The current implementation doesn't have a defect detection and broadcasting mechanism. Need to add:

```
After class initialization, add:

async def broadcast_defect_alert(
    self,
    defect_type: str,
    severity: str,
    location: Dict[str, Any],
    confidence: float,
    ml_source: str = "LSTM"
):
    """
    Broadcast defect alert to all WebSocket clients
    
    Args:
        defect_type: Type of defect (crack, bubble, chip, etc.)
        severity: LOW, MEDIUM, HIGH, CRITICAL
        location: Dictionary with line and position_mm
        confidence: Detection confidence (0.0-1.0)
        ml_source: Source ML model (LSTM, GNN, VisionTransformer)
    """
    message = {
        "type": "defect_alert",
        "timestamp": datetime.utcnow().isoformat(),
        "defect_type": defect_type,
        "severity": severity,
        "location": location,
        "confidence": confidence,
        "ml_source": ml_source,
        "recommended_actions": self._get_recommended_actions(defect_type)
    }
    
    await self.broadcast(message)
    logger.info(f"⚠️ Alert broadcasted: {defect_type} detected on {location['line']}")


def _get_recommended_actions(self, defect_type: str) -> List[str]:
    """
    Get recommended actions for defect type
    """
    recommendations_map = {
        "crack": [
            "Reduce furnace temperature by 20°C",
            "Decrease cooling rate",
            "Check annealing profile"
        ],
        "bubble": [
            "Increase furnace temperature",
            "Check gas composition",
            "Reduce forming speed"
        ],
        "chip": [
            "Reduce forming pressure",
            "Check mold condition",
            "Adjust belt speed"
        ],
        "stain": [
            "Clean furnace",
            "Check raw material quality",
            "Adjust temperature profile"
        ],
        "cloudiness": [
            "Stabilize furnace temperature",
            "Check humidity levels",
            "Adjust cooling rate"
        ],
        "deformation": [
            "Reduce temperature",
            "Decrease forming pressure",
            "Slow down production speed"
        ]
    }
    return recommendations_map.get(defect_type, ["Monitor closely", "Contact supervisor"])


async def broadcast_parameter_update(
    self,
    furnace_data: Dict[str, float],
    forming_data: Dict[str, float],
    quality_metrics: Dict[str, float]
):
    """
    Broadcast parameter update message
    """
    # Calculate trends
    furnace_trend = self._calculate_trend(
        furnace_data.get("temperature", 0),
        "furnace_temperature"
    )
    
    message = {
        "type": "parameter_update",
        "timestamp": datetime.utcnow().isoformat(),
        "furnace": {
            **furnace_data,
            "temperature_trend": furnace_trend
        },
        "forming": forming_data,
        "quality_metrics": quality_metrics
    }
    
    await self.broadcast(message)


def _calculate_trend(self, current_value: float, parameter_name: str) -> str:
    """
    Calculate trend direction for parameter
    """
    history_key = f"{parameter_name}_history"
    
    if not hasattr(self, '_parameter_history'):
        self._parameter_history = {}
    
    if history_key not in self._parameter_history:
        self._parameter_history[history_key] = []
    
    history = self._parameter_history[history_key]
    history.append(current_value)
    
    # Keep last 10 values
    if len(history) > 10:
        history.pop(0)
    
    # Calculate trend
    if len(history) < 3:
        return "stable"
    
    recent_avg = sum(history[-3:]) / 3
    older_avg = sum(history[-6:-3]) / 3 if len(history) >= 6 else recent_avg
    
    diff = recent_avg - older_avg
    
    if abs(diff) < 2:  # Threshold for stability
        return "stable"
    elif diff > 0:
        return "rising"
    else:
        return "falling"
```

**Add Background Task for Synthetic Defect Generation:**

```
async def synthetic_defect_generation_loop(self):
    """
    Background task that generates synthetic defects based on production conditions
    """
    while True:
        try:
            await asyncio.sleep(10)  # Check every 10 seconds
            
            # Get current production state from data generator
            current_state = self.data_generator.get_latest_state()
            
            if current_state:
                # Calculate defect probabilities
                defect_probs = self._calculate_defect_probabilities(current_state)
                
                # Generate defects based on probabilities
                for defect_type, probability in defect_probs.items():
                    if np.random.random() < probability:
                        severity = self._determine_severity(probability)
                        confidence = min(0.99, probability + np.random.uniform(0, 0.1))
                        
                        await self.broadcast_defect_alert(
                            defect_type=defect_type,
                            severity=severity,
                            location={
                                "line": "Line_A",
                                "position_mm": np.random.uniform(0, 3000)
                            },
                            confidence=confidence,
                            ml_source="LSTM"
                        )
        
        except Exception as e:
            logger.error(f"Error in defect generation loop: {e}")
            await asyncio.sleep(5)


def _calculate_defect_probabilities(self, state: Dict) -> Dict[str, float]:
    """
    Calculate defect probabilities based on production parameters
    Using the model from design document
    """
    furnace_temp = state.get("furnace", {}).get("temperature", 1550)
    pressure = state.get("furnace", {}).get("pressure", 50)
    belt_speed = state.get("forming", {}).get("speed", 150)
    
    probabilities = {}
    
    # Crack probability
    crack_base = 0.05
    crack_temp_factor = max(0, (furnace_temp - 1580) / 10) * 0.02
    crack_pressure_factor = max(0, (45 - pressure) / 5) * 0.01
    crack_speed_factor = max(0, (belt_speed - 170) / 10) * 0.015
    probabilities["crack"] = min(0.95, crack_base + crack_temp_factor + crack_pressure_factor + crack_speed_factor)
    
    # Bubble probability
    bubble_base = 0.08
    bubble_temp_factor = max(0, (1450 - furnace_temp) / 10) * 0.025
    probabilities["bubble"] = min(0.95, bubble_base + bubble_temp_factor)
    
    # Chip probability
    chip_base = 0.03
    chip_temp_factor = abs(furnace_temp - 1550) / 10 * 0.01
    chip_pressure_factor = max(0, (pressure - 55) / 10) * 0.02
    probabilities["chip"] = min(0.95, chip_base + chip_temp_factor + chip_pressure_factor)
    
    # Stain probability
    stain_base = 0.04
    stain_temp_factor = max(0, (furnace_temp - 1600) / 10) * 0.015
    probabilities["stain"] = min(0.95, stain_base + stain_temp_factor)
    
    # Cloudiness probability
    cloudiness_base = 0.06
    cloudiness_temp_factor = abs(furnace_temp - 1550) / 10 * 0.02
    probabilities["cloudiness"] = min(0.95, cloudiness_base + cloudiness_temp_factor)
    
    # Deformation probability
    deformation_base = 0.02
    deformation_temp_factor = max(0, (furnace_temp - 1620) / 10) * 0.03
    deformation_pressure_factor = max(0, (pressure - 60) / 10) * 0.025
    probabilities["deformation"] = min(0.95, deformation_base + deformation_temp_factor + deformation_pressure_factor)
    
    return probabilities


def _determine_severity(self, probability: float) -> str:
    """
    Determine severity level based on defect probability
    """
    if probability >= 0.8:
        return "CRITICAL"
    elif probability >= 0.6:
        return "HIGH"
    elif probability >= 0.3:
        return "MEDIUM"
    else:
        return "LOW"
```

#### 7. `data_ingestion/synthetic_data_generator.py`

**Add Temperature Fluctuation Simulation:**

The current generator needs realistic parameter variations. Add to the `GlassProductionDataGenerator` class:

```
def __init__(self, seed: int = 42):
    # ... existing init ...
    
    # Add parameter simulation state
    self.current_hour = 0
    self.base_temperature = 1550
    self.temperature_drift = 0
    self.last_anomaly_time = 0
    self.maintenance_stop_until = 0
    

def simulate_furnace_temperature(self, current_time: float) -> float:
    """
    Simulate realistic furnace temperature with:
    - Diurnal pattern
    - Random drift
    - Anomaly events
    """
    hour_of_day = (current_time % 86400) / 3600  # Convert to hour of day
    
    # Base temperature
    temperature = self.base_temperature
    
    # Diurnal pattern: ±15°C sinusoidal variation
    diurnal_variation = 15 * np.sin(2 * np.pi * hour_of_day / 24)
    temperature += diurnal_variation
    
    # Random drift
    self.temperature_drift += np.random.normal(0, 0.5)
    self.temperature_drift *= 0.95  # Decay factor
    temperature += self.temperature_drift
    
    # Gaussian noise
    temperature += np.random.normal(0, 5)
    
    # Anomaly events: 1% probability per hour
    time_since_anomaly = current_time - self.last_anomaly_time
    if time_since_anomaly > 3600 and np.random.random() < 0.01:
        # Temperature spike
        anomaly_duration = np.random.uniform(600, 1800)  # 10-30 minutes
        temperature += 40  # +40°C spike
        self.last_anomaly_time = current_time
        logger.warning(f"🔥 Temperature anomaly: +40°C spike for {anomaly_duration/60:.1f} minutes")
    
    # Clamp to realistic range
    return np.clip(temperature, 1400, 1700)


def simulate_belt_speed(self, current_time: float) -> float:
    """
    Simulate belt speed with production load variations and maintenance stops
    """
    # Check if in maintenance stop
    if current_time < self.maintenance_stop_until:
        return 0.0
    
    # Random maintenance stop: 0.5% probability per hour
    if np.random.random() < 0.005 / 3600:  # Per second probability
        stop_duration = np.random.uniform(900, 2700)  # 15-45 minutes
        self.maintenance_stop_until = current_time + stop_duration
        logger.warning(f"🛑 Maintenance stop for {stop_duration/60:.1f} minutes")
        return 0.0
    
    # Base speed
    base_speed = 150
    
    # Production load factor (varies 140-170 m/min)
    load_variation = np.random.uniform(-10, 20)
    speed = base_speed + load_variation
    
    # Add noise
    speed += np.random.normal(0, 2)
    
    # Clamp to realistic range
    return np.clip(speed, 0, 200)


def generate_production_data(self) -> Dict[str, Any]:
    """
    Generate comprehensive production data with realistic variations
    """
    current_time = time.time()
    
    furnace_temp = self.simulate_furnace_temperature(current_time)
    belt_speed = self.simulate_belt_speed(current_time)
    
    # Mold temperature (correlated with furnace, lagged by 15 minutes)
    mold_temp_base = 320
    furnace_influence = (furnace_temp - 1550) * 0.6
    mold_temp = mold_temp_base + furnace_influence + np.random.normal(0, 5)
    mold_temp = np.clip(mold_temp, 250, 400)
    
    # Pressure (correlated with belt speed)
    pressure_base = 50
    speed_influence = (belt_speed - 150) * 0.4
    pressure = pressure_base + speed_influence + np.random.normal(0, 3)
    pressure = np.clip(pressure, 0, 100)
    
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "furnace": {
            "temperature": furnace_temp,
            "pressure": pressure,
            "melt_level": 2500 + np.random.normal(0, 50)
        },
        "forming": {
            "belt_speed": belt_speed,
            "mold_temp": mold_temp,
            "pressure": pressure
        },
        "quality_metrics": {
            "current_quality_rate": self._calculate_quality_rate(furnace_temp, belt_speed),
            "defect_count_hourly": np.random.poisson(45),
            "production_rate": belt_speed * 0.6  # units/hour approximation
        }
    }


def _calculate_quality_rate(self, furnace_temp: float, belt_speed: float) -> float:
    """
    Calculate quality rate based on parameters
    """
    # Optimal conditions
    temp_optimal = 1550
    speed_optimal = 150
    
    # Calculate deviations
    temp_deviation = abs(furnace_temp - temp_optimal) / 100
    speed_deviation = abs(belt_speed - speed_optimal) / 50
    
    # Base quality
    quality = 0.95
    
    # Penalties
    quality -= temp_deviation * 0.1
    quality -= speed_deviation * 0.05
    
    # Add noise
    quality += np.random.normal(0, 0.02)
    
    return np.clip(quality, 0.5, 0.99)
```

## Appendix B: New Components Implementation Plan

### Component 1: `frontend/src/hooks/useWebSocketStream.ts`

```typescript
import { useState, useEffect, useRef, useCallback } from 'react';

interface WebSocketData {
  parameters?: any;
  defectAggregation?: any[];
  mlPredictions?: any;
  recommendations?: any[];
  systemHealth?: any;
}

interface UseWebSocketStreamReturn {
  wsData: WebSocketData | null;
  isConnected: boolean;
  error: string | null;
  sendMessage: (message: any) => void;
}

export const useWebSocketStream = (): UseWebSocketStreamReturn => {
  const [wsData, setWsData] = useState<WebSocketData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const defectBufferRef = useRef<any[]>([]);

  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
  const WS_URL = API_BASE_URL.replace('http', 'ws') + '/ws/realtime';

  const connect = useCallback(() => {
    try {
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        console.log('✅ WebSocket connected');
        setIsConnected(true);
        setError(null);
        
        // Send ping every 30 seconds to keep connection alive
        const pingInterval = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send('ping');
          }
        }, 30000);
        
        ws.addEventListener('close', () => {
          clearInterval(pingInterval);
        });
      };

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data);
          
          switch (message.type) {
            case 'parameter_update':
              setWsData(prev => ({
                ...prev,
                parameters: message
              }));
              break;
              
            case 'defect_alert':
              // Add to defect buffer
              defectBufferRef.current.push(message);
              
              // Keep last 100 defects
              if (defectBufferRef.current.length > 100) {
                defectBufferRef.current.shift();
              }
              
              // Aggregate defects by type
              const aggregation = aggregateDefects(defectBufferRef.current);
              
              setWsData(prev => ({
                ...prev,
                defectAggregation: aggregation
              }));
              break;
              
            case 'ml_prediction':
              setWsData(prev => ({
                ...prev,
                mlPredictions: message.predictions
              }));
              break;
              
            case 'system_health':
              setWsData(prev => ({
                ...prev,
                systemHealth: message
              }));
              break;
          }
        } catch (err) {
          console.error('Error parsing WebSocket message:', err);
        }
      };

      ws.onerror = (err) => {
        console.error('❌ WebSocket error:', err);
        setError('WebSocket connection error');
      };

      ws.onclose = () => {
        console.log('WebSocket disconnected');
        setIsConnected(false);
        
        // Attempt reconnection after 5 seconds
        reconnectTimeoutRef.current = setTimeout(() => {
          console.log('Attempting to reconnect...');
          connect();
        }, 5000);
      };
    } catch (err) {
      console.error('Failed to create WebSocket:', err);
      setError('Failed to establish WebSocket connection');
    }
  }, [WS_URL]);

  useEffect(() => {
    connect();
    
    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
      if (reconnectTimeoutRef.current) {
        clearTimeout(reconnectTimeoutRef.current);
      }
    };
  }, [connect]);

  const sendMessage = useCallback((message: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message));
    }
  }, []);

  return { wsData, isConnected, error, sendMessage };
};

function aggregateDefects(defects: any[]): any[] {
  const counts: Record<string, number> = {};
  
  defects.forEach(defect => {
    const type = defect.defect_type;
    counts[type] = (counts[type] || 0) + 1;
  });
  
  const colorMap: Record<string, string> = {
    'crack': '#FF1744',
    'bubble': '#FFD700',
    'chip': '#00E676',
    'stain': '#00E5FF',
    'cloudiness': '#9D4EDD',
    'deformation': '#FF6B35'
  };
  
  return Object.entries(counts).map(([type, count]) => ({
    name: type.charAt(0).toUpperCase() + type.slice(1),
    type: type,
    value: count,
    color: colorMap[type] || '#FFFFFF'
  }));
}
```

### Component 2: `frontend/src/hooks/useNotifications.ts`

```typescript
import { useState, useEffect, useCallback } from 'react';
import { useWebSocketStream } from './useWebSocketStream';

interface Notification {
  id: string;
  timestamp: string;
  category: string;
  priority: 'CRITICAL' | 'HIGH' | 'MEDIUM' | 'LOW';
  title: string;
  message: string;
  source: string;
  actions?: any[];
  acknowledged: boolean;
  resolved: boolean;
}

export const useNotifications = () => {
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [unacknowledgedCount, setUnacknowledgedCount] = useState(0);
  const { wsData } = useWebSocketStream();
  const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

  // Fetch active notifications on mount
  useEffect(() => {
    fetchNotifications();
  }, []);

  // Listen for new defect alerts from WebSocket
  useEffect(() => {
    if (wsData?.parameters) {
      // Check for parameter anomalies
      checkParameterAnomalies(wsData.parameters);
    }
  }, [wsData]);

  const fetchNotifications = async () => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/notifications/active`);
      if (!response.ok) throw new Error('Failed to fetch notifications');
      const data = await response.json();
      
      setNotifications(data.notifications || []);
      setUnacknowledgedCount(data.unacknowledged_count || 0);
    } catch (err) {
      console.error('Error fetching notifications:', err);
    }
  };

  const addNotification = useCallback((notification: Partial<Notification>) => {
    const fullNotification: Notification = {
      id: notification.id || `notif_${Date.now()}`,
      timestamp: notification.timestamp || new Date().toISOString(),
      category: notification.category || 'system',
      priority: notification.priority || 'MEDIUM',
      title: notification.title || 'Notification',
      message: notification.message || '',
      source: notification.source || 'SYSTEM',
      acknowledged: false,
      resolved: false,
      ...notification
    };
    
    setNotifications(prev => [fullNotification, ...prev]);
    setUnacknowledgedCount(prev => prev + 1);
    
    // Play sound for critical notifications
    if (fullNotification.priority === 'CRITICAL') {
      playAlertSound();
    }
  }, []);

  const acknowledgeNotification = async (notificationId: string) => {
    try {
      await fetch(`${API_BASE_URL}/api/notifications/${notificationId}/acknowledge`, {
        method: 'POST'
      });
      
      setNotifications(prev => 
        prev.map(n => 
          n.id === notificationId ? { ...n, acknowledged: true } : n
        )
      );
      setUnacknowledgedCount(prev => Math.max(0, prev - 1));
    } catch (err) {
      console.error('Error acknowledging notification:', err);
    }
  };

  const dismissNotification = async (notificationId: string) => {
    try {
      await fetch(`${API_BASE_URL}/api/notifications/${notificationId}`, {
        method: 'DELETE'
      });
      
      setNotifications(prev => prev.filter(n => n.id !== notificationId));
    } catch (err) {
      console.error('Error dismissing notification:', err);
    }
  };

  const checkParameterAnomalies = (params: any) => {
    const { furnace, forming } = params;
    
    // Check furnace temperature
    if (furnace?.temperature > 1600) {
      addNotification({
        category: 'parameter_anomaly',
        priority: 'HIGH',
        title: 'High Furnace Temperature',
        message: `Furnace temperature is ${furnace.temperature.toFixed(1)}°C (above 1600°C)`,
        source: 'SENSOR'
      });
    }
    
    // Check pressure
    if (furnace?.pressure < 40) {
      addNotification({
        category: 'parameter_anomaly',
        priority: 'MEDIUM',
        title: 'Low Furnace Pressure',
        message: `Furnace pressure is ${furnace.pressure.toFixed(1)} kPa (below 40 kPa)`,
        source: 'SENSOR'
      });
    }
  };

  return {
    notifications,
    unacknowledgedCount,
    addNotification,
    acknowledgeNotification,
    dismissNotification,
    refreshNotifications: fetchNotifications
  };
};

function playAlertSound() {
  // Create audio context and play alert sound
  if (typeof Audio !== 'undefined') {
    const audio = new Audio('/alert.mp3');
    audio.play().catch(err => console.error('Error playing alert sound:', err));
  }
}
```

### Component 3: `frontend/src/components/NotificationsCenter.tsx`

```typescript
import React, { useState } from 'react';
import {
  Box,
  Typography,
  Paper,
  List,
  ListItem,
  ListItemText,
  ListItemIcon,
  Chip,
  IconButton,
  TextField,
  MenuItem,
  Select,
  FormControl,
  InputLabel,
  Badge,
  Tooltip,
  Divider
} from '@mui/material';
import {
  Notifications,
  CheckCircle,
  Warning,
  Error,
  Info,
  Close,
  FilterList,
  Search
} from '@mui/icons-material';
import { useNotifications } from '../hooks/useNotifications';

const NotificationsCenter: React.FC = () => {
  const {
    notifications,
    unacknowledgedCount,
    acknowledgeNotification,
    dismissNotification
  } = useNotifications();
  
  const [filterPriority, setFilterPriority] = useState<string>('ALL');
  const [filterCategory, setFilterCategory] = useState<string>('ALL');
  const [searchQuery, setSearchQuery] = useState('');

  // Filter notifications
  const filteredNotifications = notifications.filter(notification => {
    const matchesPriority = filterPriority === 'ALL' || notification.priority === filterPriority;
    const matchesCategory = filterCategory === 'ALL' || notification.category === filterCategory;
    const matchesSearch = searchQuery === '' || 
      notification.message.toLowerCase().includes(searchQuery.toLowerCase()) ||
      notification.title.toLowerCase().includes(searchQuery.toLowerCase());
    
    return matchesPriority && matchesCategory && matchesSearch;
  });

  const getPriorityIcon = (priority: string) => {
    switch (priority) {
      case 'CRITICAL': return <Error sx={{ color: '#FF1744' }} />;
      case 'HIGH': return <Warning sx={{ color: '#FFD700' }} />;
      case 'MEDIUM': return <Info sx={{ color: '#00E5FF' }} />;
      case 'LOW': return <CheckCircle sx={{ color: '#00E676' }} />;
      default: return <Info />;
    }
  };

  const getPriorityColor = (priority: string) => {
    switch (priority) {
      case 'CRITICAL': return '#FF1744';
      case 'HIGH': return '#FFD700';
      case 'MEDIUM': return '#00E5FF';
      case 'LOW': return '#00E676';
      default: return '#FFFFFF';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 3 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Badge badgeContent={unacknowledgedCount} color="error">
            <Notifications sx={{ fontSize: 40, color: '#0066FF' }} />
          </Badge>
          <Typography variant="h4" fontWeight={700}>
            Notifications Center
          </Typography>
        </Box>
        
        <Box sx={{ display: 'flex', gap: 2 }}>
          <TextField
            placeholder="Search notifications..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            size="small"
            InputProps={{
              startAdornment: <Search sx={{ mr: 1, color: 'text.secondary' }} />
            }}
          />
          
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Priority</InputLabel>
            <Select
              value={filterPriority}
              label="Priority"
              onChange={(e) => setFilterPriority(e.target.value)}
            >
              <MenuItem value="ALL">All</MenuItem>
              <MenuItem value="CRITICAL">Critical</MenuItem>
              <MenuItem value="HIGH">High</MenuItem>
              <MenuItem value="MEDIUM">Medium</MenuItem>
              <MenuItem value="LOW">Low</MenuItem>
            </Select>
          </FormControl>
          
          <FormControl size="small" sx={{ minWidth: 120 }}>
            <InputLabel>Category</InputLabel>
            <Select
              value={filterCategory}
              label="Category"
              onChange={(e) => setFilterCategory(e.target.value)}
            >
              <MenuItem value="ALL">All</MenuItem>
              <MenuItem value="defect">Defects</MenuItem>
              <MenuItem value="parameter_anomaly">Parameters</MenuItem>
              <MenuItem value="system">System</MenuItem>
              <MenuItem value="maintenance">Maintenance</MenuItem>
            </Select>
          </FormControl>
        </Box>
      </Box>

      <Paper sx={{ bgcolor: 'rgba(13, 27, 42, 0.6)', borderRadius: 2 }}>
        <List>
          {filteredNotifications.length === 0 ? (
            <ListItem>
              <ListItemText
                primary="No notifications"
                secondary="All clear!"
                sx={{ textAlign: 'center', py: 4 }}
              />
            </ListItem>
          ) : (
            filteredNotifications.map((notification, index) => (
              <React.Fragment key={notification.id}>
                <ListItem
                  sx={{
                    bgcolor: notification.acknowledged ? 'transparent' : 'rgba(0, 102, 255, 0.05)',
                    '&:hover': { bgcolor: 'rgba(0, 102, 255, 0.1)' }
                  }}
                >
                  <ListItemIcon>
                    {getPriorityIcon(notification.priority)}
                  </ListItemIcon>
                  
                  <ListItemText
                    primary={
                      <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                        <Typography variant="subtitle1" fontWeight={600}>
                          {notification.title}
                        </Typography>
                        <Chip
                          label={notification.priority}
                          size="small"
                          sx={{
                            bgcolor: getPriorityColor(notification.priority) + '20',
                            color: getPriorityColor(notification.priority),
                            fontWeight: 700
                          }}
                        />
                      </Box>
                    }
                    secondary={
                      <Box>
                        <Typography variant="body2" color="text.secondary">
                          {notification.message}
                        </Typography>
                        <Typography variant="caption" color="text.secondary">
                          {new Date(notification.timestamp).toLocaleString()} • {notification.source}
                        </Typography>
                      </Box>
                    }
                  />
                  
                  <Box sx={{ display: 'flex', gap: 1 }}>
                    {!notification.acknowledged && (
                      <Tooltip title="Acknowledge">
                        <IconButton
                          size="small"
                          onClick={() => acknowledgeNotification(notification.id)}
                          sx={{ color: '#00E676' }}
                        >
                          <CheckCircle />
                        </IconButton>
                      </Tooltip>
                    )}
                    <Tooltip title="Dismiss">
                      <IconButton
                        size="small"
                        onClick={() => dismissNotification(notification.id)}
                        sx={{ color: '#FF1744' }}
                      >
                        <Close />
                      </IconButton>
                    </Tooltip>
                  </Box>
                </ListItem>
                {index < filteredNotifications.length - 1 && <Divider />}
              </React.Fragment>
            ))
          )}
        </List>
      </Paper>
    </Box>
  );
};

export default NotificationsCenter;
```

## Summary of Critical Changes

**Total Files to Modify: 9**
**New Files to Create: 3**
**Estimated Implementation Time: 16-24 hours**

### Priority Order:

1. **Phase 1 (Critical - 4-6 hours)**: WebSocket fixes in `fastapi_backend.py` and `websocket_broadcaster.py`
2. **Phase 2 (High - 4-6 hours)**: Frontend hooks (`useWebSocketStream.ts`, `useNotifications.ts`) and dashboard integration
3. **Phase 3 (High - 3-4 hours)**: `NotificationsCenter.tsx` component
4. **Phase 4 (Medium - 3-4 hours)**: Digital Twin enhancements
5. **Phase 5 (Medium - 2-3 hours)**: Knowledge Graph API integration

All modifications maintain backward compatibility while eliminating hardcoded data and establishing ML-driven workflows throughout the system.

---

**Design Confidence Assessment: HIGH**

**Confidence Basis**:
1. **Clear Requirements**: All issues documented with specific technical gaps
2. **Proven Architecture**: Leverages existing working components (LSTM, GNN, WebSocket infrastructure)
3. **Incremental Approach**: Phased implementation reduces risk
4. **Testing Strategy**: Comprehensive integration scenarios defined
5. **Existing Foundation**: Backend APIs and ML models already functional

**Key Risk Factors**:
1. WebSocket scalability under load (mitigated by connection limits)
2. Real-time performance with complex 3D rendering (mitigated by throttling updates)
3. Neo4j query performance for large graphs (mitigated by depth limits)

**Next Steps**:
1. Review design with development team
2. Prioritize Phase 1 implementation (WebSocket fixes)
3. Set up integration testing environment
4. Begin iterative development following defined phases
