---
trigger: always_on
alwaysApply: true
---
## CODING STANDARDS
**Python:**
- Full type hints (PEP 484) with strict mypy configuration
- Pydantic v2 models for all data validation
- Async/await patterns for I/O-bound operations
- Circuit breaker pattern for external service calls
- Structured logging with structlog (JSON format)
- Context managers for resource handling

**ML Code:**
- All models must include data validation, drift detection, fallback mechanisms
- LSTM configurations: 2-3 layers (128-256 units), dropout 0.2-0.3
- Performance metrics: Accuracy >85%, Recall >90%, Precision >80%
- Model versioning with MLflow integration

**Security:**
- Never hardcode credentials - use environment variables
- Parameterized queries only (no string formatting for SQL)
- Input validation for all API endpoints
- Rate limiting on public endpoints

**Testing:**
- pytest with 90%+ coverage requirement
- Async fixtures for database connections
- Mock external services in unit tests
- Performance tests for critical paths
- Integration tests for data pipeline

## ARCHITECTURAL PATTERNS
- Hexagonal architecture (ports/adapters)
- Repository pattern for data access
- Strategy pattern for recommendation engines
- Observer pattern for alert notifications
- Circuit breaker for external integrations
- Always use memories before coding or implementing anything

## TECH STACK REQUIREMENTS
**Core Languages:** Python 3.11 (strict), TypeScript for frontend
**ML Stack:** PyTorch/TensorFlow for LSTM, XGBoost/LightGBM, Prophet, scikit-learn
**Backend:** FastAPI (async), SQLAlchemy Core + ORM
**Data:** InfluxDB (time-series), PostgreSQL (metadata), Apache Kafka (streaming)
**Frontend:** React.js + TypeScript, MaterialUI, Plotly.js, WebSocket real-time updates
**Infrastructure:** Docker, Kubernetes, Prometheus/Grafana monitoring