# Data Ingestion System for Glass Production

## Overview

The Data Ingestion System is responsible for collecting, routing, and processing data from various industrial sources in glass production facilities. It provides a unified interface for connecting to OPC UA servers, Modbus devices, MIK-1 camera systems, and MQTT brokers.

## Architecture

```
Data Ingestion System
├── Data Collector
│   ├── OPC UA Client
│   ├── Modbus Driver
│   ├── MIK-1 Camera Stream
│   └── MQTT Connector
├── Data Router
│   ├── Kafka Producer
│   ├── Feature Extractor
│   ├── Data Validator
│   └── Data Buffer
└── Configuration & Setup
    ├── System Configuration
    ├── Environment Setup
    └── Dependency Management
```

## Key Components

### 1. Data Collector
- **OPC UA Client**: Connects to industrial automation systems
- **Modbus Driver**: Communicates with PLCs and industrial devices
- **MIK-1 Camera Stream**: Processes visual inspection data
- **MQTT Connector**: Handles IoT device communications

### 2. Data Router
- **Kafka Producer**: Streams data to Kafka topics
- **Feature Extractor**: Computes real-time features from sensor data
- **Data Validator**: Validates and detects anomalies in incoming data
- **Data Buffer**: Temporarily stores data during system issues

### 3. Configuration & Setup
- **System Configuration**: Manages system-wide settings
- **Environment Setup**: Handles environment-specific configurations
- **Dependency Management**: Ensures all required packages are available

## Features

- **Multi-Protocol Support**: OPC UA, Modbus TCP/RTU, MQTT, Camera Streams
- **Real-Time Processing**: Low-latency data collection and routing
- **Fault Tolerance**: Automatic reconnection and data buffering
- **Scalable Architecture**: Horizontal scaling support
- **Monitoring & Metrics**: Built-in health checks and performance metrics
- **Security**: TLS/SSL support and authentication

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run setup
python setup.py
```

## Configuration

The system can be configured through:
1. **Configuration File**: `data_ingestion_config.json`
2. **Environment Variables**: `.env` file
3. **Command Line Arguments**: Runtime parameters

## Usage

### Basic Usage
```python
from data_ingestion.main import DataIngestionSystem

# Create and start system
system = DataIngestionSystem()
await system.initialize_system()
await system.start_system()
```

### Custom Configuration
```python
from data_ingestion.setup import DataIngestionSetup

# Load custom configuration
setup = DataIngestionSetup("custom_config.json")
config = setup.load_config()
```

## Data Flow

1. **Collection**: Data is collected from all configured sources
2. **Validation**: Incoming data is validated and checked for anomalies
3. **Routing**: Valid data is routed to appropriate destinations
4. **Processing**: Real-time features are extracted from sensor data
5. **Storage**: Data is streamed to Kafka for further processing

## Monitoring

The system provides:
- **Health Checks**: Periodic system health verification
- **Performance Metrics**: Collection rates, error rates, throughput
- **Logging**: Comprehensive logging with configurable levels
- **Alerts**: Critical issue notifications

## Security

- **Authentication**: Username/password and certificate-based auth
- **Encryption**: TLS/SSL support for all connections
- **Access Control**: Role-based access to system components

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black .
flake8 .
```

### Type Checking
```bash
mypy .
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For support, please open an issue on the GitHub repository or contact the development team.