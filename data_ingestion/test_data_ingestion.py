"""
Tests for Data Ingestion System
"""

import asyncio
import pytest
import json
from datetime import datetime
from unittest.mock import AsyncMock, Mock, patch

# Import system components
from data_ingestion.data_collector import DataCollector
from data_ingestion.data_router import DataRouter, DataBuffer
from data_ingestion.setup import DataIngestionSetup


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data for testing"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "production_line": "Line_A",
        "sensors": {
            "furnace_temperature": {"value": 1500.0, "status": "OK"},
            "furnace_pressure": {"value": 15.0, "status": "OK"},
            "melt_level": {"value": 2500.0, "status": "OK"}
        }
    }


@pytest.fixture
def sample_defect_data():
    """Sample defect data for testing"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "production_line": "Line_A",
        "defect_type": "bubble",
        "severity": "MEDIUM",
        "position": {"x": 100.0, "y": 50.0},
        "size_mm": 2.5,
        "confidence": 0.85
    }


class TestDataCollector:
    """Tests for DataCollector class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test DataCollector initialization"""
        collector = DataCollector(collection_interval=2.0, buffer_size=500)
        
        assert collector.collection_interval == 2.0
        assert collector.buffer_size == 500
        assert collector.running == False
    
    @pytest.mark.asyncio
    async def test_data_callback(self):
        """Test data callback functionality"""
        callback_called = False
        callback_data = None
        
        async def test_callback(data):
            nonlocal callback_called, callback_data
            callback_called = True
            callback_data = data
        
        collector = DataCollector(data_callback=test_callback)
        
        # Simulate calling the callback
        test_data = {"test": "data"}
        if collector.data_callback:
            await collector.data_callback(test_data)
        
        assert callback_called == True
        assert callback_data == test_data


class TestDataRouter:
    """Tests for DataRouter class"""
    
    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test DataRouter initialization"""
        router = DataRouter()
        
        assert router.running == False
        assert isinstance(router.routing_rules, dict)
        assert isinstance(router.default_routes, list)
    
    @pytest.mark.asyncio
    async def test_route_data(self, sample_sensor_data):
        """Test data routing functionality"""
        router = DataRouter()
        
        # Mock destination handlers
        router.destinations["kafka"] = AsyncMock()
        router.destinations["feature_extractor"] = AsyncMock()
        router.destinations["validator"] = AsyncMock()
        
        # Test routing
        result = await router.route_data(sample_sensor_data, "sensor_data")
        
        # Verify routing occurred
        assert result == True
        assert router.destinations["kafka"]._route_to_kafka.called
        assert router.destinations["feature_extractor"]._route_to_feature_extractor.called
        assert router.destinations["validator"]._route_to_validator.called


class TestDataBuffer:
    """Tests for DataBuffer class"""
    
    @pytest.mark.asyncio
    async def test_buffer_operations(self):
        """Test buffer add and retrieve operations"""
        buffer = DataBuffer(max_size=5)
        
        # Add data
        test_data = {"test": "data"}
        await buffer.add_data(test_data)
        
        # Check buffer stats
        stats = buffer.get_buffer_stats()
        assert stats["current_size"] == 1
        assert stats["total_added"] == 1
        
        # Retrieve data
        retrieved = await buffer.get_buffered_data(limit=1)
        assert len(retrieved) == 1
        assert retrieved[0]["data"] == test_data
        
        # Check stats after retrieval
        stats = buffer.get_buffer_stats()
        assert stats["current_size"] == 0
        assert stats["total_removed"] == 1
    
    @pytest.mark.asyncio
    async def test_buffer_limit(self):
        """Test buffer size limit"""
        buffer = DataBuffer(max_size=3)
        
        # Add more data than buffer can hold
        for i in range(5):
            await buffer.add_data({"item": i})
        
        # Check buffer is at capacity
        stats = buffer.get_buffer_stats()
        assert stats["current_size"] == 3
        assert stats["peak_size"] == 3


class TestDataIngestionSetup:
    """Tests for DataIngestionSetup class"""
    
    def test_default_config(self):
        """Test default configuration loading"""
        setup = DataIngestionSetup()
        config = setup._load_default_config()
        
        # Check required sections exist
        assert "system" in config
        assert "collector" in config
        assert "router" in config
        assert "kafka" in config
        
        # Check default values
        assert config["system"]["name"] == "GlassProductionDataIngestion"
        assert config["collector"]["collection_interval"] == 1.0
    
    def test_config_validation(self):
        """Test configuration validation"""
        setup = DataIngestionSetup()
        config = setup._load_default_config()
        
        # Validate default config
        is_valid = setup.validate_config()
        assert is_valid == True
    
    def test_environment_detection(self):
        """Test environment detection"""
        setup = DataIngestionSetup()
        environment = setup._detect_environment()
        
        # Should default to development
        assert environment in ["development", "docker", "kubernetes", "production"]


# Integration tests
@pytest.mark.asyncio
async def test_end_to_end_flow(sample_sensor_data, sample_defect_data):
    """Test end-to-end data flow"""
    # This would test the complete flow from collection to routing
    # For now, we'll test the components can be instantiated together
    
    # Initialize components
    collector = DataCollector()
    router = DataRouter()
    
    # Verify they can be created
    assert collector is not None
    assert router is not None
    
    # Test configuration
    setup = DataIngestionSetup()
    config = setup.load_config()
    assert config is not None


# Performance tests
@pytest.mark.asyncio
async def test_buffer_performance():
    """Test buffer performance with large datasets"""
    buffer = DataBuffer(max_size=1000)
    
    # Add many items quickly
    for i in range(100):
        await buffer.add_data({"item": i, "timestamp": datetime.utcnow().isoformat()})
    
    # Retrieve all items
    retrieved = await buffer.get_buffered_data(limit=100)
    
    assert len(retrieved) == 100
    assert buffer.get_buffer_stats()["total_added"] == 100


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])