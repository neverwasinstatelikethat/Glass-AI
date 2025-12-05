"""
Tests for Storage Layer Database Connections
"""

import asyncio
import pytest
import os
from datetime import datetime
from storage.influxdb_client import GlassInfluxDBClient
from storage.postgres_client import GlassPostgresClient


@pytest.fixture
def sample_sensor_data():
    """Sample sensor data for testing"""
    return {
        "timestamp": datetime.utcnow().isoformat(),
        "production_line": "Line_A",
        "sensors": {
            "furnace_temperature": {"value": 1520.5, "status": "OK"},
            "furnace_pressure": {"value": 15.2, "status": "OK"},
            "melt_level": {"value": 2500.0, "status": "OK"},
            "forming_belt_speed": 150.0
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


@pytest.fixture
def sample_quality_data():
    """Sample quality metrics data for testing"""
    return {
        "timestamp": datetime.utcnow(),
        "production_line": "Line_A",
        "total_units": 1000,
        "defective_units": 25,
        "quality_rate": 97.5
    }


class TestInfluxDBClient:
    """Tests for InfluxDB client"""
    
    @pytest.mark.asyncio
    async def test_influxdb_connection(self):
        """Test InfluxDB connection"""
        client = GlassInfluxDBClient()
        success = await client.connect()
        assert success, "Failed to connect to InfluxDB"
        
        # Close connection
        await client.close()
    
    @pytest.mark.asyncio
    async def test_write_sensor_data(self, sample_sensor_data):
        """Test writing sensor data to InfluxDB"""
        client = GlassInfluxDBClient()
        await client.connect()
        
        success = await client.write_sensor_data(sample_sensor_data)
        assert success, "Failed to write sensor data to InfluxDB"
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_write_defect_data(self, sample_defect_data):
        """Test writing defect data to InfluxDB"""
        client = GlassInfluxDBClient()
        await client.connect()
        
        success = await client.write_defect_data(sample_defect_data)
        assert success, "Failed to write defect data to InfluxDB"
        
        await client.close()


class TestPostgresClient:
    """Tests for PostgreSQL client"""
    
    @pytest.mark.asyncio
    async def test_postgres_connection(self):
        """Test PostgreSQL connection"""
        client = GlassPostgresClient()
        success = await client.connect()
        assert success, "Failed to connect to PostgreSQL"
        
        # Close connection
        await client.close()
    
    @pytest.mark.asyncio
    async def test_insert_quality_metrics(self, sample_quality_data):
        """Test inserting quality metrics to PostgreSQL"""
        client = GlassPostgresClient()
        await client.connect()
        
        success = await client.insert_quality_metrics(sample_quality_data)
        assert success, "Failed to insert quality metrics to PostgreSQL"
        
        await client.close()
    
    @pytest.mark.asyncio
    async def test_query_quality_metrics(self):
        """Test querying quality metrics from PostgreSQL"""
        client = GlassPostgresClient()
        await client.connect()
        
        # Query recent metrics
        metrics = await client.get_recent_quality_metrics("Line_A", limit=5)
        assert isinstance(metrics, list), "Query result should be a list"
        
        await client.close()


async def main():
    """Main test function"""
    print("🧪 Testing Storage Layer Database Connections...")
    
    # Test InfluxDB
    print("\n🔄 Testing InfluxDB...")
    influx_client = GlassInfluxDBClient()
    influx_success = await influx_client.connect()
    print(f"   InfluxDB Connection: {'✅ Success' if influx_success else '❌ Failed'}")
    
    if influx_success:
        # Test writing sensor data
        sensor_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "production_line": "Line_A",
            "sensors": {
                "furnace_temperature": 1520.5,
                "furnace_pressure": 15.2
            }
        }
        write_success = await influx_client.write_sensor_data(sensor_data)
        print(f"   Write Sensor Data: {'✅ Success' if write_success else '❌ Failed'}")
        
        await influx_client.close()
    
    # Test PostgreSQL
    print("\n🔄 Testing PostgreSQL...")
    postgres_client = GlassPostgresClient()
    postgres_success = await postgres_client.connect()
    print(f"   PostgreSQL Connection: {'✅ Success' if postgres_success else '❌ Failed'}")
    
    if postgres_success:
        # Test inserting quality metrics
        quality_data = {
            "timestamp": datetime.utcnow(),
            "production_line": "Line_A",
            "total_units": 1000,
            "defective_units": 25,
            "quality_rate": 97.5
        }
        insert_success = await postgres_client.insert_quality_metrics(quality_data)
        print(f"   Insert Quality Metrics: {'✅ Success' if insert_success else '❌ Failed'}")
        
        # Test querying quality metrics
        metrics = await postgres_client.get_recent_quality_metrics("Line_A", limit=1)
        print(f"   Query Quality Metrics: {'✅ Success' if isinstance(metrics, list) else '❌ Failed'}")
        
        await postgres_client.close()
    
    print("\n✅ Storage Layer Tests Complete!")


if __name__ == "__main__":
    asyncio.run(main())