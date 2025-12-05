"""
Example usage of the Data Ingestion System
Demonstrates how to use the system with sample data
"""

import sys
import os
from pathlib import Path

# Add parent directory to Python path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

import asyncio
import logging
from datetime import datetime
import numpy as np

# Import system components
from data_ingestion.setup import DataIngestionSetup

# Import only the classes we need, without hardware dependencies
try:
    from data_ingestion.data_router import DataRouter, DataBuffer
    ROUTER_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import DataRouter: {e}")
    ROUTER_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def main():
    """Main example function"""
    logger.info("🧪 Data Ingestion System Example")
    logger.info("=" * 40)
    
    # Setup configuration
    setup = DataIngestionSetup()
    config = setup.load_config()
    logger.info("✅ Configuration loaded")
    
    # Check dependencies
    dependencies = setup.check_dependencies()
    logger.info(f"📦 Dependencies: {sum(dependencies.values())}/{len(dependencies)} satisfied")
    
    # Show what we can demonstrate
    logger.info(f"🔄 Router available: {ROUTER_AVAILABLE}")
    
    if ROUTER_AVAILABLE:
        # Initialize data router
        router = DataRouter()
        logger.info("✅ Data router created")
        
        # Show routing statistics
        stats = await router.get_routing_stats()
        logger.info(f"📊 Initial routing statistics: {stats}")
        
        # Test DataBuffer
        buffer = DataBuffer(max_size=5)
        logger.info("✅ Data buffer created")
        
        # Add data to buffer
        test_data = {"test": "data", "timestamp": datetime.utcnow().isoformat()}
        await buffer.add_data(test_data)
        
        # Check buffer stats
        buffer_stats = buffer.get_buffer_stats()
        logger.info(f"📊 Buffer stats: {buffer_stats}")
        
        # Retrieve data from buffer
        retrieved = await buffer.get_buffered_data(limit=1)
        logger.info(f"📥 Retrieved {len(retrieved)} items from buffer")
    
    # Show connection information
    connection_info = setup.get_connection_info()
    logger.info(f"🔌 Configured sources: {list(connection_info['sources'].keys())}")
    
    # Show final system status
    logger.info(f"\n🏁 Example completed successfully!")
    logger.info("💡 In a real deployment, this system would:")
    logger.info("   • Connect to actual industrial equipment")
    logger.info("   • Collect real-time sensor data")
    logger.info("   • Process and validate data streams")
    logger.info("   • Route data to Kafka for analytics")
    logger.info("   • Extract features for ML models")
    logger.info("   • Handle system failures gracefully")


if __name__ == "__main__":
    asyncio.run(main())