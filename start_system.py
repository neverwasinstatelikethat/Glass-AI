#!/usr/bin/env python3
"""
Unified entry point for the Glass Production Predictive Analytics System.
This script starts all components in the correct order and manages their lifecycle.
"""

import asyncio
import logging
import os
import sys
from pathlib import Path

# Add the current directory to Python path
sys.path.append(str(Path(__file__).parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def start_backend_server(port=8000):
    """Start the FastAPI backend server"""
    try:
        # Import here to avoid issues with dependencies
        from backend.fastapi_backend import app
        import uvicorn
        
        logger.info(f"🚀 Starting FastAPI backend server on port {port}...")
        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    except Exception as e:
        logger.error(f"❌ Failed to start backend server: {e}")
        raise

async def start_system_components():
    """Start the core analytics system components"""
    try:
        # Import here to avoid issues with dependencies
        from unified_main import UnifiedGlassProductionSystem
        
        logger.info("🔧 Starting core analytics system...")
        system = UnifiedGlassProductionSystem()
        await system.start()
    except Exception as e:
        logger.error(f"❌ Failed to start core system: {e}")
        raise

async def main():
    """Main entry point that starts all system components"""
    logger.info("🔬 Glass Production Predictive Analytics System - Unified Startup")
    logger.info("=" * 60)
    
    # Try different ports if 8000 is in use
    ports_to_try = [8000, 8001, 8002, 8003]
    
    for port in ports_to_try:
        try:
            logger.info(f"Starting unified system with embedded backend on port {port}...")
            
            # Import and start the FastAPI backend which will initialize the unified system
            from backend.fastapi_backend import app
            import uvicorn
            
            logger.info(f"🚀 Starting unified system on http://0.0.0.0:{port}")
            logger.info(f"📄 API Documentation will be available at http://0.0.0.0:{port}/docs")
            
            # Start the server
            config = uvicorn.Config(
                "backend.fastapi_backend:app",
                host="0.0.0.0",
                port=port,
                log_level="info",
                reload=True  # Enable auto-reload for development
            )
            server = uvicorn.Server(config)
            await server.serve()
            
            # If we reach here, the server started successfully
            break
            
        except OSError as e:
            if "address already in use" in str(e).lower() or "[winerror 10048]" in str(e).lower():
                logger.warning(f"Port {port} is already in use, trying next port...")
                if port == ports_to_try[-1]:
                    logger.error("No available ports found. Please free up ports 8000-8003.")
                    raise
                continue
            else:
                raise
        except KeyboardInterrupt:
            logger.info("🛑 System interrupted by user")
            break
        except Exception as e:
            logger.error(f"💥 Unexpected error: {e}")
            sys.exit(1)

def run_with_docker_instructions():
    """Provide instructions for running with Docker"""
    print("\n🐳 To run with Docker (recommended for production):")
    print("1. Make sure Docker is installed and running")
    print("2. Run the following command in the project root directory:")
    print("   docker-compose up -d")
    print("3. Access the system at http://localhost:8000")
    print("4. Access the frontend at http://localhost:3000")

if __name__ == "__main__":
    # Check if Docker is preferred
    if len(sys.argv) > 1 and sys.argv[1] == "--docker":
        run_with_docker_instructions()
    else:
        print("🔬 Glass Production Predictive Analytics System")
        print("=" * 60)
        print("Starting unified system...")
        print("💡 Tip: Use '--docker' flag for Docker instructions")
        print()
        
        # Run the async main function
        try:
            asyncio.run(main())
        except KeyboardInterrupt:
            logger.info("🛑 System stopped by user")
        except Exception as e:
            logger.error(f"💥 System error: {e}")
            sys.exit(1)