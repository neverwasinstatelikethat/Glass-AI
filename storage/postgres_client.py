"""
PostgreSQL Client for Glass Production System
Handles metadata, configuration, and non-time-series data storage
"""

import os
import logging
import asyncio
import asyncpg
from typing import Dict, List, Any, Optional
from datetime import datetime

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GlassPostgresClient:
    """PostgreSQL client for storing metadata and configuration data"""
    
    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        connection_url: Optional[str] = None
    ):
        # If connection_url is provided, parse it
        if connection_url:
            from urllib.parse import urlparse
            parsed = urlparse(connection_url)
            self.host = host or parsed.hostname or "localhost"
            self.port = port or parsed.port or 5432
            self.database = database or parsed.path.lstrip('/') or "glass_production"
            self.user = user or parsed.username or "glass_admin"
            self.password = password or parsed.password or "glass_secure_pass"
        else:
            # Use individual parameters or environment variables
            self.host = host or os.getenv("POSTGRES_HOST", "localhost")
            self.port = port or int(os.getenv("POSTGRES_PORT", "5432"))
            self.database = database or os.getenv("POSTGRES_DB", "glass_production")
            self.user = user or os.getenv("POSTGRES_USER", "glass_admin")
            self.password = password or os.getenv("POSTGRES_PASSWORD", "glass_secure_pass")
        
        # Connection pool
        self.pool = None
        self.connected = False
        
        # Task requirements - defect types and severity levels
        self.defect_types = {
            "bubble": {"description": "Air bubbles in glass", "severity_level": "MEDIUM"},
            "crack": {"description": "Cracks in glass surface", "severity_level": "HIGH"},
            "scratch": {"description": "Surface scratches", "severity_level": "LOW"},
            "stain": {"description": "Color stains", "severity_level": "MEDIUM"},
            "deformation": {"description": "Shape deformation", "severity_level": "HIGH"},
            "chip": {"description": "Chipped edges", "severity_level": "MEDIUM"},
            "cloudiness": {"description": "Cloudy appearance", "severity_level": "LOW"}
        }
        
        # Task requirements - production line specifications
        self.production_lines = ["Line_A", "Line_B"]
        
    async def connect(self):
        """Establish connection pool to PostgreSQL"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                min_size=1,
                max_size=10
            )
            
            # Set connected flag before creating tables
            self.connected = True
            
            # Create tables if they don't exist
            await self._create_tables()
            
            logger.info("✅ Connected to PostgreSQL")
            return True
        except Exception as e:
            logger.warning(f"⚠️ Failed to connect to PostgreSQL (working in simulation mode): {e}")
            self.connected = False
            return False
    
    async def _create_tables(self):
        """Create required tables if they don't exist"""
        # If not connected, skip table creation
        if not self.pool:  # Remove self.connected check since it's set before this method is called
            return
            
        try:
            async with self.pool.acquire() as conn:
                # Production lines table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS production_lines (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(50) UNIQUE NOT NULL,
                        status VARCHAR(20) DEFAULT 'ACTIVE',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Equipment table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS equipment (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(100) NOT NULL,
                        type VARCHAR(50) NOT NULL,
                        production_line_id INTEGER REFERENCES production_lines(id),
                        serial_number VARCHAR(100) UNIQUE,
                        installation_date DATE,
                        status VARCHAR(20) DEFAULT 'OPERATIONAL',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Defect types table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS defect_types (
                        id SERIAL PRIMARY KEY,
                        name VARCHAR(50) UNIQUE NOT NULL,
                        description TEXT,
                        severity_level VARCHAR(20) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Quality metrics table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS quality_metrics (
                        id SERIAL PRIMARY KEY,
                        production_line_id INTEGER REFERENCES production_lines(id),
                        timestamp TIMESTAMP NOT NULL,
                        total_units INTEGER NOT NULL,
                        defective_units INTEGER NOT NULL,
                        quality_rate DECIMAL(5,2) NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Recommendations table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS recommendations (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        action_type VARCHAR(50) NOT NULL,
                        description TEXT NOT NULL,
                        urgency VARCHAR(20) NOT NULL,
                        expected_impact TEXT,
                        confidence DECIMAL(5,2),
                        status VARCHAR(20) DEFAULT 'PENDING',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Alerts table
                await conn.execute('''
                    CREATE TABLE IF NOT EXISTS alerts (
                        id SERIAL PRIMARY KEY,
                        timestamp TIMESTAMP NOT NULL,
                        alert_type VARCHAR(50) NOT NULL,
                        message TEXT NOT NULL,
                        priority VARCHAR(20) NOT NULL,
                        production_line_id INTEGER REFERENCES production_lines(id),
                        resolved BOOLEAN DEFAULT FALSE,
                        resolved_at TIMESTAMP,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                ''')
                
                # Insert default data if needed
                await self._insert_default_data(conn)
                
            logger.info("✅ PostgreSQL tables created/verified")
        except Exception as e:
            logger.error(f"❌ Error creating PostgreSQL tables: {e}")
            # If table creation fails, we should still be able to work in simulation mode
            # Don't set connected to False here as the connection itself is fine
    
    async def _insert_default_data(self, conn):
        """Insert default data if tables are empty"""
        try:
            # Check if production lines exist
            line_count = await conn.fetchval("SELECT COUNT(*) FROM production_lines")
            if line_count == 0:
                for line_name in self.production_lines:
                    await conn.execute(
                        "INSERT INTO production_lines (name) VALUES ($1)",
                        line_name
                    )
            
            # Check if defect types exist
            defect_count = await conn.fetchval("SELECT COUNT(*) FROM defect_types")
            if defect_count == 0:
                for defect_name, defect_info in self.defect_types.items():
                    await conn.execute('''
                        INSERT INTO defect_types (name, description, severity_level) VALUES
                        ($1, $2, $3)
                    ''', defect_name, defect_info["description"], defect_info["severity_level"])
                
        except Exception as e:
            logger.error(f"❌ Error inserting default data: {e}")
    
    async def insert_quality_metrics(self, data: Dict[str, Any]) -> bool:
        """Insert quality metrics data"""
        # If not connected, work in simulation mode
        if not self.connected:
            logger.debug("⏭️ PostgreSQL not connected, working in simulation mode")
            return True
            
        try:
            # Validate quality metrics data
            production_line = data.get("production_line", "Line_A")
            total_units = data.get("total_units", 0)
            defective_units = data.get("defective_units", 0)
            quality_rate = data.get("quality_rate", 0.0)
            
            # Validate production line
            if production_line not in self.production_lines:
                logger.warning(f"⚠️ Unknown production line: {production_line}")
            
            # Validate quality metrics ranges
            if total_units < 0:
                logger.warning(f"⚠️ Invalid total_units: {total_units}")
            
            if defective_units < 0 or defective_units > total_units:
                logger.warning(f"⚠️ Invalid defective_units: {defective_units}")
            
            if not (0 <= quality_rate <= 100):
                logger.warning(f"⚠️ Invalid quality_rate: {quality_rate}")
            
            async with self.pool.acquire() as conn:
                # Get production line ID
                line_record = await conn.fetchrow(
                    "SELECT id FROM production_lines WHERE name = $1",
                    production_line
                )
                line_id = line_record["id"] if line_record else None
                
                if not line_id:
                    logger.warning("⚠️ Production line not found")
                    return False
                
                # Insert quality metrics
                await conn.execute('''
                    INSERT INTO quality_metrics 
                    (production_line_id, timestamp, total_units, defective_units, quality_rate)
                    VALUES ($1, $2, $3, $4, $5)
                ''', line_id, data.get("timestamp"), total_units, defective_units, quality_rate)
                
                logger.debug("✅ Quality metrics inserted into PostgreSQL")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error inserting quality metrics: {e}")
            return False
    
    async def insert_recommendation(self, data: Dict[str, Any]) -> bool:
        """Insert recommendation data"""
        # If not connected, work in simulation mode
        if not self.connected:
            logger.debug("⏭️ PostgreSQL not connected, working in simulation mode")
            return True
            
        try:
            # Validate recommendation data
            action_type = data.get("action_type", "unknown")
            urgency = data.get("urgency", "LOW")
            confidence = data.get("confidence", 0.0)
            
            # Validate urgency levels
            if urgency not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                logger.warning(f"⚠️ Invalid urgency level: {urgency}")
            
            # Validate confidence range
            if not (0 <= confidence <= 1):
                logger.warning(f"⚠️ Invalid confidence value: {confidence}")
            
            async with self.pool.acquire() as conn:
                await conn.execute('''
                    INSERT INTO recommendations 
                    (timestamp, action_type, description, urgency, expected_impact, confidence)
                    VALUES ($1, $2, $3, $4, $5, $6)
                ''', data.get("timestamp"), action_type, data.get("description"),
                   urgency, data.get("expected_impact"), confidence)
                
                logger.debug("✅ Recommendation inserted into PostgreSQL")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error inserting recommendation: {e}")
            return False
    
    async def insert_alert(self, data: Dict[str, Any]) -> bool:
        """Insert alert data"""
        # If not connected, work in simulation mode
        if not self.connected:
            logger.debug("⏭️ PostgreSQL not connected, working in simulation mode")
            return True
            
        try:
            # Validate alert data
            alert_type = data.get("alert_type", "unknown")
            priority = data.get("priority", "LOW")
            production_line = data.get("production_line", "Line_A")
            
            # Validate priority levels
            if priority not in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
                logger.warning(f"⚠️ Invalid priority level: {priority}")
            
            # Validate production line
            if production_line not in self.production_lines:
                logger.warning(f"⚠️ Unknown production line: {production_line}")
            
            async with self.pool.acquire() as conn:
                # Get production line ID
                line_record = await conn.fetchrow(
                    "SELECT id FROM production_lines WHERE name = $1",
                    production_line
                )
                line_id = line_record["id"] if line_record else None
                
                # Insert alert
                await conn.execute('''
                    INSERT INTO alerts 
                    (timestamp, alert_type, message, priority, production_line_id)
                    VALUES ($1, $2, $3, $4, $5)
                ''', data.get("timestamp"), alert_type, data.get("message"),
                   priority, line_id)
                
                logger.debug("✅ Alert inserted into PostgreSQL")
                return True
                
        except Exception as e:
            logger.error(f"❌ Error inserting alert: {e}")
            return False
    
    async def get_recent_quality_metrics(self, production_line: str, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent quality metrics"""
        # If not connected, return empty list
        if not self.connected:
            logger.debug("⏭️ PostgreSQL not connected, returning empty results")
            return []
            
        try:
            # Validate production line
            if production_line not in self.production_lines:
                logger.warning(f"⚠️ Unknown production line: {production_line}")
            
            async with self.pool.acquire() as conn:
                records = await conn.fetch('''
                    SELECT q.*, p.name as production_line_name
                    FROM quality_metrics q
                    JOIN production_lines p ON q.production_line_id = p.id
                    WHERE p.name = $1
                    ORDER BY q.timestamp DESC
                    LIMIT $2
                ''', production_line, limit)
                
                return [dict(record) for record in records]
                
        except Exception as e:
            logger.error(f"❌ Error querying quality metrics: {e}")
            return []
    
    async def get_pending_recommendations(self) -> List[Dict[str, Any]]:
        """Get pending recommendations"""
        # If not connected, return empty list
        if not self.connected:
            logger.debug("⏭️ PostgreSQL not connected, returning empty results")
            return []
            
        try:
            async with self.pool.acquire() as conn:
                records = await conn.fetch('''
                    SELECT * FROM recommendations
                    WHERE status = 'PENDING'
                    ORDER BY timestamp DESC
                ''')
                
                return [dict(record) for record in records]
                
        except Exception as e:
            logger.error(f"❌ Error querying recommendations: {e}")
            return []
    
    async def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get active (unresolved) alerts"""
        # If not connected, return empty list
        if not self.connected:
            logger.debug("⏭️ PostgreSQL not connected, returning empty results")
            return []
            
        try:
            async with self.pool.acquire() as conn:
                records = await conn.fetch('''
                    SELECT a.*, p.name as production_line_name
                    FROM alerts a
                    LEFT JOIN production_lines p ON a.production_line_id = p.id
                    WHERE a.resolved = FALSE
                    ORDER BY a.timestamp DESC
                ''')
                
                return [dict(record) for record in records]
                
        except Exception as e:
            logger.error(f"❌ Error querying alerts: {e}")
            return []
    
    async def close(self):
        """Close PostgreSQL connection pool"""
        try:
            if self.pool:
                await self.pool.close()
                self.connected = False
                logger.info("✅ Closed PostgreSQL connection pool")
        except Exception as e:
            logger.error(f"❌ Error closing PostgreSQL connection pool: {e}")