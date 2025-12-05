"""
Modbus RTU/TCP Driver for Industrial Equipment Communication
Supports both serial (RTU) and network (TCP) Modbus protocols
"""

import asyncio
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import struct
from pymodbus.client import AsyncModbusTcpClient, AsyncModbusSerialClient
from pymodbus.exceptions import ModbusException

# Try to import payload module - fallback to older structure if needed
try:
    from pymodbus.payload import BinaryPayloadDecoder, BinaryPayloadBuilder
    PAYLOAD_AVAILABLE = True
    BUILDER_AVAILABLE = True
except ImportError:
    try:
        import pymodbus.payload as payload
        BinaryPayloadDecoder = payload.BinaryPayloadDecoder
        BinaryPayloadBuilder = payload.BinaryPayloadBuilder
        PAYLOAD_AVAILABLE = True
        BUILDER_AVAILABLE = True
    except ImportError:
        PAYLOAD_AVAILABLE = False
        BUILDER_AVAILABLE = False
        BinaryPayloadDecoder = None
        BinaryPayloadBuilder = None

# Try to import Endian - fallback to older structure if needed
try:
    from pymodbus.constants import Endian
    ENDIAN_AVAILABLE = True
except ImportError:
    try:
        from pymodbus import constants
        Endian = constants.Endian
        ENDIAN_AVAILABLE = True
    except (ImportError, AttributeError):
        ENDIAN_AVAILABLE = False
        # Define a simple enum-like class for Endian
        class Endian:
            BIG = "big"
            LITTLE = "little"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModbusDriver:
    """Asynchronous Modbus driver supporting both TCP and RTU protocols"""
    
    def __init__(
        self,
        protocol: str = "tcp",  # "tcp" or "rtu"
        host: str = "localhost",
        port: int = 502,
        serial_port: str = "/dev/ttyUSB0",
        baudrate: int = 9600,
        parity: str = "N",
        stopbits: int = 1,
        timeout: float = 3.0,
        callback: Optional[Callable] = None
    ):
        self.protocol = protocol.lower()
        self.host = host
        self.port = port
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.parity = parity
        self.stopbits = stopbits
        self.timeout = timeout
        self.callback = callback
        self.client: Optional[AsyncModbusTcpClient | AsyncModbusSerialClient] = None
        self.connected = False
        self.slave_id = 1
        self.running = False
        
        # Define register mappings for glass production equipment
        self.register_mappings = {
            # Furnace registers
            "furnace_temperature": {"address": 0, "type": "holding", "format": "float"},
            "furnace_pressure": {"address": 2, "type": "holding", "format": "float"},
            "furnace_melt_level": {"address": 4, "type": "holding", "format": "uint16"},
            "furnace_power": {"address": 5, "type": "holding", "format": "uint16"},
            "furnace_o2_percent": {"address": 6, "type": "holding", "format": "float"},
            "furnace_co2_percent": {"address": 8, "type": "holding", "format": "float"},
            
            # Forming machine registers
            "forming_belt_speed": {"address": 10, "type": "holding", "format": "uint16"},
            "forming_mold_temp": {"address": 11, "type": "holding", "format": "float"},
            "forming_pressure": {"address": 13, "type": "holding", "format": "float"},
            
            # Annealing oven registers
            "annealing_temp_zone1": {"address": 15, "type": "holding", "format": "float"},
            "annealing_temp_zone2": {"address": 17, "type": "holding", "format": "float"},
            "annealing_temp_zone3": {"address": 19, "type": "holding", "format": "float"},
            
            # Quality control registers
            "quality_score": {"address": 21, "type": "input", "format": "float"},
            "defect_count": {"address": 23, "type": "input", "format": "uint16"},
            "production_rate": {"address": 24, "type": "input", "format": "uint16"},
            
            # Control registers
            "emergency_stop": {"address": 100, "type": "coil", "format": "bool"},
            "auto_mode": {"address": 101, "type": "coil", "format": "bool"},
            "maintenance_mode": {"address": 102, "type": "coil", "format": "bool"}
        }
    
    async def connect(self) -> bool:
        """Establish connection to Modbus device"""
        try:
            if self.protocol == "tcp":
                self.client = AsyncModbusTcpClient(
                    host=self.host,
                    port=self.port,
                    timeout=self.timeout
                )
            elif self.protocol == "rtu":
                self.client = AsyncModbusSerialClient(
                    port=self.serial_port,
                    baudrate=self.baudrate,
                    parity=self.parity,
                    stopbits=self.stopbits,
                    timeout=self.timeout
                )
            else:
                raise ValueError(f"Unsupported protocol: {self.protocol}")
            
            # Attempt connection
            await self.client.connect()
            self.connected = self.client.connected
            
            if self.connected:
                logger.info(f"✅ Connected to Modbus {self.protocol.upper()} device")
                logger.info(f"   Host: {self.host}:{self.port}" if self.protocol == "tcp" 
                           else f"   Serial: {self.serial_port} ({self.baudrate} baud)")
                return True
            else:
                logger.error("❌ Failed to connect to Modbus device")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error connecting to Modbus device: {e}")
            # Continue with simulation mode
            self.connected = False
            return False
    
    async def disconnect(self):
        """Disconnect from Modbus device"""
        if self.client:
            self.running = False
            try:
                if self.client is not None:
                    await self.client.close()
            except Exception as e:
                logger.error(f"❌ Error closing Modbus connection: {e}")
            self.connected = False
            logger.info("✅ Disconnected from Modbus device")
    
    async def read_register(self, register_name: str, slave_id: int = None) -> Any:
        """Read a single register by name"""
        if not self.connected or not self.client:
            logger.error("❌ Not connected to Modbus device")
            return None
        
        if register_name not in self.register_mappings:
            logger.error(f"❌ Unknown register: {register_name}")
            return None
        
        # Check if required modules are available
        if not PAYLOAD_AVAILABLE:
            logger.error("❌ Payload module not available")
            return None
            
        slave_id = slave_id or self.slave_id
        reg_info = self.register_mappings[register_name]
        address = reg_info["address"]
        reg_type = reg_info["type"]
        format_type = reg_info["format"]
        
        try:
            # Check connection before reading
            if not self.client.connected:
                logger.warning("⚠️ Modbus client disconnected, attempting to reconnect...")
                await self.connect()
                if not self.connected:
                    logger.error("❌ Failed to reconnect to Modbus device")
                    return None
            
            if reg_type == "holding":
                if format_type == "float":
                    # Read two consecutive registers for float
                    response = await self.client.read_holding_registers(address, 2, slave=slave_id)
                    if not response.isError():
                        decoder = BinaryPayloadDecoder.fromRegisters(
                            response.registers, byteorder=Endian.BIG, wordorder=Endian.LITTLE
                        )
                        value = decoder.decode_32bit_float()
                        return value
                elif format_type in ["uint16", "int16"]:
                    response = await self.client.read_holding_registers(address, 1, slave=slave_id)
                    if not response.isError():
                        return response.registers[0]
                        
            elif reg_type == "input":
                if format_type == "float":
                    response = await self.client.read_input_registers(address, 2, slave=slave_id)
                    if not response.isError():
                        decoder = BinaryPayloadDecoder.fromRegisters(
                            response.registers, byteorder=Endian.BIG, wordorder=Endian.LITTLE
                        )
                        value = decoder.decode_32bit_float()
                        return value
                elif format_type in ["uint16", "int16"]:
                    response = await self.client.read_input_registers(address, 1, slave=slave_id)
                    if not response.isError():
                        return response.registers[0]
                        
            elif reg_type == "coil":
                response = await self.client.read_coils(address, 1, slave=slave_id)
                if not response.isError():
                    return response.bits[0]
            
            logger.error(f"❌ Error reading register {register_name}")
            return None
            
        except ModbusException as e:
            logger.error(f"❌ Modbus error reading {register_name}: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error reading {register_name}: {e}")
            return None
    
    async def write_register(self, register_name: str, value: Any, slave_id: int = None) -> bool:
        """Write a value to a register"""
        if not self.connected:
            logger.error("❌ Not connected to Modbus device")
            return False
        
        if register_name not in self.register_mappings:
            logger.error(f"❌ Unknown register: {register_name}")
            return False
            
        # Check if required modules are available
        if not PAYLOAD_AVAILABLE:
            logger.error("❌ Payload module not available")
            return False
        
        slave_id = slave_id or self.slave_id
        reg_info = self.register_mappings[register_name]
        address = reg_info["address"]
        reg_type = reg_info["type"]
        format_type = reg_info["format"]
        
        try:
            if reg_type == "holding":
                if format_type == "float":
                    # Convert float to registers using BinaryPayloadBuilder if available
                    if BUILDER_AVAILABLE:
                        builder = BinaryPayloadBuilder(byteorder=Endian.BIG, wordorder=Endian.LITTLE)
                        builder.add_32bit_float(float(value))
                        registers = builder.to_registers()
                    else:
                        # Fallback manual encoding
                        packed = struct.pack('>f', float(value))
                        registers = [struct.unpack('>H', packed[i:i+2])[0] for i in range(0, 4, 2)]
                    response = await self.client.write_registers(address, registers, slave=slave_id)
                elif format_type in ["uint16", "int16"]:
                    response = await self.client.write_register(address, int(value), slave=slave_id)
                else:
                    logger.error(f"❌ Unsupported format for writing: {format_type}")
                    return False
                    
            elif reg_type == "coil":
                response = await self.client.write_coil(address, bool(value), slave=slave_id)
            else:
                logger.error(f"❌ Cannot write to {reg_type} registers")
                return False
            
            if response and not response.isError():
                logger.debug(f"✅ Wrote {value} to {register_name}")
                return True
            else:
                logger.error(f"❌ Error writing to {register_name}")
                return False
                
        except ModbusException as e:
            logger.error(f"❌ Modbus error writing to {register_name}: {e}")
            return False
        except Exception as e:
            logger.error(f"❌ Error writing to {register_name}: {e}")
            return False
    
    async def read_all_sensors(self, slave_id: int = None) -> Dict[str, Any]:
        """Read all sensor values in one operation"""
        if not self.connected or not self.client:
            logger.error("❌ Not connected to Modbus device")
            return {}
        
        # Check if required modules are available
        if not PAYLOAD_AVAILABLE:
            logger.warning("⚠️ Payload module not available, using basic register reading")
            # Fallback to reading individual registers
            sensor_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "slave_id": slave_id or self.slave_id,
                "sensors": {}
            }
            
            # Read individual registers as fallback
            for reg_name, reg_info in self.register_mappings.items():
                try:
                    value = await self.read_register(reg_name, slave_id)
                    sensor_data["sensors"][reg_name] = value
                except Exception as e:
                    logger.warning(f"⚠️ Error reading {reg_name}: {e}")
                    sensor_data["sensors"][reg_name] = None
            
            return sensor_data
        
        slave_id = slave_id or self.slave_id
        sensor_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "slave_id": slave_id,
            "sensors": {}
        }
        
        # Read all holding registers first (more common for sensor data)
        try:
            # Group consecutive registers for efficient reading
            grouped_reads = self._group_consecutive_registers()
            
            for start_addr, count, reg_names in grouped_reads:
                response = await self.client.read_holding_registers(start_addr, count, slave=slave_id)
                if not response.isError():
                    # Decode the values
                    decoder = BinaryPayloadDecoder.fromRegisters(
                        response.registers, byteorder=Endian.BIG, wordorder=Endian.LITTLE
                    )
                    
                    # Assign values to corresponding registers
                    for reg_name in reg_names:
                        reg_info = self.register_mappings[reg_name]
                        format_type = reg_info["format"]
                        
                        try:
                            if format_type == "float":
                                value = decoder.decode_32bit_float()
                            elif format_type == "uint16":
                                value = decoder.decode_16bit_uint()
                            elif format_type == "int16":
                                value = decoder.decode_16bit_int()
                            else:
                                value = None
                            
                            sensor_data["sensors"][reg_name] = value
                        except Exception as decode_error:
                            logger.warning(f"⚠️ Error decoding {reg_name}: {decode_error}")
                            sensor_data["sensors"][reg_name] = None
                else:
                    logger.error(f"❌ Error reading registers {start_addr}-{start_addr+count-1}")
            
            # Read coils
            coil_response = await self.client.read_coils(100, 3, slave=slave_id)
            if not coil_response.isError():
                sensor_data["sensors"]["emergency_stop"] = coil_response.bits[0]
                sensor_data["sensors"]["auto_mode"] = coil_response.bits[1]
                sensor_data["sensors"]["maintenance_mode"] = coil_response.bits[2]
            
        except Exception as e:
            logger.error(f"❌ Error reading all sensors: {e}")
        
        return sensor_data
    
    def _group_consecutive_registers(self) -> List[tuple]:
        """Group consecutive holding registers for efficient reading"""
        # Filter holding registers and sort by address
        holding_regs = [(name, info) for name, info in self.register_mappings.items() 
                       if info["type"] == "holding"]
        holding_regs.sort(key=lambda x: x[1]["address"])
        
        if not holding_regs:
            return []
        
        groups = []
        current_group = [holding_regs[0]]
        current_end_addr = holding_regs[0][1]["address"]
        
        # Account for register size (floats take 2 registers)
        reg_size = 2 if holding_regs[0][1]["format"] == "float" else 1
        current_end_addr += reg_size
        
        for name, info in holding_regs[1:]:
            reg_size = 2 if info["format"] == "float" else 1
            
            # Check if this register is consecutive to the current group
            if info["address"] == current_end_addr:
                current_group.append((name, info))
                current_end_addr += reg_size
            else:
                # Finish current group
                start_addr = current_group[0][1]["address"]
                total_regs = current_end_addr - start_addr
                reg_names = [item[0] for item in current_group]
                groups.append((start_addr, total_regs, reg_names))
                
                # Start new group
                current_group = [(name, info)]
                current_end_addr = info["address"] + reg_size
        
        # Add last group
        if current_group:
            start_addr = current_group[0][1]["address"]
            total_regs = current_end_addr - start_addr
            reg_names = [item[0] for item in current_group]
            groups.append((start_addr, total_regs, reg_names))
        
        return groups
    
    async def start_polling(self, interval_seconds: int = 60, slave_id: int = None):
        """Start periodic polling of all sensors"""
        self.running = True
        slave_id = slave_id or self.slave_id
        
        logger.info(f"🔄 Starting Modbus polling every {interval_seconds}s")
        
        while self.running:
            try:
                sensor_data = await self.read_all_sensors(slave_id)
                
                if sensor_data and self.callback:
                    await self.callback(sensor_data)
                
                await asyncio.sleep(interval_seconds)
                
            except asyncio.CancelledError:
                logger.info("⏹️ Polling cancelled")
                break
            except Exception as e:
                logger.error(f"❌ Error in polling loop: {e}")
                await asyncio.sleep(5)  # Wait before retrying
    
    async def stop_polling(self):
        """Stop periodic polling"""
        self.running = False
        logger.info("⏹️ Stopped Modbus polling")


class ModbusSimulator:
    """Simulator for Modbus devices for testing"""
    
    def __init__(self):
        # Simulated register values with realistic ranges
        self.registers = {
            # Furnace
            "furnace_temperature": 1500.0 + (50.0 * (0.5 - np.random.random())),  # 1450-1550°C
            "furnace_pressure": 15.0 + (3.0 * (0.5 - np.random.random())),       # 12-18 bar
            "furnace_melt_level": 2500 + int(200 * (0.5 - np.random.random())),   # 2300-2700 mm
            "furnace_power": 85 + int(15 * (0.5 - np.random.random())),           # 70-100%
            "furnace_o2_percent": 5.0 + (1.0 * (0.5 - np.random.random())),      # 4-6%
            "furnace_co2_percent": 10.0 + (2.0 * (0.5 - np.random.random())),    # 8-12%
            
            # Forming
            "forming_belt_speed": 150 + int(30 * (0.5 - np.random.random())),     # 120-180 m/min
            "forming_mold_temp": 320.0 + (40.0 * (0.5 - np.random.random())),    # 280-360°C
            "forming_pressure": 50.0 + (10.0 * (0.5 - np.random.random())),      # 40-60 bar
            
            # Annealing
            "annealing_temp_zone1": 600.0 + (50.0 * (0.5 - np.random.random())), # 550-650°C
            "annealing_temp_zone2": 550.0 + (50.0 * (0.5 - np.random.random())), # 500-600°C
            "annealing_temp_zone3": 500.0 + (50.0 * (0.5 - np.random.random())), # 450-550°C
            
            # Quality
            "quality_score": 0.95 + (0.05 * (0.5 - np.random.random())),         # 0.90-1.00
            "defect_count": int(5 * np.random.random()),                         # 0-5 defects
            "production_rate": 1200 + int(200 * (0.5 - np.random.random())),      # 1000-1400 units/hr
            
            # Controls
            "emergency_stop": False,
            "auto_mode": True,
            "maintenance_mode": False
        }
    
    def get_register_value(self, register_name: str) -> Any:
        """Get simulated register value with some drift"""
        if register_name in self.registers:
            value = self.registers[register_name]
            
            # Add small random drift to simulate real sensors
            if isinstance(value, float):
                drift = 0.01 * value * (0.5 - np.random.random())
                return value + drift
            elif isinstance(value, int):
                drift = int(0.01 * value * (0.5 - np.random.random()))
                return max(0, value + drift)
            else:
                return value
        return None


async def main_example():
    """Example usage of Modbus driver"""
    
    async def sensor_callback(data):
        """Callback for sensor data"""
        print(f"📡 Sensor data received at {data['timestamp']}")
        for name, value in data['sensors'].items():
            if value is not None:
                print(f"  {name}: {value}")
    
    # Create Modbus driver
    driver = ModbusDriver(
        protocol="tcp",
        host="localhost",
        port=502,
        callback=sensor_callback
    )
    
    try:
        # For demo, we'll show how it would work
        print("🧪 Modbus driver initialized for TCP connection")
        print("   Host: localhost:502")
        print("   Protocol: Modbus TCP")
        print()
        print("📝 Register mappings:")
        for name, info in driver.register_mappings.items():
            print(f"  {name}: addr={info['address']}, type={info['type']}, format={info['format']}")
        
        # Simulate a few reads
        simulator = ModbusSimulator()
        print("\n🔍 Simulated sensor readings:")
        sample_regs = ["furnace_temperature", "forming_belt_speed", "quality_score"]
        for reg in sample_regs:
            value = simulator.get_register_value(reg)
            print(f"  {reg}: {value}")
            
    except KeyboardInterrupt:
        logger.info("⏹️ Demo interrupted by user")


if __name__ == "__main__":
    import numpy as np
    asyncio.run(main_example())