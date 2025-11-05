# plc_packet.py
"""PLC packet generation (32-byte binary frame)."""

import struct
import time
from typing import Dict, Any

PACKET_SIZE = 32
MAGIC = b"ACSI"  # "41435349" in hex
VERSION = 0x0001
COMMAND_ID_SORT_DECISION = 0x0001

# Decision class to enum mapping
DECISION_ENUM = {
    "BACKGROUND_TRASH": 10,
    "UNKNOWN_VARIANT": 20,
    "HIGH_CONFIDENCE_SORT": 30,
    "EMBEDDING_RESCUE": 40,
}


def create_plc_packet(decision_obj: Dict[str, Any], ts_ms: int = None) -> bytes:
    """
    Create 32-byte PLC packet from decision object.
    
    Format (all fields BIG-ENDIAN):
    - Bytes 0-3: ASCII "ACSI"
    - Bytes 4-5: version (0x0001) BE
    - Bytes 6-7: command_id (0x0001 = sort decision) BE
    - Bytes 8-11: class_id (u32 BE)
    - Bytes 12-15: decision_class enum (u32 BE)
    - Bytes 16-19: target_bin (u32 BE)
    - Bytes 20-23: confidence_scaled (u32 BE, conf*10000)
    - Bytes 24-31: timestamp_ms (u64 BE)
    
    Args:
        decision_obj: Decision object with keys:
            - class_id: Classification ID
            - conf: Confidence [0.0, 1.0]
            - decision_class: Decision class name (e.g., "BACKGROUND_TRASH")
            - target_bin: Target bin number
        ts_ms: Timestamp in milliseconds (if None, uses current time)
    
    Returns:
        32-byte binary packet
    """
    packet = bytearray(PACKET_SIZE)
    
    # 0-3: ASCII "ACSI"
    packet[0:4] = MAGIC
    
    # 4-5: version (0x0001) BE
    packet[4:6] = struct.pack(">H", VERSION)
    
    # 6-7: command_id (0x0001) BE
    packet[6:8] = struct.pack(">H", COMMAND_ID_SORT_DECISION)
    
    # 8-11: class_id (u32 BE)
    class_id = decision_obj.get("class_id", 9999)
    packet[8:12] = struct.pack(">I", class_id)
    
    # 12-15: decision_class enum (u32 BE)
    decision_class = decision_obj.get("decision_class", "UNKNOWN_VARIANT")
    decision_enum = DECISION_ENUM.get(decision_class, 20)  # Default to UNKNOWN_VARIANT
    packet[12:16] = struct.pack(">I", decision_enum)
    
    # 16-19: target_bin (u32 BE)
    target_bin = decision_obj.get("target_bin", 0)
    packet[16:20] = struct.pack(">I", target_bin)
    
    # 20-23: confidence_scaled (u32 BE, conf*10000)
    conf = decision_obj.get("conf", 0.0)
    confidence_scaled = int(conf * 10000)
    packet[20:24] = struct.pack(">I", confidence_scaled)
    
    # 24-31: timestamp_ms (u64 BE)
    if ts_ms is None:
        ts_ms = int(time.time() * 1000)
    packet[24:32] = struct.pack(">Q", ts_ms)
    
    return bytes(packet)


def packet_to_hex(packet: bytes) -> str:
    """Convert packet to hex string for display."""
    return packet.hex().upper()

