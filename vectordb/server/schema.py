"""
Schema definitions and validation for VectorDB collections.

Ref: Technical_Design_VectorDB.md §1.1 核心数据结构定义 (L22-L76)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional

# --- Reserved field names (system-internal columns) ---
RESERVED_FIELD_NAMES = {"_lsn", "_deleted", "_offset", "_segment_id"}


class FieldType(Enum):
    """Supported field data types."""
    INT = "int"         # int64
    STRING = "string"   # variable-length UTF-8 string
    VECTOR = "vector"   # float32 fixed-dimension vector


class MetricType(Enum):
    """Distance metric types for vector fields."""
    L2 = "L2"
    IP = "IP"           # Inner Product
    COSINE = "COSINE"


@dataclass
class FieldSchema:
    """Single field definition within a collection schema.

    Ref: Technical Design L37-L45
    """
    name: str
    dtype: FieldType
    is_primary: bool = False
    # Only valid for VECTOR type fields
    dim: Optional[int] = None
    metric: Optional[MetricType] = None


@dataclass
class CollectionSchema:
    """Complete schema for a collection.

    Ref: Technical Design L48-L68
    """
    collection_name: str
    fields: List[FieldSchema]
    # Auto-filled by the system
    created_at: Optional[float] = None   # timestamp
    version: int = 0                      # schema version

    def primary_field(self) -> FieldSchema:
        """Return the primary key field."""
        for f in self.fields:
            if f.is_primary:
                return f
        raise ValueError("No primary field defined")

    def vector_field(self) -> FieldSchema:
        """Return the first vector field."""
        for f in self.fields:
            if f.dtype == FieldType.VECTOR:
                return f
        raise ValueError("No vector field defined")


def validate_schema(schema: CollectionSchema) -> None:
    """Validate a CollectionSchema against all rules.

    Rules (Ref: Technical Design §1.1 校验规则表 L71-L76):
      1. Must have exactly one is_primary=True field; PK type must be INT or STRING.
      2. Must have at least one VECTOR field with dim and metric specified.
      3. Field names must be unique and not use reserved names.

    Raises:
        ValueError: If any rule is violated.
    """
    if not schema.fields:
        raise ValueError("Schema must have at least one field")

    # --- Rule 1: Exactly one primary key, INT or STRING only ---
    primary_fields = [f for f in schema.fields if f.is_primary]
    if len(primary_fields) == 0:
        raise ValueError("Schema must have exactly one primary key field")
    if len(primary_fields) > 1:
        raise ValueError(
            f"Schema must have exactly one primary key field, "
            f"found {len(primary_fields)}"
        )
    pk = primary_fields[0]
    if pk.dtype not in (FieldType.INT, FieldType.STRING):
        raise ValueError(
            f"Primary key field '{pk.name}' must be INT or STRING, "
            f"got {pk.dtype.value}"
        )

    # --- Rule 2: At least one VECTOR field with dim + metric ---
    vector_fields = [f for f in schema.fields if f.dtype == FieldType.VECTOR]
    if len(vector_fields) == 0:
        raise ValueError("Schema must have at least one VECTOR field")
    for vf in vector_fields:
        if vf.dim is None:
            raise ValueError(
                f"VECTOR field '{vf.name}' must specify dim"
            )
        if vf.dim <= 0:
            raise ValueError(
                f"VECTOR field '{vf.name}' dim must be positive, got {vf.dim}"
            )
        if vf.metric is None:
            raise ValueError(
                f"VECTOR field '{vf.name}' must specify metric"
            )

    # --- Rule 3: Unique field names, no reserved names ---
    seen_names: set = set()
    for f in schema.fields:
        if f.name in RESERVED_FIELD_NAMES:
            raise ValueError(
                f"Field name '{f.name}' is reserved by the system"
            )
        if f.name in seen_names:
            raise ValueError(f"Duplicate field name: '{f.name}'")
        seen_names.add(f.name)
