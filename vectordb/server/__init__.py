"""
vectordb.server - Vector Database Server Engine

Core storage engine providing:
- Schema management (FieldType, CollectionSchema)
- Segment-based storage (SegmentData, CollectionEngine)
- MVCC snapshot reads (LSNGenerator, ReadSnapshot)
- Dual index strategies (HNSWLib Per-Segment / IVF+RaBitQ Global)
"""
