"""Shared subtitle-timing cache, labels, metrics, and segment utilities.

The package intentionally stays import-light because H.264 feature extraction
uses its hashing module while feature-cache validation also understands H.264
visual feature metadata.
"""

__all__: list[str] = []
