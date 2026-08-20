"""Single home for cross-module constants (Design Law L3).

Any threshold shared by two modules must live here and be imported by both.
Duplicating a literal elsewhere is caught by tests/test_source_discipline.py.
"""

# Recipe schema version accepted by this build.
RECIPE_SCHEMA_VERSION = "1.0"

# Setting-change severity thresholds (fractions, shared by advisor & change log).
NOTICE_CHANGE_FRAC = 0.15
LARGE_CHANGE_FRAC = 0.30

# Alignment defaults / validation bounds.
MAX_ROTATION_DEG_LIMIT = 10.0     # recipe may not ask for more than this
MIN_ANCHOR_SEPARATION_FRAC = 0.20  # anchor centers, as fraction of golden diagonal

# Fingerprint length (hex chars of sha256 prefix) — same convention as the run-block provenance fingerprints.
FINGERPRINT_LEN = 12
