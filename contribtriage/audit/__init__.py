"""
contribtriage/audit/__init__.py

Stage 3 — Environment Audit.

Exposes two public entry points:
  - audit_environment  : Full host environment snapshot → EnvReport
  - check_system_tools : Per-service availability check → List[SystemToolStatus]
"""

from contribtriage.audit.env_auditor import audit_environment
from contribtriage.audit.tool_checker import check_system_tools

__all__ = ["audit_environment", "check_system_tools"]
