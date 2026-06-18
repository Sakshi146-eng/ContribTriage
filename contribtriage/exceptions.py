"""
contribtriage/exceptions.py

Clean exception hierarchy for the ContribTriage pipeline.
The orchestrator's retry loop and all stage modules raise and catch
these — never bare Exception — so control flow stays readable.
"""


class ContribTriageError(Exception):
    """Base exception for all ContribTriage errors."""


# ---------------------------------------------------------------------------
# Dependency Resolution Exceptions
# ---------------------------------------------------------------------------

class DependencyInstallError(ContribTriageError):
    """
    Raised when `uv pip install <package>` fails.

    The orchestrator catches this and decrements the retry counter.
    If retries are exhausted, it promotes this to MaxRetriesExceeded.

    Attributes:
        package:    The package name that failed to install.
        stderr:     Raw stderr output from the uv subprocess.
        returncode: Process exit code from uv.
    """

    def __init__(self, package: str, stderr: str = "", returncode: int = 1):
        self.package = package
        self.stderr = stderr
        self.returncode = returncode
        super().__init__(
            f"Failed to install '{package}' (exit {returncode}).\n"
            f"uv stderr: {stderr.strip() or '(none)'}"
        )


class MaxRetriesExceeded(ContribTriageError):
    """
    Raised by the orchestrator when the dependency reconciliation loop
    has hit its --max-retries ceiling without achieving a clean environment.

    Attributes:
        attempts: How many install attempts were made.
        last_error: The DependencyInstallError that triggered the final failure.
    """

    def __init__(self, attempts: int, last_error: DependencyInstallError):
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"Dependency resolution failed after {attempts} attempt(s). "
            f"Last failure was for package '{last_error.package}'. "
            "See SETUP_DIAGNOSTICS.md for manual remediation steps."
        )


class UserDeclinedInstall(ContribTriageError):
    """
    Raised when the user answers [N] at the interactive install prompt.

    For APPLICATION dependencies: the orchestrator halts the install loop
    and proceeds to report generation, logging the declined package.

    For SYSTEM dependencies: the orchestrator skips ALL intermediate
    stages (test execution, test generation) and jumps directly to
    report generation, logging the missing system tool as a manual action.

    Attributes:
        package:     The package/tool name the user declined.
        is_system:   True if this was a system-level dependency (postgres,
                     docker, redis, etc.) — triggers the full skip-to-report
                     shortcut in the orchestrator.
    """

    def __init__(self, package: str, is_system: bool = False):
        self.package = package
        self.is_system = is_system
        skip_msg = (
            " Skipping all intermediate stages and jumping to report generation."
            if is_system
            else " Package logged as unresolved in the final report."
        )
        super().__init__(
            f"User declined installation of '{package}'.{skip_msg}"
        )


# ---------------------------------------------------------------------------
# Environment Exceptions
# ---------------------------------------------------------------------------

class IncompatibleEnvironmentError(ContribTriageError):
    """
    Raised during the Environment Audit stage when a hard blocker is found
    that cannot be auto-resolved — e.g., Python version mismatch so severe
    that no packages can be installed safely.

    Attributes:
        reason: Human-readable explanation of the incompatibility.
    """

    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(f"Incompatible environment: {reason}")


# ---------------------------------------------------------------------------
# Ingestion Exceptions
# ---------------------------------------------------------------------------

class IngestionError(ContribTriageError):
    """
    Raised when AST parsing or vector store ingestion fails on a file.
    Non-fatal: the orchestrator logs the offending file and continues.

    Attributes:
        filepath: Path to the file that caused the failure.
        cause:    The underlying exception.
    """

    def __init__(self, filepath: str, cause: Exception):
        self.filepath = filepath
        self.cause = cause
        super().__init__(
            f"Failed to ingest '{filepath}': {type(cause).__name__}: {cause}"
        )


# ---------------------------------------------------------------------------
# Verification Exceptions
# ---------------------------------------------------------------------------

class TestExecutionError(ContribTriageError):
    """
    Raised when the test runner subprocess itself fails to launch
    (distinct from tests failing — that is a normal result, not an exception).

    Attributes:
        command:    The command list that was attempted.
        returncode: Process exit code.
        stderr:     Raw stderr from pytest.
    """

    def __init__(self, command: list, returncode: int, stderr: str = ""):
        self.command = command
        self.returncode = returncode
        self.stderr = stderr
        super().__init__(
            f"Test runner crashed (exit {returncode}) running: {' '.join(command)}\n"
            f"stderr: {stderr.strip() or '(none)'}"
        )


class ImportValidationError(ContribTriageError):
    """
    Raised when the dynamically generated import-validation test script
    itself cannot be written or executed — not when imports fail
    (those failures are captured as normal TestResult entries).
    """
