class SimgrepError(Exception):
    """Base typed error."""

    def __init__(self, message: str, *, hint: str | None = None, exit_code: int = 1) -> None:
        super().__init__(message)
        self.hint = hint
        self.exit_code = exit_code


class ConfigError(SimgrepError):
    pass


class ProjectError(SimgrepError):
    pass


class MetadataError(SimgrepError):
    pass


class SearchError(SimgrepError):
    pass


class ClustersError(SimgrepError):
    pass


class DiffError(SimgrepError):
    pass


class ExprError(SimgrepError):
    pass


class ExpandError(SimgrepError):
    """expand / --whole-unit failures."""


class DebtError(SimgrepError):
    """Debt radar failure."""


class RerankError(SimgrepError):
    """`simgrep rerank` failures: chunk cap exceeded, model load failure, zero readable files."""
