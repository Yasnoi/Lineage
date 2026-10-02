"""Actionable diagnostics made from trusted field names and fixed descriptions."""


class ConfigurationError(ValueError):
    """Never include a supplied value, arbitrary key or original exception text."""

    def __init__(self, path: str, code: str, constraint: str, hint: str):
        self.diagnostic = {"path": path, "code": code, "constraint": constraint, "hint": hint}
        super().__init__(f"{path} [{code}]：{constraint} {hint}")

    def as_issue(self, stage="preflight"):
        return {"stage": stage, "code": self.diagnostic["code"], "message": str(self),
                "diagnostic": dict(self.diagnostic), "evidence": []}
