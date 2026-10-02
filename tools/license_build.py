from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class LicenseBuildHook(BuildHookInterface):
    def initialize(self, version: str, build_data: dict[str, object]) -> None:
        license_path = Path(self.root) / "LICENSE"
        if license_path.is_symlink():
            source_path = license_path.resolve()
        elif license_path.is_file():
            return
        else:
            source_path = license_path.parent.parent / "LICENSE"
        if not source_path.is_file():
            raise FileNotFoundError(source_path)
        force_include = build_data.setdefault("force_include", {})
        assert isinstance(force_include, dict)
        for source, target in list(force_include.items()):
            if target == "LICENSE" and Path(source).resolve() == source_path:
                del force_include[source]
        force_include[str(source_path)] = "LICENSE"
