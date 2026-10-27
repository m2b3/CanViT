# Changelog

Changes that reach users of the `canvit-mlx` package, newest first. A release moves the Unreleased
entries under its version. Entries marked **Breaking** change what existing code or saved files do;
**Hub** entries are steps on the Hugging Face Hub that must follow the release, in order, because
released versions load each repo's `main` revision.

## Unreleased

### Removed

- **Breaking**: `canvit_mlx.model.config`, which only re-exported `CanViTConfig`; import it from
  `canvit_mlx` or `canvit_core.config`.
