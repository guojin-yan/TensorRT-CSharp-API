# tests

This directory contains the managed test projects used to keep the repository
shape, package metadata, manifest rules, and smoke-runner contracts honest.

- `JYPPX.ProjectQuality.Tests` validates repository-level rules such as managed
  package content, runtime manifest hygiene, and native bridge path resolution.
- `JYPPX.TensorRtSharp.Tests`, `JYPPX.CudaSharp.Tests`, and
  `JYPPX.IntegrationTests` are reserved for API and integration coverage as the
  wrapper surface expands.

## Fixtures

`fixtures/` contains inputs copied into repository-external validation workspaces. These directories are not user
samples and are not added to `TensorRtSharp.sln`:

- `package-consumers` contains active managed plus bridge package-isolation fixtures for callback, allocator,
  stream-reader, and refitted-plan validation.
- `mnist-onnx-runtime-reference` contains the independent ONNX Runtime reference program used by MNIST checks.
- `legacy-package-consumers` preserves the pre-release Classification and YoloVision package layouts only so
  historical evidence remains reproducible. Classification and YoloVision themselves are non-packable applications.

No fixture is published to NuGet, GitHub Packages, or a Release.
