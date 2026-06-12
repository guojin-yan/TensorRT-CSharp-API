# tests

This directory contains the managed test projects used to keep the repository
shape, package metadata, manifest rules, and smoke-runner contracts honest.

- `JYPPX.ProjectQuality.Tests` validates repository-level rules such as managed
  package content, runtime manifest hygiene, and native bridge path resolution.
- `JYPPX.TensorRtSharp.Tests`, `JYPPX.CudaSharp.Tests`, and
  `JYPPX.IntegrationTests` are reserved for API and integration coverage as the
  wrapper surface expands.
