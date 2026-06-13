# TensorRtSharp4.0

TensorRtSharp4.0 is a production-oriented TensorRT and CUDA bridge for .NET.

## Current Status

Verified on 2026-06-12:

- TensorRT interface coverage: `0` missing rows.
- CUDA runtime interface coverage: `0` missing rows.
- Manifest inventory: `3271` API records across `102` manifests.
- Latest coverage report: `artifacts/interface-coverage/interface-coverage-summary.md`.
- Native validation: `win-x64-trt11-cuda13-release` configures and builds.
- Managed validation: solution build and project quality tests pass.

The project is now past raw interface coverage closure for the locally scanned headers. The active work is release hardening: sample coverage, documentation, package validation, and promotion of high-value deferred boundaries into safe public wrappers.

## Start Here

English:

- [Getting Started](articles/en/getting-started.md)
- [Installation Layout](articles/en/installation-layout.md)
- [Windows API Completion](articles/en/windows-api-completion.md)
- [API Coverage and Deferred Boundaries](articles/en/api-coverage-and-deferred-boundaries.md)
- [Sample Runners](articles/en/sample-runners.md)
- [Runtime Package Strategy](articles/en/runtime-packages.md)
- [Runtime Distribution Strategy](articles/en/runtime-distribution-strategy.md)
- [Package Consumer Validation](articles/en/package-consumer-validation.md)
- [Release Candidate Gate](articles/en/release-candidate-gate.md)

Chinese articles:

- [Getting Started](articles/zh-cn/getting-started.md)
- [Installation Layout](articles/zh-cn/installation-layout.md)
- [Windows API Completion](articles/zh-cn/windows-api-completion.md)
- [Latest Windows API Status](articles/zh-cn/windows-api-completion-latest.md)
- [Runtime Packages](articles/zh-cn/runtime-packages.md)
- [Runtime Distribution Strategy](articles/zh-cn/runtime-distribution-strategy.md)
- [Package Consumer Validation](articles/zh-cn/package-consumer-validation.md)
- [Release Candidate Gate](articles/zh-cn/release-candidate-gate.md)
- [Sample Runners](articles/zh-cn/sample-runners.md)

## Example And Smoke Path

Recommended common-example order:

1. `MultiStream`
2. `DynamicShape`
3. `InferenceBindings`
4. `OnnxToEngine`

Recommended smoke order:

1. `CudaSmokeRunner`
2. `TensorRtSmokeRunner`
3. `LifecycleSmokeRunner`
4. `OnnxToEngineSmokeRunner`
5. `NetworkBuilderSmokeRunner`
6. Layer-specific network runners

See [Sample Runners](articles/en/sample-runners.md), `samples/README.md`, and `smoke/README.md` for commands and expected evidence lines.
