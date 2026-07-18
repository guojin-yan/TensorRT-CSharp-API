# TensorRT Builder-Config Scalar Candidate Audit

Decision: `promote-by-explicit-alias-history`

This batch closes the four scalar `IBuilderConfig` records below by routing the
official TensorRT interfaces to existing real bridge entry points. The old
deferred records remain in their historical manifests and stubs; deferred
records remain callable as diagnostic compatibility entries, but they are not
the supported route used by the managed wrapper.

## Candidate Surface

| Official interface | TRT8 real entry | TRT10 real entry | TRT11 real entry | Ownership risk |
| --- | --- | --- | --- | --- |
| `getAvgTimingIterations` | `jyppx_trt8_builder_config_get_average_timing_iterations` | `jyppx_trt10_builder_config_get_average_timing_iterations` | `jyppx_trt11_builder_config_get_average_timing_iterations` | Low: scalar out value, no returned pointer |
| `setAvgTimingIterations` | `jyppx_trt8_builder_config_set_average_timing_iterations` | `jyppx_trt10_builder_config_set_average_timing_iterations` | `jyppx_trt11_builder_config_set_average_timing_iterations` | Low: scalar input, no retained memory |
| `getBuilderOptimizationLevel` | `jyppx_trt8_builder_config_get_optimization_level` | `jyppx_trt10_builder_config_get_optimization_level` | `jyppx_trt11_builder_config_get_optimization_level` | Low: scalar out value, no returned pointer |
| `setBuilderOptimizationLevel` | `jyppx_trt8_builder_config_set_optimization_level` | `jyppx_trt10_builder_config_set_optimization_level` | `jyppx_trt11_builder_config_set_optimization_level` | Low: scalar input, no retained memory |

## Vendor Evidence

All three version lines expose these methods on `nvinfer1::IBuilderConfig` in
the local `NvInfer.h` headers. The underlying C++ methods are vtable scalar
controls; the bridge accepts only an existing typed opaque config handle and
primitive integers. No vendor pointer, callback, array, or external resource
crosses the ABI.

| Version line | Header evidence | Import-library evidence | DLL evidence | Native implementation |
| --- | --- | --- | --- | --- |
| TRT8 | `third_party/nvidia/TensorRT-8.6.1.6-cuda 11.8/include/NvInfer.h` and the CUDA 12.1 sibling | `.../lib/nvinfer.lib` | `.../lib/nvinfer.dll` | `native/src/tensorrt/v8/modules/builder/builder_config.inc` |
| TRT10 | `third_party/nvidia/TensorRT-10.11.0.33-cuda 11.8/include/NvInfer.h` and the CUDA 12.9 sibling | `.../lib/nvinfer_10.lib` | `.../lib/nvinfer_10.dll` | `native/src/tensorrt/v10/modules/builder/builder_config.inc` |
| TRT11 | `third_party/nvidia/TensorRT-11.0.0.114-cuda 12.9/include/NvInfer.h` and the CUDA 13.2 sibling | `.../lib/nvinfer_11.lib` | `TensorRT-11.0.0.114-cuda 13.2/bin/nvinfer_11.dll` | `native/src/tensorrt/v11/api.cpp` |

The TRT11 CUDA 12.9 vendor folder contains the import library and headers but
does not contain a runtime DLL in this checkout; that is a package-material
availability constraint, not a scalar ABI mismatch. The CUDA 13.2 sibling is
the DLL-backed runtime evidence for the same TRT11 version line.

## Manifest and History Mapping

Real records are in:

- `native/manifests/tensorrt/v8/trt8-minimal.manifest.json`
- `native/manifests/tensorrt/v10/trt10-minimal.manifest.json`
- `native/manifests/tensorrt/v11/trt11-deployment.manifest.json`

Historical deferred records remain in:

- `native/manifests/tensorrt/v8/trt8-twenty-third-batch-deferred-coverage.manifest.json`
- `native/manifests/tensorrt/v10/trt10-twenty-third-batch-deferred-coverage.manifest.json`
- `native/manifests/tensorrt/v11/trt11-twenty-third-batch-deferred-coverage.manifest.json`

`eng/Export-InterfaceCoverageMatrix.ps1` now keeps real IDs in the explicit
alias map and deferred IDs in `deferredHistoryAliasMap`. This makes
`Get-ResolvedImplementationStatus` report
`implemented-with-deferred-history` only when both real and historical
entry points are present and exported.

## Managed Route and Verification Contract

- Native interop: `NativeBridgeApi.Set/GetAverageTimingIterations` and
  `Set/GetBuilderOptimizationLevel` select TRT8, TRT10, or TRT11 by the
  version guard.
- Public wrapper: `TensorRtBuilderConfig.Set/GetAverageTimingIterations` and
  `Set/GetOptimizationLevel` expose primitive scalar semantics only.
- Snapshot and smoke: builder-config deployment snapshots read the same real
  controls; scalar round-trip is covered by `NetworkBuilderSmokeRunner`.
- Quality gate: ProjectQuality asserts the three version manifests, version
  guards, real/history alias split, no public raw pointer, and retained
  deferred records.

## Deliberately Out of Scope

This audit does not promote callback-owned algorithm selectors, calibrators,
progress monitors, plugin mutation, allocator/resource APIs, or borrowed
pointer surfaces. Their ownership contracts remain deferred.
