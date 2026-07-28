# CUDA Runtime Compilation (NVRTC) Roadmap

## Goal and Current Boundary

The project now ships the first owner-safe NVRTC compile stage so .NET callers can submit CUDA C++ source, headers, compile options, and name expressions and receive copied PTX, CUBIN, or LTO IR artifacts. Owner-safe kernel launch and readback remain the next stage.

The repository already provides `CudaKernelLibrary.Load(byte[])`, copied library inventory, name-based queries, and kernel attribute updates. Its native owner retains a copy of input code and never exposes borrowed `cudaKernel_t` values through the public C# API. This CUDA Runtime library path requires CUDA Toolkit 12.9 or later. The existing raw `cudaLaunchKernel` entry point remains an internal/generated boundary and is not an acceptable public RTC launch API.

NVRTC integration is larger than one P/Invoke. It covers compiler-program lifetime, variable-length log and artifact copying, reproducible source/options metadata, compiled-code module ownership, typed kernel arguments, optional dynamic dependencies, and runtime packaging.

## Verified Local Baseline

The following installed toolkits were audited on 2026-07-28 without downloading new files:

| Toolkit | Header / import library | Windows runtime |
| --- | --- | --- |
| CUDA 12.9 | `include/nvrtc.h`, `lib/x64/nvrtc.lib` | `nvrtc64_120_0.dll`, `nvrtc-builtins64_129.dll` |
| CUDA 13.2 | `include/nvrtc.h`, `lib/x64/nvrtc.lib` | `bin/x64/nvrtc64_130_0.dll`, `nvrtc-builtins64_132.dll` |

Both headers expose version/error APIs, program create/destroy, compile, program logs, PTX, CUBIN, LTO IR, name expressions, and lowered names. The CUDA 12.9 header still declares deprecated NVVM output; that output will not be the primary path of the new public API.

Windows headers, import libraries, DLLs, builtins, and exports for CUDA 11.8, 12.1, 12.9, and 13.2 are now audited by `eng/Export-CudaRtcCapabilityMatrix.ps1`. No Linux `.so` asset was found on the audited E drive or under the Windows Toolkit roots, so Linux SONAME and symbol support remain explicitly unverified and are not inferred from Windows.

## Implemented Baseline (2026-07-28)

- Native code now provides an optional dynamic loader and `JYPPX_CudaRtcProgram` owner. `JYPPX_NVRTC_LIBRARY` selects an exact library, while the core bridge has no static NVRTC link.
- The ABI covers capability and dependency diagnostics, retained source/program names, virtual headers, name expressions, compile, logs, copied PTX/CUBIN/LTO IR, and copied lowered names. UTF-8, embedded NUL, duplicate, count, and byte limits are enforced, and C++ exceptions plus Windows SEH stay inside the bridge.
- Managed code now exposes `CudaRtcCompiler`, `CudaRtcProgram`, `CudaRtcProgramSource`, `CudaRtcCompileOptions`, `CudaRtcCompilationResult`, and `CudaRtcArtifact` without public `IntPtr`, `SafeHandle`, or vendor program/kernel handles.
- `samples/CudaRuntimeCompilation` exercises virtual headers, a template lowered name, PTX, `sm_75` CUBIN, LTO IR where supported, repeated PTX SHA256 determinism, and an intentional compiler-failure log.
- All four local compilers complete the compile smoke. PTX from 11.8, 12.1, and 12.9 loads through the current CUDA 12.9 `CudaKernelLibrary`; 13.2 PTX is rejected by the current runtime/driver with `cudaErrorUnsupportedPtxVersion`, so 13.2 remains compile proof only.
- Evidence is stored in `artifacts/cuda-runtime-compilation/capability-matrix.json` and `local-smoke.json`, with `kernelLaunch=false`, `gpuReadback=false`, and `correctnessProof=false` retained explicitly.

The remaining RTC work is owner-bound named-kernel launch, typed argument packing, GPU readback, Linux runtime proof, materialized full-runtime `cuda-rtc` components, clean package consumers, and post-publish verification. Compile/load success does not promote any of those lanes.

## Design Invariants

1. `JYPPX_CudaRtcProgram` owns the native compiler program. Create, destroy, and compile are no-throw and contain C++ exceptions, Windows SEH, and NVRTC errors inside the bridge.
2. The owner copies source, program name, header sources/names, compile options, and name expressions. It never retains caller memory or returns `nvrtcProgram` or borrowed lowered-name pointers.
3. Text uses caller-buffer size/copy APIs and binary artifacts use count/copy APIs. Compile logs remain available after compilation failure.
4. Public C# APIs expose no `IntPtr`, `nint`, `UIntPtr`, `SafeHandle`, `nvrtcProgram`, `cudaKernel_t`, `CUmodule`, `CUfunction`, or kernel-argument pointer array.
5. Compilation failure is a diagnostic result that retains the complete log. Invalid lifetime, missing dependencies, and bridge ABI failures continue through the existing status/exception mapping.
6. NVRTC is an optional, diagnosable capability. A consumer that does not use RTC must not fail to load the core bridge solely because NVRTC is absent.
7. Every artifact contains immutable copied bytes and source/options/header/compiler/target/output hashes. Artifacts remain valid after the native program owner is disposed.

## Planned Managed Surface

The target surface is subject to public API review:

- `CudaRtcCompiler` for capability/version queries and one-shot compilation.
- `CudaRtcProgram` for an explicit reusable `IDisposable` native owner.
- `CudaRtcProgramSource` for immutable source, virtual name, headers, and name expressions.
- `CudaRtcCompileOptions` for immutable target and compiler options.
- `CudaRtcCompilationResult` for status, full log, lowered names, versions, and artifacts.
- `CudaRtcArtifact` and `CudaRtcArtifactKind` for copied PTX, CUBIN, and LTO IR payloads with lengths and SHA256 values.

PTX, CUBIN, and LTO IR are option- and target-dependent. The API must represent an unavailable artifact explicitly. A PTX hash proves artifact identity, not kernel-output correctness.

## Delivery Phases

### A. Capability and Vendor Audit

- Audit CUDA 11.8, 12.1, 12.9, and 13.2 headers, libraries, DLLs/`.so` files, builtins, symbols, versions, and CMake targets.
- Add a machine-readable capability matrix for logs, PTX, CUBIN, LTO IR, deprecated NVVM, and name expressions.
- Define Windows delay-load/late binding and Linux `dlopen`/SONAME behavior so missing NVRTC produces a focused diagnostic without breaking non-RTC consumers.

### B. Native Owner and Caller-Buffer ABI

- Add the `JYPPX_CudaRtcProgram` owner, bounded allocation limits, idempotent destruction, and no-throw guards.
- Add create, name-expression, compile, log size/copy, artifact size/copy, lowered-name size/copy, version, and error diagnostics.
- Integrate manifests, generated entry points, ABI declaration parity, PE export parity, and Linux symbol parity.

### C. Managed High-Level API

- Add an internal SafeHandle while exposing only the compiler/program abstractions and immutable DTOs.
- Copy logs, lowered names, and artifact bytes; validate UTF-8, embedded NULs, duplicate inputs, disposed owners, and bounded sizes.
- Add bilingual XML documentation, API snapshots, dependency diagnostics, and deterministic artifact contracts.

### D. Compile to Load to Launch

- On CUDA 12.9/13.2, first verify that NVRTC PTX/CUBIN artifacts load through `CudaKernelLibrary.Load(byte[])`.
- Add an owner-bound launch-by-name surface to `CudaKernelLibrary`; borrowed `cudaKernel_t` values must stay inside the bridge.
- CUDA 11.8/12.1 lack the current Runtime library API. Provide a unified CUDA Driver module/function owner or document a deliberate support limit; never fill the gap with raw function pointers.
- Use typed argument packing and bridge-owned launch storage with explicit buffer, scalar, stream, and module lifetimes.

### E. Samples and Runtime Proof

Add `samples/CudaRuntimeCompilation` with a vector-add/elementwise compile-load-launch-readback check, an intentional compiler-error log check, a C++ name-expression/lowered-name check, and artifact metadata/hash export.

Compile-only output, artifact hashes, synthetic kernels, and a local Toolkit are not clean consumer, public package, or post-publish proof.

### F. Packaging and Cross-Platform Proof

- The bridge-only NuGet package does not bundle NVRTC; callers install a matching CUDA Toolkit and receive focused dependency diagnostics.
- Full GitHub runtime packages add `nvrtc` and the matching `nvrtc-builtins`, with Windows/Linux manifests, split-package roles, hashes, size checks, and redistribution review.
- Validate Windows x64, Linux x64, and CUDA 11.8/12.1/12.9/13.2. Runtime-key declarations must match actual native dependencies and package assets.
- A clean consumer must compile and launch without ProjectReference or development probing. Repeat the smoke against the public package after publishing.

## Evidence Ladder

Header/library audit proves vendor surface only. Compile-only proves compiler ownership and output copying only. Artifact hashes prove identity only. Local compile-to-launch proves one host path only. Clean-package-consumer and post-publish checks remain separate evidence classes, and Owner approval is still required for final publication and issue closure.

CUDA RTC becomes release-ready only after compile, load, launch, readback, clean consumer, cross-platform package, and post-publish evidence all close.
