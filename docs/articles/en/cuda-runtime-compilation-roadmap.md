# CUDA Runtime Compilation (NVRTC) Roadmap

## Goal and Current Boundary

The project now ships an owner-safe NVRTC compile API and two owner-bound launch/readback paths. .NET callers can submit CUDA C++ source, headers, compile options, and name expressions, receive copied PTX, CUBIN, or LTO IR artifacts, and launch named kernels with typed arguments through either the CUDA 12.9+ Runtime library or a dynamically loaded CUDA Driver module owner. A repository-external, local-feed-only bridge-package consumer is verified on Windows; Linux, public-package, and post-publish proof remain future work.

The repository provides `CudaKernelLibrary.Load(byte[])`, copied library inventory, name-based queries, kernel attribute updates, and an owner-safe `CudaKernelLibrary.Launch(...)` path. This Runtime-library path requires CUDA Toolkit 12.9 or later. `CudaDriverModule.Load(...)` and `Launch(...)` add a unified module path backed by dynamic `nvcuda.dll` / `libcuda.so.1` loading, a retained primary context, typed arguments, and a completion-event owner. Neither path exposes borrowed `cudaKernel_t`, `CUmodule`, or `CUfunction` values through the public C# API. The raw generated launch entry point remains internal.

NVRTC integration is larger than one P/Invoke. It covers compiler-program lifetime, variable-length log and artifact copying, reproducible source/options metadata, compiled-code module ownership, typed kernel arguments, optional dynamic dependencies, and runtime packaging.

## Verified Local Baseline

The following installed toolkits were audited on 2026-07-28 without downloading new files:

| Toolkit | Header / import library | Windows runtime |
| --- | --- | --- |
| CUDA 12.9 | `include/nvrtc.h`, `lib/x64/nvrtc.lib` | `nvrtc64_120_0.dll`, `nvrtc-builtins64_129.dll` |
| CUDA 13.2 | `include/nvrtc.h`, `lib/x64/nvrtc.lib` | `bin/x64/nvrtc64_130_0.dll`, `nvrtc-builtins64_132.dll` |

Both headers expose version/error APIs, program create/destroy, compile, program logs, PTX, CUBIN, LTO IR, name expressions, and lowered names. The CUDA 12.9 header still declares deprecated NVVM output; that output will not be the primary path of the new public API.

Windows headers, import libraries, DLLs, builtins, and exports for CUDA 11.8, CUDA 12.1, CUDA 12.9, and CUDA 13.2 are now audited by `eng/Export-CudaRtcCapabilityMatrix.ps1`. No Linux `.so` asset was found on the audited E drive or under the Windows Toolkit roots, so Linux SONAME and symbol support remain explicitly unverified and are not inferred from Windows.

The matching `cuda.h` and `cuda.lib` Driver surface is audited by `eng/Export-CudaDriverCapabilityMatrix.ps1`, including context, module, launch, failure-cleanup stream synchronization, and event symbols. `driver.cpp` also compiles independently against all four installed header lines through the `jyppx_cuda_driver_compile_probe` target. These are vendor-surface and compile-compatibility proofs, not Linux or runtime proofs.

## Implemented Baseline (2026-07-28)

- Native code now provides an optional dynamic loader and `JYPPX_CudaRtcProgram` owner. `JYPPX_NVRTC_LIBRARY` selects an exact library, while the core bridge has no static NVRTC link.
- The ABI covers capability and dependency diagnostics, retained source/program names, virtual headers, name expressions, compile, logs, copied PTX/CUBIN/LTO IR, and copied lowered names. UTF-8, embedded NUL, duplicate, count, and byte limits are enforced, and C++ exceptions plus Windows SEH stay inside the bridge.
- Managed code now exposes `CudaRtcCompiler`, `CudaRtcProgram`, `CudaRtcProgramSource`, `CudaRtcCompileOptions`, `CudaRtcCompilationResult`, and `CudaRtcArtifact` without public `IntPtr`, `SafeHandle`, or vendor program/kernel handles.
- Native code now also provides an optional dynamic CUDA Driver loader, retained-primary-context `JYPPX_CudaDriverModule`, typed launch storage, and Driver event completion ownership through 9 ABI entry points. Managed code exposes `CudaDriver`, `CudaDriverModule`, and `CudaDriverKernelLaunch` without raw Driver handles.
- `samples/Cuda/01.RuntimeCompilation` exercises virtual headers, a template lowered name, PTX, `sm_75` CUBIN, LTO IR where supported, repeated PTX SHA256 determinism, and an intentional compiler-failure log.
- All four local compilers complete the compile smoke. PTX from 11.8, 12.1, and 12.9 loads through both the current CUDA 12.9 `CudaKernelLibrary` and the current system Driver 12090, launches by name, and reads back 257 validated floats with the same output SHA256; 13.2 PTX is rejected with the corresponding unsupported-PTX-version diagnostic, so 13.2 remains compile-only/load-rejected proof.
- `eng/Test-CudaRtcBridgePackageConsumer.ps1` creates a repository-external consumer with a cleared, local-only NuGet source and only managed/bridge `PackageReference` entries. It verifies that the bridge package contains only `jyppxtrtbridge.dll`, copies the exact packaged bridge without `JYPPX_NATIVE_BRIDGE_PATH`, diagnoses missing NVRTC while Driver 12090 remains available, and then uses the installed NVRTC 12.9 library for compile, intentional-failure log capture, and Runtime-library/Driver launch/readback/correctness with matching output hashes.
- Evidence is stored in `artifacts/cuda-runtime-compilation/capability-matrix.json`, `driver-capability-matrix.json`, `local-smoke.json`, `native-abi-surface.json`, `kernel-launch-native-abi-surface.json`, and `driver-native-abi-surface.json`. The first three smoke records have Runtime-library and Driver launch/readback/correctness/owner-retention fields set to true; 13.2 explicitly retains them as false.
- The clean consumer record is stored in `artifacts/cuda-runtime-compilation/bridge-package-consumer.json` and `.md`. Its classification is `local-feed-clean-package-consumer-candidate`; it cannot promote public-package or post-publish proof.
- `eng/Test-CudaRtcFullRuntimePackagingPreflight.ps1` now covers all 18 runtime keys without copying or packaging assets. The four Windows version lines match their manifest path, capability-matrix size, and SHA256 pairs and have local license text; Linux remains `0/4`, redistribution and package-host size review are pending, and the split role remains unmaterialized. The report therefore keeps `canMaterializeFullRuntimeCudaRtcRole=false` with four explicit blockers.

The remaining RTC work is Linux runtime proof, materialized full-runtime `cuda-rtc` components, public-source clean consumers, and post-publish verification. Local Windows bridge/Driver launch/readback success and the local-feed package candidate do not promote those lanes.

## Design Invariants

1. `JYPPX_CudaRtcProgram` owns the native compiler program. Create, destroy, and compile are no-throw and contain C++ exceptions, Windows SEH, and NVRTC errors inside the bridge.
2. The owner copies source, program name, header sources/names, compile options, and name expressions. It never retains caller memory or returns `nvrtcProgram` or borrowed lowered-name pointers.
3. Text uses caller-buffer size/copy APIs and binary artifacts use count/copy APIs. Compile logs remain available after compilation failure.
4. Public C# APIs expose no `IntPtr`, `nint`, `UIntPtr`, `SafeHandle`, `nvrtcProgram`, `cudaKernel_t`, `CUmodule`, `CUfunction`, or kernel-argument pointer array.
5. Compilation failure is a diagnostic result that retains the complete log. Invalid lifetime, missing dependencies, and bridge ABI failures continue through the existing status/exception mapping.
6. NVRTC is an optional, diagnosable capability. A consumer that does not use RTC must not fail to load the core bridge solely because NVRTC is absent.
7. Every artifact contains immutable copied bytes and source/options/header/compiler/target/output hashes. Artifacts remain valid after the native program owner is disposed.

## Implemented Managed Surface

The following high-level surface is implemented and covered by the public API gate:

- `CudaRtcCompiler` for capability/version queries and one-shot compilation.
- `CudaRtcProgram` for an explicit reusable `IDisposable` native owner.
- `CudaRtcProgramSource` for immutable source, virtual name, headers, and name expressions.
- `CudaRtcCompileOptions` for immutable target and compiler options.
- `CudaRtcCompilationResult` for status, full log, lowered names, versions, and artifacts.
- `CudaRtcArtifact` and `CudaRtcArtifactKind` for copied PTX, CUBIN, and LTO IR payloads with lengths and SHA256 values.
- `CudaKernelLibrary` for named owner-bound `Launch(...)`.
- `CudaKernelLaunchConfiguration` and `CudaDim3` for non-zero grid/block and dynamic shared-memory configuration.
- `CudaKernelArgument` for copied scalars or owner-bound `CudaMemory` plus a byte offset.
- `CudaKernelLaunch` for completion ownership and leases over library, stream, and device memory.
- `CudaDriver` for optional Driver capability and dependency diagnostics.
- `CudaDriverModule` and `CudaDriverKernelLaunch` for copied module bytes, retained primary context, typed named-kernel launch, and completion ownership.

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

### D. Compile to Load to Launch (local Windows complete)

- On CUDA 12.9/13.2, first verify that NVRTC PTX/CUBIN artifacts load through `CudaKernelLibrary.Load(byte[])`.
- `CudaKernelLibrary.Launch(...)` now provides owner-bound launch by name; borrowed `cudaKernel_t` values stay inside the bridge, while typed scalar/device-memory arguments and completion events remain owner-bound.
- CUDA 11.8/12.1 lack the current Runtime library API. The dynamically loaded `CudaDriverModule` owner now provides the unified module/function path; borrowed functions stay inside the bridge and all module, context, stream, memory, and event lifetimes remain owner-bound.
- Use typed argument packing and bridge-owned launch storage with explicit buffer, scalar, stream, and module lifetimes.

### E. Samples and Runtime Proof (local Windows complete)

Add `samples/Cuda/01.RuntimeCompilation` with a vector-add/elementwise compile-load-launch-readback check, an intentional compiler-error log check, a C++ name-expression/lowered-name check, and artifact metadata/hash export.

The sample also releases the participating owners before synchronization and validates every readback value. Its classification remains local Toolkit runtime proof.

Compile-only output, artifact hashes, synthetic kernels, and a local Toolkit are not clean consumer, public package, or post-publish proof.

### F. Packaging and Cross-Platform Proof

- The bridge-only NuGet package does not bundle NVRTC; callers install a matching CUDA Toolkit and receive focused dependency diagnostics.
- `.Bridge` packages never carry `nvrtc` or `nvrtc-builtins`. Consumers install a matching CUDA Toolkit, and the Windows/Linux diagnostics record machine paths, versions, sizes, and hashes.
- The historical packaging preflight is retained only as a host-dependency identity audit. The `cuda-rtc` package role is `retired-not-packable`, and every explicit non-bridge pack request fails before asset collection.
- Validate Windows x64, Linux x64, and CUDA 11.8/12.1/12.9/13.2. Runtime-key declarations must match actual host dependencies; vendor files are never package assets.
- The Windows CUDA 12.9 local-feed clean consumer now compiles and launches without `ProjectReference` or development probing. Repeat the same smoke against a public source after publishing; the current result must remain a local candidate.

## Evidence Ladder

Header/library audit proves vendor surface only. Compile-only proves compiler ownership and output copying only. Artifact hashes prove identity only. Local compile-to-launch/readback proves one host Toolkit/GPU path only. The local-feed clean consumer proves repository-external `PackageReference` consumption of the current local package pair only. Public-source clean-consumer and post-publish checks remain separate evidence classes, and Owner approval is still required for final publication and issue closure.

CUDA RTC becomes release-ready only after compile, load, launch, readback, clean consumer, cross-platform package, and post-publish evidence all close.
