# Bridge Package Distribution Strategy

TensorRtSharp publishes only the managed C# package, project-owned `.Bridge` packages, and tracked source archives. CUDA, cuDNN, TensorRT, NVRTC, parser, plugin, and builder-resource libraries are consumer-installed prerequisites and never package or Release assets.

## Compatibility keys

Every bridge build is identified by the exact deployment combination:

- TensorRT major/minor, for example `trt10.11`;
- CUDA major/minor, for example `cuda12.9`;
- cuDNN major/minor, for example `cudnn9.22`;
- RID and, on Linux, the target distribution version.

Example key and package ID:

```text
win-x64-trt10.11-cuda12.9-cudnn9.22
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt10.11.cuda12.9.cudnn9.22.Bridge
```

The key selects bridge compilation headers and import libraries and documents the machine-installed runtime expected by smoke tests. It does not authorize bundling those NVIDIA libraries.

## Windows matrix

Current Windows x64 bridge targets:

- `win-x64-trt8.6-cuda11.8-cudnn8.9`;
- `win-x64-trt8.6-cuda12.1-cudnn8.9`;
- `win-x64-trt10.11-cuda11.8-cudnn8.9`;
- `win-x64-trt10.11-cuda12.9-cudnn9.22`;
- `win-x64-trt11.0-cuda12.9-cudnn9.22`;
- `win-x64-trt11.0-cuda13.2-cudnn9.22`.

A successful compile is build evidence only. Runtime proof for each row requires a compatible driver, the matching machine-installed TensorRT/CUDA/cuDNN libraries, a repository-external consumer, enqueue/readback, logs, and hashes. A CUDA 12.9 host cannot close the CUDA 13.2 runtime row.

## Linux matrix

Linux keys include distribution and architecture. Modeled x64 lines cover Ubuntu 20.04, 22.04, and 24.04 where matching NVIDIA repositories exist. ARM64/SBSA, Jetson/L4T, and non-Ubuntu distributions require separate bridge identities, runners or containers, dependency sources, and runtime evidence.

`pack/runtime/linux-runtime-targets.manifest.json` remains the target catalog. `runtime_key_set` aliases may select a matrix for build validation, but every published `.Bridge` package still has its own exact key and proof row.

## Package contents

The sole packable native role is `Bridge`:

- Windows: `runtimes/win-x64/native/jyppxtrtbridge.dll`;
- Linux: `runtimes/<linux-rid>/native/libjyppxtrtbridge.so`.

Historical `CudaCudnn`, `TensorRt`, `CudaRtc`, collection, meta, and full-vendor identities remain in selected manifests only for cleanup and compatibility audit. Their projects are non-packable. `eng/Test-ExternalVendorRuntimePackagePolicy.ps1` rejects those identities and NVIDIA binaries on every current pack or upload path.

## Public channels

The same managed plus bridge-only boundary is available through two channels:

1. A NuGet-compatible source using normal `PackageReference` restore.
2. GitHub Release `.nupkg` assets with immutable download URLs and GitHub SHA256 digests.

GitHub Release is not a NuGet feed. `eng/Invoke-PublicReleaseBridgePackageConsumer.ps1` downloads the managed and bridge assets, verifies the remote digests, package identities, nuspec repository URL/commit, and bridge-only contents, then places the verified files in isolated restore staging. A direct `.nupkg` or DLL reference is not accepted.

Managed and bridge nuspec files must identify the formal repository and the same source commit. `-AllowCrossCommitPair` is diagnostic-only and cannot promote public asset consumer evidence, package-consumer-runtime proof, or post-publish proof.

## Build workflow

Resolve and validate machine-local inputs, then pack only the bridge role:

```powershell
$roots = powershell -ExecutionPolicy Bypass -File .\eng\Resolve-RuntimeRoots.ps1 `
  -RuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22 | ConvertFrom-Json

powershell -ExecutionPolicy Bypass -File .\eng\Validate-WindowsRuntimeInputs.ps1 `
  -RuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22 `
  -TensorRtRoot $roots.tensorRtRoot `
  -CudaRoot $roots.cudaRoot `
  -CudnnRoot $roots.cudnnRoot

powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -RuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22 `
  -SplitPackageRole bridge
```

`eng/Invoke-LocalRuntimePackage.ps1` is retired and fails closed. Non-bridge split roles also fail closed.

Formal releases run only from `guojin-yan/TensorRT-CSharp-API`. The `grape-yan` repository is validation-only and has no package push or Release upload lane. Current release workflows may publish:

- `JYPPX.TensorRT.CSharp.API`;
- matching `.Bridge` packages;
- tracked-files-only source archives.

## Historical cleanup

Vendor-bearing GitHub Package versions and matching Release assets published in June 2026 were removed on 2026-07-30 after exact fingerprint review. Historical release tags and manifest identities may still appear in audits, but they are not a source of current packages and must not be republished.

## Evidence boundary

Public release closure requires all of the following for the intended rows:

- owner-authorized managed, bridge, and source publication;
- public URL, package identity, version, size, and SHA256;
- same-commit managed/bridge provenance;
- repository-external restore/build/runtime smoke;
- host driver, GPU, TensorRT, CUDA, cuDNN, and optional NVRTC metadata;
- runtime stdout/stderr and structured report hashes;
- post-publish clean consumer verification;
- strict validators and final Owner decision.

Local pack, local feed, ProjectReference, direct `.nupkg`, dependency probe, build-only output, historical vendor-package evidence, or a green dashboard cannot substitute for these records.
