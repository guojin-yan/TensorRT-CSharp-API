# Bridge packages

Only project-owned native bridge packages are active in this directory. A bridge package contains exactly one native binary under `runtimes/<rid>/native`:

- Windows: `jyppxtrtbridge.dll`
- Linux: `libjyppxtrtbridge.so`

The package ID records the TensorRT and CUDA build line, for example:

```text
JYPPX.TensorRT.CSharp.API.Runtime.win-x64.trt11.0.cuda12.9.cudnn9.22.Bridge
```

CUDA, cuDNN, TensorRT, NVRTC, parser, plugin, and builder-resource libraries are never included. Consumers install a matching NVIDIA stack and expose it through the operating system loader paths or the documented TensorRtSharp dependency configuration.

Build one bridge package:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Invoke-LocalSplitRuntimePackage.ps1 `
  -SourceRuntimeKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -Version 4.0.0 `
  -SplitPackageRole bridge
```

Build and validate the complete Windows bridge matrix without publishing:

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Invoke-WindowsBridgePackageMatrix.ps1 `
  -Version 4.0.0
```

The matrix runner derives the six Windows bridge combinations from the manifest, creates one report directory per runtime key, checks the exact package allowlist, and rejects any NVIDIA vendor runtime entry. The Linux workflow uses the same bridge-only contract for all modeled Linux keys: nine hosted Ubuntu 22.04/24.04 keys (`runtime_key_set=hosted-all`) and three Ubuntu 20.04 container keys (`runtime_key_set=hosted-container-ubuntu20`). A mixed Linux dispatch may pass all twelve keys explicitly with `runner_mode=any`. Its output is local candidate evidence only: `IsRuntimeExecutionProof=False`, `IsPackageConsumerRuntimeProof=False`, `CanPublishPublicly=False`, and `PublicationExecuted=False`. Use `-RunDependencyProbe` only for dependency diagnostics; it still does not execute TensorRT inference or publish a package.

The `4.0.0-preview.1` candidate publishes all twelve Linux `.Bridge.nupkg` files to the GitHub Packages NuGet feed and attaches them to the matching GitHub Release. The six Windows packages remain blocked until a compatible Windows/x64 self-hosted runner with the required NVIDIA SDKs is online; the workflow must not substitute an incompatible hosted image. No runtime package is pushed to nuget.org, and no CUDA, cuDNN, TensorRT, NVRTC, parser, plugin, or builder-resource library is included.

Validate package contents before any upload:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-ExternalVendorRuntimePackagePolicy.ps1 `
  -PackagePath .\artifacts\runtime-split-nupkg\win-x64-trt11.0-cuda12.9-cudnn9.22
```

Non-bridge project files have been removed. Their manifest entries remain only as non-packable historical identities for compatibility audits and interpretation of old evidence. Requests for `all`, `cuda-cudnn`, `tensorrt`, `cuda-rtc`, `collection`, or `meta` fail before asset collection.

`eng/Test-BridgePackageConsumer.ps1` validates restore, bridge layout, managed wrapper compile surface, and dependency diagnostics. `eng/Test-BridgePackageRuntimeConsumer.ps1` adds runtime execution against compatible system-installed NVIDIA dependencies from a bridge-only package set. For TensorRT 10/11 it also requires a real DebugListener attach/invoke/detach cycle from the external PackageReference-only consumer: invocation must be positive, failure and in-flight counts must be zero, copied metadata must be pointer-free, and detach must succeed. The report keeps `source-tree`, `local-package`, `public-package`, and `post-publish` scopes separate. A successful local package callback run does not prove a public package or post-publish install.

Per-feature design gates and runtime evidence live in the dedicated articles and validation reports. They are intentionally not copied into this package inventory: internal readiness markers do not change bridge contents and are not publication evidence.
