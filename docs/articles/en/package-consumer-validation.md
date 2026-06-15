# Package Consumer Validation

`eng/Test-PackageConsumer.ps1` creates a temporary consumer console application for each selected runtime key, installs the managed package and matching runtime package from local package folders, builds the app, and verifies copied managed/native assets.

Runtime keys now include TensorRT / CUDA / cuDNN `major.minor` versions.

Current stable Windows smoke example:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 `
  -RuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9 `
  -RunSmoke `
  -SmokeRuntimePackageKey win-x64-trt10.11-cuda11.8-cudnn8.9
```

Matrix example:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 `
  -RuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22,win-x64-trt11.0-cuda12.9-cudnn9.22,win-x64-trt11.0-cuda13.2-cudnn9.22 `
  -RunSmoke `
  -SmokeRuntimePackageKey win-x64-trt10.11-cuda12.9-cudnn9.22,win-x64-trt11.0-cuda12.9-cudnn9.22
```

The script reports package ID/version, expected native asset count, found native asset count, missing assets, elapsed time, and optional smoke result.

By default, the generated consumer project and the short-lived NuGet restore cache are deleted after each runtime key. This keeps self-hosted runners from accumulating several TensorRT/CUDA/cuDNN copies during matrix packaging. When investigating a local restore or native-copy issue, pass `-KeepConsumerOutput` to preserve `build-out/package-consumer/<runtime-key>`.

Latest local evidence, generated on 2026-06-12:

- `win-x64-trt10.11-cuda11.8-cudnn8.9`: `16/16` native assets, smoke `passed`, probe output TensorRT `10.11.0`, CUDA `11.8`.
- `win-x64-trt10.11-cuda12.9-cudnn9.22`: `19/19` native asset patterns, smoke `passed`, probe output TensorRT `10.11.0`, CUDA `12.9`.
- `win-x64-trt11.0-cuda12.9-cudnn9.22`: `19/19` native asset patterns, smoke `passed`, probe output TensorRT `11.0.0`, CUDA `12.9`.
- `win-x64-trt11.0-cuda13.2-cudnn9.22`: `19/19` native asset patterns, restore/build/native-copy passed; smoke is `not-requested` on this driver/runtime stack.
- Managed package: `JYPPX.TensorRT.CSharp.API 4.0.0-alpha.1`.
- Report: `artifacts/package-consumer/package-consumer-validation-summary.md`.

Consumer validation is a release gate after interface coverage reaches zero missing rows. A package that restores and copies native assets correctly is not sufficient by itself: at least one intended release runtime key should also have smoke evidence on a compatible machine before it is treated as release-candidate usable.

On Windows machines with WDAC / application-control policies, unsigned freshly built consumer outputs can be blocked with `0x800711C7` even when package restore and native asset copy are correct. Pass `-SignConsumerOutput` to sign the generated consumer app, managed assemblies, and bridge DLL with the local development code-signing certificate before running smoke. If the local policy also requires the certificate to be trusted for the current user, add `-TrustSigningCertificate` and `-TrustSigningCertificateRoot`:

```powershell
powershell -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumer.ps1 `
  -RuntimePackageKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -RunSmoke `
  -SmokeRuntimePackageKey win-x64-trt11.0-cuda12.9-cudnn9.22 `
  -SignConsumerOutput `
  -TrustSigningCertificate `
  -TrustSigningCertificateRoot
```

CUDA `12.9` target packages must be validated with the installed CUDA `12.9` toolkit and matching TensorRT/cuDNN assets. The earlier CUDA `12.3` interim fallback is retired.

Current local package-consumer expectations:

- `win-x64-trt10.11-cuda11.8-cudnn8.9` is the current stable real vendor-backed smoke path and has 2026-06-12 package-consumer smoke evidence.
- `win-x64-trt10.11-cuda12.9-cudnn9.22` and `win-x64-trt11.0-cuda12.9-cudnn9.22` have current CUDA `12.9` package-consumer smoke evidence.
- `win-x64-trt11.0-cuda13.2-cudnn9.22` packed the full split package set and passed restore/build/native-copy validation on 2026-06-14 with `19/19` native asset patterns copied, but readiness remains blocked until CUDA 13 runtime/builder smoke is validated on a compatible driver/runtime stack.
