# Linux Runner Setup

## Purpose

Linux runtime packaging is currently prepared structurally, but it is expected to run on a self-hosted Linux x64 runner.

## Required runner capabilities

- Linux x64 host
- self-hosted runner labels: `self-hosted`, `linux`, `x64`
- .NET 10 SDK
- `pwsh`
- CMake
- matching CUDA Toolkit installation
- matching TensorRT unpacked root

## Expected workflow inputs

For `manual-pack-runtime-linux.yml`, provide:

- `runtime_key`
- `configure_preset`
- `build_preset`
- `tensorrt_root`
- `cuda_root`

## Expected root examples

- `tensorrt_root=/opt/tensorrt/trt10-cuda11`
- `cuda_root=/usr/local/cuda-11.8`

## Dry-run validation idea

Before trying a full pack:

1. run `pwsh -File ./eng/Validate-RuntimeManifest.ps1`
2. run `pwsh -File ./eng/Validate-LinuxRuntimeInputs.ps1 -RuntimePackageKey <linux key> -TensorRtRoot <path> -CudaRoot <path>`
3. run `pwsh -File ./eng/Invoke-LinuxRuntimeDryRun.ps1 -RuntimePackageKey <linux key> -TensorRtRoot <path> -CudaRoot <path>`
4. run `pwsh -File ./eng/Validate-LinuxDryRunArtifacts.ps1 -RuntimePackageKey <linux key>`
5. inspect `artifacts/linux-dry-run/<key>/linux-runtime-dry-run.json`
6. inspect `artifacts/linux-dry-run/<key>/linux-runner-checklist.md`
7. run `pwsh -File ./eng/Export-LinuxPreflightSummary.ps1 -RuntimePackageKey <linux key>`
8. run `pwsh -File ./eng/Export-LinuxPackageConsumerPlan.ps1 -RuntimePackageKey <linux key>`
9. run `pwsh -File ./eng/Export-LinuxRunnerExecutionStatus.ps1 -RuntimePackageKey <linux key>`
10. run `pwsh -File ./eng/Export-LinuxHandoffIndex.ps1 -RuntimePackageKey <linux key>`
11. then run:
   - `cmake --preset <linux preset>`
   - `cmake --build --preset <linux preset>`
   - `pwsh -File ./eng/Collect-RuntimeAssets.ps1 -RuntimePackageKey <linux key> -TensorRtRoot <path> -CudaRoot <path>`
   - `dotnet pack ./pack/runtime/<key>/<packageId>.csproj -c Release -o ./artifacts/runtime-nupkg`

## Current dry-run outputs

- `artifacts/linux-dry-run/<key>/linux-runtime-dry-run.json`
- `artifacts/linux-dry-run/<key>/README.md`
- `artifacts/linux-dry-run/<key>/linux-runner-checklist.md`
- `artifacts/linux-dry-run/<key>/linux-preflight-summary.md`
- `artifacts/linux-dry-run/<key>/linux-workflow-contract.md`
- `artifacts/linux-dry-run/<key>/linux-handoff-index.md`
- `artifacts/linux-dry-run/<key>/linux-package-consumer-plan.md`
- `artifacts/linux-dry-run/<key>/linux-package-consumer-plan.json`
- `artifacts/linux-dry-run/<key>/linux-runner-execution-status.md`
- `artifacts/linux-dry-run/<key>/linux-runner-execution-status.json`

Supporting validation script:

- `eng/Validate-LinuxDryRunArtifacts.ps1`
- `eng/Export-LinuxPreflightSummary.ps1`
- `eng/Test-LinuxRuntimeWorkflowContract.ps1`
- `eng/Export-LinuxHandoffIndex.ps1`
- `eng/Export-LinuxPackageConsumerPlan.ps1`
- `eng/Export-LinuxRunnerExecutionStatus.ps1`

Current workflow consumption order:

1. `Validate-RuntimeManifest`
2. `Validate-LinuxRuntimeInputs`
3. `Invoke-LinuxRuntimeDryRun`
4. `Validate-LinuxDryRunArtifacts`
5. `Test-LinuxRuntimeWorkflowContract`
6. `Export-LinuxPreflightSummary`
7. `Export-LinuxPackageConsumerPlan`
8. `Export-LinuxRunnerExecutionStatus`
9. `Collect-RuntimeAssets`
10. `dotnet pack` for the managed package
11. `dotnet pack` for the selected Linux runtime package
12. `Test-PackageConsumer` without smoke
13. `Test-RuntimePublishReadiness`

Those files serve different audiences:

- JSON summary: machine-readable handoff for scripts and workflow inspection
- README: human-readable quick overview of roots, commands, and expected outputs
- checklist: step-by-step preflight / build / pack instructions for the Linux runner maintainer

## Expected artifact rules

- native bridge output: `build-out/<preset>/bin/Release/<bridgeFile>`
- import libraries and adjacent native outputs: `build-out/<preset>/lib/Release/`
- collected runtime asset root: `artifacts/runtime/<key>/runtimes/<rid>/native/`
- collected asset manifest: `artifacts/runtime/<key>/artifact-manifest.json`
- runtime nupkg output: `artifacts/runtime-nupkg/`
- Linux package consumer report: `artifacts/package-consumer/package-consumer-validation-summary.md`
- Linux package consumer output: `build-out/package-consumer/<key>/bin/Release/<tfm>/linux-x64/`

## Linux package consumer validation

After runtime assets have been collected and the Linux runtime package has been packed, run:

```bash
pwsh -File ./eng/Test-PackageConsumer.ps1 -RuntimePackageKey <linux key>
```

This checks local package restore, consumer build, managed assembly copy, bridge copy, and TensorRT/CUDA `.so` asset copy.

Only add `-RunSmoke` when the self-hosted runner has a usable NVIDIA driver, a compatible GPU, and runtime access to the selected CUDA/TensorRT combination.

Linux packages cannot move from `dry-run-only` to `local-validated` until this non-smoke consumer validation passes on a real Linux x64 runner.

## Current status

The repository now includes Linux package manifest entries and Linux pack workflows, but they are not yet validated on this Windows workstation.

## Common failure cases to check first

- `TensorRT root` does not match the requested runtime key line
- `CUDA root` does not match the requested runtime key line
- expected `.so` wildcard patterns do not resolve
- self-hosted runner is missing `pwsh`
- self-hosted runner is missing `dotnet` or `cmake`
- native build succeeds, but `Collect-RuntimeAssets.ps1` still fails because the supplied roots do not match the manifest layout assumptions
- artifact upload succeeds but native assets are incomplete because the roots were wrong
