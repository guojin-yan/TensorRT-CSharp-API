# Linux Runner Setup

## Purpose

Linux runtime packaging now supports both GitHub-hosted and self-hosted execution paths. The default release line uses GitHub-hosted Ubuntu 22.04 x64 runners; other distributions and architectures must be selected through explicit runtime keys.

## Linux Matrix Boundaries

- Ubuntu 22.04 x64 is the default release line. `runtime-linux.yml` builds the six configured TensorRT / CUDA / cuDNN combinations for this distribution by default.
- Ubuntu 24.04 x64 can run on GitHub-hosted runners, but NVIDIA apt repositories do not cover the older TensorRT 8.6 / CUDA 11.8 / CUDA 12.0 lines there. Keep Ubuntu 24.04 to the newer TensorRT 10.11 / 11.0 combinations.
- Ubuntu 20.04 x64 is no longer available as a GitHub-hosted runner. The manifest keeps those runtime keys, but publishing them requires a self-hosted runner or explicitly prepared official NVIDIA roots.
- ARM64 / Jetson / L4T should not be mixed into the generic `linux-x64` package line. They need separate runtime keys, RIDs, runner labels, NVIDIA repo architecture, and L4T dependency handling.

## Required runner capabilities

- Linux x64 host; Ubuntu 22.04 / 24.04 can use GitHub-hosted runners, while Ubuntu 20.04 requires self-hosted execution
- recommended self-hosted runner labels: `self-hosted`, `linux`, `x64`, plus the target distribution label such as `ubuntu-20.04`
- .NET 10 SDK
- `pwsh`
- CMake
- matching CUDA Toolkit installed from an official NVIDIA distribution
- matching TensorRT Linux package downloaded from NVIDIA and unpacked
- matching cuDNN Linux package downloaded from NVIDIA and installed or unpacked

Do not commit CUDA, cuDNN, or TensorRT binaries to Git. The hosted Linux path installs publicly available runtime libraries and development headers from NVIDIA's official apt repositories; the self-hosted path reads already prepared official package roots from the runner.

## Ubuntu 20.04 self-hosted runner bootstrap

Use the helper below on the Ubuntu 20.04 x64 machine after signing in with `gh` as an account that can request repository runner registration tokens:

```bash
pwsh -File ./eng/Install-GitHubSelfHostedRunner.ps1 \
  -Repository guojin-yan/TensorRT-CSharp-API \
  -UseGhRegistrationToken \
  -InstallService
```

The helper downloads the official GitHub Actions runner, configures it with `ubuntu-20.04,tensorrt-csharp` custom labels, and leaves the built-in `self-hosted,linux,x64` labels intact. It never writes the short-lived registration token to disk. Use `-DryRun` first when you want to preview the exact commands.

Before dispatching the Ubuntu 20.04 runtime lane, run:

```bash
pwsh -File ./eng/Test-LinuxSelfHostedRunnerReadiness.ps1 \
  -RequireRegisteredRunner \
  -CheckGitHubRunner
```

The readiness check writes `artifacts/linux-self-hosted-runner-readiness/linux-self-hosted-runner-readiness.json` and `.md`. It verifies the OS, x64 architecture, `pwsh`, `.NET SDK 10.0.300+`, CMake, Ninja, Git, `gh`, local runner configuration, GitHub runner labels, and the NVIDIA roots resolved for `self-hosted-ubuntu20`.

After the runner is visible in GitHub, you can run the same check remotely with `linux-self-hosted-runner-readiness.yml`. It targets `self-hosted,linux,x64,ubuntu-20.04`, uploads the same readiness artifacts, and does not publish packages. Use it as the final gate before enabling `run_linux_self_hosted_ubuntu20_runtime_packaging=true` in `release-bundle.yml`.

## Expected workflow inputs

For `runtime-linux.yml`, provide:

- `version`
- `runtime_keys`: comma-separated Linux runtime keys
- `runtime_key_set`: use `ubuntu22-hosted`, `hosted-all`, `ubuntu24-hosted`, `self-hosted-ubuntu20`, or `custom` when `runtime_keys` is empty
- `runner_mode`: defaults to `hosted`; use `self-hosted` for Ubuntu 20.04 or manually prepared roots
- `split_package_roles`: use `all` for a dependency refresh, or `bridge,collection` when reusing existing CUDA/cuDNN and TensorRT component packages
- `cuda_cudnn_package_version` and `tensorrt_package_version`: required when publishing `collection` or `meta` without rebuilding those stable dependencies and every requested runtime key uses the same stable dependency version
- `cuda_cudnn_package_version_map` and `tensorrt_package_version_map`: use these instead of one global version when requested runtime keys reference different dependency publication versions, for example `linux-x64-ubuntu22.04-*=4.0.6167;linux-x64-ubuntu24.04-*=4.0.6169`
- `run_smoke`: enable only when the runner has a compatible NVIDIA GPU, driver, and runtime stack
- `publish_to_github_packages`: defaults to false because large Linux runtime packages should normally stay as GitHub Release assets
- `release_tag`
- `attach_to_github_release`

For `release-bundle.yml`, the hosted Linux lane uses `run_linux_runtime_packaging=true` and defaults to `linux_runtime_key_set=hosted-all`, which dispatches Ubuntu 22.04 x64 plus the modeled Ubuntu 24.04 x64 packages. Ubuntu 20.04 is a separate self-hosted lane: use `run_linux_self_hosted_ubuntu20_runtime_packaging=true`; it defaults to `self-hosted-ubuntu20` and dispatches `runner_mode=self-hosted`.

`release-bundle.yml` checks runner availability before creating a GitHub Release when repository secret `RUNNER_AUDIT_TOKEN` is available. GitHub documents the repository self-hosted runner list API as requiring a fine-grained token with `Administration` repository permission set to `read`: <https://docs.github.com/rest/actions/self-hosted-runners>. Ubuntu 20.04 self-hosted packaging is strict: it requires that audit token and an online repository runner with `self-hosted`, `linux`, `x64`, and `ubuntu-20.04` labels.

`release-bundle.yml` keeps common choices as top-level inputs and accepts advanced overrides through `release_config_json`. Use that JSON object for less common values such as `linux_runtime_delivery_mode`, `linux_self_hosted_ubuntu20_runtime_key_set`, package release-tag overrides, package version maps, bridge/meta package version overrides, and skip-validation toggles.

## Expected root examples

- TensorRT: `/opt/tensorrt/trt10-cuda12.9`
- CUDA: `/usr/local/cuda-12.9`
- cuDNN: `/opt/cudnn/cuda12`

Root resolution order:

1. `pack/runtime/runtime-packages.local.json`
2. the JSON file pointed to by `JYPPX_RUNTIME_PACKAGE_ROOTS_FILE`
3. `~/.jyppx/runtime-packages.local.json`
4. the default Linux roots in `pack/runtime/runtime-packages.manifest.json`

Start from `pack/runtime/runtime-packages.local.example.json` and adjust it for the runner. The example now includes both Windows and Linux matrices. `Resolve-RuntimeRoots.ps1` expands `<repo-root>`, environment variables, and `~`.

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
2. `Prepare-LinuxNvidiaDependencies`, which installs CUDA / cuDNN / TensorRT from NVIDIA official apt repositories on hosted runners
3. `Validate-LinuxRuntimeInputs`
4. `cmake --preset`
5. `cmake --build --preset`
6. `Invoke-LinuxRuntimeDryRun`
7. `Validate-LinuxDryRunArtifacts`
8. `Test-LinuxRuntimeWorkflowContract`
9. `Export-LinuxPreflightSummary`
10. `Export-LinuxPackageConsumerPlan`
11. `Export-LinuxRunnerExecutionStatus`
12. `Collect-RuntimeAssets`
13. split runtime pack
14. GitHub Release asset upload

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

The repository now includes Linux package manifest entries and Linux pack workflows. Ubuntu 22.04 hosted runners are the default remote packaging validation line; Ubuntu 20.04, ARM64, and Jetson still need dedicated runner and dependency strategies before publishing.

## Common failure cases to check first

- `TensorRT root` does not match the requested runtime key line
- `CUDA root` does not match the requested runtime key line
- expected `.so` wildcard patterns do not resolve
- self-hosted runner is missing `pwsh`
- self-hosted runner is missing `dotnet` or `cmake`
- native build succeeds, but `Collect-RuntimeAssets.ps1` still fails because the supplied roots do not match the manifest layout assumptions
- artifact upload succeeds but native assets are incomplete because the roots were wrong
