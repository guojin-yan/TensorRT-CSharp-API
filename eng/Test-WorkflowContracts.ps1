[CmdletBinding()]
param(
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function New-Requirement {
  param(
    [string]$Needle,
    [string]$Description
  )

  return [pscustomobject]@{
    needle = $Needle
    description = $Description
  }
}

function Test-Workflow {
  param(
    [string]$RelativePath,
    [object[]]$Requirements
  )

  $path = Join-Path $RepositoryRoot $RelativePath
  $checks = New-Object System.Collections.Generic.List[object]

  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    $checks.Add([pscustomobject]@{
      workflow = $RelativePath
      requirement = "file exists"
      status = "failed"
      detail = "Workflow file is missing."
    })
    return @($checks.ToArray())
  }

  $content = Get-Content -LiteralPath $path -Raw -Encoding utf8
  foreach ($requirement in $Requirements) {
    $present = $content.IndexOf($requirement.needle, [System.StringComparison]::OrdinalIgnoreCase) -ge 0
    $checks.Add([pscustomobject]@{
      workflow = $RelativePath
      requirement = $requirement.description
      status = if ($present) { "passed" } else { "failed" }
      detail = $requirement.needle
    })
  }

  return @($checks.ToArray())
}

$workflowContracts = @(
  [pscustomobject]@{
    path = ".github\workflows\docs-release.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "dotnet tool restore" -Description "restore local tools"
      New-Requirement -Needle "dotnet docfx" -Description "DocFX build"
      New-Requirement -Needle "actions/deploy-pages" -Description "GitHub Pages deploy"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\package-managed.yml"
    requirements = @(
      New-Requirement -Needle "workflow_call" -Description "reusable workflow entrypoint"
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "Resolve-PackageVersion.ps1" -Description "version normalization"
      New-Requirement -Needle "Test-BindingGeneratorOutputs.ps1" -Description "binding generator determinism"
      New-Requirement -Needle "Test-ManagedPackageContent.ps1" -Description "managed package content validation"
      New-Requirement -Needle "Push-NuGetPackages.ps1" -Description "package publication"
      New-Requirement -Needle "release_tag" -Description "GitHub Release tag input"
      New-Requirement -Needle "attach_to_github_release" -Description "GitHub Release asset upload toggle"
      New-Requirement -Needle "gh release upload" -Description "managed package release asset upload"
      New-Requirement -Needle "-ApiKeyEnvironmentVariable NUGET_API_KEY" -Description "nuget.org publish uses repository secret"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\runtime-windows.yml"
    requirements = @(
      New-Requirement -Needle "workflow_call" -Description "reusable workflow entrypoint"
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "self-hosted" -Description "self-hosted runner"
      New-Requirement -Needle "windows" -Description "windows runner label"
      New-Requirement -Needle "Invoke-LocalRuntimePackage.ps1" -Description "full runtime packaging entrypoint"
      New-Requirement -Needle "Invoke-LocalSplitRuntimePackage.ps1" -Description "split runtime packaging entrypoint"
      New-Requirement -Needle "Restore-PublishedSplitPackageSource.ps1" -Description "published stable dependency release asset package source"
      New-Requirement -Needle "cuda_cudnn_package_version" -Description "CUDA/cuDNN split version input"
      New-Requirement -Needle "tensorrt_package_version" -Description "TensorRT split version input"
      New-Requirement -Needle "cuda_cudnn_package_release_tag" -Description "CUDA/cuDNN split release tag input"
      New-Requirement -Needle "tensorrt_package_release_tag" -Description "TensorRT split release tag input"
      New-Requirement -Needle "sign_consumer_output" -Description "consumer signing toggle"
      New-Requirement -Needle "Push-NuGetPackages.ps1" -Description "package publication"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\runtime-linux.yml"
    requirements = @(
      New-Requirement -Needle "workflow_call" -Description "reusable workflow entrypoint"
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "Resolve-RuntimeMatrix.ps1" -Description "runtime matrix generation"
      New-Requirement -Needle "Resolve-RuntimeKeySet.ps1" -Description "runtime key set generation"
      New-Requirement -Needle "runtime_key_set" -Description "Linux runtime key set input"
      New-Requirement -Needle "Resolve-RuntimeRoots.ps1" -Description "runtime root resolution"
      New-Requirement -Needle "Pack managed package" -Description "Linux runtime managed package artifact build"
      New-Requirement -Needle "managed-packages-runtime-linux" -Description "Linux runtime managed package artifact"
      New-Requirement -Needle "runner_mode" -Description "hosted/self-hosted runner mode"
      New-Requirement -Needle "Prepare-LinuxNvidiaDependencies.ps1" -Description "hosted Linux NVIDIA dependency preparation"
      New-Requirement -Needle "fromJson(matrix.runsOnJson)" -Description "manifest-driven Linux runner labels"
      New-Requirement -Needle "Restore-PublishedSplitPackageSource.ps1" -Description "published stable dependency release asset package source"
      New-Requirement -Needle "cuda_cudnn_package_release_tag" -Description "Linux split CUDA/cuDNN release tag input"
      New-Requirement -Needle "tensorrt_package_release_tag" -Description "Linux split TensorRT release tag input"
      New-Requirement -Needle "linux" -Description "linux runner label"
      New-Requirement -Needle "Validate-LinuxRuntimeInputs.ps1" -Description "Linux input validation"
      New-Requirement -Needle "CudnnRoot" -Description "Linux cuDNN input validation"
      New-Requirement -Needle "Invoke-LinuxRuntimeDryRun.ps1" -Description "Linux dry-run"
      New-Requirement -Needle "Collect-RuntimeAssets.ps1" -Description "runtime asset collection"
      New-Requirement -Needle "Test-PackageConsumer.ps1" -Description "package consumer validation"
      New-Requirement -Needle "Push-NuGetPackages.ps1" -Description "package publication"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\release-bundle.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "gh workflow run" -Description "child workflow dispatch"
      New-Requirement -Needle "package-managed" -Description "managed package child workflow"
      New-Requirement -Needle "runtime-windows" -Description "Windows runtime child workflow"
      New-Requirement -Needle "runtime-linux" -Description "Linux runtime child workflow"
      New-Requirement -Needle "gh release create" -Description "release creation"
      New-Requirement -Needle "gh release upload" -Description "release asset upload"
      New-Requirement -Needle "linux_runner_mode" -Description "Linux runner mode input"
      New-Requirement -Needle "linux_runtime_key_set" -Description "Linux runtime key set input"
      New-Requirement -Needle "linux_split_package_roles" -Description "Linux split package roles input"
      New-Requirement -Needle "run_linux_self_hosted_ubuntu20_runtime_packaging" -Description "Ubuntu 20.04 self-hosted Linux packaging toggle"
      New-Requirement -Needle "release_config_json" -Description "advanced release configuration JSON input"
      New-Requirement -Needle "get_config" -Description "advanced release configuration parser"
      New-Requirement -Needle "Test-GitHubRunnerAvailability.ps1" -Description "runner availability preflight"
      New-Requirement -Needle "RUNNER_AUDIT_TOKEN" -Description "runner audit token secret"
      New-Requirement -Needle "self-hosted,windows,x64" -Description "Windows self-hosted runner label preflight"
      New-Requirement -Needle "self-hosted,linux,x64,ubuntu-20.04" -Description "Ubuntu 20.04 self-hosted runner label preflight"
      New-Requirement -Needle "Self-hosted Linux runtime packaging requires RUNNER_AUDIT_TOKEN" -Description "strict Linux self-hosted runner preflight token guard"
      New-Requirement -Needle "linux_cuda_cudnn_package_version" -Description "Linux split CUDA/cuDNN version input"
      New-Requirement -Needle "linux_tensorrt_package_version" -Description "Linux split TensorRT version input"
      New-Requirement -Needle "runtime-linux-hosted" -Description "hosted Linux child workflow label"
      New-Requirement -Needle "runtime-linux-self-hosted-ubuntu20" -Description "Ubuntu 20.04 self-hosted Linux child workflow label"
      New-Requirement -Needle "hosted-all" -Description "hosted Linux default key set"
      New-Requirement -Needle "self-hosted-ubuntu20" -Description "Ubuntu 20.04 self-hosted key set"
      New-Requirement -Needle "runtime_key_set=$LINUX_RUNTIME_KEY_SET" -Description "Linux runtime key set dispatch"
      New-Requirement -Needle "runner_mode=$LINUX_RUNNER_MODE" -Description "Linux runner mode dispatch"
      New-Requirement -Needle "runtime_key_set=$LINUX_SELF_HOSTED_UBUNTU20_RUNTIME_KEY_SET" -Description "Ubuntu 20.04 self-hosted Linux runtime key set dispatch"
      New-Requirement -Needle "runner_mode=self-hosted" -Description "Ubuntu 20.04 self-hosted Linux runner mode dispatch"
      New-Requirement -Needle "linux_split_package_roles includes collection/meta but no CUDA/cuDNN package version was provided" -Description "Linux split collection CUDA/cuDNN version guard"
      New-Requirement -Needle "linux_split_package_roles includes collection/meta but no TensorRT package version was provided" -Description "Linux split collection TensorRT version guard"
      New-Requirement -Needle "windows_split_package_roles includes collection/meta but no CUDA/cuDNN package version was provided" -Description "split collection CUDA/cuDNN version guard"
      New-Requirement -Needle "windows_split_package_roles includes collection/meta but no TensorRT package version was provided" -Description "split collection TensorRT version guard"
      New-Requirement -Needle "include cuda-cudnn/all in windows_split_package_roles" -Description "split collection same-run CUDA/cuDNN refresh guidance"
      New-Requirement -Needle "include tensorrt/all in windows_split_package_roles" -Description "split collection same-run TensorRT refresh guidance"
      New-Requirement -Needle "windows_cuda_cudnn_package_release_tag" -Description "split collection CUDA/cuDNN release tag override"
      New-Requirement -Needle "windows_tensorrt_package_release_tag" -Description "split collection TensorRT release tag override"
      New-Requirement -Needle 'attach_to_github_release=$ATTACH_RUNTIME_TO_GITHUB_RELEASE' -Description "managed package release asset toggle dispatch"
      New-Requirement -Needle "publish_managed_to_nuget=true requires the repository secret NUGET_API_KEY" -Description "nuget.org secret guard before release creation"
    )
  }
  [pscustomobject]@{
    path = "eng\Resolve-RuntimeKeySet.ps1"
    requirements = @(
      New-Requirement -Needle "auto" -Description "auto key set selection"
      New-Requirement -Needle "hosted-all" -Description "hosted Linux all key set"
      New-Requirement -Needle "ubuntu24-hosted" -Description "Ubuntu 24.04 hosted key set"
      New-Requirement -Needle "self-hosted-ubuntu20" -Description "Ubuntu 20.04 self-hosted key set"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\publish-release-nuget-assets.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "Publish-ReleaseNuGetAssetsToGitHubPackages.ps1" -Description "release asset publication script"
      New-Requirement -Needle "Test-GitHubPackagesCoverage.ps1" -Description "post-publish GitHub Packages coverage audit"
      New-Requirement -Needle "packages: write" -Description "GitHub Packages write permission"
      New-Requirement -Needle "asset_patterns" -Description "release asset pattern input"
      New-Requirement -Needle "asset_names" -Description "exact release asset names input"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\release-publication-audit.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "Test-ReleasePublicationState.ps1" -Description "release publication state audit script"
      New-Requirement -Needle "HAS_NUGET_API_KEY" -Description "secret availability is passed without listing secrets"
      New-Requirement -Needle "runner_required_label_sets" -Description "runner label audit input"
      New-Requirement -Needle "RUNNER_AUDIT_TOKEN" -Description "runner audit token secret"
      New-Requirement -Needle "Test-GitHubRunnerAvailability.ps1" -Description "runner availability audit script"
      New-Requirement -Needle "Test-LinuxRuntimeTargetCoverage.ps1" -Description "Linux target coverage audit script"
      New-Requirement -Needle "RequireRuntimeGitHubPackagesCoverage" -Description "runtime GitHub Packages coverage gate"
      New-Requirement -Needle "actions/upload-artifact" -Description "audit artifact upload"
      New-Requirement -Needle "packages: read" -Description "GitHub Packages read permission"
    )
  }
  [pscustomobject]@{
    path = "eng\Publish-ReleaseNuGetAssetsToGitHubPackages.ps1"
    requirements = @(
      New-Requirement -Needle '$ErrorActionPreference = "Stop"' -Description "fail-fast PowerShell errors"
      New-Requirement -Needle "A selected release asset has an empty name" -Description "empty release asset guard"
      New-Requirement -Needle "Failed to publish release asset" -Description "NuGet push exit-code guard"
      New-Requirement -Needle 'Selected $(' -Description "selected asset count logging"
    )
  }
  [pscustomobject]@{
    path = "eng\Test-GitHubPackagesCoverage.ps1"
    requirements = @(
      New-Requirement -Needle '$PackageOwnerKind = "auto"' -Description "automatic user/org package owner detection"
      New-Requirement -Needle "GitHub Packages coverage audit" -Description "coverage report generation"
      New-Requirement -Needle "Missing package versions" -Description "missing package version reporting"
      New-Requirement -Needle "versionExists" -Description "package version presence check"
      New-Requirement -Needle "AssetName" -Description "exact asset name filtering"
    )
  }
  [pscustomobject]@{
    path = "eng\Test-ReleasePublicationState.ps1"
    requirements = @(
      New-Requirement -Needle "NUGET_API_KEY" -Description "nuget.org secret audit"
      New-Requirement -Needle "NuGetApiKeyAvailable" -Description "workflow-provided secret availability"
      New-Requirement -Needle "Test-GitHubPackagesCoverage.ps1" -Description "runtime GitHub Packages coverage delegation"
      New-Requirement -Needle "api.nuget.org/v3-flatcontainer" -Description "nuget.org managed package visibility audit"
      New-Requirement -Needle "CheckFailedWorkflowRuns" -Description "failed workflow run audit"
      New-Requirement -Needle "RequireManagedReleaseAsset" -Description "managed release asset gate"
    )
  }
  [pscustomobject]@{
    path = "eng\Test-GitHubRunnerAvailability.ps1"
    requirements = @(
      New-Requirement -Needle "actions/runners" -Description "GitHub Actions runner API query"
      New-Requirement -Needle "RequiredLabelSet" -Description "required runner label set input"
      New-Requirement -Needle "onlineMatchingRunnerCount" -Description "online matching runner audit"
      New-Requirement -Needle "github-runner-availability" -Description "runner availability report"
      New-Requirement -Needle "WarnOnly" -Description "non-failing audit mode"
    )
  }
  [pscustomobject]@{
    path = "eng\Test-LinuxRuntimeTargetCoverage.ps1"
    requirements = @(
      New-Requirement -Needle "ubuntu22.04-x64-hosted" -Description "Ubuntu 22.04 hosted target coverage"
      New-Requirement -Needle "ubuntu24.04-x64-hosted" -Description "Ubuntu 24.04 hosted target coverage"
      New-Requirement -Needle "ubuntu20.04-x64-self-hosted" -Description "Ubuntu 20.04 self-hosted target coverage"
      New-Requirement -Needle "linux-arm64-sbsa" -Description "future SBSA package-line guard"
      New-Requirement -Needle "linux-jetson-l4t" -Description "future Jetson/L4T package-line guard"
      New-Requirement -Needle "non-ubuntu-linux" -Description "future non-Ubuntu package-line guard"
      New-Requirement -Needle "linux-runtime-target-coverage" -Description "target coverage report"
    )
  }
)

$results = New-Object System.Collections.Generic.List[object]
foreach ($contract in $workflowContracts) {
  foreach ($result in @(Test-Workflow -RelativePath $contract.path -Requirements $contract.requirements)) {
    $results.Add($result)
  }
}

$singleLinuxMatrixJson = (pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeMatrix.ps1") -Platform linux -RuntimeKey "linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22" | Out-String).Trim()
$results.Add([pscustomobject]@{
    workflow = "eng\Resolve-RuntimeMatrix.ps1"
    requirement = "single runtime key emits a JSON array"
    status = if ($singleLinuxMatrixJson.StartsWith("[")) { "passed" } else { "failed" }
    detail = "linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22"
  })

$ubuntu22KeySet = @((pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeKeySet.ps1") -Platform linux -RuntimeKeySet auto -RunnerMode hosted -OutputFormat json | Out-String).Trim() | ConvertFrom-Json)
$results.Add([pscustomobject]@{
    workflow = "eng\Resolve-RuntimeKeySet.ps1"
    requirement = "Auto hosted Linux key set resolves all six Ubuntu 22.04 dependency combinations"
    status = if ($ubuntu22KeySet.Count -eq 6 -and $ubuntu22KeySet -contains "linux-x64-ubuntu22.04-trt8.6-cuda11.8-cudnn8.9" -and $ubuntu22KeySet -contains "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22") { "passed" } else { "failed" }
    detail = $ubuntu22KeySet -join ", "
  })

$hostedAllKeySet = @((pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeKeySet.ps1") -Platform linux -RuntimeKeySet hosted-all -RunnerMode hosted -OutputFormat json | Out-String).Trim() | ConvertFrom-Json)
$results.Add([pscustomobject]@{
    workflow = "eng\Resolve-RuntimeKeySet.ps1"
    requirement = "Hosted-all Linux key set includes Ubuntu 22.04 and Ubuntu 24.04 package lines"
    status = if ($hostedAllKeySet.Count -eq 9 -and $hostedAllKeySet -contains "linux-x64-ubuntu22.04-trt8.6-cuda11.8-cudnn8.9" -and $hostedAllKeySet -contains "linux-x64-ubuntu24.04-trt11.0-cuda13.2-cudnn9.22") { "passed" } else { "failed" }
    detail = $hostedAllKeySet -join ", "
  })

$ubuntu24KeySet = @((pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeKeySet.ps1") -Platform linux -RuntimeKeySet ubuntu24-hosted -RunnerMode hosted -OutputFormat json | Out-String).Trim() | ConvertFrom-Json)
$results.Add([pscustomobject]@{
    workflow = "eng\Resolve-RuntimeKeySet.ps1"
    requirement = "Ubuntu 24.04 hosted key set resolves the modern hosted package line"
    status = if ($ubuntu24KeySet.Count -eq 3 -and $ubuntu24KeySet -contains "linux-x64-ubuntu24.04-trt10.11-cuda12.9-cudnn9.22" -and $ubuntu24KeySet -contains "linux-x64-ubuntu24.04-trt11.0-cuda13.2-cudnn9.22") { "passed" } else { "failed" }
    detail = $ubuntu24KeySet -join ", "
  })

$ubuntu20SelfHostedKeySet = @((pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeKeySet.ps1") -Platform linux -RuntimeKeySet self-hosted-ubuntu20 -RunnerMode self-hosted -OutputFormat json | Out-String).Trim() | ConvertFrom-Json)
$results.Add([pscustomobject]@{
    workflow = "eng\Resolve-RuntimeKeySet.ps1"
    requirement = "Ubuntu 20.04 self-hosted key set resolves the modeled self-hosted package line"
    status = if ($ubuntu20SelfHostedKeySet.Count -eq 3 -and $ubuntu20SelfHostedKeySet -contains "linux-x64-ubuntu20.04-trt8.6-cuda11.8-cudnn8.9") { "passed" } else { "failed" }
    detail = $ubuntu20SelfHostedKeySet -join ", "
  })

pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Test-LinuxRuntimeTargetCoverage.ps1") | Out-Host
$linuxTargetCoverageJson = Get-Content -LiteralPath (Join-Path $RepositoryRoot "artifacts\linux-target-coverage\linux-runtime-target-coverage.json") -Raw
$linuxTargetCoverage = $linuxTargetCoverageJson | ConvertFrom-Json
$results.Add([pscustomobject]@{
    workflow = "eng\Test-LinuxRuntimeTargetCoverage.ps1"
    requirement = "Linux runtime target coverage models Ubuntu 20.04, 22.04, and 24.04 while holding future ARM/Jetson lines"
    status = if ($linuxTargetCoverage.failedCount -eq 0 -and $linuxTargetCoverage.modeledTargets.Count -eq 3 -and ($linuxTargetCoverage.futureTargets | Where-Object { $_.target -eq "linux-jetson-l4t" }).Count -eq 1) { "passed" } else { "failed" }
    detail = "failed=$($linuxTargetCoverage.failedCount); modeled=$($linuxTargetCoverage.modeledTargets.Count); future=$($linuxTargetCoverage.futureTargets.Count)"
  })

$singleLinuxMatrix = $singleLinuxMatrixJson | ConvertFrom-Json
$results.Add([pscustomobject]@{
    workflow = "eng\Resolve-RuntimeMatrix.ps1"
    requirement = "Ubuntu 22.04 hosted matrix emits the matching runner label"
    status = if ($singleLinuxMatrix[0].runsOnJson -eq '["ubuntu-22.04"]') { "passed" } else { "failed" }
    detail = [string]$singleLinuxMatrix[0].runsOnJson
  })

$linuxDependencyPlanJson = (pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Prepare-LinuxNvidiaDependencies.ps1") -RuntimePackageKey "linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22" -DescribeDependencyPlan | Out-String).Trim()
$linuxDependencyPlan = $linuxDependencyPlanJson | ConvertFrom-Json
$requiredPinnedTensorRtPackages = @(
  "libnvinfer11=11.0.0.114-1+cuda12.9*",
  "libnvinfer-lean11=11.0.0.114-1+cuda12.9*",
  "libnvinfer-plugin11=11.0.0.114-1+cuda12.9*",
  "libnvinfer-vc-plugin11=11.0.0.114-1+cuda12.9*",
  "libnvinfer-dispatch11=11.0.0.114-1+cuda12.9*",
  "libnvonnxparsers11=11.0.0.114-1+cuda12.9*"
)
$missingPinnedTensorRtPackages = @($requiredPinnedTensorRtPackages | Where-Object { $linuxDependencyPlan.aptPackages -notcontains $_ })
$results.Add([pscustomobject]@{
    workflow = "eng\Prepare-LinuxNvidiaDependencies.ps1"
    requirement = "TensorRT runtime dependencies are pinned to CUDA 12.9"
    status = if ($missingPinnedTensorRtPackages.Count -eq 0) { "passed" } else { "failed" }
    detail = if ($missingPinnedTensorRtPackages.Count -eq 0) { "linux-x64-ubuntu22.04-trt11.0-cuda12.9-cudnn9.22" } else { $missingPinnedTensorRtPackages -join ", " }
  })
$results.Add([pscustomobject]@{
    workflow = "eng\Prepare-LinuxNvidiaDependencies.ps1"
    requirement = "Modern CUDA host compiler headers use cuda-crt"
    status = if ($linuxDependencyPlan.aptPackages -contains "cuda-crt-12-9") { "passed" } else { "failed" }
    detail = "cuda-crt-12-9"
  })
$modernCuda132PlanJson = (pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Prepare-LinuxNvidiaDependencies.ps1") -RuntimePackageKey "linux-x64-ubuntu22.04-trt11.0-cuda13.2-cudnn9.22" -DescribeDependencyPlan | Out-String).Trim()
$modernCuda132Plan = $modernCuda132PlanJson | ConvertFrom-Json
$results.Add([pscustomobject]@{
    workflow = "eng\Prepare-LinuxNvidiaDependencies.ps1"
    requirement = "CUDA 13.2 host compiler headers use cuda-crt"
    status = if ($modernCuda132Plan.aptPackages -contains "cuda-crt-13-2") { "passed" } else { "failed" }
    detail = "cuda-crt-13-2"
  })

$legacyCudaDependencyPlanJson = (pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Prepare-LinuxNvidiaDependencies.ps1") -RuntimePackageKey "linux-x64-ubuntu22.04-trt8.6-cuda11.8-cudnn8.9" -DescribeDependencyPlan | Out-String).Trim()
$legacyCudaDependencyPlan = $legacyCudaDependencyPlanJson | ConvertFrom-Json
$results.Add([pscustomobject]@{
    workflow = "eng\Prepare-LinuxNvidiaDependencies.ps1"
    requirement = "Legacy CUDA 11.8 host compiler headers use cuda-nvcc"
    status = if (($legacyCudaDependencyPlan.aptPackages -contains "cuda-nvcc-11-8") -and ($legacyCudaDependencyPlan.aptPackages -notcontains "cuda-crt-11-8")) { "passed" } else { "failed" }
    detail = "cuda-nvcc-11-8"
  })
$results.Add([pscustomobject]@{
    workflow = "eng\Prepare-LinuxNvidiaDependencies.ps1"
    requirement = "TensorRT 8.6 dependency plan pins vc plugin to CUDA 11.8"
    status = if ($legacyCudaDependencyPlan.aptPackages -contains "libnvinfer-vc-plugin8=8.6.1.6-1+cuda11.8*") { "passed" } else { "failed" }
    detail = "libnvinfer-vc-plugin8=8.6.1.6-1+cuda11.8*"
  })

$cuda121DependencyPlanJson = (pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Prepare-LinuxNvidiaDependencies.ps1") -RuntimePackageKey "linux-x64-ubuntu22.04-trt8.6-cuda12.1-cudnn8.9" -DescribeDependencyPlan | Out-String).Trim()
$cuda121DependencyPlan = $cuda121DependencyPlanJson | ConvertFrom-Json
$results.Add([pscustomobject]@{
    workflow = "eng\Prepare-LinuxNvidiaDependencies.ps1"
    requirement = "CUDA 12.1 host compiler headers use cuda-nvcc"
    status = if (($cuda121DependencyPlan.aptPackages -contains "cuda-nvcc-12-1") -and ($cuda121DependencyPlan.aptPackages -notcontains "cuda-crt-12-1")) { "passed" } else { "failed" }
    detail = "cuda-nvcc-12-1"
  })

$legacyTensorRt10PlanJson = (pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Prepare-LinuxNvidiaDependencies.ps1") -RuntimePackageKey "linux-x64-ubuntu22.04-trt10.11-cuda12.9-cudnn9.22" -DescribeDependencyPlan | Out-String).Trim()
$legacyTensorRt10Plan = $legacyTensorRt10PlanJson | ConvertFrom-Json
$legacyTensorRtPlans = @($legacyCudaDependencyPlan, $legacyTensorRt10Plan)
$legacySafeHeaders = @($legacyTensorRtPlans | ForEach-Object { $_.aptPackages } | Where-Object { $_ -like "libnvinfer-safe-headers-dev=*" })
$results.Add([pscustomobject]@{
    workflow = "eng\Prepare-LinuxNvidiaDependencies.ps1"
    requirement = "TensorRT 8.6 and 10.11 dependency plans omit unavailable safe headers package"
    status = if ($legacySafeHeaders.Count -eq 0) { "passed" } else { "failed" }
    detail = if ($legacySafeHeaders.Count -eq 0) { "trt8.6/trt10.11" } else { $legacySafeHeaders -join ", " }
  })

$outputRoot = Join-Path $RepositoryRoot "artifacts\workflow-contracts"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null

$jsonPath = Join-Path $outputRoot "workflow-contract-report.json"
$results | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Workflow Contract Report")
$lines.Add("")
$lines.Add("Generated on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')")
$lines.Add("")
$lines.Add("| Workflow | Requirement | Status | Evidence |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($result in $results) {
  $evidence = '`' + $result.detail + '`'
  $lines.Add("| $($result.workflow) | $($result.requirement) | $($result.status) | $evidence |")
}

$markdownPath = Join-Path $outputRoot "workflow-contract-report.md"
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

$failed = @($results | Where-Object { $_.status -ne "passed" })
Write-Host "Workflow contract report written to $jsonPath"
Write-Host "Workflow contract report written to $markdownPath"

if ($failed.Count -gt 0) {
  foreach ($item in $failed) {
    Write-Error "$($item.workflow): missing $($item.requirement) ($($item.detail))"
  }
  exit 1
}
