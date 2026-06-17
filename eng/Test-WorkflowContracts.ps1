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
      New-Requirement -Needle "cuda_cudnn_package_version_map" -Description "CUDA/cuDNN split runtime-key version map input"
      New-Requirement -Needle "tensorrt_package_version" -Description "TensorRT split version input"
      New-Requirement -Needle "tensorrt_package_version_map" -Description "TensorRT split runtime-key version map input"
      New-Requirement -Needle "cuda_cudnn_package_release_tag" -Description "CUDA/cuDNN split release tag input"
      New-Requirement -Needle "cuda_cudnn_package_release_tag_map" -Description "CUDA/cuDNN split runtime-key release tag map input"
      New-Requirement -Needle "tensorrt_package_release_tag" -Description "TensorRT split release tag input"
      New-Requirement -Needle "tensorrt_package_release_tag_map" -Description "TensorRT split runtime-key release tag map input"
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
      New-Requirement -Needle "cuda_cudnn_package_version_map" -Description "Linux split CUDA/cuDNN runtime-key version map input"
      New-Requirement -Needle "cuda_cudnn_package_release_tag" -Description "Linux split CUDA/cuDNN release tag input"
      New-Requirement -Needle "cuda_cudnn_package_release_tag_map" -Description "Linux split CUDA/cuDNN runtime-key release tag map input"
      New-Requirement -Needle "tensorrt_package_version_map" -Description "Linux split TensorRT runtime-key version map input"
      New-Requirement -Needle "tensorrt_package_release_tag" -Description "Linux split TensorRT release tag input"
      New-Requirement -Needle "tensorrt_package_release_tag_map" -Description "Linux split TensorRT runtime-key release tag map input"
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
    path = ".github\workflows\linux-self-hosted-runner-readiness.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "self-hosted" -Description "self-hosted runner"
      New-Requirement -Needle "linux" -Description "linux runner label"
      New-Requirement -Needle "ubuntu-20.04" -Description "Ubuntu 20.04 runner label"
      New-Requirement -Needle "Test-LinuxSelfHostedRunnerReadiness.ps1" -Description "Ubuntu 20.04 readiness script"
      New-Requirement -Needle "RUNNER_AUDIT_TOKEN" -Description "runner audit token support"
      New-Requirement -Needle "linux-self-hosted-runner-readiness" -Description "readiness artifact upload"
      New-Requirement -Needle "runtime_key_set" -Description "runtime key set input"
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
      New-Requirement -Needle "Publish-ReleaseNuGetAssetsToGitHubPackages.ps1" -Description "release asset GitHub Packages backfill"
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
      New-Requirement -Needle "linux_cuda_cudnn_package_version_map" -Description "Linux split CUDA/cuDNN runtime-key version map input"
      New-Requirement -Needle "linux_cuda_cudnn_package_release_tag_map" -Description "Linux split CUDA/cuDNN runtime-key release tag map input"
      New-Requirement -Needle "linux_tensorrt_package_version" -Description "Linux split TensorRT version input"
      New-Requirement -Needle "linux_tensorrt_package_version_map" -Description "Linux split TensorRT runtime-key version map input"
      New-Requirement -Needle "linux_tensorrt_package_release_tag_map" -Description "Linux split TensorRT runtime-key release tag map input"
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
      New-Requirement -Needle "windows_cuda_cudnn_package_version_map" -Description "split collection CUDA/cuDNN version map override"
      New-Requirement -Needle "windows_cuda_cudnn_package_release_tag_map" -Description "split collection CUDA/cuDNN release tag map override"
      New-Requirement -Needle "windows_tensorrt_package_release_tag" -Description "split collection TensorRT release tag override"
      New-Requirement -Needle "windows_tensorrt_package_version_map" -Description "split collection TensorRT version map override"
      New-Requirement -Needle "windows_tensorrt_package_release_tag_map" -Description "split collection TensorRT release tag map override"
      New-Requirement -Needle 'attach_to_github_release=$ATTACH_RUNTIME_TO_GITHUB_RELEASE' -Description "managed package release asset toggle dispatch"
      New-Requirement -Needle "publish_managed_to_nuget=true requires the repository secret NUGET_API_KEY" -Description "nuget.org secret guard before release creation"
    )
  }
  [pscustomobject]@{
    path = "eng\Invoke-RemoteReleaseBundle.ps1"
    requirements = @(
      New-Requirement -Needle "release_config_json" -Description "advanced release config serialization"
      New-Requirement -Needle "DryRun" -Description "dry-run dispatch mode"
      New-Requirement -Needle "runtime_delivery_mode" -Description "windows/linux runtime delivery mode mapping"
      New-Requirement -Needle "WindowsCudaCudnnPackageVersionMap" -Description "Windows split dependency version map parameter"
      New-Requirement -Needle "WindowsTensorRtPackageVersionMap" -Description "Windows split TensorRT version map parameter"
      New-Requirement -Needle "LinuxCudaCudnnPackageVersionMap" -Description "Linux split dependency version map parameter"
      New-Requirement -Needle "LinuxTensorRtPackageVersionMap" -Description "Linux split TensorRT version map parameter"
      New-Requirement -Needle "gh workflow run release-bundle.yml" -Description "remote workflow dispatch command"
    )
  }
  [pscustomobject]@{
    path = "eng\Resolve-SplitPackagePins.ps1"
    requirements = @(
      New-Requirement -Needle "CudaCudnnPackageVersionMap" -Description "CUDA/cuDNN version map parameter"
      New-Requirement -Needle "TensorRtPackageVersionMap" -Description "TensorRT version map parameter"
      New-Requirement -Needle "CudaCudnnPackageReleaseTagMap" -Description "CUDA/cuDNN release tag map parameter"
      New-Requirement -Needle "TensorRtPackageReleaseTagMap" -Description "TensorRT release tag map parameter"
      New-Requirement -Needle "ConvertFrom-PinMap" -Description "map parser"
      New-Requirement -Needle "Resolve-PinValue" -Description "exact and wildcard runtime-key map resolver"
      New-Requirement -Needle "map-wildcard" -Description "wildcard match provenance"
    )
  }
  [pscustomobject]@{
    path = "eng\Invoke-LocalSplitRuntimePackage.ps1"
    requirements = @(
      New-Requirement -Needle "SourceRuntimeKey is required" -Description "split runtime packaging requires an explicit source key"
      New-Requirement -Needle "Resolve-SplitPackagePins.ps1" -Description "shared split package pin resolver"
      New-Requirement -Needle "CudaCudnnPackageVersionMap" -Description "CUDA/cuDNN version map parameter"
      New-Requirement -Needle "TensorRtPackageVersionMap" -Description "TensorRT version map parameter"
      New-Requirement -Needle "Pass -CudaCudnnPackageVersion or -CudaCudnnPackageVersionMap" -Description "meta package guard accepts version map"
      New-Requirement -Needle "Pass -TensorRtPackageVersion or -TensorRtPackageVersionMap" -Description "meta package TensorRT guard accepts version map"
    )
  }
  [pscustomobject]@{
    path = "eng\Invoke-LocalRuntimePackage.ps1"
    requirements = @(
      New-Requirement -Needle "Resolve-DefaultRuntimeKeys" -Description "local runtime packaging resolves the host matrix from the manifest"
      New-Requirement -Needle "ResolveOnly" -Description "local runtime package key resolution without build side effects"
      New-Requirement -Needle "No -RuntimePackageKey was provided" -Description "default runtime key resolution is logged"
      New-Requirement -Needle "ARM/SBSA, Jetson/L4T, or non-Ubuntu" -Description "future Linux package lines require explicit runtime keys"
    )
  }
  [pscustomobject]@{
    path = "eng\Resolve-RuntimeKeySet.ps1"
    requirements = @(
      New-Requirement -Needle "linux-runtime-targets.manifest.json" -Description "Linux target catalog is consulted"
      New-Requirement -Needle "future package line" -Description "future Linux key sets fail with explicit guidance"
      New-Requirement -Needle "arm64-sbsa" -Description "ARM/SBSA key set is recognized as future"
      New-Requirement -Needle "jetson-l4t" -Description "Jetson/L4T key set is recognized as future"
      New-Requirement -Needle "non-ubuntu" -Description "non-Ubuntu key set is recognized as future"
    )
  }
  [pscustomobject]@{
    path = "eng\Export-RuntimePublicationIndex.ps1"
    requirements = @(
      New-Requirement -Needle "Runtime Publication Index" -Description "publication index report title"
      New-Requirement -Needle "runtime-publication-index" -Description "publication index artifact output"
      New-Requirement -Needle "latest managed release does not necessarily contain every runtime asset" -Description "release-tag split guidance"
      New-Requirement -Needle "Ubuntu 20.04, ARM/SBSA, Jetson/L4T, and non-Ubuntu Linux" -Description "future and infrastructure-blocked package line guidance"
    )
  }
  [pscustomobject]@{
    path = "eng\Export-RuntimeReleasePlan.ps1"
    requirements = @(
      New-Requirement -Needle "Runtime Release Plan" -Description "runtime release plan report title"
      New-Requirement -Needle "runtime-release-plan" -Description "runtime release plan artifact output"
      New-Requirement -Needle "dispatchableNextTargets" -Description "dispatchable next-target reporting"
      New-Requirement -Needle "stableDependencyPinMaps" -Description "stable dependency pin maps for bridge/collection refreshes"
      New-Requirement -Needle "ubuntu20-after-runner-is-online" -Description "Ubuntu 20.04 self-hosted runner guard command"
      New-Requirement -Needle "future separate package lines" -Description "future ARM/Jetson/non-Ubuntu release planning"
    )
  }
  [pscustomobject]@{
    path = "eng\Restore-PublishedSplitPackageSource.ps1"
    requirements = @(
      New-Requirement -Needle "Resolve-SplitPackagePins.ps1" -Description "shared split package pin resolver"
      New-Requirement -Needle "CudaCudnnPackageVersionMap" -Description "CUDA/cuDNN version map parameter"
      New-Requirement -Needle "TensorRtPackageVersionMap" -Description "TensorRT version map parameter"
      New-Requirement -Needle "CudaCudnnPackageReleaseTagMap" -Description "CUDA/cuDNN release tag map parameter"
      New-Requirement -Needle "TensorRtPackageReleaseTagMap" -Description "TensorRT release tag map parameter"
    )
  }
  [pscustomobject]@{
    path = "eng\Test-RemoteReleasePrerequisites.ps1"
    requirements = @(
      New-Requirement -Needle "NUGET_API_KEY" -Description "nuget.org secret prerequisite"
      New-Requirement -Needle "RUNNER_AUDIT_TOKEN" -Description "runner audit secret prerequisite"
      New-Requirement -Needle "NuGetApiKeyAvailable" -Description "workflow-provided nuget.org secret availability"
      New-Requirement -Needle "RunnerAuditTokenAvailable" -Description "workflow-provided runner audit token availability"
      New-Requirement -Needle "runnerQuerySource" -Description "runner query source report"
      New-Requirement -Needle "self-hosted,windows,x64" -Description "Windows runner prerequisite"
      New-Requirement -Needle "self-hosted,linux,x64,ubuntu-20.04" -Description "Ubuntu 20.04 runner prerequisite"
      New-Requirement -Needle "remote-release-prerequisites" -Description "prerequisite artifact output"
    )
  }
  [pscustomobject]@{
    path = "eng\Install-GitHubSelfHostedRunner.ps1"
    requirements = @(
      New-Requirement -Needle "actions/runner/releases" -Description "official GitHub Actions runner download"
      New-Requirement -Needle "registration-token" -Description "short-lived registration token support"
      New-Requirement -Needle "ubuntu-20.04" -Description "Ubuntu 20.04 custom runner label"
      New-Requirement -Needle "tensorrt-csharp" -Description "project-specific runner label"
      New-Requirement -Needle "The registration token is never written to disk" -Description "registration token non-persistence note"
      New-Requirement -Needle "InstallService" -Description "runner service installation option"
      New-Requirement -Needle "DryRun" -Description "safe command preview mode"
    )
  }
  [pscustomObject]@{
    path = "eng\Test-LinuxSelfHostedRunnerReadiness.ps1"
    requirements = @(
      New-Requirement -Needle "self-hosted-ubuntu20" -Description "Ubuntu 20.04 runtime key set readiness"
      New-Requirement -Needle "self-hosted,linux,x64,ubuntu-20.04" -Description "release runner label readiness"
      New-Requirement -Needle "Test-GitHubRunnerAvailability.ps1" -Description "GitHub runner label audit delegation"
      New-Requirement -Needle "Resolve-RuntimeRoots.ps1" -Description "NVIDIA root resolution"
      New-Requirement -Needle "Validate-LinuxRuntimeInputs.ps1" -Description "Linux runtime input validation delegation"
      New-Requirement -Needle ".NET SDK 10.0.300+" -Description "required .NET SDK readiness"
      New-Requirement -Needle "linux-self-hosted-runner-readiness" -Description "readiness artifact output"
      New-Requirement -Needle "WarnOnly" -Description "non-failing readiness mode"
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
    path = ".github\workflows\release-publication-audit.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "Test-ReleasePublicationState.ps1" -Description "release publication state audit script"
      New-Requirement -Needle "HAS_NUGET_API_KEY" -Description "secret availability is passed without listing secrets"
      New-Requirement -Needle 'default: "4.0.6170"' -Description "current managed package version is audited by default"
      New-Requirement -Needle 'default: "v4.0.6170"' -Description "current managed release tag is audited by default"
      New-Requirement -Needle "require_managed_nuget_org" -Description "nuget.org managed package gate"
      New-Requirement -Needle "require_nuget_api_key" -Description "nuget.org API key gate"
      New-Requirement -Needle "runner_required_label_sets" -Description "runner label audit input"
      New-Requirement -Needle "RequiredLabelSet" -Description "runner availability label-set aggregation"
      New-Requirement -Needle "RUNNER_AUDIT_TOKEN" -Description "runner audit token secret"
      New-Requirement -Needle "Test-GitHubRunnerAvailability.ps1" -Description "runner availability audit script"
      New-Requirement -Needle "Test-GitHubPublicationInventory.ps1" -Description "publication inventory audit script"
      New-Requirement -Needle "require_publication_inventory_clean" -Description "publication inventory strictness gate"
      New-Requirement -Needle "require_package_repository_association" -Description "package repository association gate"
      New-Requirement -Needle "Export-RuntimePublicationIndex.ps1" -Description "runtime publication index export script"
      New-Requirement -Needle "publication-index" -Description "runtime publication index artifact output"
      New-Requirement -Needle "Export-RuntimeReleasePlan.ps1" -Description "runtime release plan export script"
      New-Requirement -Needle "runtime-release-plan" -Description "runtime release plan artifact output"
      New-Requirement -Needle "Test-ReleaseReadiness.ps1" -Description "external release readiness audit script"
      New-Requirement -Needle "include_release_readiness" -Description "release readiness audit toggle"
      New-Requirement -Needle "Test-RemoteReleasePrerequisites.ps1" -Description "remote release prerequisites audit script"
      New-Requirement -Needle "remote-release-prerequisites" -Description "remote release prerequisites artifact output"
      New-Requirement -Needle "Test-LinuxRuntimeTargetCoverage.ps1" -Description "Linux target coverage audit script"
      New-Requirement -Needle "Test-RuntimePublicationTargetCoverage.ps1" -Description "runtime publication target coverage audit script"
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
      New-Requirement -Needle "repository secret NUGET_API_KEY availability" -Description "secret availability report distinguishes optional and required gates"
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
      New-Requirement -Needle "Expand-RequestedLabelSets" -Description "recover accidentally joined required label set arguments"
      New-Requirement -Needle "onlineMatchingRunnerCount" -Description "online matching runner audit"
      New-Requirement -Needle "github-runner-availability" -Description "runner availability report"
      New-Requirement -Needle "WarnOnly" -Description "non-failing audit mode"
    )
  }
  [pscustomobject]@{
    path = "eng\Test-GitHubPublicationInventory.ps1"
    requirements = @(
      New-Requirement -Needle "ExpectedReleaseTag" -Description "expected release whitelist input"
      New-Requirement -Needle "RequireOnlyExpectedReleases" -Description "strict release inventory gate"
      New-Requirement -Needle "RequireOnlyExpectedPackageVersions" -Description "strict package version inventory gate"
      New-Requirement -Needle "RequirePackageRepositoryAssociation" -Description "package repository association gate"
      New-Requirement -Needle "publication-inventory" -Description "publication inventory artifact output"
      New-Requirement -Needle "Unexpected GitHub Package versions" -Description "stale package version reporting"
      New-Requirement -Needle "Runtime Matrix Summary" -Description "runtime package matrix inventory report"
      New-Requirement -Needle "runtimeTargetSummary" -Description "runtime target summary JSON output"
    )
  }
  [pscustomobject]@{
    path = "eng\Test-ReleaseReadiness.ps1"
    requirements = @(
      New-Requirement -Needle "NUGET_API_KEY" -Description "nuget.org secret readiness"
      New-Requirement -Needle "RUNNER_AUDIT_TOKEN" -Description "runner audit token readiness"
      New-Requirement -Needle "runnerQuerySource" -Description "runner query source reporting"
      New-Requirement -Needle "gh-auth" -Description "local gh-auth runner query fallback"
      New-Requirement -Needle "self-hosted,linux,x64,ubuntu-20.04" -Description "Ubuntu 20.04 self-hosted runner readiness"
      New-Requirement -Needle "future-target:linux-arm64-sbsa" -Description "future SBSA readiness blocker"
      New-Requirement -Needle "future-target:linux-jetson-l4t" -Description "future Jetson readiness blocker"
      New-Requirement -Needle "future-target:non-ubuntu-linux" -Description "future non-Ubuntu readiness blocker"
      New-Requirement -Needle "release-readiness" -Description "release readiness artifact output"
      New-Requirement -Needle "WarnOnly" -Description "non-failing readiness mode"
    )
  }
  [pscustomobject]@{
    path = "eng\Test-LinuxRuntimeTargetCoverage.ps1"
    requirements = @(
      New-Requirement -Needle "linux-runtime-targets.manifest.json" -Description "target coverage is generated from the Linux target catalog"
      New-Requirement -Needle "targetCatalog.targets" -Description "modeled target rows come from the target catalog"
      New-Requirement -Needle "targetCatalog.futureTargets" -Description "future target rows come from the target catalog"
      New-Requirement -Needle "requiredEvidenceItems" -Description "structured future-target evidence checklist"
      New-Requirement -Needle "linux-runtime-target-coverage" -Description "target coverage report"
    )
  }
  [pscustomobject]@{
    path = "pack\runtime\linux-runtime-targets.manifest.json"
    requirements = @(
      New-Requirement -Needle "ubuntu22.04-x64-hosted" -Description "Ubuntu 22.04 hosted target is cataloged"
      New-Requirement -Needle "ubuntu24.04-x64-hosted" -Description "Ubuntu 24.04 hosted target is cataloged"
      New-Requirement -Needle "ubuntu20.04-x64-self-hosted" -Description "Ubuntu 20.04 self-hosted target is cataloged"
      New-Requirement -Needle "linux-arm64-sbsa" -Description "future SBSA package line is cataloged"
      New-Requirement -Needle "linux-jetson-l4t" -Description "future Jetson/L4T package line is cataloged"
      New-Requirement -Needle "non-ubuntu-linux" -Description "future non-Ubuntu package line is cataloged"
      New-Requirement -Needle "packageIdentityRule" -Description "future targets document package identity rules"
    )
  }
  [pscustomobject]@{
    path = "eng\Test-RuntimePublicationTargetCoverage.ps1"
    requirements = @(
      New-Requirement -Needle "published-required" -Description "published target requirement classification"
      New-Requirement -Needle "infrastructure-blocked" -Description "infrastructure-blocked target classification"
      New-Requirement -Needle "linux-x64.ubuntu20.04" -Description "Ubuntu 20.04 publication target coverage"
      New-Requirement -Needle "linux-x64.ubuntu22.04" -Description "Ubuntu 22.04 publication target coverage"
      New-Requirement -Needle "linux-x64.ubuntu24.04" -Description "Ubuntu 24.04 publication target coverage"
      New-Requirement -Needle "linux-arm64-sbsa" -Description "future SBSA package line coverage"
      New-Requirement -Needle "linux-jetson-l4t" -Description "future Jetson/L4T package line coverage"
      New-Requirement -Needle "runtime-publication-target-coverage" -Description "publication target coverage report"
      New-Requirement -Needle "RequireInfrastructureBlockedTargetsPublished" -Description "optional strict gate for infrastructure-blocked targets"
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

$futureKeySetOutput = (pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeKeySet.ps1") -Platform linux -RuntimeKeySet arm64-sbsa -RunnerMode self-hosted -OutputFormat json 2>&1 | Out-String).Trim()
$futureKeySetExitCode = $LASTEXITCODE
$results.Add([pscustomobject]@{
    workflow = "eng\Resolve-RuntimeKeySet.ps1"
    requirement = "Future Linux key sets fail with package-line readiness guidance instead of silently dispatching"
    status = if ($futureKeySetExitCode -ne 0 -and $futureKeySetOutput -match "future package line" -and $futureKeySetOutput -match "linux-arm64-sbsa") { "passed" } else { "failed" }
    detail = $futureKeySetOutput
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

pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Test-RuntimePublicationTargetCoverage.ps1") -InventoryJsonPath (Join-Path $RepositoryRoot "artifacts\publication-inventory\github-publication-inventory.json") | Out-Host
$publicationTargetCoverageJson = Get-Content -LiteralPath (Join-Path $RepositoryRoot "artifacts\runtime-publication-target-coverage\runtime-publication-target-coverage.json") -Raw
$publicationTargetCoverage = $publicationTargetCoverageJson | ConvertFrom-Json
$publishedTargetRows = @($publicationTargetCoverage.targets | Where-Object { $_.requirement -eq "published-required" })
$infrastructureBlockedTargetRows = @($publicationTargetCoverage.targets | Where-Object { $_.requirement -eq "infrastructure-blocked" })
$results.Add([pscustomobject]@{
    workflow = "eng\Test-RuntimePublicationTargetCoverage.ps1"
    requirement = "Runtime publication target coverage proves Windows, Ubuntu 22.04, and Ubuntu 24.04 publication while keeping Ubuntu 20.04 infrastructure-blocked"
    status = if ($publicationTargetCoverage.failedCount -eq 0 -and $publishedTargetRows.Count -eq 3 -and (@($publishedTargetRows | Where-Object { $_.coverageState -eq "complete" }).Count -eq 3) -and $infrastructureBlockedTargetRows.Count -eq 1 -and $infrastructureBlockedTargetRows[0].coverageState -eq "not-published") { "passed" } else { "failed" }
    detail = "failed=$($publicationTargetCoverage.failedCount); publishedTargets=$($publishedTargetRows.Count); blockedTargets=$($infrastructureBlockedTargetRows.Count)"
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
