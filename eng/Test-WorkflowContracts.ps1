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
      New-Requirement -Needle "Resolve-RuntimeRoots.ps1" -Description "runtime root resolution"
      New-Requirement -Needle "Pack managed package" -Description "Linux runtime managed package artifact build"
      New-Requirement -Needle "managed-packages-runtime-linux" -Description "Linux runtime managed package artifact"
      New-Requirement -Needle "runner_mode" -Description "hosted/self-hosted runner mode"
      New-Requirement -Needle "Prepare-LinuxNvidiaDependencies.ps1" -Description "hosted Linux NVIDIA dependency preparation"
      New-Requirement -Needle "ubuntu-latest" -Description "hosted Linux runner"
      New-Requirement -Needle "Use runner_mode=hosted" -Description "hosted Linux runner guard"
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
      New-Requirement -Needle "runner_mode=$LINUX_RUNNER_MODE" -Description "Linux runner mode dispatch"
      New-Requirement -Needle "windows_split_package_roles includes collection/meta but no CUDA/cuDNN package version was provided" -Description "split collection CUDA/cuDNN version guard"
      New-Requirement -Needle "windows_split_package_roles includes collection/meta but no TensorRT package version was provided" -Description "split collection TensorRT version guard"
      New-Requirement -Needle "include cuda-cudnn/all in windows_split_package_roles" -Description "split collection same-run CUDA/cuDNN refresh guidance"
      New-Requirement -Needle "include tensorrt/all in windows_split_package_roles" -Description "split collection same-run TensorRT refresh guidance"
      New-Requirement -Needle "windows_cuda_cudnn_package_release_tag" -Description "split collection CUDA/cuDNN release tag override"
      New-Requirement -Needle "windows_tensorrt_package_release_tag" -Description "split collection TensorRT release tag override"
    )
  }
)

$results = New-Object System.Collections.Generic.List[object]
foreach ($contract in $workflowContracts) {
  foreach ($result in @(Test-Workflow -RelativePath $contract.path -Requirements $contract.requirements)) {
    $results.Add($result)
  }
}

$singleLinuxMatrixJson = (pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Resolve-RuntimeMatrix.ps1") -Platform linux -RuntimeKey "linux-x64-trt11.0-cuda12.9-cudnn9.22" | Out-String).Trim()
$results.Add([pscustomobject]@{
    workflow = "eng\Resolve-RuntimeMatrix.ps1"
    requirement = "single runtime key emits a JSON array"
    status = if ($singleLinuxMatrixJson.StartsWith("[")) { "passed" } else { "failed" }
    detail = "linux-x64-trt11.0-cuda12.9-cudnn9.22"
  })

$linuxDependencyPlanJson = (pwsh -NoProfile -File (Join-Path $RepositoryRoot "eng\Prepare-LinuxNvidiaDependencies.ps1") -RuntimePackageKey "linux-x64-trt11.0-cuda12.9-cudnn9.22" -DescribeDependencyPlan | Out-String).Trim()
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
    detail = if ($missingPinnedTensorRtPackages.Count -eq 0) { "linux-x64-trt11.0-cuda12.9-cudnn9.22" } else { $missingPinnedTensorRtPackages -join ", " }
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
