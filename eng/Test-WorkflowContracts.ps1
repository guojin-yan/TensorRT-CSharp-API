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
    path = ".github\workflows\manual-quality-gate.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "Validate-RuntimeManifest.ps1" -Description "runtime manifest validation"
      New-Requirement -Needle "Test-BindingGeneratorOutputs.ps1" -Description "binding generator determinism"
      New-Requirement -Needle "dotnet docfx" -Description "DocFX build"
      New-Requirement -Needle "Test-ManagedPackageContent.ps1" -Description "managed package content validation"
      New-Requirement -Needle "Validate-SplitDeliveryPrototype.ps1" -Description "split delivery prototype validation"
      New-Requirement -Needle "run_runtime_package_checks" -Description "hosted/runtime gate split input"
      New-Requirement -Needle "Test-PackageConsumer.ps1" -Description "optional package consumer validation"
      New-Requirement -Needle "Test-RuntimePublishReadiness.ps1" -Description "runtime publish readiness"
      New-Requirement -Needle "Export-RuntimeDeliveryStrategy.ps1" -Description "runtime delivery strategy export"
      New-Requirement -Needle "Export-ReleaseCandidateChecklist.ps1" -Description "release candidate checklist"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\manual-pack-runtime.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "self-hosted" -Description "self-hosted runner"
      New-Requirement -Needle "windows" -Description "windows runner label"
      New-Requirement -Needle "cudnn_root" -Description "explicit cuDNN root input"
      New-Requirement -Needle "Validate-WindowsRuntimeInputs.ps1" -Description "Windows input validation"
      New-Requirement -Needle "-CudnnRoot" -Description "cuDNN root is passed through validation and asset collection"
      New-Requirement -Needle "Collect-RuntimeAssets.ps1" -Description "runtime asset collection"
      New-Requirement -Needle "Test-PackageConsumer.ps1" -Description "package consumer validation"
      New-Requirement -Needle "Test-RuntimePublishReadiness.ps1" -Description "runtime publish readiness"
      New-Requirement -Needle "Validate-SplitDeliveryPrototype.ps1" -Description "split delivery prototype validation"
      New-Requirement -Needle "Export-RuntimeDeliveryStrategy.ps1" -Description "runtime delivery strategy export"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\manual-pack-runtime-split.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "self-hosted" -Description "self-hosted runner"
      New-Requirement -Needle "windows" -Description "windows runner label"
      New-Requirement -Needle "cudnn_root" -Description "explicit cuDNN root input"
      New-Requirement -Needle "Validate-SplitDeliveryPrototype.ps1" -Description "split delivery prototype validation"
      New-Requirement -Needle "Validate-WindowsRuntimeInputs.ps1" -Description "source runtime input validation"
      New-Requirement -Needle "-CudnnRoot" -Description "cuDNN root is passed through validation and asset collection"
      New-Requirement -Needle "Collect-RuntimeAssets.ps1" -Description "source runtime asset collection"
      New-Requirement -Needle "Collect-SplitRuntimeAssets.ps1" -Description "split runtime asset collection"
      New-Requirement -Needle "dotnet pack" -Description "split runtime pack"
      New-Requirement -Needle "Export-RuntimeDeliveryStrategy.ps1" -Description "runtime delivery strategy export"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\manual-pack-runtime-linux.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "self-hosted" -Description "self-hosted runner"
      New-Requirement -Needle "linux" -Description "linux runner label"
      New-Requirement -Needle "x64" -Description "x64 runner label"
      New-Requirement -Needle "Validate-LinuxRuntimeInputs.ps1" -Description "Linux input validation"
      New-Requirement -Needle "Invoke-LinuxRuntimeDryRun.ps1" -Description "Linux dry-run"
      New-Requirement -Needle "Test-LinuxRuntimeWorkflowContract.ps1" -Description "Linux workflow contract check"
      New-Requirement -Needle "Export-LinuxPackageConsumerPlan.ps1" -Description "Linux package consumer handoff plan"
      New-Requirement -Needle "Export-LinuxRunnerExecutionStatus.ps1" -Description "Linux runner execution status"
      New-Requirement -Needle "Collect-RuntimeAssets.ps1" -Description "runtime asset collection"
      New-Requirement -Needle "Test-PackageConsumer.ps1" -Description "Linux package consumer validation"
      New-Requirement -Needle "Test-RuntimePublishReadiness.ps1" -Description "publish readiness"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\manual-build-native-linux.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "self-hosted" -Description "self-hosted runner"
      New-Requirement -Needle "Validate-LinuxRuntimeInputs.ps1" -Description "Linux input validation"
      New-Requirement -Needle "Invoke-LinuxRuntimeDryRun.ps1" -Description "Linux dry-run"
      New-Requirement -Needle "Export-LinuxPreflightSummary.ps1" -Description "Linux preflight summary"
      New-Requirement -Needle "Export-LinuxPackageConsumerPlan.ps1" -Description "Linux package consumer handoff plan"
      New-Requirement -Needle "Export-LinuxRunnerExecutionStatus.ps1" -Description "Linux runner execution status"
      New-Requirement -Needle "cmake --preset" -Description "CMake configure"
      New-Requirement -Needle "cmake --build --preset" -Description "CMake build"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\manual-build-docs.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "dotnet tool restore" -Description "restore local tools"
      New-Requirement -Needle "dotnet docfx" -Description "DocFX build"
    )
  }
  [pscustomobject]@{
    path = ".github\workflows\manual-deploy-docs.yml"
    requirements = @(
      New-Requirement -Needle "workflow_dispatch" -Description "manual trigger"
      New-Requirement -Needle "dotnet tool restore" -Description "restore local tools"
      New-Requirement -Needle "dotnet docfx" -Description "DocFX build"
      New-Requirement -Needle "actions/deploy-pages" -Description "GitHub Pages deploy"
    )
  }
)

$results = New-Object System.Collections.Generic.List[object]
foreach ($contract in $workflowContracts) {
  foreach ($result in @(Test-Workflow -RelativePath $contract.path -Requirements $contract.requirements)) {
    $results.Add($result)
  }
}

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
