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
      New-Requirement -Needle "Restore-PublishedSplitPackageSource.ps1" -Description "published vendor release asset package source"
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
      New-Requirement -Needle "self-hosted" -Description "self-hosted runner"
      New-Requirement -Needle "linux" -Description "linux runner label"
      New-Requirement -Needle "Validate-LinuxRuntimeInputs.ps1" -Description "Linux input validation"
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
      New-Requirement -Needle "windows_split_package_roles includes collection/meta but no vendor component version was provided" -Description "split collection vendor-version guard"
      New-Requirement -Needle "windows_vendor_package_release_tag" -Description "split collection vendor release tag override"
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
