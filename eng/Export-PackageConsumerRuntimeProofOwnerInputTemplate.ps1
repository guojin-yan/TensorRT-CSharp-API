[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$SourceQualityRunEvidenceImportPath = "artifacts\final-release\github-actions-source-quality-run-evidence-import.json",
  [string]$PackageDryRunEvidenceImportPath = "artifacts\final-release\github-actions-run-evidence-import.json",
  [string]$CurrentHeadPackageDryRunPreflightPath = "artifacts\final-release\current-head-package-dry-run-preflight.json",
  [string]$GitHubActionsRunEvidenceImportPath = "",
  [string]$CurrentHead,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Resolve-RepoPath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return $Path
  }

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)

  $resolvedPath = Resolve-RepoPath -Path $Path
  if ([string]::IsNullOrWhiteSpace($resolvedPath) -or -not (Test-Path -LiteralPath $resolvedPath -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $resolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-BoolPropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [bool]$DefaultValue
  )

  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $DefaultValue
  if ($value -is [bool]) {
    return [bool]$value
  }

  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) {
    return $parsed
  }

  return $DefaultValue
}

function Get-GitHeadOrEmpty {
  if (-not [string]::IsNullOrWhiteSpace($CurrentHead)) {
    return $CurrentHead.Trim()
  }

  try {
    $head = (& git -C $RepositoryRoot rev-parse HEAD 2>$null)
    if ($LASTEXITCODE -eq 0 -and -not [string]::IsNullOrWhiteSpace($head)) {
      return ([string]$head).Trim()
    }
  }
  catch {
  }

  return ""
}

function Test-SameSha {
  param(
    [string]$Left,
    [string]$Right
  )

  return -not [string]::IsNullOrWhiteSpace($Left) -and
    -not [string]::IsNullOrWhiteSpace($Right) -and
    $Left.Trim().Equals($Right.Trim(), [StringComparison]::OrdinalIgnoreCase)
}

if (-not [string]::IsNullOrWhiteSpace($GitHubActionsRunEvidenceImportPath)) {
  $PackageDryRunEvidenceImportPath = $GitHubActionsRunEvidenceImportPath
}

$currentHeadValue = Get-GitHeadOrEmpty
$sourceQualityEvidence = Read-JsonOrNull -Path $SourceQualityRunEvidenceImportPath
$packageDryRunEvidence = Read-JsonOrNull -Path $PackageDryRunEvidenceImportPath
$currentHeadPackageDryRunPreflight = Read-JsonOrNull -Path $CurrentHeadPackageDryRunPreflightPath

$sourceQualityRunId = if ($null -eq $sourceQualityEvidence) { "<no-source-quality-run-evidence-import>" } else { [string](Get-PropertyOrDefault -Object $sourceQualityEvidence -Name "runId" -DefaultValue "<missing-source-quality-run-id>") }
$sourceQualityRunUrl = if ($null -eq $sourceQualityEvidence) { "<no-source-quality-run-evidence-import>" } else { [string](Get-PropertyOrDefault -Object $sourceQualityEvidence -Name "runUrl" -DefaultValue "<missing-source-quality-run-url>") }
$sourceQualityHeadSha = if ($null -eq $sourceQualityEvidence) { "<no-source-quality-run-evidence-import>" } else { [string](Get-PropertyOrDefault -Object $sourceQualityEvidence -Name "headSha" -DefaultValue "<missing-source-quality-head-sha>") }
$sourceQualityRunEvidenceReady = $null -ne $sourceQualityEvidence -and (
  (Get-BoolPropertyOrDefault -Object $sourceQualityEvidence -Name "canClaimGitHubActionsSourceQualityForRun" -DefaultValue $false) -or
  ([string](Get-PropertyOrDefault -Object $sourceQualityEvidence -Name "evidenceState" -DefaultValue "")).Equals("source-quality-run-evidence-ready", [StringComparison]::OrdinalIgnoreCase)
)
$sourceQualityHeadMatchesCurrentHead = Test-SameSha -Left $sourceQualityHeadSha -Right $currentHeadValue

$dryRunPackages = if ($null -eq $packageDryRunEvidence) {
  @()
}
else {
  @(Get-PropertyOrDefault -Object $packageDryRunEvidence -Name "nupkgPackages" -DefaultValue @())
}

$dryRunManagedPackage = @($dryRunPackages | Where-Object {
    $fileName = [string](Get-PropertyOrDefault -Object $_ -Name "fileName" -DefaultValue "")
    $fileName.StartsWith("JYPPX.TensorRT.CSharp.API.", [StringComparison]::OrdinalIgnoreCase)
  } | Select-Object -First 1)
if ($dryRunManagedPackage.Count -eq 0 -and $dryRunPackages.Count -gt 0) {
  $dryRunManagedPackage = @($dryRunPackages[0])
}

$sourceGitHubActionsRunId = if ($null -eq $packageDryRunEvidence) { "<no-package-dry-run-evidence-import>" } else { [string](Get-PropertyOrDefault -Object $packageDryRunEvidence -Name "runId" -DefaultValue "<missing-package-dry-run-run-id>") }
$sourceGitHubActionsRunUrl = if ($null -eq $packageDryRunEvidence) { "<no-package-dry-run-evidence-import>" } else { [string](Get-PropertyOrDefault -Object $packageDryRunEvidence -Name "runUrl" -DefaultValue "<missing-package-dry-run-run-url>") }
$sourceHeadSha = if ($null -eq $packageDryRunEvidence) { "<no-package-dry-run-evidence-import>" } else { [string](Get-PropertyOrDefault -Object $packageDryRunEvidence -Name "headSha" -DefaultValue "<missing-package-dry-run-head-sha>") }
$packageDryRunCanClaimPack = if ($null -eq $packageDryRunEvidence) { $false } else { [bool](Get-PropertyOrDefault -Object $packageDryRunEvidence -Name "canClaimGitHubActionsPackageDryRunPackForRun" -DefaultValue $false) }
$packageDryRunHeadMatchesCurrentHead = Test-SameSha -Left $sourceHeadSha -Right $currentHeadValue
$packageDryRunCanClaimCurrentHeadPack = $packageDryRunCanClaimPack -and $packageDryRunHeadMatchesCurrentHead
$packageDryRunRequiresOwnerAuthorization = -not $packageDryRunCanClaimCurrentHeadPack
$packageDryRunArtifactPath = if ($dryRunManagedPackage.Count -eq 0) { "<no-package-managed-dry-run-artifact>" } else { [string](Get-PropertyOrDefault -Object $dryRunManagedPackage[0] -Name "fullPath" -DefaultValue "<missing-dry-run-package-path>") }
$packageDryRunManagedNupkgSha256 = if ($dryRunManagedPackage.Count -eq 0) { "<no-package-managed-dry-run-sha256>" } else { [string](Get-PropertyOrDefault -Object $dryRunManagedPackage[0] -Name "sha256" -DefaultValue "<missing-dry-run-package-sha256>") }
$currentHeadPackageDryRunPreflightPresent = $null -ne $currentHeadPackageDryRunPreflight -and
  ([string](Get-PropertyOrDefault -Object $currentHeadPackageDryRunPreflight -Name "recordKind" -DefaultValue "")).Equals("current-head-package-dry-run-preflight", [StringComparison]::Ordinal)
$currentHeadPackageDryRunPreflightState = if ($currentHeadPackageDryRunPreflightPresent) { [string](Get-PropertyOrDefault -Object $currentHeadPackageDryRunPreflight -Name "state" -DefaultValue "missing-state") } else { "missing-current-head-package-dry-run-preflight" }
$currentHeadPackageDryRunReady = $currentHeadPackageDryRunPreflightPresent -and (Get-BoolPropertyOrDefault -Object $currentHeadPackageDryRunPreflight -Name "canClaimGitHubActionsPackageDryRunPackForCurrentHead" -DefaultValue $false)
$currentHeadPackageDryRunBlockedReason = if ($currentHeadPackageDryRunPreflightPresent) { [string](Get-PropertyOrDefault -Object $currentHeadPackageDryRunPreflight -Name "blockedReason" -DefaultValue "Current HEAD package dry-run proof is not ready.") } else { "Current HEAD package dry-run preflight is missing." }
$currentHeadPackageDryRunOwnerAuthorizationRequired = if ($currentHeadPackageDryRunPreflightPresent) { Get-BoolPropertyOrDefault -Object $currentHeadPackageDryRunPreflight -Name "packageDryRunRequiresOwnerAuthorization" -DefaultValue (-not $currentHeadPackageDryRunReady) } else { $true }
$packageDryRunCanClaimCurrentHeadPack = $packageDryRunCanClaimCurrentHeadPack -and $currentHeadPackageDryRunReady
$packageDryRunRequiresOwnerAuthorization = $packageDryRunRequiresOwnerAuthorization -or $currentHeadPackageDryRunOwnerAuthorizationRequired

$template = [pscustomobject]@{
  recordKind = "package-consumer-runtime-proof-owner-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  ownerInputState = "template-owner-input-required"
  proofLineId = "package-consumer-runtime"
  currentHead = $currentHeadValue
  currentHeadPackageDryRunPreflightPath = $CurrentHeadPackageDryRunPreflightPath
  currentHeadPackageDryRunPreflightPresent = $currentHeadPackageDryRunPreflightPresent
  currentHeadPackageDryRunPreflightState = $currentHeadPackageDryRunPreflightState
  currentHeadPackageDryRunReady = $currentHeadPackageDryRunReady
  currentHeadPackageDryRunOwnerAuthorizationRequired = $currentHeadPackageDryRunOwnerAuthorizationRequired
  currentHeadPackageDryRunBlockedReason = $currentHeadPackageDryRunBlockedReason
  currentHeadPackageDryRunOwnerAction = if ($currentHeadPackageDryRunReady) { "current-head-package-dry-run-context-available" } else { "owner-authorize-non-publish-workflow-dispatch-and-import-current-head-dry-run-evidence" }
  sourceQualityRunEvidenceImportPath = $SourceQualityRunEvidenceImportPath
  sourceQualityRunId = $sourceQualityRunId
  sourceQualityRunUrl = $sourceQualityRunUrl
  sourceQualityHeadSha = $sourceQualityHeadSha
  sourceQualityRunEvidenceReady = $sourceQualityRunEvidenceReady
  sourceQualityHeadMatchesCurrentHead = $sourceQualityHeadMatchesCurrentHead
  sourceGitHubActionsRunEvidenceImportPath = $PackageDryRunEvidenceImportPath
  packageDryRunEvidenceImportPath = $PackageDryRunEvidenceImportPath
  packageDryRunRunId = $sourceGitHubActionsRunId
  packageDryRunRunUrl = $sourceGitHubActionsRunUrl
  packageDryRunHeadSha = $sourceHeadSha
  packageDryRunHeadMatchesCurrentHead = $packageDryRunHeadMatchesCurrentHead
  sourceGitHubActionsRunId = $sourceGitHubActionsRunId
  sourceGitHubActionsRunUrl = $sourceGitHubActionsRunUrl
  sourceHeadSha = $sourceHeadSha
  packageDryRunArtifactPath = $packageDryRunArtifactPath
  packageDryRunManagedNupkgSha256 = $packageDryRunManagedNupkgSha256
  packageDryRunCanClaimPack = $packageDryRunCanClaimPack
  packageDryRunCanClaimCurrentHeadPack = $packageDryRunCanClaimCurrentHeadPack
  packageDryRunRequiresOwnerAuthorization = $packageDryRunRequiresOwnerAuthorization
  manualWorkflowDispatchNotPerformed = $true
  ownerAuthorizationState = if ($packageDryRunRequiresOwnerAuthorization) { "owner-authorization-required-before-current-head-package-dry-run" } else { "current-head-package-dry-run-context-available" }
  isDryRunOnly = $true
  isPublishedPackageProof = $false
  isPackageConsumerRuntimeProof = $false
  cleanExternalConsumerRoot = "<owner-fill-clean-external-consumer-root-outside-repository>"
  consumerProjectPath = "<owner-fill-clean-consumer-csproj-path>"
  publicPackageSourceKind = "<owner-fill-nuget-or-github-packages>"
  publicPackageSource = "<owner-fill-public-package-source-url-or-id>"
  publicPackageFeedUrl = "<owner-fill-public-feed-url>"
  managedPackageUrl = "<owner-fill-public-managed-package-url>"
  managedPackageId = "JYPPX.TensorRT.CSharp.API"
  managedPackageVersion = "<owner-fill-managed-package-version>"
  managedNupkgPath = "<owner-fill-public-managed-nupkg-path>"
  managedNupkgSha256 = "<owner-fill-managed-nupkg-sha256>"
  runtimePackageUrl = "<owner-fill-public-runtime-package-url>"
  runtimePackageId = "JYPPX.TensorRT.CSharp.API.runtime.$RuntimePackageKey"
  runtimePackageVersion = "<owner-fill-runtime-package-version>"
  runtimePackageKey = $RuntimePackageKey
  runtimeNupkgPath = "<owner-fill-public-runtime-nupkg-path>"
  runtimeNupkgSha256 = "<owner-fill-runtime-nupkg-sha256>"
  ownerName = "<owner-fill-owner-name>"
  machineName = "<owner-fill-machine-name>"
  hostOs = "<owner-fill-host-os>"
  hostArchitecture = "<owner-fill-host-architecture>"
  gpuName = "<owner-fill-gpu-name>"
  cudaDriverVersion = "<owner-fill-cuda-driver-version>"
  cudaDriverSupportedRuntime = "<owner-fill-cuda-driver-supported-runtime>"
  cudaRuntimeVersion = "<owner-fill-cuda-runtime-version>"
  cudnnVersion = "<owner-fill-cudnn-version>"
  tensorRtVersion = "<owner-fill-tensorrt-version>"
  tensorRtLine = "<owner-fill-tensorrt-line>"
  restoreCommand = "dotnet restore <clean-consumer-project> --source <public-package-source>"
  buildCommand = "dotnet build <clean-consumer-project> -c Release --no-restore"
  smokeCommand = "dotnet run --project <clean-consumer-project> -- --runtime-package-key $RuntimePackageKey"
  exitCode = "<owner-fill-smoke-exit-code>"
  startedAtUtc = "<owner-fill-started-at-utc>"
  finishedAtUtc = "<owner-fill-finished-at-utc>"
  dependencyProbeStatus = "<owner-fill-dependency-probe-status>"
  smokeStatus = "<owner-fill-smoke-status>"
  nativeAssetsCopied = "<owner-fill-native-assets-copied-true-or-false>"
  smokeLogPath = "<owner-fill-package-consumer-smoke-log-path>"
  smokeLogSha256 = "<owner-fill-smoke-log-sha256>"
  stdoutSummary = "<owner-fill-stdout-summary>"
  stderrSummary = "<owner-fill-stderr-summary>"
  failureDiagnostic = "<owner-fill-failure-diagnostic-or-empty>"
  sourceRunnerQueueStatus = "<owner-fill-completed-not-queued>"
  sourceRunnerInfrastructureStatus = "<owner-fill-available-not-missing-self-hosted-runner>"
  sourceRunnerOwnerAction = "owner-infra-action-required-until-public-package-run-completes"
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteProof = $false
  forbiddenSubstitutes = @(
    "local feed",
    "ProjectReference",
    "direct .nupkg",
    "build-only",
    "dry-run",
    "queued GitHub Actions run",
    "missing self-hosted runner",
    "dashboard",
    "template",
    "skipped run"
  )
  safetyBoundary = "Owner input template only. It records source-quality run evidence, package-managed dry-run evidence, and current-head dry-run preflight separately. Source-quality evidence and current-head preflight cannot fill public package proof fields. Package dry-run context may reduce manual copying only when it is imported separately and matches currentHead, but it is not published package proof and is not package-consumer runtime proof. It does not publish packages, close the release issue, or promote package-consumer runtime proof. local feed, ProjectReference, direct .nupkg, build-only, dry-run, queued GitHub Actions run, missing self-hosted runner, dashboard, template, and skipped run cannot be used as public package proof."
}

$jsonPath = Join-Path $artifactRoot "package-consumer-runtime-proof-owner-input.template.json"
$markdownPath = Join-Path $artifactRoot "package-consumer-runtime-proof-owner-input.template.md"

$template | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Package Consumer Runtime Proof Owner Input Template

生成时间：$($template.generatedAtUtc)

## 用途

该模板用于 Owner 在真实 clean external consumer 环境中回填 package-consumer runtime proof candidate 所需输入。它不是 proof，不执行发布，不关闭 release issue，也不会把 local feed、ProjectReference 或 direct `.nupkg` 当作 public package proof。

## GitHub Actions Dry-Run 上下文

模板会分别读取 source-quality evidence 和 package dry-run evidence。source-quality evidence 只能说明当前代码已通过 source-quality run，不能预填 `packageDryRunArtifactPath` 或 `packageDryRunManagedNupkgSha256`。只有单独导入的 package dry-run evidence 才能提供 dry-run pack 对照信息；它仍是 dry-run-only，不能替代 NuGet/GitHub Packages 已发布证明，不能替代 clean external package consumer runtime smoke，也不能把 `packageDryRunArtifactPath` 直接填入 `managedNupkgPath`。

| 字段 | 当前值 |
|---|---|
| currentHead | ``$($template.currentHead)`` |
| currentHeadPackageDryRunPreflightState | ``$($template.currentHeadPackageDryRunPreflightState)`` |
| currentHeadPackageDryRunReady | ``$($template.currentHeadPackageDryRunReady)`` |
| currentHeadPackageDryRunOwnerAuthorizationRequired | ``$($template.currentHeadPackageDryRunOwnerAuthorizationRequired)`` |
| currentHeadPackageDryRunOwnerAction | ``$($template.currentHeadPackageDryRunOwnerAction)`` |
| currentHeadPackageDryRunBlockedReason | ``$($template.currentHeadPackageDryRunBlockedReason)`` |
| sourceQualityRunId | ``$($template.sourceQualityRunId)`` |
| sourceQualityHeadSha | ``$($template.sourceQualityHeadSha)`` |
| sourceQualityRunEvidenceReady | ``$($template.sourceQualityRunEvidenceReady)`` |
| sourceQualityHeadMatchesCurrentHead | ``$($template.sourceQualityHeadMatchesCurrentHead)`` |
| packageDryRunRunId | ``$($template.packageDryRunRunId)`` |
| packageDryRunHeadSha | ``$($template.packageDryRunHeadSha)`` |
| packageDryRunHeadMatchesCurrentHead | ``$($template.packageDryRunHeadMatchesCurrentHead)`` |
| sourceGitHubActionsRunId | ``$($template.sourceGitHubActionsRunId)`` |
| sourceHeadSha | ``$($template.sourceHeadSha)`` |
| packageDryRunArtifactPath | ``$($template.packageDryRunArtifactPath)`` |
| packageDryRunManagedNupkgSha256 | ``$($template.packageDryRunManagedNupkgSha256)`` |
| packageDryRunCanClaimPack | ``$($template.packageDryRunCanClaimPack)`` |
| packageDryRunCanClaimCurrentHeadPack | ``$($template.packageDryRunCanClaimCurrentHeadPack)`` |
| packageDryRunRequiresOwnerAuthorization | ``$($template.packageDryRunRequiresOwnerAuthorization)`` |
| manualWorkflowDispatchNotPerformed | ``$($template.manualWorkflowDispatchNotPerformed)`` |
| isDryRunOnly | ``$($template.isDryRunOnly)`` |
| isPublishedPackageProof | ``$($template.isPublishedPackageProof)`` |
| isPackageConsumerRuntimeProof | ``$($template.isPackageConsumerRuntimeProof)`` |

## 必填字段

- cleanExternalConsumerRoot
- consumerProjectPath
- publicPackageSourceKind / publicPackageSource / publicPackageFeedUrl
- managedPackageUrl / runtimePackageUrl
- managedPackageId / managedPackageVersion / managedNupkgPath / managedNupkgSha256
- runtimePackageId / runtimePackageVersion / runtimePackageKey / runtimeNupkgPath / runtimeNupkgSha256
- ownerName / machineName / hostOs / hostArchitecture / gpuName
- cudaDriverVersion / cudaDriverSupportedRuntime / cudaRuntimeVersion / cudnnVersion
- tensorRtVersion / tensorRtLine
- restoreCommand / buildCommand / smokeCommand
- exitCode / startedAtUtc / finishedAtUtc
- dependencyProbeStatus / smokeStatus / nativeAssetsCopied
- smokeLogPath / smokeLogSha256
- sourceRunnerQueueStatus / sourceRunnerInfrastructureStatus / sourceRunnerOwnerAction
- stdoutSummary / stderrSummary
- failureDiagnostic

## Validator

```powershell
pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PackageConsumerRuntimeProofOwnerInput.ps1 -Strict
```

## Safety Boundary

$($template.safetyBoundary)

## Forbidden Substitutes

$($template.forbiddenSubstitutes | ForEach-Object { "- ``$_``" } | Out-String)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Package consumer runtime proof owner input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
