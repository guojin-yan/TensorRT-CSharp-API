[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputRoot,
  [string]$PreReleaseReadinessMatrixPath = "artifacts\final-release\pre-release-package-proof-readiness-matrix.json"
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$runtimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22"
$runtimePackageId = "JYPPX.TensorRT.CSharp.API.runtime.$runtimePackageKey"

function Resolve-RepositoryPath {
  param([string]$Path)
  if ([string]::IsNullOrWhiteSpace($Path)) { return $Path }
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Read-JsonOrNull {
  param([string]$Path)
  $resolved = Resolve-RepositoryPath -Path $Path
  if (-not (Test-Path -LiteralPath $resolved -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $resolved -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Get-BoolPropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [bool]$DefaultValue)
  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue $DefaultValue
  if ($value -is [bool]) { return [bool]$value }
  $parsed = $false
  if ([bool]::TryParse(([string]$value).Trim(), [ref]$parsed)) { return $parsed }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$preReleaseReadinessMatrix = Read-JsonOrNull -Path $PreReleaseReadinessMatrixPath
$preReleaseReadinessMatrixState = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "matrixState" -DefaultValue "missing-pre-release-package-proof-readiness-matrix")
$preReleaseReadinessMatrixReady = $preReleaseReadinessMatrixState.Equals("pre-release-package-proof-ready", [StringComparison]::OrdinalIgnoreCase)
$preReleaseReadyLaneCount = [int](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "readyLaneCount" -DefaultValue 0)
$preReleaseBlockedLaneCount = [int](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "blockedLaneCount" -DefaultValue 0)
$preReleaseCurrentHead = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "currentHead" -DefaultValue "")
$preReleaseSourceQualityRunId = [string](Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "sourceQualityRunId" -DefaultValue "")
$preReleaseLanes = @((Get-PropertyOrDefault -Object $preReleaseReadinessMatrix -Name "lanes" -DefaultValue @()))
$blockedReadinessLanes = @($preReleaseLanes | Where-Object { -not (Get-BoolPropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) } | ForEach-Object {
    [pscustomobject]@{
      id = [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
      state = [string](Get-PropertyOrDefault -Object $_ -Name "state" -DefaultValue "")
      requiredEvidence = [string](Get-PropertyOrDefault -Object $_ -Name "requiredEvidence" -DefaultValue "")
      validatorPath = [string](Get-PropertyOrDefault -Object $_ -Name "validatorPath" -DefaultValue "")
      blockedReason = [string](Get-PropertyOrDefault -Object $_ -Name "blockedReason" -DefaultValue "")
    }
  })
$readinessRequiredEvidence = @($preReleaseLanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "requiredEvidence" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)
$readinessValidatorPaths = @($preReleaseLanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "validatorPath" -DefaultValue "") } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Unique)

$record = [pscustomobject]@{
  recordKind = "owner-publish-execution-result-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = "blocked-owner-publish-execution-result-required"
  ownerExecutionResultReady = $false
  managedPackageId = "JYPPX.TensorRT.CSharp.API"
  runtimePackageId = $runtimePackageId
  runtimePackageKey = $runtimePackageKey
  managedPackageVersion = "<owner-fill-managed-package-version>"
  runtimePackageVersion = "<owner-fill-runtime-package-version>"
  ownerName = "<owner-fill-owner-name>"
  ownerReviewedAtUtc = "<owner-fill-owner-reviewed-at-utc>"
  ownerApprovalReference = "<owner-fill-owner-approval-reference>"
  publishCommandReviewed = "<owner-fill-true>"
  publishExecutedByOwner = "<owner-fill-true>"
  publishStartedAtUtc = "<owner-fill-publish-started-at-utc>"
  publishCompletedAtUtc = "<owner-fill-publish-completed-at-utc>"
  publishExitCode = "<owner-fill-0>"
  publishCommand = "<owner-fill-redacted-command-without-token>"
  pushTranscriptPath = "<owner-fill-push-transcript-path>"
  pushTranscriptSha256 = "<owner-fill-push-transcript-sha256>"
  pushStdoutPath = "<owner-fill-push-stdout-path>"
  pushStdoutSha256 = "<owner-fill-push-stdout-sha256>"
  pushStderrPath = "<owner-fill-push-stderr-path>"
  pushStderrSha256 = "<owner-fill-push-stderr-sha256>"
  publicManagedPackageUrl = "<owner-fill-https-public-managed-package-url>"
  publicRuntimePackageUrl = "<owner-fill-https-public-runtime-package-url>"
  nugetPackageMetadataUrl = "<owner-fill-https-nuget-package-metadata-url>"
  githubPackagesMetadataUrl = "<owner-fill-https-github-packages-metadata-url>"
  downloadedManagedNupkgPath = "<owner-fill-downloaded-managed-nupkg-path>"
  downloadedManagedNupkgSha256 = "<owner-fill-downloaded-managed-nupkg-sha256>"
  downloadedRuntimeNupkgPath = "<owner-fill-downloaded-runtime-nupkg-path>"
  downloadedRuntimeNupkgSha256 = "<owner-fill-downloaded-runtime-nupkg-sha256>"
  releaseNotesPath = "<owner-fill-release-notes-path>"
  releaseNotesSha256 = "<owner-fill-release-notes-sha256>"
  rollbackPlanPath = "<owner-fill-rollback-plan-path>"
  rollbackPlanSha256 = "<owner-fill-rollback-plan-sha256>"
  rollbackDecision = "<owner-fill-rollback-decision>"
  preReleaseReadinessMatrixPath = $PreReleaseReadinessMatrixPath
  preReleaseReadinessMatrixState = $preReleaseReadinessMatrixState
  preReleaseReadinessMatrixReady = $preReleaseReadinessMatrixReady
  preReleaseReadinessCurrentHead = $preReleaseCurrentHead
  preReleaseReadinessSourceQualityRunId = $preReleaseSourceQualityRunId
  preReleaseReadyLaneCount = $preReleaseReadyLaneCount
  preReleaseBlockedLaneCount = $preReleaseBlockedLaneCount
  preReleaseBlockedLanes = @($blockedReadinessLanes)
  preReleaseRequiredEvidence = @($readinessRequiredEvidence)
  preReleaseValidatorPaths = @($readinessValidatorPaths)
  confirmsPreReleaseReadinessMatrixReviewed = "<owner-fill-true>"
  confirmsNoTokenPersisted = "<owner-fill-true>"
  confirmsNoTokenInTranscripts = "<owner-fill-true>"
  confirmsNoDryRunArtifactSubstitution = "<owner-fill-true>"
  confirmsNoLocalFeedSubstitution = "<owner-fill-true>"
  confirmsNoDirectNupkgSubstitution = "<owner-fill-true>"
  confirmsNoGitHubActionsArtifactSubstitution = "<owner-fill-true>"
  confirmsPublicPackageDownloadProofStillRequired = "<owner-fill-true>"
  confirmsPostPublishProofStillRequired = "<owner-fill-true>"
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  forbiddenSubstitutes = @(
    "local feed",
    "direct .nupkg",
    "artifacts/package-managed-dry-run",
    "GitHub Actions package artifact",
    "ProjectReference",
    "build-only",
    "dependency-probe-only",
    "dashboard-only",
    "template-only"
  )
  safetyBoundary = "Owner publish execution result input template only. It records owner-supplied publish result evidence after an owner-run publish; automation does not publish, does not use tokens, does not claim runtime proof, does not claim post-publish proof, and cannot close the release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-publish-execution-result-input.template.json"
$markdownPath = Join-Path $OutputRoot "owner-publish-execution-result-input.template.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$forbiddenRows = $record.forbiddenSubstitutes | ForEach-Object { "| $_ | rejected as real Owner publish execution result proof substitute |" }
$markdown = @"
# Owner Publish Execution Result Input Template

生成时间：$($record.generatedAtUtc)

| 项目 | 当前值 |
|---|---|
| validationState | ``$($record.validationState)`` |
| ownerExecutionResultReady | ``$($record.ownerExecutionResultReady)`` |
| managedPackageId | ``$($record.managedPackageId)`` |
| runtimePackageId | ``$($record.runtimePackageId)`` |
| preReleaseReadinessMatrixState | ``$($record.preReleaseReadinessMatrixState)`` |
| preReleaseReadinessMatrixReady | ``$($record.preReleaseReadinessMatrixReady)`` |
| preReleaseBlockedLaneCount | ``$($record.preReleaseBlockedLaneCount)`` |
| performsPublish | ``$($record.performsPublish)`` |
| usesPublishToken | ``$($record.usesPublishToken)`` |
| canPublishPublicly | ``$($record.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |

## Pre-Release Readiness Blocked Lanes

| Lane | State | Validator | Required Evidence |
|---|---|---|---|
$(@($record.preReleaseBlockedLanes | ForEach-Object { "| ``$(ConvertTo-MarkdownCell $_.id)`` | ``$(ConvertTo-MarkdownCell $_.state)`` | ``$(ConvertTo-MarkdownCell $_.validatorPath)`` | $(ConvertTo-MarkdownCell $_.requiredEvidence) |" }) -join "`r`n")

## Forbidden Substitutes

| Substitute | Boundary |
|---|---|
$($forbiddenRows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner publish execution result input template written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$($record.validationState) PerformsPublish=False UsesPublishToken=False CanCloseReleaseIssue=False"
