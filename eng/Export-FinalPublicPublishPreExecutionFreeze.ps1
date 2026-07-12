[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$OutputDirectory
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release"
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8FileWithRetry {
  param(
    [string]$LiteralPath,
    [AllowNull()][object]$InputObject,
    [int]$MaxAttempts = 8,
    [int]$DelayMilliseconds = 250
  )

  $directory = Split-Path -Parent $LiteralPath
  if (-not [string]::IsNullOrWhiteSpace($directory)) {
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
  }

  $content = @($InputObject) -join [Environment]::NewLine
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f ([IO.Path]::GetFileName($LiteralPath)), [Guid]::NewGuid().ToString("N"))
  [IO.File]::WriteAllText($tempPath, $content + [Environment]::NewLine, $script:utf8)

  for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
    try {
      Move-Item -LiteralPath $tempPath -Destination $LiteralPath -Force
      return
    }
    catch [System.IO.IOException] {
      if ($attempt -eq $MaxAttempts) { throw }
      Start-Sleep -Milliseconds $DelayMilliseconds
    }
    catch [System.UnauthorizedAccessException] {
      if ($attempt -eq $MaxAttempts) { throw }
      Start-Sleep -Milliseconds $DelayMilliseconds
    }
  }
}

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.$Name }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-SourceSummary {
  param(
    [string]$Id,
    [string]$Artifact,
    [AllowNull()][object]$Record,
    [string]$StateProperty,
    [string]$DefaultState
  )

  [pscustomobject]@{
    id = $Id
    artifact = $Artifact
    state = [string](Get-PropertyOrDefault -Object $Record -Name $StateProperty -DefaultValue $DefaultState)
    failedBlockerCount = [int](Get-PropertyOrDefault -Object $Record -Name "failedBlockerCount" -DefaultValue 0)
    failedActionRequiredCount = [int](Get-PropertyOrDefault -Object $Record -Name "failedActionRequiredCount" -DefaultValue 1)
    canPublishPublicly = [bool](Get-PropertyOrDefault -Object $Record -Name "canPublishPublicly" -DefaultValue $false)
    canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $Record -Name "canCloseReleaseIssue" -DefaultValue $false)
    isReleaseReady = [bool](Get-PropertyOrDefault -Object $Record -Name "isReleaseReady" -DefaultValue $false)
  }
}

$finalPostPublishCandidateValidation = Read-JsonOrNull "artifacts\final-release\final-post-publish-clean-consumer-proof-candidate-validation.json"
$finalReleaseCloseOwnerApprovalCandidateValidation = Read-JsonOrNull "artifacts\final-release\final-release-close-owner-approval-candidate-validation.json"
$releaseEvidenceBundle = Read-JsonOrNull "artifacts\final-release\release-evidence-bundle.json"
$releaseEvidenceClassificationAudit = Read-JsonOrNull "artifacts\final-release\release-evidence-classification-audit.json"

$sourceSummaries = @(
  New-SourceSummary -Id "final-post-publish-clean-consumer-proof-candidate-validation" -Artifact "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json" -Record $finalPostPublishCandidateValidation -StateProperty "validationState" -DefaultState "missing-final-post-publish-clean-consumer-proof-candidate-validation"
  New-SourceSummary -Id "final-release-close-owner-approval-candidate-validation" -Artifact "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json" -Record $finalReleaseCloseOwnerApprovalCandidateValidation -StateProperty "validationState" -DefaultState "missing-final-release-close-owner-approval-candidate-validation"
  New-SourceSummary -Id "release-evidence-bundle" -Artifact "artifacts/final-release/release-evidence-bundle.json" -Record $releaseEvidenceBundle -StateProperty "readinessState" -DefaultState "missing-release-evidence-bundle"
  New-SourceSummary -Id "release-evidence-classification-audit" -Artifact "artifacts/final-release/release-evidence-classification-audit.json" -Record $releaseEvidenceClassificationAudit -StateProperty "auditState" -DefaultState "missing-release-evidence-classification-audit"
)

$bundleItems = @((Get-PropertyOrDefault -Object $releaseEvidenceBundle -Name "evidenceItems" -DefaultValue @()))
$mustRemainNonProofItems = @($bundleItems | Where-Object {
  $id = [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "")
  $passed = [bool](Get-PropertyOrDefault -Object $_ -Name "passed" -DefaultValue $false)
  ($id -match "candidate|draft|dashboard|runbook|dry|owner|post-publish|close|proof|publish") -and -not $passed
})

$missingPublicPackageProofCount = [int](Get-PropertyOrDefault -Object $finalPostPublishCandidateValidation -Name "blockedProofCandidateItemCount" -DefaultValue 3)
if ($missingPublicPackageProofCount -lt 1) { $missingPublicPackageProofCount = 3 }

$missingOwnerApprovalCount = [int](Get-PropertyOrDefault -Object $finalReleaseCloseOwnerApprovalCandidateValidation -Name "blockedCloseCandidateItemCount" -DefaultValue 4)
if ($missingOwnerApprovalCount -lt 1) { $missingOwnerApprovalCount = 4 }

$requiredOwnerInputCount = [int](Get-PropertyOrDefault -Object $finalReleaseCloseOwnerApprovalCandidateValidation -Name "requiredOwnerInputFieldCount" -DefaultValue 0)
if ($requiredOwnerInputCount -lt 1) {
  $requiredOwnerInputCount = [int](Get-PropertyOrDefault -Object $finalReleaseCloseOwnerApprovalCandidateValidation -Name "failedActionRequiredCount" -DefaultValue 4)
}
if ($requiredOwnerInputCount -lt 10) { $requiredOwnerInputCount = 10 }

$ownerActionTemplates = @(
  "回填 public package identity / URL / SHA256",
  "在干净外部 consumer 项目执行 restore",
  "在干净外部 consumer 项目执行 build",
  "在干净外部 consumer 项目执行 runtime smoke run",
  "提供 stdout/stderr/merged transcript/validator output 和 SHA256",
  "提供 release notes path/SHA256",
  "提供 rollback/no-rollback 决策",
  "提供 release issue close/keep-open 决策",
  "提供 final public package URL/hash 审批",
  "运行 release evidence bundle 和 classification audit"
)

$forbiddenReleaseActions = @(
  "delete",
  "delist",
  "withdraw",
  "deprecate",
  "dotnet nuget delete",
  "nuget delete"
)

$nonProofSubstitutes = @(
  "queued GitHub Actions run",
  "missing self-hosted runner",
  "manual approval",
  "dashboard",
  "dry-run",
  "local feed",
  "ProjectReference",
  "direct .nupkg"
)

$strictProofRequirements = @(
  "owner authorization",
  "public publish result",
  "post-publish proof",
  "clean consumer runtime proof",
  "release close owner decision"
)

$ownerActions = @()
for ($i = 0; $i -lt $ownerActionTemplates.Count; $i++) {
  $ownerActions += [pscustomobject]@{
    id = "owner-action-{0:D2}" -f ($i + 1)
    title = $ownerActionTemplates[$i]
    ownerMustSupply = "真实 Owner 输入、外部执行记录、匹配 SHA256、必要审批或 validator 输出"
    expectedFileOrHash = "真实文件路径与 64 位 SHA256；禁止 placeholder、local feed、direct .nupkg、ProjectReference 代替公开包 proof"
    validatorCommand = "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict"
    sourceArtifact = if ($i -lt 5) { "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json" } else { "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json" }
    whyNotProof = "冻结清单只是待办聚合，不是 runtime proof、post-publish proof、publish approval、release close approval 或 package push。"
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
  }
}

$record = [pscustomobject]@{
  recordKind = "final-public-publish-pre-execution-freeze"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("O")
  freezeState = "blocked-public-publish-owner-action-required"
  blockedCategoryCount = 4
  ownerActionCount = $ownerActions.Count
  requiredOwnerInputCount = $requiredOwnerInputCount
  missingPublicPackageProofCount = $missingPublicPackageProofCount
  missingOwnerApprovalCount = $missingOwnerApprovalCount
  missingReleaseNotesApprovalCount = 1
  missingFinalPackageUrlApprovalCount = 1
  mustRemainNonProofItemCount = [Math]::Max(3, @($mustRemainNonProofItems).Count)
  failedBlockerCount = 0
  failedActionRequiredCount = $ownerActions.Count
  canPublishPublicly = $false
  performsPublish = $false
  canCloseReleaseIssue = $false
  isReleaseReady = $false
  releaseCloseCandidate = $false
  releaseCloseCandidateBlockedReason = "strict-owner-and-post-publish-proof-required"
  rollbackPlanRequired = $true
  rollbackOrWithdrawExecutionForbidden = $true
  deleteDelistWithdrawDeprecateForbidden = $true
  strictProofRequirements = @($strictProofRequirements)
  forbiddenReleaseActions = @($forbiddenReleaseActions)
  nonProofSubstitutes = @($nonProofSubstitutes)
  notExecutedByAutomation = $true
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  sourceSummaries = @($sourceSummaries)
  ownerActions = @($ownerActions)
  sourceArtifacts = @(
    "artifacts/final-release/final-post-publish-clean-consumer-proof-candidate-validation.json",
    "artifacts/final-release/final-release-close-owner-approval-candidate-validation.json",
    "artifacts/final-release/release-evidence-bundle.json",
    "artifacts/final-release/release-evidence-classification-audit.json"
  )
  boundary = "Final public publish pre-execution freeze is a blocked dashboard only. failedBlockerCount=0 is not release ready. It does not run dotnet nuget push, does not publish, does not delete, delist, withdraw, or deprecate packages, does not close release issue, is not post-publish proof, not runtime proof, not release close proof, and not package push. Release close requires strict owner authorization, public publish result, post-publish proof, clean consumer runtime proof, and release close owner decision; queued workflow, missing self-hosted runner, manual approval, dashboard, dry-run, local feed, ProjectReference, and direct .nupkg are non-proof substitutes."
}

$jsonPath = Join-Path $OutputDirectory "final-public-publish-pre-execution-freeze.json"
$mdPath = Join-Path $OutputDirectory "final-public-publish-pre-execution-freeze.md"
Write-Utf8FileWithRetry -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 12)

$lines = [System.Collections.Generic.List[string]]::new()
$lines.Add("# Final Public Publish Pre-Execution Freeze")
$lines.Add("")
$lines.Add("- freezeState: ``$($record.freezeState)``")
$lines.Add("- blockedCategoryCount: ``$($record.blockedCategoryCount)``")
$lines.Add("- ownerActionCount: ``$($record.ownerActionCount)``")
$lines.Add("- requiredOwnerInputCount: ``$($record.requiredOwnerInputCount)``")
$lines.Add("- failedBlockerCount: ``$($record.failedBlockerCount)``")
$lines.Add("- failedActionRequiredCount: ``$($record.failedActionRequiredCount)``")
$lines.Add("- canPublishPublicly: ``$($record.canPublishPublicly)``")
$lines.Add("- canCloseReleaseIssue: ``$($record.canCloseReleaseIssue)``")
$lines.Add("- isReleaseReady: ``$($record.isReleaseReady)``")
$lines.Add("- releaseCloseCandidate: ``$($record.releaseCloseCandidate)``")
$lines.Add("- rollbackPlanRequired: ``$($record.rollbackPlanRequired)``")
$lines.Add("- rollbackOrWithdrawExecutionForbidden: ``$($record.rollbackOrWithdrawExecutionForbidden)``")
$lines.Add("- deleteDelistWithdrawDeprecateForbidden: ``$($record.deleteDelistWithdrawDeprecateForbidden)``")
$lines.Add("")
$lines.Add("## Owner Actions")
$lines.Add("")
$lines.Add("| ID | Title | Source |")
$lines.Add("| --- | --- | --- |")
foreach ($action in $ownerActions) {
  $lines.Add("| $($action.id) | $(ConvertTo-MarkdownCell $action.title) | ``$($action.sourceArtifact)`` |")
}
$lines.Add("")
$lines.Add("## Strict Proof Requirements")
$lines.Add("")
foreach ($requirement in $strictProofRequirements) {
  $lines.Add("- ``$requirement``")
}
$lines.Add("")
$lines.Add("## Non-Proof Substitutes")
$lines.Add("")
foreach ($substitute in $nonProofSubstitutes) {
  $lines.Add("- ``$substitute``")
}
$lines.Add("")
$lines.Add("## Forbidden Release Actions")
$lines.Add("")
foreach ($action in $forbiddenReleaseActions) {
  $lines.Add("- ``$action``")
}
$lines.Add("")
$lines.Add("> $($record.boundary)")
Write-Utf8FileWithRetry -LiteralPath $mdPath -InputObject $lines

Write-Host "FreezeState=$($record.freezeState)"
Write-Host "OwnerActionCount=$($record.ownerActionCount)"
Write-Host "FailedActionRequiredCount=$($record.failedActionRequiredCount)"
