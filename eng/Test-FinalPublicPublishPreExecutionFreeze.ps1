[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$InputPath,
  [string]$OutputDirectory,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) { $OutputDirectory = Join-Path $RepositoryRoot "artifacts\final-release" }
if ([string]::IsNullOrWhiteSpace($InputPath)) { $InputPath = Join-Path $OutputDirectory "final-public-publish-pre-execution-freeze.json" }

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8FileWithRetry {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  $directory = Split-Path -Parent $LiteralPath
  if (-not [string]::IsNullOrWhiteSpace($directory)) { New-Item -ItemType Directory -Path $directory -Force | Out-Null }
  [IO.File]::WriteAllText($LiteralPath, (@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine, $script:utf8)
}

function Add-Finding {
  param([System.Collections.Generic.List[object]]$Findings, [string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  $Findings.Add([pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }) | Out-Null
}

function Test-StringArrayContainsAll {
  param(
    [AllowNull()][object]$Array,
    [string[]]$Expected
  )

  $values = @($Array | ForEach-Object { [string]$_ })
  foreach ($item in $Expected) {
    if (-not ($values | Where-Object { $_.Equals($item, [StringComparison]::OrdinalIgnoreCase) })) {
      return $false
    }
  }

  return $true
}

if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-FinalPublicPublishPreExecutionFreeze.ps1") -RepositoryRoot $RepositoryRoot -OutputDirectory $OutputDirectory
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$findings = [System.Collections.Generic.List[object]]::new()
$boundary = [string]$record.boundary
$requiredNonProofSubstitutes = @("queued GitHub Actions run", "missing self-hosted runner", "manual approval", "dashboard", "dry-run", "local feed", "ProjectReference", "direct .nupkg")
$requiredForbiddenActions = @("delete", "delist", "withdraw", "deprecate", "dotnet nuget delete", "nuget delete")
$requiredStrictProofs = @("owner authorization", "public publish result", "post-publish proof", "clean consumer runtime proof", "release close owner decision")
Add-Finding $findings "record-kind" ($record.recordKind -eq "final-public-publish-pre-execution-freeze") "blocker" "recordKind must match final-public-publish-pre-execution-freeze."
Add-Finding $findings "freeze-state-blocked" ($record.freezeState -eq "blocked-public-publish-owner-action-required") "blocker" "Freeze must remain blocked until Owner action is supplied."
Add-Finding $findings "owner-action-count" ([int]$record.ownerActionCount -ge 10) "blocker" "Owner action count must be at least 10."
Add-Finding $findings "required-owner-input" ([int]$record.requiredOwnerInputCount -gt 0) "action-required" "Required Owner input count must be greater than zero."
Add-Finding $findings "missing-public-package-proof" ([int]$record.missingPublicPackageProofCount -gt 0) "action-required" "Public package proof must still be missing."
Add-Finding $findings "missing-owner-approval" ([int]$record.missingOwnerApprovalCount -gt 0) "action-required" "Owner approval must still be missing."
Add-Finding $findings "non-proof-items" ([int]$record.mustRemainNonProofItemCount -ge 3) "action-required" "At least freeze/worklist/dry contract surfaces must remain non-proof."
Add-Finding $findings "does-not-publish" (-not [bool]$record.performsPublish -and -not [bool]$record.canPublishPublicly) "blocker" "Freeze must not publish or allow public publish."
Add-Finding $findings "does-not-close" (-not [bool]$record.canCloseReleaseIssue -and -not [bool]$record.isReleaseReady) "blocker" "Freeze must not close release issue or mark release ready."
Add-Finding $findings "failed-blocker-zero" ([int]$record.failedBlockerCount -eq 0) "blocker" "failedBlockerCount should remain zero for structural freeze."
Add-Finding $findings "action-required-positive" ([int]$record.failedActionRequiredCount -gt 0) "action-required" "failedActionRequiredCount must remain positive."
Add-Finding $findings "release-close-candidate-blocked" (-not [bool]$record.releaseCloseCandidate -and [string]$record.releaseCloseCandidateBlockedReason -eq "strict-owner-and-post-publish-proof-required") "blocker" "Freeze must not become a release close candidate without strict proof."
Add-Finding $findings "rollback-plan-required" ([bool]$record.rollbackPlanRequired) "action-required" "Rollback/no-rollback plan must be explicitly required."
Add-Finding $findings "rollback-execution-forbidden" ([bool]$record.rollbackOrWithdrawExecutionForbidden -and [bool]$record.deleteDelistWithdrawDeprecateForbidden) "blocker" "Rollback/delete/delist/withdraw/deprecate execution must be forbidden by automation."
Add-Finding $findings "forbidden-release-actions" (Test-StringArrayContainsAll $record.forbiddenReleaseActions $requiredForbiddenActions) "blocker" "Forbidden release actions must include delete/delist/withdraw/deprecate command surfaces."
Add-Finding $findings "strict-proof-requirements" (Test-StringArrayContainsAll $record.strictProofRequirements $requiredStrictProofs) "blocker" "Strict close requirements must include owner authorization, publish result, post-publish proof, clean consumer proof, and owner close decision."
Add-Finding $findings "non-proof-substitutes" (Test-StringArrayContainsAll $record.nonProofSubstitutes $requiredNonProofSubstitutes) "action-required" "Queued runs, missing runners, manual approval, dashboards, dry-runs, local feed, ProjectReference, and direct nupkg must remain non-proof substitutes."
Add-Finding $findings "boundary-forbids-release-actions" ($boundary.Contains("does not delete", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("delist", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("withdraw", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("deprecate", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must explicitly forbid delete/delist/withdraw/deprecate."
Add-Finding $findings "boundary-requires-strict-proof" ($boundary.Contains("strict owner authorization", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("clean consumer runtime proof", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must require strict owner and post-publish proof chain."

$failedBlockerCount = @($findings | Where-Object { $_.severity -eq "blocker" -and -not $_.passed }).Count
$failedActionRequiredCount = @($findings | Where-Object { $_.severity -eq "action-required" -and -not $_.passed }).Count
$validation = [pscustomobject]@{
  recordKind = "final-public-publish-pre-execution-freeze-validation"
  validationState = if ($failedBlockerCount -eq 0) { "blocked-public-publish-owner-action-required" } else { "failed-final-public-publish-pre-execution-freeze" }
  freezeState = [string]$record.freezeState
  blockedCategoryCount = [int]$record.blockedCategoryCount
  ownerActionCount = [int]$record.ownerActionCount
  requiredOwnerInputCount = [int]$record.requiredOwnerInputCount
  missingPublicPackageProofCount = [int]$record.missingPublicPackageProofCount
  missingOwnerApprovalCount = [int]$record.missingOwnerApprovalCount
  missingReleaseNotesApprovalCount = [int]$record.missingReleaseNotesApprovalCount
  missingFinalPackageUrlApprovalCount = [int]$record.missingFinalPackageUrlApprovalCount
  mustRemainNonProofItemCount = [int]$record.mustRemainNonProofItemCount
  failedBlockerCount = $failedBlockerCount
  failedActionRequiredCount = [Math]::Max($failedActionRequiredCount, [int]$record.failedActionRequiredCount)
  canPublishPublicly = [bool]$record.canPublishPublicly
  performsPublish = [bool]$record.performsPublish
  canCloseReleaseIssue = [bool]$record.canCloseReleaseIssue
  isReleaseReady = [bool]$record.isReleaseReady
  releaseCloseCandidate = [bool]$record.releaseCloseCandidate
  releaseCloseCandidateBlockedReason = [string]$record.releaseCloseCandidateBlockedReason
  rollbackPlanRequired = [bool]$record.rollbackPlanRequired
  rollbackOrWithdrawExecutionForbidden = [bool]$record.rollbackOrWithdrawExecutionForbidden
  deleteDelistWithdrawDeprecateForbidden = [bool]$record.deleteDelistWithdrawDeprecateForbidden
  strictProofRequirements = @($record.strictProofRequirements)
  forbiddenReleaseActions = @($record.forbiddenReleaseActions)
  nonProofSubstitutes = @($record.nonProofSubstitutes)
  findings = @($findings)
  boundary = "Validation confirms the freeze is structurally safe but still blocked by Owner action and strict post-publish proof; failedBlockerCount=0 is not release ready and rollback/delete/delist/withdraw/deprecate execution is forbidden."
}

$jsonPath = Join-Path $OutputDirectory "final-public-publish-pre-execution-freeze-validation.json"
$mdPath = Join-Path $OutputDirectory "final-public-publish-pre-execution-freeze-validation.md"
Write-Utf8FileWithRetry -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8FileWithRetry -LiteralPath $mdPath -InputObject @(
  "# Final Public Publish Pre-Execution Freeze Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- ownerActionCount: ``$($validation.ownerActionCount)``",
  "- failedBlockerCount: ``$($validation.failedBlockerCount)``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "- canPublishPublicly: ``$($validation.canPublishPublicly)``",
  "- canCloseReleaseIssue: ``$($validation.canCloseReleaseIssue)``",
  "- releaseCloseCandidate: ``$($validation.releaseCloseCandidate)``",
  "- rollbackOrWithdrawExecutionForbidden: ``$($validation.rollbackOrWithdrawExecutionForbidden)``",
  "- deleteDelistWithdrawDeprecateForbidden: ``$($validation.deleteDelistWithdrawDeprecateForbidden)``",
  "",
  "> $($validation.boundary)"
)

Write-Host "ValidationState=$($validation.validationState)"
Write-Host "FailedBlockerCount=$($validation.failedBlockerCount)"
Write-Host "FailedActionRequiredCount=$($validation.failedActionRequiredCount)"
if ($Strict -and $failedBlockerCount -gt 0) { throw "Final public publish pre-execution freeze validation failed." }
