[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\release-close-final-candidate-audit-pack.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
if (-not [System.IO.Path]::IsPathRooted($InputPath)) { $InputPath = Join-Path $RepositoryRoot $InputPath }
if (-not (Test-Path -LiteralPath $InputPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Export-ReleaseCloseFinalCandidateAuditPack.ps1") -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
}

$record = Get-Content -LiteralPath $InputPath -Raw -Encoding utf8 | ConvertFrom-Json
$checks = @(Convert-ToArray (Get-PropertyOrDefault -Object $record -Name "checks" -DefaultValue @()))
$checkIds = @($checks | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
$requiredCheckIds = @(
  "release-evidence-bundle-hash-lock",
  "owner-import-readiness-no-proof-ready",
  "final-landing-pack-non-proof",
  "public-package-url-hash-proof",
  "external-clean-consumer-post-publish-proof",
  "strict-close-dashboard-and-bridge",
  "release-issue-close-owner-decision",
  "rollback-and-known-limitations",
  "forbidden-substitute-final-scan"
)

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "release-close-final-candidate-audit-pack") "blocker" "recordKind must match.")) | Out-Null
$items.Add((New-OwnerValidationItem "audit-state" ([string](Get-PropertyOrDefault -Object $record -Name "auditState" -DefaultValue "") -eq "blocked-release-close-final-candidate-real-owner-evidence-required") "blocker" "Audit pack must remain blocked until real Owner evidence exists.")) | Out-Null
$items.Add((New-OwnerValidationItem "check-count" ($checks.Count -ge 9 -and [int](Get-PropertyOrDefault -Object $record -Name "checkCount" -DefaultValue 0) -eq $checks.Count) "blocker" "Audit pack must include all final close candidate checks.")) | Out-Null
$items.Add((New-OwnerValidationItem "blocked-checks" ([int](Get-PropertyOrDefault -Object $record -Name "blockedCheckCount" -DefaultValue 0) -eq $checks.Count -and [int](Get-PropertyOrDefault -Object $record -Name "passedCheckCount" -DefaultValue -1) -eq 0) "blocker" "All final close candidate checks must remain blocked by default.")) | Out-Null
$items.Add((New-OwnerValidationItem "bundle-hash" (Test-Sha256Text (Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue "")) "blocker" "Audit pack must carry a SHA256 of release-evidence-bundle.json.")) | Out-Null
$items.Add((New-OwnerValidationItem "classification-hash" (Test-Sha256Text (Get-PropertyOrDefault -Object $record -Name "classificationAuditSha256" -DefaultValue "")) "blocker" "Audit pack must carry a SHA256 of release-evidence-classification-audit.json.")) | Out-Null
$items.Add((New-OwnerValidationItem "non-proof-flags" ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isRuntimeExecutionProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseProof" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "isReleaseCloseRecordProof" -DefaultValue $true))) "blocker" "Audit pack must not publish, close, or promote proof.")) | Out-Null

foreach ($id in $requiredCheckIds) {
  $items.Add((New-OwnerValidationItem "check-$id" ($checkIds -contains $id) "blocker" "Missing required final close check: $id")) | Out-Null
}

foreach ($check in $checks) {
  $id = [string](Get-PropertyOrDefault -Object $check -Name "id" -DefaultValue "")
  $boundary = [string](Get-PropertyOrDefault -Object $check -Name "boundary" -DefaultValue "")
  $items.Add((New-OwnerValidationItem "check-$id-blocked" ([string](Get-PropertyOrDefault -Object $check -Name "checkState" -DefaultValue "") -eq "blocked-real-owner-evidence-required" -and -not [bool](Get-PropertyOrDefault -Object $check -Name "passed" -DefaultValue $true)) "blocker" "Check must remain blocked and failed by default.")) | Out-Null
  $items.Add((New-OwnerValidationItem "check-$id-inputs" ([int](Get-PropertyOrDefault -Object $check -Name "requiredRealInputCount" -DefaultValue 0) -ge 3 -and [int](Get-PropertyOrDefault -Object $check -Name "blockedReasonCount" -DefaultValue 0) -ge 2) "blocker" "Check must describe required real inputs and blockers.")) | Out-Null
  $items.Add((New-OwnerValidationItem "check-$id-boundary" ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Check boundary must be explicit.")) | Out-Null
}

$boundary = [string](Get-PropertyOrDefault -Object $record -Name "boundary" -DefaultValue "")
$items.Add((New-OwnerValidationItem "boundary" ($boundary.Contains("not runtime proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not post-publish proof", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not publish approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not release close approval", [StringComparison]::OrdinalIgnoreCase) -and $boundary.Contains("not package push", [StringComparison]::OrdinalIgnoreCase)) "blocker" "Boundary must preserve all non-proof classifications.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$state = if ($failedBlockers.Count -eq 0) { "release-close-final-candidate-audit-pack-ready-non-proof" } else { "blocked-release-close-final-candidate-audit-pack-invalid" }
$validation = [pscustomobject]@{
  recordKind = "release-close-final-candidate-audit-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = [int]$failedBlockers.Count
  validationItemCount = [int]$validationItems.Count
  checkCount = [int]$checks.Count
  blockedCheckCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedCheckCount" -DefaultValue 0)
  releaseEvidenceBundleSha256 = [string](Get-PropertyOrDefault -Object $record -Name "releaseEvidenceBundleSha256" -DefaultValue "")
  classificationAuditSha256 = [string](Get-PropertyOrDefault -Object $record -Name "classificationAuditSha256" -DefaultValue "")
  validationItems = @($validationItems)
  ownerActionRequired = $true
  passed = $false
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  isReleaseCloseRecordProof = $false
  boundary = "Release close final candidate audit pack validation is non-proof structure validation only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "release-close-final-candidate-audit-pack-validation.json"
$mdPath = Join-Path $OutputRoot "release-close-final-candidate-audit-pack-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 12)
Write-Utf8File -LiteralPath $mdPath -InputObject @(
  "# Release Close Final Candidate Audit Pack Validation",
  "",
  "- validationState: ``$state``",
  "- failedBlockerCount: ``$($failedBlockers.Count)``",
  "- checkCount: ``$($validation.checkCount)``",
  "- canCloseReleaseIssue: ``False``",
  "",
  $validation.boundary
)
Write-Host "ReleaseCloseFinalCandidateAuditPackValidationState=$state FailedBlockers=$($failedBlockers.Count) Checks=$($validation.checkCount)"
if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) { throw "Release close final candidate audit pack validation failed." }
