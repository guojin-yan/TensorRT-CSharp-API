[CmdletBinding()]
param(
  [string]$ImportPath = "artifacts\final-release\owner-real-proof-staging-workspace-import.json",
  [string]$CandidatePath = "artifacts\final-release\owner-real-proof-staging-workspace-candidate.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict,
  [switch]$FailOnNotProof
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")
$ctx = Initialize-OwnerRealProofScript -RepositoryRoot $RepositoryRoot -OutputRoot $OutputRoot
$RepositoryRoot = $ctx.RepositoryRoot
$OutputRoot = $ctx.OutputRoot
foreach ($name in @("ImportPath", "CandidatePath")) {
  if (-not [System.IO.Path]::IsPathRooted((Get-Variable $name).Value)) {
    Set-Variable -Name $name -Value (Join-Path $RepositoryRoot (Get-Variable $name).Value)
  }
}
if (-not (Test-Path -LiteralPath $ImportPath -PathType Leaf) -or -not (Test-Path -LiteralPath $CandidatePath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-OwnerRealProofStagingWorkspace.ps1") -RepositoryRoot $RepositoryRoot
}

$import = Get-Content -LiteralPath $ImportPath -Raw -Encoding utf8 | ConvertFrom-Json
$candidate = Get-Content -LiteralPath $CandidatePath -Raw -Encoding utf8 | ConvertFrom-Json
$boundary = [string](Get-PropertyOrDefault $import "boundary" "")
$validationItems = @(
  New-OwnerValidationItem "record-kind" ([string](Get-PropertyOrDefault $import "recordKind" "") -eq "owner-real-proof-staging-workspace-import" -and [string](Get-PropertyOrDefault $candidate "recordKind" "") -eq "owner-real-proof-staging-workspace-candidate") "blocker" "Import and candidate recordKind values must match."
  New-OwnerValidationItem "default-blocked" (([string](Get-PropertyOrDefault $import "importState" "")).Contains("blocked") -or [bool](Get-PropertyOrDefault $import "readyForStrictImport" $false)) "blocker" "Default import must remain blocked unless Owner staging is supplied."
  New-OwnerValidationItem "non-proof" (-not [bool](Get-PropertyOrDefault $import "proofCandidateReady" $true) -and -not [bool](Get-PropertyOrDefault $import "canCloseReleaseIssue" $true) -and -not [bool](Get-PropertyOrDefault $import "isRuntimeExecutionProof" $true)) "blocker" "Staging import must not claim proof."
  New-OwnerValidationItem "findings" ([int](Get-PropertyOrDefault $import "failedActionRequiredCount" 0) -gt 0 -or [bool](Get-PropertyOrDefault $import "readyForStrictImport" $false)) "blocker" "Default state should report owner action findings."
  New-OwnerValidationItem "boundary" ($boundary.Contains("not runtime proof") -and $boundary.Contains("not post-publish proof") -and $boundary.Contains("not package push")) "blocker" "Boundary must preserve non-proof status."
)
$failed = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$ready = [bool](Get-PropertyOrDefault $import "readyForStrictImport" $false)
$state = if ($failed.Count -eq 0) { "owner-real-proof-staging-workspace-validation-ready" } else { "blocked-owner-real-proof-staging-workspace-validation-invalid" }
$validation = [pscustomobject]@{
  recordKind = "owner-real-proof-staging-workspace-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $state
  failedBlockerCount = $failed.Count
  readyForStrictImport = $ready
  proofCandidateReady = $false
  validationItems = @($validationItems)
  ownerActionRequired = -not $ready
  passed = $false
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Validation checks Owner staging import shape only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}
$jsonPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-validation.json"
$mdPath = Join-Path $OutputRoot "owner-real-proof-staging-workspace-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 10)
Write-Utf8File -LiteralPath $mdPath -InputObject @("# Owner Real Proof Staging Workspace Validation", "", "- validationState: ``$state``", "- failedBlockerCount: ``$($failed.Count)``", "- readyForStrictImport: ``$ready``", "", "## Boundary", "", $validation.boundary)
Write-Host "OwnerRealProofStagingWorkspaceValidationState=$state FailedBlockers=$($failed.Count) ReadyForStrictImport=$ready"
if ($Strict.IsPresent -and $failed.Count -gt 0) { throw "Owner real proof staging workspace validation failed." }
if ($FailOnNotProof.IsPresent -and -not $ready) { throw "Owner real proof staging workspace is not ready for strict import." }
