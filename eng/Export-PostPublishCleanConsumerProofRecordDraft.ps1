[CmdletBinding()]
param(
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

function New-OwnerField {
  param([string]$Name, [string]$Description, [string[]]$ForbiddenSubstitutes = @())
  [pscustomobject]@{
    name = $Name
    description = $Description
    value = $null
    required = $true
    ready = $false
    valueState = "owner-real-input-required"
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
  }
}

$fields = @(
  New-OwnerField -Name "cleanConsumerProjectPath" -Description "Repository-external clean consumer project path or archive." -ForbiddenSubstitutes @("repo sample", "ProjectReference")
  New-OwnerField -Name "packageSourceUrl" -Description "Public package source used by clean consumer restore." -ForbiddenSubstitutes @("local feed")
  New-OwnerField -Name "restoredPackageId" -Description "Package id resolved by clean consumer restore." -ForbiddenSubstitutes @("ProjectReference")
  New-OwnerField -Name "restoredPackageVersion" -Description "Package version resolved by clean consumer restore." -ForbiddenSubstitutes @("local build version")
  New-OwnerField -Name "restoredPackageSha256" -Description "SHA256 of restored package from public source." -ForbiddenSubstitutes @("direct nupkg", "build output hash")
  New-OwnerField -Name "restoreCommand" -Description "Owner captured restore command." -ForbiddenSubstitutes @("runbook")
  New-OwnerField -Name "buildCommand" -Description "Owner captured build command." -ForbiddenSubstitutes @("dry-run")
  New-OwnerField -Name "smokeCommand" -Description "Owner captured runtime smoke command with runtime package key." -ForbiddenSubstitutes @("build-only")
  New-OwnerField -Name "smokeExitCode" -Description "Exit code of clean consumer runtime smoke." -ForbiddenSubstitutes @("skipped", "not run")
  New-OwnerField -Name "stdoutPath" -Description "Path to smoke stdout log." -ForbiddenSubstitutes @("summary-only")
  New-OwnerField -Name "stdoutSha256" -Description "SHA256 of smoke stdout log." -ForbiddenSubstitutes @("missing log hash")
  New-OwnerField -Name "stderrPath" -Description "Path to smoke stderr log." -ForbiddenSubstitutes @("summary-only")
  New-OwnerField -Name "stderrSha256" -Description "SHA256 of smoke stderr log." -ForbiddenSubstitutes @("missing log hash")
  New-OwnerField -Name "hostOs" -Description "Clean consumer host operating system." -ForbiddenSubstitutes @("unknown host")
  New-OwnerField -Name "cudaDriverVersion" -Description "CUDA driver version on clean consumer host." -ForbiddenSubstitutes @("dependency probe only")
  New-OwnerField -Name "runtimePackageKey" -Description "Runtime package key used by smoke command." -ForbiddenSubstitutes @("placeholder key")
  New-OwnerField -Name "noProjectReference" -Description "Owner confirmation that clean consumer uses no ProjectReference." -ForbiddenSubstitutes @("ProjectReference")
  New-OwnerField -Name "noLocalFeed" -Description "Owner confirmation that clean consumer uses no local package feed." -ForbiddenSubstitutes @("local feed")
  New-OwnerField -Name "noDirectNupkg" -Description "Owner confirmation that clean consumer uses no direct nupkg path." -ForbiddenSubstitutes @("direct nupkg")
)

$record = [pscustomobject]@{
  recordKind = "post-publish-clean-consumer-proof-record-draft"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  draftState = "blocked-post-publish-clean-consumer-proof-record-required"
  requiredFieldCount = $fields.Count
  blockedRequiredFieldCount = $fields.Count
  readyRequiredFieldCount = 0
  ownerFields = @($fields)
  sourceArtifacts = @(
    "artifacts/final-release/post-publish-clean-consumer-proof-record-contract-validation.json",
    "artifacts/final-release/public-publish-real-result-record-draft-validation.json",
    "artifacts/final-release/post-publish-clean-consumer-result-convergence-validation.json"
  )
  expectedFollowUpValidators = @(
    "eng\Test-PostPublishCleanConsumerProofRecordDraft.ps1 -Strict",
    "eng\Export-PublicPublishForbiddenSubstituteScan.ps1",
    "eng\Test-PublicPublishForbiddenSubstituteScan.ps1 -Strict",
    "eng\Export-ReleaseCloseRealProofImportBridge.ps1",
    "eng\Test-ReleaseCloseRealProofImportBridge.ps1 -Strict"
  )
  notExecutedByAutomation = $true
  ownerExecutionOnly = $true
  performsPublish = $false
  approvesPublicRelease = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  isPostPublishProof = $false
  boundary = "This draft waits for owner-filled repository-external clean consumer proof fields. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-record-draft.json"
$markdownPath = Join-Path $OutputRoot "post-publish-clean-consumer-proof-record-draft.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Post-Publish Clean Consumer Proof Record Draft",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| draftState | ``$($record.draftState)`` |",
  "| requiredFieldCount | ``$($record.requiredFieldCount)`` |",
  "| blockedRequiredFieldCount | ``$($record.blockedRequiredFieldCount)`` |",
  "| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |",
  "",
  "## Owner Fields",
  "",
  "| Field | State | Description |",
  "| --- | --- | --- |"
)

foreach ($field in $fields) {
  $markdown += "| $($field.name) | ``$($field.valueState)`` | $($field.description) |"
}

$markdown += @("", "## Boundary", "", $record.boundary)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Post-publish clean consumer proof record draft written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "DraftState=$($record.draftState) RequiredFields=$($record.requiredFieldCount) Blocked=$($record.blockedRequiredFieldCount)"
