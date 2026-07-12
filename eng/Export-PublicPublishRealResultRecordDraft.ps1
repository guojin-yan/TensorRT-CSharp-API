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
  param(
    [string]$Name,
    [string]$Description,
    [string[]]$ForbiddenSubstitutes = @()
  )

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
  New-OwnerField -Name "packageId" -Description "Package id observed from the public package channel." -ForbiddenSubstitutes @("ProjectReference", "local nupkg")
  New-OwnerField -Name "packageVersion" -Description "Package version observed from the public package channel." -ForbiddenSubstitutes @("local build version", "dry-run")
  New-OwnerField -Name "publicSource" -Description "NuGet/GitHub Packages/GitHub Release source used by the owner." -ForbiddenSubstitutes @("local feed", "direct nupkg")
  New-OwnerField -Name "publicPackageUrl" -Description "Stable URL for the published package." -ForbiddenSubstitutes @("local path", "artifact scan")
  New-OwnerField -Name "publicPackageSha256" -Description "SHA256 of package downloaded from the public source." -ForbiddenSubstitutes @("build output hash", "hash slot")
  New-OwnerField -Name "publishedAtUtc" -Description "UTC timestamp captured after public package availability." -ForbiddenSubstitutes @("template timestamp")
  New-OwnerField -Name "publishCommandTranscriptPath" -Description "Path to owner captured publish command transcript." -ForbiddenSubstitutes @("runbook", "dashboard")
  New-OwnerField -Name "publishCommandTranscriptSha256" -Description "SHA256 of owner captured publish command transcript." -ForbiddenSubstitutes @("missing log hash")
  New-OwnerField -Name "packageOwnerAccount" -Description "Human-owned account or organization used for the public publish." -ForbiddenSubstitutes @("automation placeholder")
  New-OwnerField -Name "reviewer" -Description "Human reviewer who checked the public publish result." -ForbiddenSubstitutes @("unreviewed local artifact")
  New-OwnerField -Name "rollbackPlanReviewed" -Description "Owner confirmation that rollback plan was reviewed after publish." -ForbiddenSubstitutes @("unchecked rollback")
  New-OwnerField -Name "forbiddenSubstituteScanResult" -Description "Result proving no local feed, ProjectReference, direct nupkg, template, dry-run, dashboard, audit pack, or local-only artifact scan was substituted." -ForbiddenSubstitutes @("local feed", "ProjectReference", "direct nupkg", "template", "dry-run", "dashboard", "audit pack", "local-only artifact scan")
)

$record = [pscustomobject]@{
  recordKind = "public-publish-real-result-record-draft"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  draftState = "blocked-public-publish-real-result-record-required"
  requiredFieldCount = $fields.Count
  blockedRequiredFieldCount = $fields.Count
  readyRequiredFieldCount = 0
  ownerFields = @($fields)
  sourceArtifacts = @(
    "artifacts/final-release/public-publish-real-result-owner-input-contract-validation.json",
    "artifacts/final-release/public-publish-final-owner-execution-pack-validation.json",
    "artifacts/final-release/public-publish-command-cross-check-validation.json"
  )
  expectedFollowUpValidators = @(
    "eng\Test-PublicPublishRealResultRecordDraft.ps1 -Strict",
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
  boundary = "This draft waits for owner-filled real public publish result fields. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-publish-real-result-record-draft.json"
$markdownPath = Join-Path $OutputRoot "public-publish-real-result-record-draft.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Public Publish Real Result Record Draft",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| draftState | ``$($record.draftState)`` |",
  "| requiredFieldCount | ``$($record.requiredFieldCount)`` |",
  "| blockedRequiredFieldCount | ``$($record.blockedRequiredFieldCount)`` |",
  "| performsPublish | ``$($record.performsPublish)`` |",
  "| canPublishPublicly | ``$($record.canPublishPublicly)`` |",
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

Write-Host "Public publish real result record draft written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "DraftState=$($record.draftState) RequiredFields=$($record.requiredFieldCount) Blocked=$($record.blockedRequiredFieldCount)"
