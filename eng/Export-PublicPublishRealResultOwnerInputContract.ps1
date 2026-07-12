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

function New-RequiredField {
  param(
    [string]$Name,
    [string]$Description,
    [string[]]$ForbiddenSubstitutes = @()
  )

  [pscustomobject]@{
    name = $Name
    description = $Description
    required = $true
    valueState = "owner-input-required"
    ready = $false
    forbiddenSubstitutes = @($ForbiddenSubstitutes)
  }
}

$requiredFields = @(
  New-RequiredField -Name "packageId" -Description "Published managed package id from the public channel." -ForbiddenSubstitutes @("local nupkg", "ProjectReference")
  New-RequiredField -Name "packageVersion" -Description "Published managed package version from the public channel." -ForbiddenSubstitutes @("local version only", "dry-run")
  New-RequiredField -Name "publicSource" -Description "NuGet/GitHub Packages/GitHub Release public source used by the owner." -ForbiddenSubstitutes @("local feed", "direct nupkg")
  New-RequiredField -Name "publicPackageUrl" -Description "Stable public package URL after owner publish." -ForbiddenSubstitutes @("local path", "artifact scan")
  New-RequiredField -Name "publicPackageSha256" -Description "SHA256 of the downloaded public package." -ForbiddenSubstitutes @("build output hash", "hash slot")
  New-RequiredField -Name "publishedAtUtc" -Description "UTC timestamp observed after the owner publish." -ForbiddenSubstitutes @("template timestamp")
  New-RequiredField -Name "publishCommandTranscriptPath" -Description "Path to the owner captured publish command transcript." -ForbiddenSubstitutes @("runbook", "dashboard")
  New-RequiredField -Name "publishCommandTranscriptSha256" -Description "SHA256 of the owner captured publish command transcript." -ForbiddenSubstitutes @("missing log hash")
  New-RequiredField -Name "packageOwnerAccount" -Description "Owner account or organization used for the public publish." -ForbiddenSubstitutes @("automation placeholder")
  New-RequiredField -Name "reviewer" -Description "Human reviewer of the real public publish result." -ForbiddenSubstitutes @("unreviewed local artifact")
  New-RequiredField -Name "rollbackPlanReviewed" -Description "Owner confirmation that rollback plan was reviewed after publish." -ForbiddenSubstitutes @("unchecked rollback")
  New-RequiredField -Name "forbiddenSubstituteScanResult" -Description "Result proving no local feed, ProjectReference, or direct nupkg substitute was used." -ForbiddenSubstitutes @("local feed", "ProjectReference", "direct nupkg")
)

$record = [pscustomobject]@{
  recordKind = "public-publish-real-result-owner-input-contract"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  contractState = "blocked-public-publish-real-result-owner-input-required"
  requiredFieldCount = $requiredFields.Count
  blockedRequiredFieldCount = $requiredFields.Count
  readyRequiredFieldCount = 0
  requiredFields = @($requiredFields)
  sourceArtifacts = @(
    "artifacts/final-release/public-publish-final-owner-execution-pack-validation.json",
    "artifacts/final-release/public-publish-command-cross-check-validation.json",
    "artifacts/final-release/public-publish-result-owner-input-validation.json",
    "artifacts/final-release/release-evidence-classification-audit.json"
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
  boundary = "This contract waits for real owner public publish result input. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "public-publish-real-result-owner-input-contract.json"
$markdownPath = Join-Path $OutputRoot "public-publish-real-result-owner-input-contract.md"

$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @(
  "# Public Publish Real Result Owner Input Contract",
  "",
  "| Field | Value |",
  "| --- | --- |",
  "| contractState | ``$($record.contractState)`` |",
  "| requiredFieldCount | ``$($record.requiredFieldCount)`` |",
  "| blockedRequiredFieldCount | ``$($record.blockedRequiredFieldCount)`` |",
  "| performsPublish | ``$($record.performsPublish)`` |",
  "| canPublishPublicly | ``$($record.canPublishPublicly)`` |",
  "| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |",
  "",
  "## Required Fields",
  "",
  "| Field | State | Description |",
  "| --- | --- | --- |"
)

foreach ($field in $requiredFields) {
  $markdown += "| $($field.name) | ``$($field.valueState)`` | $($field.description) |"
}

$markdown += @(
  "",
  "## Boundary",
  "",
  $record.boundary
)

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Public publish real result owner input contract written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ContractState=$($record.contractState) RequiredFields=$($record.requiredFieldCount) Blocked=$($record.blockedRequiredFieldCount)"
