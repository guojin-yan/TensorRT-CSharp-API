[CmdletBinding()]
param(
  [string]$DashboardPath = ".\artifacts\final-release\release-rc-proof-dashboard.json",
  [switch]$FailOnReady,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
}

$dashboardFullPath = if ([System.IO.Path]::IsPathRooted($DashboardPath)) { $DashboardPath } else { Join-Path $RepositoryRoot $DashboardPath }
if (-not (Test-Path -LiteralPath $dashboardFullPath -PathType Leaf)) {
  & (Join-Path $PSScriptRoot "Export-ReleaseRcProofDashboard.ps1") -RepositoryRoot $RepositoryRoot | Out-Null
}

$dashboard = Get-Content -LiteralPath $dashboardFullPath -Raw -Encoding utf8 | ConvertFrom-Json
$requiredIds = @(
  "owner-authorization",
  "package-consumer-runtime",
  "callback-runtime-proof",
  "linux-runner-proof",
  "real-model-runtime",
  "post-publish-verification",
  "stale-release-claims",
  "release-close-preflight"
)

$requiredNonSubstitutes = @(
  "managed-readiness",
  "CallbackAllocatorReadinessSnapshot",
  "precheck-only",
  "dry-run-only",
  "schema-only"
)

$ids = @($dashboard.proofBlockers | ForEach-Object { $_.id })
$missingIds = @($requiredIds | Where-Object { $ids -notcontains $_ })
$readyItems = @($dashboard.proofBlockers | Where-Object { [bool]$_.ready })
$missingFields = New-Object System.Collections.Generic.List[string]

foreach ($item in @($dashboard.proofBlockers)) {
  foreach ($property in @("requiredValidator", "requiredOwnerInputs", "sourceArtifacts", "nextCommand", "cannotUse", "nonSubstituteProofKinds")) {
    if ($item.PSObject.Properties.Name -notcontains $property) {
      $missingFields.Add("$($item.id).$property")
    }
  }
}

$nonSubstitutes = @($dashboard.nonSubstituteProofKinds)
$missingNonSubstitutes = @($requiredNonSubstitutes | Where-Object { $nonSubstitutes -notcontains $_ })
$sourceArtifacts = @($dashboard.sourceArtifacts)
$ownerProofInputArtifactsPresent =
  ($sourceArtifacts -contains "artifacts/final-release/release-owner-proof-input-record-template.json") -and
  ($sourceArtifacts -contains "artifacts/final-release/release-owner-proof-input-record-validation.json")

$validation = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "release-rc-proof-dashboard-validation"
  sourceDashboardPath = $dashboardFullPath
  validationState = if ($missingIds.Count -eq 0 -and $missingFields.Count -eq 0 -and $missingNonSubstitutes.Count -eq 0 -and $readyItems.Count -eq 0 -and $ownerProofInputArtifactsPresent) { "blocked-valid-owner-handoff" } else { "blocked-dashboard-repair-required" }
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  readyProofBlockerCount = $readyItems.Count
  missingRequiredIdCount = $missingIds.Count
  missingRequiredIds = @($missingIds)
  missingFieldCount = $missingFields.Count
  missingFields = @($missingFields.ToArray())
  missingNonSubstituteCount = $missingNonSubstitutes.Count
  missingNonSubstitutes = @($missingNonSubstitutes)
  ownerProofInputArtifactsPresent = $ownerProofInputArtifactsPresent
  boundary = "Dashboard validation checks owner handoff completeness only; it does not promote proof or publication readiness."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null
$jsonPath = Join-Path $artifactRoot "release-rc-proof-dashboard-validation.json"
$markdownPath = Join-Path $artifactRoot "release-rc-proof-dashboard-validation.md"

$validation | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Release RC Proof Dashboard Validation

- validation state: ``$($validation.validationState)``
- canPublishPublicly=false
- canCloseReleaseIssue=false
- ready proof blocker count: ``$($validation.readyProofBlockerCount)``
- missing required id count: ``$($validation.missingRequiredIdCount)``
- missing field count: ``$($validation.missingFieldCount)``
- missing non-substitute count: ``$($validation.missingNonSubstituteCount)``
- owner proof input artifacts present: ``$ownerProofInputArtifactsPresent``

## Boundary

$($validation.boundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Release RC proof dashboard validation written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "ValidationState=$($validation.validationState)"
Write-Output "CanPublishPublicly=False"

if ($FailOnReady -and $readyItems.Count -gt 0) {
  throw "Release RC proof dashboard unexpectedly contains ready proof blockers."
}
