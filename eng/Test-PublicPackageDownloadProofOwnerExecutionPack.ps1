[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\public-package-download-proof-owner-execution-pack.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Public package download proof owner execution pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$steps = @((Get-PropertyOrDefault -Object $record -Name "ownerSteps" -DefaultValue @()))
$stepIds = @($steps | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "stepId" -DefaultValue "") })
$manualCommands = @((Get-PropertyOrDefault -Object $record -Name "manualCommands" -DefaultValue @()) | ForEach-Object { [string]$_ })
$joinedCommands = $manualCommands -join "`n"

$requiredStepIds = @(
  "preflight-claim-safety",
  "confirm-public-publish-result-exists",
  "generate-public-download-input-template",
  "download-managed-package",
  "download-runtime-package",
  "hash-downloaded-packages",
  "fill-public-download-input",
  "validate-public-download-input",
  "import-public-download-candidate",
  "validate-public-download-candidate",
  "refresh-post-publish-verification",
  "refresh-release-evidence"
)
$missingStepIds = @($requiredStepIds | Where-Object { $stepIds -notcontains $_ })

$requiredCommandMarkers = @(
  "Test-PublicDocsAndPackageMetadataGate.ps1 -Strict",
  "Export-ReleaseDocsAndNuGetMetadataAudit.ps1",
  "Test-ReleaseDocsAndNuGetMetadataAudit.ps1 -Strict",
  "Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict",
  "Export-PublicPackageDownloadProofInputTemplate.ps1",
  "Invoke-WebRequest",
  "Get-FileHash -Algorithm SHA256",
  "Test-PublicPackageDownloadProofInput.ps1 -Strict",
  "Import-PublicPackageDownloadProofCandidate.ps1",
  "Test-PublicPackageDownloadProofCandidate.ps1 -Strict",
  "Export-PostPublishUserVerificationPack.ps1",
  "Test-PostPublishUserVerificationPack.ps1 -Strict",
  "Export-ReleaseEvidenceBundle.ps1",
  "Test-ReleaseEvidenceClassificationAudit.ps1 -Strict"
)
$missingCommandMarkers = @($requiredCommandMarkers | Where-Object { -not $joinedCommands.Contains($_, [StringComparison]::OrdinalIgnoreCase) })

$forbiddenCommandPatterns = @(
  "dotnet\s+nuget\s+push",
  "Push-NuGetPackages",
  "NUGET_API_KEY",
  "GITHUB_PACKAGES_TOKEN",
  "--api-key",
  "gh\s+workflow\s+run",
  "gh\s+release\s+create",
  "gh\s+release\s+upload"
)
$forbiddenCommandHits = @(
  foreach ($pattern in $forbiddenCommandPatterns) {
    if ($joinedCommands -match $pattern) {
      $pattern
    }
  }
)

$unsafeSteps = @($steps | Where-Object {
  [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "usesPublishToken" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "isPostPublishProof" -DefaultValue $true)
})
$blockedSteps = @($steps | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "public-package-download-proof-owner-execution-pack") "blocker" "recordKind must be public-package-download-proof-owner-execution-pack.")) | Out-Null
$items.Add((New-ValidationItem "required-owner-steps-present" ($missingStepIds.Count -eq 0) "blocker" ("Missing required owner steps: " + ($missingStepIds -join ", ")))) | Out-Null
$items.Add((New-ValidationItem "minimum-owner-step-count" ($steps.Count -ge 12 -and [int](Get-PropertyOrDefault -Object $record -Name "ownerStepCount" -DefaultValue 0) -eq $steps.Count) "blocker" "Execution pack must cover preflight, publish-result confirmation, downloads, hash capture, input validation, candidate import, post-publish pack refresh, and evidence refresh.")) | Out-Null
$items.Add((New-ValidationItem "manual-command-markers-present" ($missingCommandMarkers.Count -eq 0) "blocker" ("Missing required command markers: " + ($missingCommandMarkers -join ", ")))) | Out-Null
$items.Add((New-ValidationItem "manual-commands-no-publish-side-effects" ($forbiddenCommandHits.Count -eq 0) "blocker" ("Forbidden command markers: " + ($forbiddenCommandHits -join ", ")))) | Out-Null
$items.Add((New-ValidationItem "blocked-until-owner-download-proof" ([string](Get-PropertyOrDefault -Object $record -Name "packState" -DefaultValue "") -eq "blocked-public-package-download-proof-owner-execution-required" -and $blockedSteps.Count -ge 8) "blocker" "Execution pack must stay blocked until Owner supplies real public download proof.")) | Out-Null
$items.Add((New-ValidationItem "steps-safe" ($unsafeSteps.Count -eq 0) "blocker" "Every owner step must keep publish/proof/close flags false.")) | Out-Null
$items.Add((New-ValidationItem "pack-no-side-effects" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) "blocker" "Pack must not publish, use token, promote proof, claim post-publish proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($blockedSteps)
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-public-package-download-proof-owner-execution-pack"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-public-package-download-proof-owner-execution-required"
}
else {
  "public-package-download-proof-owner-execution-ready-for-owner-review"
}

$validation = [ordered]@{
  recordKind = "public-package-download-proof-owner-execution-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  packState = [string](Get-PropertyOrDefault -Object $record -Name "packState" -DefaultValue "")
  ownerStepCount = $steps.Count
  readyOwnerStepCount = @($steps | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
  blockedOwnerStepCount = $blockedSteps.Count
  manualCommandCount = $manualCommands.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  missingStepIds = $missingStepIds
  missingCommandMarkers = $missingCommandMarkers
  forbiddenCommandHits = $forbiddenCommandHits
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  canPromotePublicProof = $false
  canPromotePostPublishProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  validationItems = @($items.ToArray())
  safetyBoundary = "Public package download proof owner execution pack validation checks manual command coverage and non-publish boundaries only. It does not publish, download packages, run clean consumer smoke, promote proof, or close release issues."
}

$jsonPath = Join-Path $OutputRoot "public-package-download-proof-owner-execution-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "public-package-download-proof-owner-execution-pack-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Public Package Download Proof Owner Execution Pack Validation

Generated at: ``$($validation.generatedAtUtc)``

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| packState | ``$($validation.packState)`` |
| ownerStepCount | ``$($validation.ownerStepCount)`` |
| readyOwnerStepCount | ``$($validation.readyOwnerStepCount)`` |
| blockedOwnerStepCount | ``$($validation.blockedOwnerStepCount)`` |
| manualCommandCount | ``$($validation.manualCommandCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``False`` |
| usesPublishToken | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Boundary

$($validation.safetyBoundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Public package download proof owner execution pack validation written to $jsonPath"
Write-Output "ValidationState=$validationState Steps=$($validation.ownerStepCount) Blocked=$($validation.blockedOwnerStepCount) FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Public package download proof owner execution pack has blocker validation failures."
}
