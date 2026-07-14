[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\post-publish-user-verification-pack.json",
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
  throw "Post-publish user verification pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$lanes = @((Get-PropertyOrDefault -Object $record -Name "lanes" -DefaultValue @()))
$laneIds = @($lanes | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "") })
$ownerInputFields = @((Get-PropertyOrDefault -Object $record -Name "ownerInputFields" -DefaultValue @()))
$ownerInputFieldNames = @($ownerInputFields | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "fieldName" -DefaultValue "") })
$requiredLaneIds = @(
  "release-docs-and-nuget-metadata-audit",
  "public-docs-package-metadata-gate",
  "pre-release-package-proof-readiness-matrix",
  "owner-publish-authorization",
  "owner-publish-execution-result",
  "public-package-download-proof",
  "clean-external-consumer-smoke",
  "post-publish-clean-consumer-proof",
  "final-post-publish-audit-pack"
)
$missingLaneIds = @($requiredLaneIds | Where-Object { $laneIds -notcontains $_ })
$requiredOwnerInputFields = @(
  "publicPackageRestoreLogPath",
  "publicPackageRestoreLogSha256",
  "dotnetRestoreTranscriptPath",
  "dotnetRestoreTranscriptSha256",
  "dotnetBuildTranscriptPath",
  "dotnetBuildTranscriptSha256",
  "dotnetTestTranscriptPath",
  "dotnetTestTranscriptSha256",
  "hostMetadataPath",
  "cudaRuntimeMetadata",
  "tensorRtRuntimeMetadata",
  "packageIdentity",
  "packageUrl",
  "packageSha256",
  "cleanWorkspaceProofPath",
  "resultValidationPath",
  "resultValidationSha256",
  "ownerReviewer",
  "ownerReviewedAtUtc"
)
$missingOwnerInputFields = @($requiredOwnerInputFields | Where-Object { $ownerInputFieldNames -notcontains $_ })
$unsafeOwnerInputFields = @($ownerInputFields | Where-Object {
  [string](Get-PropertyOrDefault -Object $_ -Name "state" -DefaultValue "") -ne "blocked-post-publish-real-input-required" -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "acceptsPlaceholder" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "acceptsLocalFeed" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "acceptsProjectReference" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "acceptsDirectNupkg" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "acceptsDryRun" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "acceptsTemplate" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "acceptsDashboard" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "isPostPublishProof" -DefaultValue $true)
})
$unsafeLanes = @($lanes | Where-Object {
  [bool](Get-PropertyOrDefault -Object $_ -Name "performsPublish" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "usesPublishToken" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPublishPublicly" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canCloseReleaseIssue" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteRuntimeProof" -DefaultValue $true) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "isPostPublishProof" -DefaultValue $true)
})
$docsLane = @($lanes | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "") -eq "release-docs-and-nuget-metadata-audit" } | Select-Object -First 1)
$blockedProofLaneIds = @($lanes |
  Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "") -in @("public-package-download-proof", "clean-external-consumer-smoke", "post-publish-clean-consumer-proof", "final-post-publish-audit-pack") -and -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) } |
  ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "laneId" -DefaultValue "") })

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem "record-kind" ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "post-publish-user-verification-pack") "blocker" "recordKind must be post-publish-user-verification-pack.")) | Out-Null
$items.Add((New-ValidationItem "required-lanes-present" ($missingLaneIds.Count -eq 0) "blocker" ("Missing required lanes: " + ($missingLaneIds -join ", ")))) | Out-Null
$items.Add((New-ValidationItem "minimum-lane-count" ($lanes.Count -ge 9 -and [int](Get-PropertyOrDefault -Object $record -Name "verificationLaneCount" -DefaultValue 0) -eq $lanes.Count) "blocker" "Pack must include the docs, pre-release, owner publish, public download, clean consumer, post-publish, and final audit lanes.")) | Out-Null
$items.Add((New-ValidationItem "required-owner-input-fields-present" ($missingOwnerInputFields.Count -eq 0) "blocker" ("Missing required owner input fields: " + ($missingOwnerInputFields -join ", ")))) | Out-Null
$items.Add((New-ValidationItem "owner-input-field-counts" ($ownerInputFields.Count -ge 19 -and [int](Get-PropertyOrDefault -Object $record -Name "requiredOwnerFieldCount" -DefaultValue 0) -eq $ownerInputFields.Count -and [int](Get-PropertyOrDefault -Object $record -Name "blockedRequiredOwnerFieldCount" -DefaultValue 0) -eq $ownerInputFields.Count) "blocker" "Pack must expose public restore/build/test transcripts, host/runtime metadata, package URL/hash, clean workspace, validation hash, and owner review fields as blocked real-owner inputs.")) | Out-Null
$items.Add((New-ValidationItem "owner-input-fields-safe" ($unsafeOwnerInputFields.Count -eq 0) "blocker" "Owner input fields must reject local/template/dashboard substitutes and keep publish/proof/close flags false.")) | Out-Null
$items.Add((New-ValidationItem "blocked-until-real-user-proof" ([string](Get-PropertyOrDefault -Object $record -Name "packState" -DefaultValue "") -eq "blocked-post-publish-user-verification-required") "blocker" "Current pack must stay blocked until real public download, clean consumer, and post-publish evidence exists.")) | Out-Null
$items.Add((New-ValidationItem "docs-audit-lane-ready" ($docsLane.Count -eq 1 -and [bool](Get-PropertyOrDefault -Object $docsLane[0] -Name "ready" -DefaultValue $false)) "blocker" "Docs/NuGet metadata audit lane should be ready-non-proof before user verification handoff.")) | Out-Null
$items.Add((New-ValidationItem "real-proof-lanes-still-blocked" ($blockedProofLaneIds.Count -ge 4) "blocker" "Public download, clean external consumer, post-publish proof, and final post-publish audit lanes must stay blocked until real owner evidence exists.")) | Out-Null
$items.Add((New-ValidationItem "lanes-safe" ($unsafeLanes.Count -eq 0) "blocker" "Every lane must keep publish/proof/close flags false.")) | Out-Null
$items.Add((New-ValidationItem "pack-no-side-effects" (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "usesPublishToken" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "isPostPublishProof" -DefaultValue $true)) "blocker" "Pack must not publish, use token, promote proof, claim post-publish proof, or close release issue.")) | Out-Null

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedActionRequired = @($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) })
$validationState = if ($failedBlockers.Count -gt 0) {
  "invalid-post-publish-user-verification-pack"
}
elseif ($failedActionRequired.Count -gt 0) {
  "blocked-post-publish-user-verification-required"
}
else {
  "post-publish-user-verification-ready-for-owner-close-review"
}

$validation = [ordered]@{
  recordKind = "post-publish-user-verification-pack-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputPath = $resolvedInputPath
  validationState = $validationState
  packState = [string](Get-PropertyOrDefault -Object $record -Name "packState" -DefaultValue "")
  verificationLaneCount = $lanes.Count
  readyVerificationLaneCount = @($lanes | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
  blockedVerificationLaneCount = @($lanes | Where-Object { -not [bool](Get-PropertyOrDefault -Object $_ -Name "ready" -DefaultValue $false) }).Count
  requiredOwnerFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "requiredOwnerFieldCount" -DefaultValue 0)
  blockedRequiredOwnerFieldCount = [int](Get-PropertyOrDefault -Object $record -Name "blockedRequiredOwnerFieldCount" -DefaultValue 0)
  rejectedSubstituteCount = [int](Get-PropertyOrDefault -Object $record -Name "rejectedSubstituteCount" -DefaultValue 0)
  sourceReadinessSignalCount = [int](Get-PropertyOrDefault -Object $record -Name "sourceReadinessSignalCount" -DefaultValue 0)
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
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
  safetyBoundary = "Post-publish user verification pack validation checks aggregation shape only. It does not publish, download public packages, run clean consumer smoke, promote proof, or close release issues."
}

$jsonPath = Join-Path $OutputRoot "post-publish-user-verification-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "post-publish-user-verification-pack-validation.md"
$validation | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = @"
# Post-Publish User Verification Pack Validation

Generated at: ``$($validation.generatedAtUtc)``

| Item | Value |
|---|---|
| validationState | ``$($validation.validationState)`` |
| packState | ``$($validation.packState)`` |
| verificationLaneCount | ``$($validation.verificationLaneCount)`` |
| readyVerificationLaneCount | ``$($validation.readyVerificationLaneCount)`` |
| blockedVerificationLaneCount | ``$($validation.blockedVerificationLaneCount)`` |
| requiredOwnerFieldCount | ``$($validation.requiredOwnerFieldCount)`` |
| blockedRequiredOwnerFieldCount | ``$($validation.blockedRequiredOwnerFieldCount)`` |
| rejectedSubstituteCount | ``$($validation.rejectedSubstituteCount)`` |
| sourceReadinessSignalCount | ``$($validation.sourceReadinessSignalCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($validation.failedActionRequiredCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |

## Boundary

$($validation.safetyBoundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Post-publish user verification pack validation written to $jsonPath"
Write-Output "ValidationState=$validationState Lanes=$($validation.verificationLaneCount) Blocked=$($validation.blockedVerificationLaneCount) FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Post-publish user verification pack has blocker validation failures."
}
