[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-proof-input-draft-pack.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$Strict,
  [switch]$RequireAllProofLines,
  [switch]$FailOnPromotedProof
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) {
    $scriptRoot = (Get-Location).Path
  }
  else {
    $scriptRoot = $PSScriptRoot
  }

  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
  }
}

$resolvedInputPath = if ([System.IO.Path]::IsPathRooted($InputPath)) {
  $InputPath
}
else {
  Join-Path $RepositoryRoot $InputPath
}

if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner proof input draft pack not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]
$requiredLineIds = @(
  "owner-authorization",
  "package-consumer-runtime",
  "linux-runner-proof",
  "real-model-runtime",
  "post-publish-verification",
  "release-issue-close-record"
)

$recordKind = [string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "")
$draftPackState = [string](Get-PropertyOrDefault -Object $record -Name "draftPackState" -DefaultValue "")
$draftSpecs = @(Get-PropertyOrDefault -Object $record -Name "draftSpecs" -DefaultValue @())
$performsPublish = [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $false)
$canPublishPublicly = [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $false)
$canCloseReleaseIssue = [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $false)

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-proof-input-draft-pack") -Severity "blocker" -Detail "recordKind must be owner-proof-input-draft-pack.")) | Out-Null
$items.Add((New-ValidationItem -Id "draft-pack-state" -Passed ($draftPackState -eq "blocked-draft-non-proof") -Severity "blocker" -Detail "draftPackState must remain blocked-draft-non-proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "Draft pack must not publish, approve public publication, or close the release issue.")) | Out-Null

if ($Strict -or $RequireAllProofLines) {
  $ids = @($draftSpecs | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") })
  foreach ($id in $requiredLineIds) {
    $items.Add((New-ValidationItem -Id "draft-spec-$id-present" -Passed ($ids -contains $id) -Severity "blocker" -Detail "Draft spec '$id' must exist.")) | Out-Null
  }

  $items.Add((New-ValidationItem -Id "draft-spec-count" -Passed ($draftSpecs.Count -eq 6) -Severity "blocker" -Detail "Strict draft pack validation expects exactly six draft specs.")) | Out-Null
}

foreach ($spec in $draftSpecs) {
  $id = [string](Get-PropertyOrDefault -Object $spec -Name "id" -DefaultValue "")
  $inputDraftIsProof = [bool](Get-PropertyOrDefault -Object $spec -Name "inputDraftIsProof" -DefaultValue $true)
  $canPromoteProof = [bool](Get-PropertyOrDefault -Object $spec -Name "canPromoteProof" -DefaultValue $true)
  $strictValidationCommand = [string](Get-PropertyOrDefault -Object $spec -Name "strictValidationCommand" -DefaultValue "")
  $inputDraftPath = [string](Get-PropertyOrDefault -Object $spec -Name "inputDraftPath" -DefaultValue "")
  $markers = Convert-ToStringArray (Get-PropertyOrDefault -Object $spec -Name "blockedByNonProofMarkers" -DefaultValue @())

  $items.Add((New-ValidationItem -Id "draft-$id-non-proof" -Passed (-not $inputDraftIsProof -and -not $canPromoteProof) -Severity "blocker" -Detail "Draft spec '$id' must remain non-proof and non-promotable.")) | Out-Null
  $items.Add((New-ValidationItem -Id "draft-$id-path" -Passed (-not [string]::IsNullOrWhiteSpace($inputDraftPath)) -Severity "blocker" -Detail "Draft spec '$id' must expose inputDraftPath.")) | Out-Null
  $items.Add((New-ValidationItem -Id "draft-$id-strict-validator" -Passed (-not [string]::IsNullOrWhiteSpace($strictValidationCommand)) -Severity "blocker" -Detail "Draft spec '$id' must expose strictValidationCommand.")) | Out-Null

  foreach ($marker in @("template", "draft", "ProjectReference", "local feed", "direct .nupkg reference", "release-issue-close-record-template.json")) {
    $items.Add((New-ValidationItem -Id "draft-$id-blocks-$($marker -replace '[^A-Za-z0-9]+','-')" -Passed ($markers -contains $marker) -Severity "blocker" -Detail "Draft spec '$id' must block '$marker' as proof substitute.")) | Out-Null
  }
}

$packageConsumer = $draftSpecs | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "package-consumer-runtime" } | Select-Object -First 1
if ($null -ne $packageConsumer) {
  $rules = Convert-ToStringArray (Get-PropertyOrDefault -Object $packageConsumer -Name "requiredRealInputRules" -DefaultValue @())
  foreach ($rule in @("cleanExternalConsumerIdentity", "noProjectReference", "noLocalFeedAsPublicProof", "managedNupkgSha256", "runtimeNupkgSha256", "runtimePackageKeyMatches", "compatibleHostMetadata", "smokeCommandIncludesRuntimePackageKey", "smokeLogPath", "smokeLogSha256")) {
    $items.Add((New-ValidationItem -Id "package-consumer-rule-$rule" -Passed ($rules -contains $rule) -Severity "blocker" -Detail "package-consumer-runtime must require '$rule'.")) | Out-Null
  }
}

$closeRecord = $draftSpecs | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "id" -DefaultValue "") -eq "release-issue-close-record" } | Select-Object -First 1
if ($null -ne $closeRecord) {
  $rules = Convert-ToStringArray (Get-PropertyOrDefault -Object $closeRecord -Name "requiredRealInputRules" -DefaultValue @())
  foreach ($rule in @("releaseEvidenceBundleSha256", "releaseClosePreflightPathAndHash", "staleClaimsAuditPathAndHash", "postPublishProofValidationPathAndHash", "rollbackPlan", "ownerFinalCloseDecision", "strictCloseValidatorCommand")) {
    $items.Add((New-ValidationItem -Id "close-record-rule-$rule" -Passed ($rules -contains $rule) -Severity "blocker" -Detail "release-issue-close-record must require '$rule'.")) | Out-Null
  }
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$promotedProofItems = @($draftSpecs | Where-Object {
  [bool](Get-PropertyOrDefault -Object $_ -Name "inputDraftIsProof" -DefaultValue $false) -or
  [bool](Get-PropertyOrDefault -Object $_ -Name "canPromoteProof" -DefaultValue $false)
})

$validationState = if ($failedBlockers.Count -eq 0 -and $promotedProofItems.Count -eq 0) {
  "valid-draft-non-proof"
}
else {
  "invalid-draft-pack"
}

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  recordKind = "owner-proof-input-draft-pack-validation"
  inputPath = $InputPath
  validationState = $validationState
  isValidDraftPack = ($validationState -eq "valid-draft-non-proof")
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  strict = [bool]$Strict
  requireAllProofLines = [bool]$RequireAllProofLines
  draftSpecCount = $draftSpecs.Count
  failedBlockerCount = $failedBlockers.Count
  promotedProofItemCount = $promotedProofItems.Count
  validationItems = @($items.ToArray())
  boundary = "This validator checks draft-pack shape and proof-substitute blockers only. It does not create proof, publish packages, approve public publication, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-proof-input-draft-pack-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-proof-input-draft-pack-validation.md"

$summary | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Proof Input Draft Pack Validation")
$lines.Add("")
$lines.Add("- generated: ``$($summary.generatedAtUtc)``")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- valid draft pack: ``$($summary.isValidDraftPack)``")
$lines.Add("- draft spec count: ``$($summary.draftSpecCount)``")
$lines.Add("- failed blocker count: ``$($summary.failedBlockerCount)``")
$lines.Add("- promoted proof item count: ``$($summary.promotedProofItemCount)``")
$lines.Add("- performs publish: ``False``")
$lines.Add("- can publish publicly: ``False``")
$lines.Add("- can close release issue: ``False``")
$lines.Add("")
$lines.Add("## Validation Items")
$lines.Add("")
$lines.Add("| ID | Passed | Severity | Detail |")
$lines.Add("|---|---:|---|---|")
foreach ($item in $items) {
  $detail = ([string]$item.detail).Replace("|", "\|")
  $lines.Add("| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $detail |")
}
$lines.Add("")
$lines.Add($summary.boundary)

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner proof input draft pack validation written to $jsonPath"
Write-Host "Owner proof input draft pack validation written to $markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) PromotedProofItems=$($promotedProofItems.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($FailOnPromotedProof -and $promotedProofItems.Count -gt 0) {
  Write-Error "Owner proof input draft pack contains promoted proof markers."
  exit 1
}

if ($Strict -and $validationState -ne "valid-draft-non-proof") {
  Write-Error "Owner proof input draft pack strict validation failed. ValidationState=$validationState"
  exit 1
}
