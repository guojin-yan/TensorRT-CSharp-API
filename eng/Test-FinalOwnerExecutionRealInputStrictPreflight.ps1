[CmdletBinding()]
param(
  [string]$CandidatePath = "artifacts\final-release\final-owner-execution-real-input-candidate.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict,
  [switch]$RequireExistingFiles,
  [switch]$RequireHashMatch,
  [switch]$FailOnNotReady,
  [switch]$RejectLocalFeed,
  [switch]$RejectProjectReference,
  [switch]$RejectDirectNupkg,
  [switch]$RejectPrePublishSmokeAsPostPublishProof
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("CandidatePath", "OutputRoot")) {
  if (-not [System.IO.Path]::IsPathRooted((Get-Variable $pathName).Value)) {
    Set-Variable -Name $pathName -Value (Join-Path $RepositoryRoot (Get-Variable $pathName).Value)
  }
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-Finding {
  param([string]$Id, [string]$Severity, [string]$Category, [string]$Message)
  [pscustomobject]@{ id = $Id; severity = $Severity; category = $Category; message = $Message; ownerActionRequired = $true }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

if (-not (Test-Path -LiteralPath $CandidatePath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Import-FinalOwnerExecutionRealInput.ps1") -RepositoryRoot $RepositoryRoot
}

$candidate = Get-Content -LiteralPath $CandidatePath -Raw -Encoding utf8 | ConvertFrom-Json
$fieldResults = @(Convert-ToArray (Get-PropertyOrDefault -Object $candidate -Name "fieldResults" -DefaultValue @()))
$nonSubstituteConfirmations = @(Convert-ToArray (Get-PropertyOrDefault -Object $candidate -Name "nonSubstituteConfirmations" -DefaultValue @()))
$findings = New-Object System.Collections.Generic.List[object]

$dualPackageRouteIds = @("nuget-small-bridge-core", "github-packages-full-runtime")
$dualPackageRouteProofResults = @($fieldResults | Where-Object {
    [string](Get-PropertyOrDefault -Object $_ -Name "fieldPath" -DefaultValue "") -like "dualPackageRoutes.*"
  })
$dualPackageRouteProofFieldIds = @($dualPackageRouteProofResults | ForEach-Object {
    [string](Get-PropertyOrDefault -Object $_ -Name "fieldId" -DefaultValue "")
  } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
$dualPackageRouteProofFieldPaths = @($dualPackageRouteProofResults | ForEach-Object {
    [string](Get-PropertyOrDefault -Object $_ -Name "fieldPath" -DefaultValue "")
  } | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
$dualPackageRouteProofReadyFieldCount = @($dualPackageRouteProofResults | Where-Object {
    [bool](Get-PropertyOrDefault -Object $_ -Name "readyForImport" -DefaultValue $false)
  }).Count
$dualPackageRouteProofPlaceholderFieldCount = @($dualPackageRouteProofResults | Where-Object {
    -not [bool](Get-PropertyOrDefault -Object $_ -Name "placeholderReplaced" -DefaultValue $false)
  }).Count

foreach ($result in $fieldResults) {
  $fieldId = [string](Get-PropertyOrDefault -Object $result -Name "fieldId" -DefaultValue "")
  $kind = [string](Get-PropertyOrDefault -Object $result -Name "kind" -DefaultValue "")
  if (-not [bool](Get-PropertyOrDefault -Object $result -Name "placeholderReplaced" -DefaultValue $false)) {
    $findings.Add((New-Finding -Id "$fieldId-placeholder" -Severity "action-required" -Category "placeholder" -Message "Required field still contains placeholder.")) | Out-Null
  }
  if ($kind -eq "sha256" -and -not [bool](Get-PropertyOrDefault -Object $result -Name "sha256FormatValid" -DefaultValue $false)) {
    $findings.Add((New-Finding -Id "$fieldId-sha256" -Severity "action-required" -Category "SHA256 invalid" -Message "SHA256 is missing or not 64 hex characters.")) | Out-Null
  }
  if ($kind -eq "path" -and ($RequireExistingFiles.IsPresent -or -not [bool](Get-PropertyOrDefault -Object $result -Name "pathExists" -DefaultValue $false))) {
    if (-not [bool](Get-PropertyOrDefault -Object $result -Name "pathExists" -DefaultValue $false)) {
      $findings.Add((New-Finding -Id "$fieldId-path" -Severity "action-required" -Category "path missing" -Message "Path evidence is missing or not verified.")) | Out-Null
    }
  }
  if ($RequireHashMatch.IsPresent -and $kind -eq "sha256" -and -not [bool](Get-PropertyOrDefault -Object $result -Name "hashMatchesFile" -DefaultValue $false)) {
    $findings.Add((New-Finding -Id "$fieldId-hash-match" -Severity "action-required" -Category "SHA256 mismatch" -Message "Hash match is required but not proven.")) | Out-Null
  }
}

$candidateText = $candidate | ConvertTo-Json -Depth 20
$rejects = @()
if ($RejectLocalFeed.IsPresent -or $true) { $rejects += "local feed" }
if ($RejectProjectReference.IsPresent -or $true) { $rejects += "ProjectReference" }
if ($RejectDirectNupkg.IsPresent -or $true) { $rejects += "direct .nupkg"; $rejects += "direct nupkg" }
if ($RejectPrePublishSmokeAsPostPublishProof.IsPresent -or $true) { $rejects += "pre-publish smoke reused as post-publish proof" }
foreach ($reject in ($rejects | Select-Object -Unique)) {
  $matchingConfirmations = @($nonSubstituteConfirmations | Where-Object {
      [string](Get-PropertyOrDefault -Object $_ -Name "marker" -DefaultValue "") -eq $reject
    })
  if ($matchingConfirmations.Count -eq 0) {
    $findings.Add((New-Finding -Id "reject-$($reject.Replace(' ', '-').Replace('.', 'dot'))" -Severity "action-required" -Category "forbidden substitute rejected" -Message "Forbidden substitute marker is missing required owner confirmation: $reject")) | Out-Null
    continue
  }

  foreach ($confirmation in $matchingConfirmations) {
    $confirmedAbsent = [bool](Get-PropertyOrDefault -Object $confirmation -Name "confirmedAbsent" -DefaultValue $false)
    if (-not $confirmedAbsent) {
      $findings.Add((New-Finding -Id "reject-$($reject.Replace(' ', '-').Replace('.', 'dot'))" -Severity "action-required" -Category "forbidden substitute rejected" -Message "Forbidden substitute marker is present or still not explicitly cleared: $reject")) | Out-Null
    }
  }
}

if (-not [bool](Get-PropertyOrDefault -Object $candidate -Name "readyForImport" -DefaultValue $false)) {
  $findings.Add((New-Finding -Id "candidate-not-ready" -Severity "action-required" -Category "not ready" -Message "Candidate is not ready for import.")) | Out-Null
}

$failedBlockers = @($findings | Where-Object { [string]$_.severity -eq "blocker" })
$failedActionRequired = @($findings | Where-Object { [string]$_.severity -eq "action-required" })
$readyForCloseValidation = $failedBlockers.Count -eq 0 -and $failedActionRequired.Count -eq 0
$preflightState = if ($readyForCloseValidation) { "ready-for-final-close-validation-candidate" } else { "blocked-final-owner-real-input-required" }

$preflight = [pscustomobject]@{
  recordKind = "final-owner-execution-real-input-strict-preflight"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  preflightState = $preflightState
  candidatePath = $CandidatePath
  checkedFieldCount = $fieldResults.Count
  nonSubstituteConfirmationCount = $nonSubstituteConfirmations.Count
  findingCount = $findings.Count
  failedBlockerCount = $failedBlockers.Count
  failedActionRequiredCount = $failedActionRequired.Count
  requireExistingFiles = $RequireExistingFiles.IsPresent
  requireHashMatch = $RequireHashMatch.IsPresent
  failOnNotReady = $FailOnNotReady.IsPresent
  dualPackageRouteIds = @($dualPackageRouteIds)
  dualPackageRouteProofRouteCount = $dualPackageRouteIds.Count
  dualPackageRouteProofFieldCount = $dualPackageRouteProofResults.Count
  dualPackageRouteProofReadyFieldCount = $dualPackageRouteProofReadyFieldCount
  dualPackageRouteProofPlaceholderFieldCount = $dualPackageRouteProofPlaceholderFieldCount
  dualPackageRouteProofFieldIds = @($dualPackageRouteProofFieldIds)
  dualPackageRouteProofFieldPaths = @($dualPackageRouteProofFieldPaths)
  dualPackageRouteProofReadyForCloseValidation = $dualPackageRouteProofReadyFieldCount -eq $dualPackageRouteProofResults.Count -and $dualPackageRouteProofResults.Count -gt 0
  readyForCloseValidation = $readyForCloseValidation
  ownerActionRequired = -not $readyForCloseValidation
  findings = @($findings.ToArray())
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner execution real input strict preflight validates owner input candidate only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-real-input-strict-preflight.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-real-input-strict-preflight.md"
$preflight | ConvertTo-Json -Depth 14 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = foreach ($finding in $findings) {
  "| ``$(ConvertTo-MarkdownCell $finding.id)`` | ``$(ConvertTo-MarkdownCell $finding.category)`` | $(ConvertTo-MarkdownCell $finding.message) |"
}

$markdown = @"
# Final Owner Execution Real Input Strict Preflight

| Field | Value |
|---|---|
| preflightState | ``$($preflight.preflightState)`` |
| checkedFieldCount | ``$($preflight.checkedFieldCount)`` |
| findingCount | ``$($preflight.findingCount)`` |
| failedBlockerCount | ``$($preflight.failedBlockerCount)`` |
| failedActionRequiredCount | ``$($preflight.failedActionRequiredCount)`` |
| dualPackageRouteProofRouteCount | ``$($preflight.dualPackageRouteProofRouteCount)`` |
| dualPackageRouteProofFieldCount | ``$($preflight.dualPackageRouteProofFieldCount)`` |
| dualPackageRouteProofReadyFieldCount | ``$($preflight.dualPackageRouteProofReadyFieldCount)`` |
| dualPackageRouteProofPlaceholderFieldCount | ``$($preflight.dualPackageRouteProofPlaceholderFieldCount)`` |
| readyForCloseValidation | ``$($preflight.readyForCloseValidation)`` |

## Dual-Package Route Proof

| Field Path |
|---|
$(($dualPackageRouteProofFieldPaths | ForEach-Object { "| ``$(ConvertTo-MarkdownCell $_)`` |" }) -join "`r`n")

## Findings

| ID | Category | Message |
|---|---|---|
$($rows -join "`r`n")

## Boundary

$($preflight.boundary)
"@
Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown

Write-Host "Final owner execution real input strict preflight written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "PreflightState=$preflightState ActionRequired=$($failedActionRequired.Count) Ready=$readyForCloseValidation"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  throw "Final owner execution real input strict preflight failed with $($failedBlockers.Count) blocker(s)."
}
if ($FailOnNotReady.IsPresent -and -not $readyForCloseValidation) {
  throw "Final owner execution real input strict preflight is not ready."
}
