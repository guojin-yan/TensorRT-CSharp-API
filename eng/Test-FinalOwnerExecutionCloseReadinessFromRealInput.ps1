[CmdletBinding()]
param(
  [string]$StrictPreflightPath = "artifacts\final-release\final-owner-execution-real-input-strict-preflight.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot,
  [switch]$Strict
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

foreach ($pathName in @("StrictPreflightPath", "OutputRoot")) {
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

  $directory = Split-Path -Parent $LiteralPath
  if ([string]::IsNullOrWhiteSpace($directory)) {
    $directory = "."
  }
  New-Item -ItemType Directory -Path $directory -Force | Out-Null

  $lines = New-Object System.Collections.Generic.List[string]
  foreach ($item in @($InputObject)) {
    if ($null -eq $item) {
      $lines.Add("") | Out-Null
    }
    elseif ($item -is [string]) {
      $lines.Add($item) | Out-Null
    }
    elseif ($item -is [System.Collections.IEnumerable]) {
      foreach ($child in $item) { $lines.Add([string]$child) | Out-Null }
    }
    else {
      $lines.Add([string]$item) | Out-Null
    }
  }

  $content = (($lines.ToArray() -join [Environment]::NewLine) + [Environment]::NewLine)
  $fileName = Split-Path -Leaf $LiteralPath
  $tempPath = Join-Path $directory (".{0}.{1}.tmp" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  $backupPath = Join-Path $directory (".{0}.{1}.bak" -f $fileName, [System.Guid]::NewGuid().ToString("N"))
  try {
    [System.IO.File]::WriteAllText($tempPath, $content, $script:utf8)
    for ($attempt = 1; $attempt -le 10; $attempt++) {
      try {
        if (Test-Path -LiteralPath $LiteralPath -PathType Leaf) {
          [System.IO.File]::Replace($tempPath, $LiteralPath, $backupPath)
          Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
        }
        else {
          [System.IO.File]::Move($tempPath, $LiteralPath)
        }

        return
      }
      catch {
        if ($attempt -eq 10) { throw }
        Start-Sleep -Milliseconds ([Math]::Min(250, 25 * $attempt))
      }
    }
  }
  finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path -LiteralPath $backupPath -PathType Leaf) {
      Remove-Item -LiteralPath $backupPath -Force -ErrorAction SilentlyContinue
    }
  }
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Read-JsonOrNull {
  param([string]$Path)
  if (-not [System.IO.Path]::IsPathRooted($Path)) {
    $Path = Join-Path $RepositoryRoot $Path
  }
  if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

if (-not (Test-Path -LiteralPath $StrictPreflightPath -PathType Leaf)) {
  & (Join-Path $RepositoryRoot "eng\Test-FinalOwnerExecutionRealInputStrictPreflight.ps1") -RepositoryRoot $RepositoryRoot
}

$preflight = Get-Content -LiteralPath $StrictPreflightPath -Raw -Encoding utf8 | ConvertFrom-Json
$readyForCloseValidation = [bool](Get-PropertyOrDefault -Object $preflight -Name "readyForCloseValidation" -DefaultValue $false)

$externalCleanConsumerImport = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-result-import.json"
$externalCleanConsumerCandidate = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-result-candidate.json"
$externalCleanConsumerValidation = Read-JsonOrNull "artifacts\final-release\external-clean-consumer-execution-result-validation.json"
$postPublishProofImport = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-import.json"
$postPublishProofCandidate = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-candidate.json"
$postPublishProofValidation = Read-JsonOrNull "artifacts\final-release\post-publish-clean-consumer-proof-result-validation.json"

$externalCleanConsumerProofReady =
  [bool](Get-PropertyOrDefault -Object $externalCleanConsumerImport -Name "proofCandidateReady" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $externalCleanConsumerCandidate -Name "proofCandidateReady" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $externalCleanConsumerCandidate -Name "isPackageConsumerRuntimeProof" -DefaultValue $false) -and
  [int](Get-PropertyOrDefault -Object $externalCleanConsumerValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0

$postPublishProofReady =
  [bool](Get-PropertyOrDefault -Object $postPublishProofImport -Name "proofCandidateReady" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $postPublishProofCandidate -Name "proofCandidateReady" -DefaultValue $false) -and
  [bool](Get-PropertyOrDefault -Object $postPublishProofCandidate -Name "isPostPublishProof" -DefaultValue $false) -and
  [int](Get-PropertyOrDefault -Object $postPublishProofValidation -Name "failedBlockerCount" -DefaultValue 999) -eq 0

$checks = @(
  [pscustomobject]@{ id = "real-input-strict-preflight-ready"; passed = $readyForCloseValidation; ownerAction = "Owner must clear strict preflight findings." },
  [pscustomobject]@{ id = "external-clean-consumer-proof-present"; passed = $externalCleanConsumerProofReady; ownerAction = "Owner must provide accepted repository-external CleanConsumer restore/build/run/smoke logs, hashes, host metadata, and strict validation." },
  [pscustomobject]@{ id = "post-publish-proof-present"; passed = $postPublishProofReady; ownerAction = "Owner must provide accepted post-publish clean consumer proof from the public package source." },
  [pscustomobject]@{ id = "rollback-review-present"; passed = $false; ownerAction = "Owner must provide rollback review." },
  [pscustomobject]@{ id = "final-close-decision-present"; passed = $false; ownerAction = "Owner must provide final close decision." },
  [pscustomobject]@{ id = "release-evidence-classification-clean"; passed = $false; ownerAction = "Owner must refresh release evidence and classification audit after real proof import." }
)

$blockedChecks = @($checks | Where-Object { -not [bool]$_.passed })
$readiness = [pscustomobject]@{
  recordKind = "final-owner-execution-close-readiness-from-real-input"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  readinessState = if ($blockedChecks.Count -eq 0) { "ready-for-final-owner-close-review-candidate" } else { "blocked-final-owner-real-input-close-readiness-required" }
  strictPreflightPath = $StrictPreflightPath
  readinessCheckCount = $checks.Count
  blockedReadinessCheckCount = $blockedChecks.Count
  checks = @($checks)
  externalCleanConsumerProofReady = $externalCleanConsumerProofReady
  externalCleanConsumerImportState = [string](Get-PropertyOrDefault -Object $externalCleanConsumerImport -Name "importState" -DefaultValue "missing-external-clean-consumer-execution-result-import")
  externalCleanConsumerCandidateState = [string](Get-PropertyOrDefault -Object $externalCleanConsumerCandidate -Name "candidateState" -DefaultValue "missing-external-clean-consumer-execution-result-candidate")
  externalCleanConsumerValidationState = [string](Get-PropertyOrDefault -Object $externalCleanConsumerValidation -Name "validationState" -DefaultValue "missing-external-clean-consumer-execution-result-validation")
  postPublishProofReady = $postPublishProofReady
  postPublishProofImportState = [string](Get-PropertyOrDefault -Object $postPublishProofImport -Name "importState" -DefaultValue "missing-post-publish-clean-consumer-proof-result-import")
  postPublishProofCandidateState = [string](Get-PropertyOrDefault -Object $postPublishProofCandidate -Name "candidateState" -DefaultValue "missing-post-publish-clean-consumer-proof-result-candidate")
  postPublishProofValidationState = [string](Get-PropertyOrDefault -Object $postPublishProofValidation -Name "validationState" -DefaultValue "missing-post-publish-clean-consumer-proof-result-validation")
  ownerActionRequired = $blockedChecks.Count -gt 0
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner execution close readiness from real input is a release-close readiness view only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-close-readiness-from-real-input.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-close-readiness-from-real-input.md"
$validationJsonPath = Join-Path $OutputRoot "final-owner-execution-close-readiness-from-real-input-validation.json"
$validationMarkdownPath = Join-Path $OutputRoot "final-owner-execution-close-readiness-from-real-input-validation.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($readiness | ConvertTo-Json -Depth 12)
$validation = [pscustomobject]@{
  recordKind = "final-owner-execution-close-readiness-from-real-input-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = "blocked-final-owner-real-input-close-readiness-required"
  readinessState = $readiness.readinessState
  readinessCheckCount = $checks.Count
  blockedReadinessCheckCount = $blockedChecks.Count
  externalCleanConsumerProofReady = $externalCleanConsumerProofReady
  postPublishProofReady = $postPublishProofReady
  failedBlockerCount = 0
  failedActionRequiredCount = $blockedChecks.Count
  ownerActionRequired = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = $readiness.boundary
}
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $validationJsonPath -Encoding utf8

$rows = foreach ($check in $checks) {
  "| ``$(ConvertTo-MarkdownCell $check.id)`` | ``$($check.passed)`` | $(ConvertTo-MarkdownCell $check.ownerAction) |"
}

$markdown = @"
# Final Owner Execution Close Readiness From Real Input

| Field | Value |
|---|---|
| readinessState | ``$($readiness.readinessState)`` |
| readinessCheckCount | ``$($readiness.readinessCheckCount)`` |
| blockedReadinessCheckCount | ``$($readiness.blockedReadinessCheckCount)`` |
| externalCleanConsumerProofReady | ``$($readiness.externalCleanConsumerProofReady)`` |
| postPublishProofReady | ``$($readiness.postPublishProofReady)`` |
| canCloseReleaseIssue | ``$($readiness.canCloseReleaseIssue)`` |

## Checks

| ID | Passed | Owner Action |
|---|---:|---|
$($rows -join "`r`n")

## Boundary

$($readiness.boundary)
"@
Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Utf8File -LiteralPath $validationMarkdownPath -InputObject @(
  "# Final Owner Execution Close Readiness From Real Input Validation",
  "",
  "- validationState: ``$($validation.validationState)``",
  "- failedBlockerCount: ``0``",
  "- failedActionRequiredCount: ``$($validation.failedActionRequiredCount)``",
  "- canCloseReleaseIssue: ``False``",
  "- Boundary: $($validation.boundary)"
)

Write-Host "Final owner execution close readiness written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Validation=$validationJsonPath"
Write-Host "ReadinessState=$($readiness.readinessState) Blocked=$($blockedChecks.Count)"

if ($Strict.IsPresent -and [int]$validation.failedBlockerCount -gt 0) {
  throw "Final owner execution close readiness validation failed."
}
