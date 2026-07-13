[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-owner-execution-external-result-input-contract.json",
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
  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

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

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)
  return @($Values | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_) } | ForEach-Object { [string]$_ })
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)
  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function Test-Placeholder {
  param([AllowNull()][object]$Value)
  $text = [string]$Value
  return [string]::IsNullOrWhiteSpace($text) -or $text.StartsWith("<owner-fill-", [StringComparison]::OrdinalIgnoreCase)
}

function Test-PathUnderRoot {
  param([string]$RootPath, [string]$CandidatePath)
  if (Test-Placeholder $RootPath) { return $false }
  if (Test-Placeholder $CandidatePath) { return $false }
  try {
    $resolvedRoot = (Resolve-Path -LiteralPath (Resolve-RepositoryPath -Path $RootPath) -ErrorAction Stop).Path
    $resolvedCandidate = (Resolve-Path -LiteralPath (Resolve-RepositoryPath -Path $CandidatePath) -ErrorAction Stop).Path
    return $resolvedCandidate.StartsWith($resolvedRoot.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase) -or $resolvedCandidate.Equals($resolvedRoot, [StringComparison]::OrdinalIgnoreCase)
  } catch {
    return $false
  }
}

function Get-FileSha256 {
  param([string]$Path)
  try {
    return (Get-FileHash -LiteralPath (Resolve-RepositoryPath -Path $Path) -Algorithm SHA256 -ErrorAction Stop).Hash.ToLowerInvariant()
  } catch {
    return ""
  }
}

function Test-Sha256 {
  param([string]$Path, [string]$Expected)
  if (Test-Placeholder $Path) { return $false }
  if (Test-Placeholder $Expected) { return $false }
  if ($Expected -notmatch '^[0-9a-fA-F]{64}$') { return $false }
  return (Get-FileSha256 -Path $Path) -eq $Expected.ToLowerInvariant()
}

function Test-IdentityFilled {
  param([AllowNull()][object]$Identity, [string[]]$Fields)
  if ($null -eq $Identity) { return $false }
  foreach ($field in $Fields) {
    if (-not ($Identity.PSObject.Properties.Name -contains $field)) { return $false }
    if (Test-Placeholder (Get-PropertyOrDefault -Object $Identity -Name $field -DefaultValue "")) { return $false }
  }
  return $true
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Final Owner external result input contract not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$laneResults = New-Object System.Collections.Generic.List[object]
$contractItems = @((Get-PropertyOrDefault -Object $record -Name "contractItems" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-owner-execution-external-result-input-contract") -Severity "blocker" -Detail "Preflight input must be final-owner-execution-external-result-input-contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-top-level" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true))) -Severity "blocker" -Detail "Input contract must stay non-proof and non-publish.")) | Out-Null

foreach ($marker in @("local feed", "ProjectReference", "direct .nupkg", "template", "draft", "dry-run", "dashboard", "candidate", "build-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "forbidden-marker-listed-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed ($raw.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Contract must carry forbidden substitute marker $marker.")) | Out-Null
}

foreach ($item in $contractItems) {
  $id = [string](Get-PropertyOrDefault -Object $item -Name "sourceDraftItemId" -DefaultValue "")
  if ([string]::IsNullOrWhiteSpace($id)) { $id = "unknown-lane-$($laneResults.Count)" }

  $laneChecks = New-Object System.Collections.Generic.List[object]
  $root = [string](Get-PropertyOrDefault -Object $item -Name "realExecutionRoot" -DefaultValue "")
  $stdoutPath = [string](Get-PropertyOrDefault -Object $item -Name "stdoutPath" -DefaultValue "")
  $stderrPath = [string](Get-PropertyOrDefault -Object $item -Name "stderrPath" -DefaultValue "")
  $mergedTranscriptPath = [string](Get-PropertyOrDefault -Object $item -Name "mergedTranscriptPath" -DefaultValue "")
  $validatorOutputPath = [string](Get-PropertyOrDefault -Object $item -Name "validatorOutputPath" -DefaultValue "")
  $hostIdentity = Get-PropertyOrDefault -Object $item -Name "hostIdentity" -DefaultValue $null
  $packageIdentity = Get-PropertyOrDefault -Object $item -Name "packageIdentity" -DefaultValue $null
  $confirmations = @((Get-PropertyOrDefault -Object $item -Name "ownerProvidedNonSubstituteConfirmations" -DefaultValue @()))
  $forbiddenMarkers = Convert-ToStringArray (Get-PropertyOrDefault -Object $item -Name "forbiddenSubstituteMarkers" -DefaultValue @())

  $laneChecks.Add((New-ValidationItem -Id "$id-root-filled" -Passed (-not (Test-Placeholder $root)) -Severity "action-required" -Detail "$id realExecutionRoot must be filled with an existing evidence root.")) | Out-Null
  $laneChecks.Add((New-ValidationItem -Id "$id-root-exists" -Passed ((-not (Test-Placeholder $root)) -and (Test-Path -LiteralPath (Resolve-RepositoryPath -Path $root) -PathType Container)) -Severity "action-required" -Detail "$id realExecutionRoot must exist.")) | Out-Null

  foreach ($pathPair in @(
    @{ Name = "stdout"; Path = $stdoutPath; HashName = "stdoutSha256" },
    @{ Name = "stderr"; Path = $stderrPath; HashName = "stderrSha256" },
    @{ Name = "mergedTranscript"; Path = $mergedTranscriptPath; HashName = "mergedTranscriptSha256" },
    @{ Name = "validatorOutput"; Path = $validatorOutputPath; HashName = "validatorOutputSha256" }
  )) {
    $name = [string]$pathPair.Name
    $path = [string]$pathPair.Path
    $hashName = [string]$pathPair.HashName
    $expectedHash = [string](Get-PropertyOrDefault -Object $item -Name $hashName -DefaultValue "")
    $laneChecks.Add((New-ValidationItem -Id "$id-$name-path-filled" -Passed (-not (Test-Placeholder $path)) -Severity "action-required" -Detail "$id $name path must be filled.")) | Out-Null
    $laneChecks.Add((New-ValidationItem -Id "$id-$name-path-exists" -Passed ((-not (Test-Placeholder $path)) -and (Test-Path -LiteralPath (Resolve-RepositoryPath -Path $path) -PathType Leaf)) -Severity "action-required" -Detail "$id $name path must exist.")) | Out-Null
    $laneChecks.Add((New-ValidationItem -Id "$id-$name-under-root" -Passed (Test-PathUnderRoot -RootPath $root -CandidatePath $path) -Severity "action-required" -Detail "$id $name path must stay under realExecutionRoot.")) | Out-Null
    $laneChecks.Add((New-ValidationItem -Id "$id-$name-sha256-match" -Passed (Test-Sha256 -Path $path -Expected $expectedHash) -Severity "action-required" -Detail "$id $name SHA256 must match the real file.")) | Out-Null
  }

  $exitCodeText = [string](Get-PropertyOrDefault -Object $item -Name "exitCode" -DefaultValue "")
  $exitCodeValue = 1
  $exitCodeParsed = [int]::TryParse($exitCodeText, [ref]$exitCodeValue)
  $laneChecks.Add((New-ValidationItem -Id "$id-exit-code-zero" -Passed ($exitCodeParsed -and $exitCodeValue -eq 0) -Severity "action-required" -Detail "$id exitCode must be real integer 0.")) | Out-Null

  foreach ($field in @("executedCommand", "executedAtUtc", "ownerReviewer", "ownerReviewTimestampUtc")) {
    $laneChecks.Add((New-ValidationItem -Id "$id-$field-filled" -Passed (-not (Test-Placeholder (Get-PropertyOrDefault -Object $item -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$id $field must be filled.")) | Out-Null
  }

  $laneChecks.Add((New-ValidationItem -Id "$id-host-identity-filled" -Passed (Test-IdentityFilled -Identity $hostIdentity -Fields @("machineName", "os", "architecture", "cudaVersion", "tensorrtVersion", "driverVersion")) -Severity "action-required" -Detail "$id host identity must be complete.")) | Out-Null
  $laneChecks.Add((New-ValidationItem -Id "$id-package-identity-filled" -Passed (Test-IdentityFilled -Identity $packageIdentity -Fields @("packageId", "packageVersion", "packageSource", "nupkgPath", "nupkgSha256", "publishedPackageUrl")) -Severity "action-required" -Detail "$id package identity must be complete.")) | Out-Null

  $confirmedCount = @($confirmations | Where-Object { [bool](Get-PropertyOrDefault -Object $_ -Name "ownerConfirmed" -DefaultValue $false) }).Count
  $laneChecks.Add((New-ValidationItem -Id "$id-confirmations-complete" -Passed ($confirmations.Count -gt 0 -and $confirmedCount -eq $confirmations.Count) -Severity "action-required" -Detail "$id all non-substitute confirmations must be true.")) | Out-Null
  foreach ($marker in @("local feed", "ProjectReference", "direct .nupkg", "template", "draft", "dry-run", "dashboard", "candidate", "build-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
    $laneChecks.Add((New-ValidationItem -Id "$id-forbidden-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed (($forbiddenMarkers -join "`n").IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "action-required" -Detail "$id must explicitly reject $marker as substitute proof.")) | Out-Null
  }

  $failedLaneChecks = @($laneChecks | Where-Object { -not $_.passed })
  $laneReady = $failedLaneChecks.Count -eq 0
  foreach ($check in $laneChecks) { $items.Add($check) | Out-Null }

  $laneResults.Add([pscustomobject]@{
    sourceDraftItemId = $id
    sourceExecutionStepId = [string](Get-PropertyOrDefault -Object $item -Name "sourceExecutionStepId" -DefaultValue "")
    laneId = [string](Get-PropertyOrDefault -Object $item -Name "laneId" -DefaultValue "")
    preflightState = if ($laneReady) { "owner-external-result-preflight-ready-candidate" } else { "blocked-owner-real-external-result-required" }
    readyForCandidate = $laneReady
    failedCheckCount = $failedLaneChecks.Count
    checks = @($laneChecks.ToArray())
    boundary = "Preflight result only. A ready candidate is still not runtime proof, not post-publish proof, not release close proof, not publish approval, and not package push."
  }) | Out-Null
}

$readyLaneCount = @($laneResults | Where-Object { $_.readyForCandidate }).Count
$blockedLaneCount = $laneResults.Count - $readyLaneCount
$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$failedActionRequired = @($failedItems | Where-Object { $_.severity -eq "action-required" }).Count
$preflightState = if ($failedBlockers -gt 0) { "invalid-final-owner-execution-external-result-input-preflight" } elseif ($readyLaneCount -gt 0) { "owner-external-result-candidate-ready" } else { "blocked-owner-real-external-result-required" }

$validation = [ordered]@{
  recordKind = "final-owner-execution-external-result-input-preflight"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  preflightState = $preflightState
  sourceInputPath = $InputPath
  contractItemCount = $contractItems.Count
  readyCandidateLaneCount = $readyLaneCount
  blockedLaneCount = $blockedLaneCount
  failedBlockerCount = $failedBlockers
  failedActionRequiredCount = $failedActionRequired
  findingCount = $failedItems.Count
  laneResults = @($laneResults.ToArray())
  validationItems = @($items.ToArray())
  failedItems = @($failedItems)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner external result input preflight only. It can identify candidate-ready lanes, but it is not runtime proof, not post-publish proof, not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "final-owner-execution-external-result-input-preflight.json"
$markdownPath = Join-Path $OutputRoot "final-owner-execution-external-result-input-preflight.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 32)
$markdown = @(
  "# Final Owner Execution External Result Input Preflight",
  "",
  "- preflightState: ``$preflightState``",
  "- contractItemCount: ``$($contractItems.Count)``",
  "- readyCandidateLaneCount: ``$readyLaneCount``",
  "- blockedLaneCount: ``$blockedLaneCount``",
  "- failedBlockerCount: ``$failedBlockers``",
  "- failedActionRequiredCount: ``$failedActionRequired``",
  "- boundary: $($validation.boundary)"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"

if ($Strict -and $failedBlockers -gt 0) {
  throw "Final Owner external result input preflight failed with $failedBlockers blocker(s)."
}
