[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\final-post-publish-clean-consumer-proof-record-contract.json",
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
  throw "Final post-publish clean consumer proof record contract not found: $resolvedInputPath"
}

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$raw = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8
$items = New-Object System.Collections.Generic.List[object]
$proofResults = New-Object System.Collections.Generic.List[object]
$contractItems = @((Get-PropertyOrDefault -Object $record -Name "contractItems" -DefaultValue @()))

$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "final-post-publish-clean-consumer-proof-record-contract") -Severity "blocker" -Detail "Preflight input must be final-post-publish-clean-consumer-proof-record-contract.")) | Out-Null
$items.Add((New-ValidationItem -Id "non-proof-top-level" -Passed ((-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true)) -and (-not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true))) -Severity "blocker" -Detail "Input contract must stay non-proof and non-publish.")) | Out-Null

foreach ($marker in @("local feed", "ProjectReference", "direct .nupkg", "source checkout reference", "template", "draft", "dry-run", "dashboard", "candidate", "build-only", "dependency-probe-only", "blocked-by-cuda-driver")) {
  $items.Add((New-ValidationItem -Id "forbidden-marker-listed-$($marker.Replace(' ', '-').Replace('.', 'dot'))" -Passed ($raw.IndexOf($marker, [StringComparison]::OrdinalIgnoreCase) -ge 0) -Severity "blocker" -Detail "Contract must carry forbidden substitute marker $marker.")) | Out-Null
}

foreach ($contractItem in $contractItems) {
  $id = [string](Get-PropertyOrDefault -Object $contractItem -Name "id" -DefaultValue "")
  if ([string]::IsNullOrWhiteSpace($id)) { $id = "unknown-clean-consumer-$($proofResults.Count)" }

  $checks = New-Object System.Collections.Generic.List[object]
  $projectRoot = [string](Get-PropertyOrDefault -Object $contractItem -Name "cleanConsumerProjectRoot" -DefaultValue "")
  $packageIdentity = Get-PropertyOrDefault -Object $contractItem -Name "packageIdentity" -DefaultValue $null
  $hostIdentity = Get-PropertyOrDefault -Object $contractItem -Name "hostIdentity" -DefaultValue $null

  $checks.Add((New-ValidationItem -Id "$id-project-root-filled" -Passed (-not (Test-Placeholder $projectRoot)) -Severity "action-required" -Detail "$id cleanConsumerProjectRoot must be filled.")) | Out-Null
  $checks.Add((New-ValidationItem -Id "$id-project-root-exists" -Passed ((-not (Test-Placeholder $projectRoot)) -and (Test-Path -LiteralPath (Resolve-RepositoryPath -Path $projectRoot) -PathType Container)) -Severity "action-required" -Detail "$id cleanConsumerProjectRoot must exist.")) | Out-Null

  foreach ($pathPair in @(
    @{ Name = "projectManifest"; PathName = "cleanConsumerProjectSha256Manifest"; HashName = "" },
    @{ Name = "restoreLog"; PathName = "cleanConsumerRestoreLogPath"; HashName = "cleanConsumerRestoreLogSha256" },
    @{ Name = "buildLog"; PathName = "cleanConsumerBuildLogPath"; HashName = "cleanConsumerBuildLogSha256" },
    @{ Name = "runLog"; PathName = "cleanConsumerRunLogPath"; HashName = "cleanConsumerRunLogSha256" },
    @{ Name = "mergedTranscript"; PathName = "cleanConsumerMergedTranscriptPath"; HashName = "cleanConsumerMergedTranscriptSha256" },
    @{ Name = "validatorOutput"; PathName = "cleanConsumerValidatorOutputPath"; HashName = "cleanConsumerValidatorOutputSha256" }
  )) {
    $name = [string]$pathPair.Name
    $pathName = [string]$pathPair.PathName
    $hashName = [string]$pathPair.HashName
    $path = [string](Get-PropertyOrDefault -Object $contractItem -Name $pathName -DefaultValue "")
    $checks.Add((New-ValidationItem -Id "$id-$name-path-filled" -Passed (-not (Test-Placeholder $path)) -Severity "action-required" -Detail "$id $name path must be filled.")) | Out-Null
    $checks.Add((New-ValidationItem -Id "$id-$name-path-exists" -Passed ((-not (Test-Placeholder $path)) -and (Test-Path -LiteralPath (Resolve-RepositoryPath -Path $path) -PathType Leaf)) -Severity "action-required" -Detail "$id $name path must exist.")) | Out-Null
    $checks.Add((New-ValidationItem -Id "$id-$name-under-root" -Passed (Test-PathUnderRoot -RootPath $projectRoot -CandidatePath $path) -Severity "action-required" -Detail "$id $name path must stay under cleanConsumerProjectRoot.")) | Out-Null
    if (-not [string]::IsNullOrWhiteSpace($hashName)) {
      $expectedHash = [string](Get-PropertyOrDefault -Object $contractItem -Name $hashName -DefaultValue "")
      $checks.Add((New-ValidationItem -Id "$id-$name-sha256-match" -Passed (Test-Sha256 -Path $path -Expected $expectedHash) -Severity "action-required" -Detail "$id $name SHA256 must match real file.")) | Out-Null
    }
  }

  $exitCodeText = [string](Get-PropertyOrDefault -Object $contractItem -Name "exitCode" -DefaultValue "")
  $exitCodeValue = 1
  $exitCodeParsed = [int]::TryParse($exitCodeText, [ref]$exitCodeValue)
  $checks.Add((New-ValidationItem -Id "$id-exit-code-zero" -Passed ($exitCodeParsed -and $exitCodeValue -eq 0) -Severity "action-required" -Detail "$id exitCode must be real integer 0.")) | Out-Null

  foreach ($field in @("executedCommand", "executedAtUtc", "ownerReviewer", "ownerReviewTimestampUtc")) {
    $checks.Add((New-ValidationItem -Id "$id-$field-filled" -Passed (-not (Test-Placeholder (Get-PropertyOrDefault -Object $contractItem -Name $field -DefaultValue ""))) -Severity "action-required" -Detail "$id $field must be filled.")) | Out-Null
  }

  $checks.Add((New-ValidationItem -Id "$id-host-identity-filled" -Passed (Test-IdentityFilled -Identity $hostIdentity -Fields @("machineName", "os", "architecture", "cudaVersion", "tensorrtVersion", "driverVersion")) -Severity "action-required" -Detail "$id host identity must be complete.")) | Out-Null
  $checks.Add((New-ValidationItem -Id "$id-package-identity-filled" -Passed (Test-IdentityFilled -Identity $packageIdentity -Fields @("packageId", "packageVersion", "packageSource", "publicPackageUrl", "nupkgSha256")) -Severity "action-required" -Detail "$id package identity must be complete.")) | Out-Null

  $packageSource = [string](Get-PropertyOrDefault -Object $packageIdentity -Name "packageSource" -DefaultValue "")
  $checks.Add((New-ValidationItem -Id "$id-package-source-not-local-feed" -Passed ((-not (Test-Placeholder $packageSource)) -and $packageSource.IndexOf("local", [StringComparison]::OrdinalIgnoreCase) -lt 0 -and $packageSource.IndexOf("feed", [StringComparison]::OrdinalIgnoreCase) -lt 0) -Severity "action-required" -Detail "$id packageSource must be a public package source, not local feed.")) | Out-Null
  foreach ($confirmation in @("noProjectReferenceConfirmation", "noLocalFeedConfirmation", "noDirectNupkgConfirmation", "noSourceCheckoutReferenceConfirmation")) {
    $checks.Add((New-ValidationItem -Id "$id-$confirmation" -Passed ([bool](Get-PropertyOrDefault -Object $contractItem -Name $confirmation -DefaultValue $false)) -Severity "action-required" -Detail "$id $confirmation must be true.")) | Out-Null
  }

  $failedChecks = @($checks | Where-Object { -not $_.passed })
  $ready = $failedChecks.Count -eq 0
  foreach ($check in $checks) { $items.Add($check) | Out-Null }

  $proofResults.Add([pscustomobject]@{
    id = $id
    preflightState = if ($ready) { "post-publish-clean-consumer-proof-candidate-ready" } else { "blocked-post-publish-clean-consumer-proof-required" }
    readyForProofCandidate = $ready
    failedCheckCount = $failedChecks.Count
    checks = @($checks.ToArray())
    boundary = "Clean consumer preflight result only. A ready proof candidate is still not release close approval, not publish approval, and not package push."
  }) | Out-Null
}

$readyCount = @($proofResults | Where-Object { $_.readyForProofCandidate }).Count
$blockedCount = $proofResults.Count - $readyCount
$failedItems = @($items | Where-Object { -not $_.passed })
$failedBlockers = @($failedItems | Where-Object { $_.severity -eq "blocker" }).Count
$failedActionRequired = @($failedItems | Where-Object { $_.severity -eq "action-required" }).Count
$preflightState = if ($failedBlockers -gt 0) { "invalid-final-post-publish-clean-consumer-proof-preflight" } elseif ($readyCount -gt 0) { "post-publish-clean-consumer-proof-candidate-ready" } else { "blocked-post-publish-clean-consumer-proof-required" }

$validation = [ordered]@{
  recordKind = "final-post-publish-clean-consumer-proof-preflight"
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  preflightState = $preflightState
  sourceInputPath = $InputPath
  contractItemCount = $contractItems.Count
  readyProofCandidateCount = $readyCount
  blockedProofCandidateCount = $blockedCount
  failedBlockerCount = $failedBlockers
  failedActionRequiredCount = $failedActionRequired
  findingCount = $failedItems.Count
  proofResults = @($proofResults.ToArray())
  validationItems = @($items.ToArray())
  failedItems = @($failedItems)
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final post-publish clean consumer proof preflight only. It screens Owner-filled clean consumer evidence; it is not publish approval, not release close approval, not package push, and cannot close the release."
}

$jsonPath = Join-Path $OutputRoot "final-post-publish-clean-consumer-proof-preflight.json"
$markdownPath = Join-Path $OutputRoot "final-post-publish-clean-consumer-proof-preflight.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($validation | ConvertTo-Json -Depth 32)
$markdown = @(
  "# Final Post-Publish Clean Consumer Proof Preflight",
  "",
  "- preflightState: ``$preflightState``",
  "- contractItemCount: ``$($contractItems.Count)``",
  "- readyProofCandidateCount: ``$readyCount``",
  "- blockedProofCandidateCount: ``$blockedCount``",
  "- failedBlockerCount: ``$failedBlockers``",
  "- failedActionRequiredCount: ``$failedActionRequired``",
  "- boundary: $($validation.boundary)"
)

Write-Utf8File -LiteralPath $markdownPath -InputObject $markdown
Write-Host "Wrote $jsonPath"
Write-Host "Wrote $markdownPath"

if ($Strict -and $failedBlockers -gt 0) {
  throw "Final post-publish clean consumer proof preflight failed with $failedBlockers blocker(s)."
}
