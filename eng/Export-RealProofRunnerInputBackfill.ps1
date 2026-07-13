[CmdletBinding()]
param(
  [string]$OutputRoot,
  [string]$RepositoryRoot
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

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-PlaceholderCommand {
  param([string]$Command)

  [pscustomobject]@{
    command = $Command
    exitCode = "<owner-fill-exit-code>"
    startedAtUtc = "<owner-fill-started-at-utc>"
    completedAtUtc = "<owner-fill-completed-at-utc>"
    workingDirectory = "<owner-fill-working-directory>"
    logPath = "<owner-fill-command-log-path>"
    logSha256 = "<owner-fill-command-log-sha256>"
  }
}

function New-PlaceholderLog {
  param([string]$Name)

  [pscustomobject]@{
    name = $Name
    path = "<owner-fill-$($Name.Replace(' ', '-').ToLowerInvariant())-path>"
    sha256 = "<owner-fill-$($Name.Replace(' ', '-').ToLowerInvariant())-sha256>"
    stdoutSummary = "<owner-fill-stdout-summary>"
    stderrSummary = "<owner-fill-stderr-summary>"
  }
}

function New-PlaceholderHash {
  param([string]$Name)

  [pscustomobject]@{
    name = $Name
    path = "<owner-fill-$($Name.Replace(' ', '-').ToLowerInvariant())-path>"
    sha256 = "<owner-fill-$($Name.Replace(' ', '-').ToLowerInvariant())-sha256>"
    computedSha256 = "<validator-computed-sha256>"
    matches = $false
  }
}

function New-ForbiddenSubstituteCheck {
  param([string]$Name)

  [pscustomobject]@{
    name = $Name
    checked = $false
    present = $true
    ownerEvidence = "<owner-fill-forbidden-substitute-check-evidence>"
    passed = $false
  }
}

function New-TrackInput {
  param([object]$Track)

  $id = [string](Get-PropertyOrDefault -Object $Track -Name "id" -DefaultValue "unknown-track")
  $proofKind = [string](Get-PropertyOrDefault -Object $Track -Name "proofKind" -DefaultValue "unknown-proof-kind")
  $commands = @(Get-PropertyOrDefault -Object $Track -Name "requiredCommands" -DefaultValue @() | ForEach-Object { New-PlaceholderCommand -Command ([string]$_) })
  $logs = @(Get-PropertyOrDefault -Object $Track -Name "requiredLogs" -DefaultValue @() | ForEach-Object { New-PlaceholderLog -Name ([string]$_) })
  $hashes = @(Get-PropertyOrDefault -Object $Track -Name "requiredHashes" -DefaultValue @() | ForEach-Object { New-PlaceholderHash -Name ([string]$_) })
  $forbiddenChecks = @(Get-PropertyOrDefault -Object $Track -Name "forbiddenSubstitutes" -DefaultValue @() | ForEach-Object { New-ForbiddenSubstituteCheck -Name ([string]$_) })

  [pscustomobject]@{
    trackId = $id
    proofKind = $proofKind
    ownerStatus = "owner-action-required"
    currentState = [string](Get-PropertyOrDefault -Object $Track -Name "currentState" -DefaultValue "missing-current-state")
    hostOs = "<owner-fill-host-os>"
    hostArchitecture = "<owner-fill-host-architecture>"
    cudaDriverVersion = "<owner-fill-cuda-driver-version>"
    cudaRuntimeVersion = "<owner-fill-cuda-runtime-version>"
    tensorRtVersion = "<owner-fill-tensorrt-version>"
    runtimePackageKey = "<owner-fill-runtime-package-key>"
    commands = @($commands)
    logs = @($logs)
    hashes = @($hashes)
    stdoutSummary = "<owner-fill-track-stdout-summary>"
    stderrSummary = "<owner-fill-track-stderr-summary>"
    validatorOutputs = @(
      [pscustomobject]@{
        command = "<owner-fill-validator-command>"
        exitCode = "<owner-fill-validator-exit-code>"
        logPath = "<owner-fill-validator-log-path>"
        logSha256 = "<owner-fill-validator-log-sha256>"
        passed = $false
      }
    )
    forbiddenSubstituteChecks = @($forbiddenChecks)
    sourceArtifacts = @(Get-PropertyOrDefault -Object $Track -Name "sourceArtifacts" -DefaultValue @())
    canPromoteRuntimeProof = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    boundary = [string](Get-PropertyOrDefault -Object $Track -Name "boundary" -DefaultValue "Owner input backfill is not proof.")
  }
}

$executionBundle = Read-JsonOrNull "artifacts\final-release\real-external-proof-backfill-execution-bundle.json"
if ($null -eq $executionBundle) {
  throw "Missing real external proof backfill execution bundle. Run Export-RealExternalProofBackfillExecutionBundle.ps1 first."
}

$tracks = @(Get-PropertyOrDefault -Object $executionBundle -Name "proofTracks" -DefaultValue @())
$trackInputs = @($tracks | ForEach-Object { New-TrackInput -Track $_ })
$forbiddenSubstitutes = @(Get-PropertyOrDefault -Object $executionBundle -Name "forbiddenSubstitutes" -DefaultValue @())
$sourceArtifacts = @(
  "artifacts/final-release/real-external-proof-backfill-execution-bundle.json",
  "artifacts/final-release/release-close-proof-worklist.json",
  "artifacts/final-release/package-consumer-runtime-proof-worklist.json",
  "artifacts/final-release/post-publish-verification-validation.json",
  "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
  "artifacts/user-acceptance/real-model-owner-handoff.json",
  "artifacts/final-release/release-issue-close-record-owner-input-validation.json",
  "artifacts/final-release/release-issue-final-close-decision-validation.json",
  "artifacts/final-release/release-issue-close-record-validation.json"
)

$recordOut = [pscustomobject]@{
  recordKind = "real-proof-runner-input-backfill"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  inputState = "blocked-owner-runner-input-required"
  sourceExecutionBundleState = [string](Get-PropertyOrDefault -Object $executionBundle -Name "bundleState" -DefaultValue "missing-execution-bundle-state")
  trackCount = $trackInputs.Count
  blockedTrackCount = $trackInputs.Count
  trackInputs = @($trackInputs)
  forbiddenSubstitutes = $forbiddenSubstitutes
  sourceArtifacts = $sourceArtifacts
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This owner input backfill template records fields needed for future real proof validation. It is not proof, not a package push, not post-publish verification, and not release-close approval."
}

$jsonPath = Join-Path $OutputRoot "real-proof-runner-input-backfill.template.json"
$markdownPath = Join-Path $OutputRoot "real-proof-runner-input-backfill.template.md"
$recordOut | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Real Proof Runner Input Backfill Template")
$lines.Add("")
$lines.Add("`real-proof-runner-input-backfill` 是 Owner 输入模板，用于记录真实 proof runner 所需的路径、hash、host metadata、commands、stdout/stderr summary、validator output 和 forbidden substitute 检查。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| inputState | ``$(ConvertTo-MarkdownCell $recordOut.inputState)`` |")
$lines.Add("| trackCount | ``$($recordOut.trackCount)`` |")
$lines.Add("| blockedTrackCount | ``$($recordOut.blockedTrackCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($recordOut.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($recordOut.isRuntimeExecutionProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($recordOut.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Tracks")
$lines.Add("")
$lines.Add("| Track | Proof Kind | Owner Status | Commands | Logs | Hashes |")
$lines.Add("| --- | --- | --- | --- | --- | --- |")
foreach ($track in $trackInputs) {
  $lines.Add("| $(ConvertTo-MarkdownCell $track.trackId) | $(ConvertTo-MarkdownCell $track.proofKind) | $(ConvertTo-MarkdownCell $track.ownerStatus) | ``$(@($track.commands).Count)`` | ``$(@($track.logs).Count)`` | ``$(@($track.hashes).Count)`` |")
}
$lines.Add("")
$lines.Add("## Forbidden Substitutes")
$lines.Add("")
foreach ($item in $forbiddenSubstitutes) {
  $lines.Add("- ``$(ConvertTo-MarkdownCell $item)``")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof runner input backfill template written to $jsonPath"
Write-Host "Real proof runner input backfill template markdown written to $markdownPath"
Write-Host "InputState=$($recordOut.inputState) Tracks=$($recordOut.trackCount) Blocked=$($recordOut.blockedTrackCount)"
