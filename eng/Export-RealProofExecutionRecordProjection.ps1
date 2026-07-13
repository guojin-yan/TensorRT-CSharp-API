[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\real-proof-runner-input-backfill.template.json",
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

function Resolve-InputPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
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

function Convert-CommandProjection {
  param([object]$Command)

  [pscustomobject]@{
    command = [string](Get-PropertyOrDefault -Object $Command -Name "command" -DefaultValue "<owner-fill-command>")
    exitCode = Get-PropertyOrDefault -Object $Command -Name "exitCode" -DefaultValue "<owner-fill-exit-code>"
    startedAtUtc = [string](Get-PropertyOrDefault -Object $Command -Name "startedAtUtc" -DefaultValue "<owner-fill-started-at-utc>")
    completedAtUtc = [string](Get-PropertyOrDefault -Object $Command -Name "completedAtUtc" -DefaultValue "<owner-fill-completed-at-utc>")
    workingDirectory = [string](Get-PropertyOrDefault -Object $Command -Name "workingDirectory" -DefaultValue "<owner-fill-working-directory>")
    logPath = [string](Get-PropertyOrDefault -Object $Command -Name "logPath" -DefaultValue "<owner-fill-command-log-path>")
    logSha256 = [string](Get-PropertyOrDefault -Object $Command -Name "logSha256" -DefaultValue "<owner-fill-command-log-sha256>")
  }
}

function Convert-LogProjection {
  param([object]$Log)

  [pscustomobject]@{
    name = [string](Get-PropertyOrDefault -Object $Log -Name "name" -DefaultValue "unnamed-log")
    path = [string](Get-PropertyOrDefault -Object $Log -Name "path" -DefaultValue "<owner-fill-log-path>")
    sha256 = [string](Get-PropertyOrDefault -Object $Log -Name "sha256" -DefaultValue "<owner-fill-log-sha256>")
    computedSha256 = [string](Get-PropertyOrDefault -Object $Log -Name "computedSha256" -DefaultValue "<validator-computed-sha256>")
    matches = [bool](Get-PropertyOrDefault -Object $Log -Name "matches" -DefaultValue $false)
    stdoutSummary = [string](Get-PropertyOrDefault -Object $Log -Name "stdoutSummary" -DefaultValue "<owner-fill-stdout-summary>")
    stderrSummary = [string](Get-PropertyOrDefault -Object $Log -Name "stderrSummary" -DefaultValue "<owner-fill-stderr-summary>")
  }
}

function Convert-HashProjection {
  param([object]$Hash)

  [pscustomobject]@{
    name = [string](Get-PropertyOrDefault -Object $Hash -Name "name" -DefaultValue "unnamed-hash")
    path = [string](Get-PropertyOrDefault -Object $Hash -Name "path" -DefaultValue "<owner-fill-hash-path>")
    sha256 = [string](Get-PropertyOrDefault -Object $Hash -Name "sha256" -DefaultValue "<owner-fill-hash-sha256>")
    computedSha256 = [string](Get-PropertyOrDefault -Object $Hash -Name "computedSha256" -DefaultValue "<validator-computed-sha256>")
    matches = [bool](Get-PropertyOrDefault -Object $Hash -Name "matches" -DefaultValue $false)
  }
}

function Convert-ValidatorProjection {
  param([object]$ValidatorOutput)

  [pscustomobject]@{
    command = [string](Get-PropertyOrDefault -Object $ValidatorOutput -Name "command" -DefaultValue "<owner-fill-validator-command>")
    exitCode = Get-PropertyOrDefault -Object $ValidatorOutput -Name "exitCode" -DefaultValue "<owner-fill-validator-exit-code>"
    logPath = [string](Get-PropertyOrDefault -Object $ValidatorOutput -Name "logPath" -DefaultValue "<owner-fill-validator-log-path>")
    logSha256 = [string](Get-PropertyOrDefault -Object $ValidatorOutput -Name "logSha256" -DefaultValue "<owner-fill-validator-log-sha256>")
    passed = [bool](Get-PropertyOrDefault -Object $ValidatorOutput -Name "passed" -DefaultValue $false)
  }
}

function Convert-ForbiddenCheckProjection {
  param([object]$Check)

  [pscustomobject]@{
    name = [string](Get-PropertyOrDefault -Object $Check -Name "name" -DefaultValue "unknown-forbidden-substitute")
    checked = [bool](Get-PropertyOrDefault -Object $Check -Name "checked" -DefaultValue $false)
    present = [bool](Get-PropertyOrDefault -Object $Check -Name "present" -DefaultValue $true)
    ownerEvidence = [string](Get-PropertyOrDefault -Object $Check -Name "ownerEvidence" -DefaultValue "<owner-fill-forbidden-substitute-check-evidence>")
    passed = [bool](Get-PropertyOrDefault -Object $Check -Name "passed" -DefaultValue $false)
  }
}

function New-ProofExecutionRecord {
  param([object]$Track)

  $trackId = [string](Get-PropertyOrDefault -Object $Track -Name "trackId" -DefaultValue "unknown-track")
  $commands = @(Get-PropertyOrDefault -Object $Track -Name "commands" -DefaultValue @() | ForEach-Object { Convert-CommandProjection -Command $_ })
  $logs = @(Get-PropertyOrDefault -Object $Track -Name "logs" -DefaultValue @() | ForEach-Object { Convert-LogProjection -Log $_ })
  $hashes = @(Get-PropertyOrDefault -Object $Track -Name "hashes" -DefaultValue @() | ForEach-Object { Convert-HashProjection -Hash $_ })
  $validators = @(Get-PropertyOrDefault -Object $Track -Name "validatorOutputs" -DefaultValue @() | ForEach-Object { Convert-ValidatorProjection -ValidatorOutput $_ })
  $forbiddenChecks = @(Get-PropertyOrDefault -Object $Track -Name "forbiddenSubstituteChecks" -DefaultValue @() | ForEach-Object { Convert-ForbiddenCheckProjection -Check $_ })

  [pscustomobject]@{
    recordId = "$trackId-record"
    sourceTrackId = $trackId
    proofKind = [string](Get-PropertyOrDefault -Object $Track -Name "proofKind" -DefaultValue "unknown-proof-kind")
    recordState = "blocked-real-proof-execution-record-input-required"
    ownerStatus = "owner-action-required"
    requiredExternalHost = $true
    requiredRuntimePackageKey = $true
    hostMetadata = [pscustomobject]@{
      hostOs = [string](Get-PropertyOrDefault -Object $Track -Name "hostOs" -DefaultValue "<owner-fill-host-os>")
      hostArchitecture = [string](Get-PropertyOrDefault -Object $Track -Name "hostArchitecture" -DefaultValue "<owner-fill-host-architecture>")
      cudaDriverVersion = [string](Get-PropertyOrDefault -Object $Track -Name "cudaDriverVersion" -DefaultValue "<owner-fill-cuda-driver-version>")
      cudaRuntimeVersion = [string](Get-PropertyOrDefault -Object $Track -Name "cudaRuntimeVersion" -DefaultValue "<owner-fill-cuda-runtime-version>")
      tensorRtVersion = [string](Get-PropertyOrDefault -Object $Track -Name "tensorRtVersion" -DefaultValue "<owner-fill-tensorrt-version>")
      runtimePackageKey = [string](Get-PropertyOrDefault -Object $Track -Name "runtimePackageKey" -DefaultValue "<owner-fill-runtime-package-key>")
    }
    execution = [pscustomobject]@{
      commands = @($commands)
      startedAtUtc = "<owner-fill-record-started-at-utc>"
      completedAtUtc = "<owner-fill-record-completed-at-utc>"
      workingDirectory = "<owner-fill-record-working-directory>"
      exitCodes = @($commands | ForEach-Object { $_.exitCode })
    }
    logs = @($logs)
    hashes = @($hashes)
    validatorOutputs = @($validators)
    forbiddenSubstituteChecks = @($forbiddenChecks)
    sourceArtifacts = @(Get-PropertyOrDefault -Object $Track -Name "sourceArtifacts" -DefaultValue @())
    promotionFlags = [pscustomobject]@{
      canPromoteRuntimeProof = $false
      canPublishPublicly = $false
      canCloseReleaseIssue = $false
      isRuntimeExecutionProof = $false
      isReleaseCloseProof = $false
    }
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    boundary = "This projected record is a strict owner-fill shape for future external proof. It is not proof, not a package publish, not post-publish verification, and not release-close approval."
  }
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Real proof runner input backfill template not found: $resolvedInputPath"
}

$inputRecord = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$trackInputs = @(Get-PropertyOrDefault -Object $inputRecord -Name "trackInputs" -DefaultValue @())
$records = @($trackInputs | ForEach-Object { New-ProofExecutionRecord -Track $_ })

$recordOut = [pscustomobject]@{
  recordKind = "real-proof-execution-record-projection"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  sourceInputPath = $resolvedInputPath
  sourceInputState = [string](Get-PropertyOrDefault -Object $inputRecord -Name "inputState" -DefaultValue "missing-real-proof-runner-input-backfill-state")
  projectionState = "blocked-real-proof-execution-record-input-required"
  recordCount = $records.Count
  blockedRecordCount = $records.Count
  readyRecordCount = 0
  proofExecutionRecords = @($records)
  forbiddenSubstitutes = @(Get-PropertyOrDefault -Object $inputRecord -Name "forbiddenSubstitutes" -DefaultValue @())
  sourceArtifacts = @(
    "artifacts/final-release/real-proof-runner-input-backfill.template.json",
    "artifacts/final-release/real-proof-runner-input-backfill-validation.json",
    "artifacts/final-release/real-external-proof-backfill-execution-bundle.json",
    "artifacts/final-release/release-close-proof-worklist.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This projection turns owner runner input placeholders into strict execution record shapes. It is not proof, not publish approval, not post-publish verification, and not release-close approval."
}

$jsonPath = Join-Path $OutputRoot "real-proof-execution-record-projection.json"
$markdownPath = Join-Path $OutputRoot "real-proof-execution-record-projection.md"
$recordOut | ConvertTo-Json -Depth 18 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Real Proof Execution Record Projection")
$lines.Add("")
$lines.Add("`real-proof-execution-record-projection` 将 runner input template 投影为严格执行记录形状，供 Owner 后续填入真实 host、commands、logs、hashes 和 validator output。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| projectionState | ``$(ConvertTo-MarkdownCell $recordOut.projectionState)`` |")
$lines.Add("| recordCount | ``$($recordOut.recordCount)`` |")
$lines.Add("| blockedRecordCount | ``$($recordOut.blockedRecordCount)`` |")
$lines.Add("| readyRecordCount | ``$($recordOut.readyRecordCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($recordOut.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($recordOut.canCloseReleaseIssue)`` |")
$lines.Add("| isRuntimeExecutionProof | ``$($recordOut.isRuntimeExecutionProof)`` |")
$lines.Add("| isReleaseCloseProof | ``$($recordOut.isReleaseCloseProof)`` |")
$lines.Add("")
$lines.Add("## Records")
$lines.Add("")
$lines.Add("| Record | Source Track | Proof Kind | State | Commands | Logs | Hashes |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($record in $records) {
  $lines.Add("| $(ConvertTo-MarkdownCell $record.recordId) | $(ConvertTo-MarkdownCell $record.sourceTrackId) | $(ConvertTo-MarkdownCell $record.proofKind) | $(ConvertTo-MarkdownCell $record.recordState) | ``$(@($record.execution.commands).Count)`` | ``$(@($record.logs).Count)`` | ``$(@($record.hashes).Count)`` |")
}
$lines.Add("")
$lines.Add("## Forbidden Substitutes")
$lines.Add("")
foreach ($item in $recordOut.forbiddenSubstitutes) {
  $lines.Add("- ``$(ConvertTo-MarkdownCell $item)``")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($recordOut.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Real proof execution record projection written to $jsonPath"
Write-Host "Real proof execution record projection markdown written to $markdownPath"
Write-Host "ProjectionState=$($recordOut.projectionState) Records=$($recordOut.recordCount) Blocked=$($recordOut.blockedRecordCount) Ready=$($recordOut.readyRecordCount)"
