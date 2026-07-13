[CmdletBinding()]
param(
  [string]$RuntimeProofInputPath = "artifacts\final-release\runtime-proof-execution-input-record.json",
  [string]$RuntimeProofInputValidationPath = "artifacts\final-release\runtime-proof-execution-input-record-validation.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

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

function New-RunbookItem {
  param([object]$ExecutionInput)

  $executionInputId = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "executionInputId" -DefaultValue "unknown-execution-input")
  $proofLane = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "proofLane" -DefaultValue "unknown-proof-lane")
  $commandLine = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "commandLine" -DefaultValue "")
  $validatorCommands = @(Get-PropertyOrDefault -Object $ExecutionInput -Name "validatorCommands" -DefaultValue @())

  [pscustomobject]@{
    runbookItemId = "$executionInputId-owner-runbook"
    executionInputId = $executionInputId
    candidateId = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "candidateId" -DefaultValue "")
    proofLane = $proofLane
    runtimePackageKey = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "runtimePackageKey" -DefaultValue "")
    commandSequence = @(
      [pscustomobject]@{
        order = 1
        command = "Prepare a clean proof working directory and record host metadata for $proofLane."
        expectedArtifacts = @("host metadata JSON", "working directory path record")
      },
      [pscustomobject]@{
        order = 2
        command = $commandLine
        expectedArtifacts = @(Get-PropertyOrDefault -Object $ExecutionInput -Name "expectedArtifacts" -DefaultValue @())
      },
      [pscustomobject]@{
        order = 3
        command = "Get-FileHash -Algorithm SHA256 <stdout> <stderr> <merged-transcript> <validator-output>"
        expectedArtifacts = @("stdout SHA256", "stderr SHA256", "merged transcript SHA256", "validator output SHA256")
      },
      [pscustomobject]@{
        order = 4
        command = (($validatorCommands | Select-Object -First 1) -as [string])
        expectedArtifacts = @("strict validator JSON", "strict validator markdown")
      }
    )
    requiredHashes = @(
      "nupkgSha256",
      "stdoutSha256",
      "stderrSha256",
      "mergedTranscriptSha256",
      "validatorOutputSha256"
    )
    validatorCommands = $validatorCommands
    nonSubstituteBoundary = [string](Get-PropertyOrDefault -Object $ExecutionInput -Name "nonSubstituteBoundary" -DefaultValue "")
    expectedArtifacts = @(Get-PropertyOrDefault -Object $ExecutionInput -Name "expectedArtifacts" -DefaultValue @())
    runbookState = "blocked-owner-runtime-proof-execution-required"
    ownerActionRequired = $true
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
  }
}

$resolvedInputPath = Resolve-InputPath -Path $RuntimeProofInputPath
$resolvedValidationPath = Resolve-InputPath -Path $RuntimeProofInputValidationPath
foreach ($path in @($resolvedInputPath, $resolvedValidationPath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Owner runtime proof runbook input not found: $path" }
}

$inputRecord = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$inputValidation = Get-Content -LiteralPath $resolvedValidationPath -Raw -Encoding utf8 | ConvertFrom-Json
$executionInputs = @(Get-PropertyOrDefault -Object $inputRecord -Name "executionInputs" -DefaultValue @())
$runbookItems = @($executionInputs | ForEach-Object { New-RunbookItem -ExecutionInput $_ })
$blockedRunbookItemCount = @($runbookItems | Where-Object { [string]$_.runbookState -eq "blocked-owner-runtime-proof-execution-required" }).Count

$runbook = [pscustomobject]@{
  recordKind = "owner-runtime-proof-execution-runbook"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  runtimeProofInputPath = $resolvedInputPath
  runtimeProofInputValidationPath = $resolvedValidationPath
  runtimeProofInputState = [string](Get-PropertyOrDefault -Object $inputRecord -Name "inputState" -DefaultValue "missing-runtime-proof-execution-input-record")
  runtimeProofInputValidationState = [string](Get-PropertyOrDefault -Object $inputValidation -Name "validationState" -DefaultValue "missing-runtime-proof-execution-input-record-validation")
  runbookState = "blocked-owner-runtime-proof-execution-required"
  runbookItemCount = $runbookItems.Count
  blockedRunbookItemCount = $blockedRunbookItemCount
  readyRunbookItemCount = 0
  runbookItems = @($runbookItems)
  sourceArtifacts = @(
    "artifacts/final-release/runtime-proof-execution-input-record.json",
    "artifacts/final-release/runtime-proof-execution-input-record-validation.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  boundary = "This runbook sequences owner execution commands and expected artifacts only. It is not runtime proof, package publish, post-publish verification, or release-close approval."
}

$jsonPath = Join-Path $OutputRoot "owner-runtime-proof-execution-runbook.json"
$markdownPath = Join-Path $OutputRoot "owner-runtime-proof-execution-runbook.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($runbook | ConvertTo-Json -Depth 18)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Runtime Proof Execution Runbook")
$lines.Add("")
$lines.Add("`owner-runtime-proof-execution-runbook` 将每条 runtime proof input 转成 Owner 可执行命令序列、预期产物和 hash 要求。它不是 proof。")
$lines.Add("")
$lines.Add("## Summary")
$lines.Add("")
$lines.Add("| Field | Value |")
$lines.Add("| --- | --- |")
$lines.Add("| runbookState | ``$(ConvertTo-MarkdownCell $runbook.runbookState)`` |")
$lines.Add("| runbookItemCount | ``$($runbook.runbookItemCount)`` |")
$lines.Add("| blockedRunbookItemCount | ``$($runbook.blockedRunbookItemCount)`` |")
$lines.Add("| readyRunbookItemCount | ``$($runbook.readyRunbookItemCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($runbook.canPromoteRuntimeProof)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($runbook.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Runbook Items")
$lines.Add("")
$lines.Add("| Item | Lane | State | Commands | Hashes |")
$lines.Add("| --- | --- | --- | ---: | ---: |")
foreach ($item in $runbookItems) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.runbookItemId) | $(ConvertTo-MarkdownCell $item.proofLane) | $(ConvertTo-MarkdownCell $item.runbookState) | ``$(@($item.commandSequence).Count)`` | ``$(@($item.requiredHashes).Count)`` |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($runbook.boundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner runtime proof execution runbook written to $jsonPath"
Write-Host "Owner runtime proof execution runbook markdown written to $markdownPath"
Write-Host "RunbookState=$($runbook.runbookState) Items=$($runbook.runbookItemCount) Blocked=$($runbook.blockedRunbookItemCount)"
