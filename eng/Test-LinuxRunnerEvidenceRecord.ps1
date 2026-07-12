[CmdletBinding()]
param(
  [Parameter(Mandatory = $true)]
  [string]$RuntimePackageKey,
  [string]$EvidenceRecordPath,
  [string]$RepositoryRoot,
  [switch]$FailOnNotProof
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

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Test-Truthy {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return $false
  }

  if ($Value -is [bool]) {
    return [bool]$Value
  }

  return [bool]::Parse([string]$Value)
}

function Get-PropertyOrNull {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  if ($null -eq $Object) {
    return $null
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.$Name
  }

  return $null
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

$manifestPath = Join-Path $RepositoryRoot "pack\runtime\runtime-packages.manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw -Encoding utf8 | ConvertFrom-Json
$package = $manifest.packages | Where-Object { $_.key -eq $RuntimePackageKey } | Select-Object -First 1
if (-not $package) {
  throw "Runtime package key '$RuntimePackageKey' was not found."
}

if ([string]$package.platform -ne "linux") {
  throw "Runtime package key '$RuntimePackageKey' is not a Linux package."
}

if ([string]::IsNullOrWhiteSpace($EvidenceRecordPath)) {
  $candidateRecord = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey\linux-runner-evidence-record.json"
  $candidateTemplate = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey\linux-runner-evidence-record-template.json"
  if (Test-Path -LiteralPath $candidateRecord -PathType Leaf) {
    $EvidenceRecordPath = $candidateRecord
  }
  else {
    $EvidenceRecordPath = $candidateTemplate
  }
}
elseif (-not [System.IO.Path]::IsPathRooted($EvidenceRecordPath)) {
  $EvidenceRecordPath = Join-Path $RepositoryRoot $EvidenceRecordPath
}

if (-not (Test-Path -LiteralPath $EvidenceRecordPath -PathType Leaf)) {
  throw "Linux runner evidence record '$EvidenceRecordPath' was not found."
}

$record = Get-Content -LiteralPath $EvidenceRecordPath -Raw -Encoding utf8 | ConvertFrom-Json

$recordKind = [string](Get-PropertyOrNull -Object $record -Name "recordKind")
$templateOnlyDeclared = Test-Truthy (Get-PropertyOrNull -Object $record -Name "templateOnly")
$recordState = [string](Get-PropertyOrNull -Object $record -Name "recordState")
$executionState = [string](Get-PropertyOrNull -Object $record -Name "executionState")
$recordRuntimeKey = [string](Get-PropertyOrNull -Object $record -Name "runtimeKey")
if ([string]::IsNullOrWhiteSpace($recordRuntimeKey)) {
  $recordRuntimeKey = [string](Get-PropertyOrNull -Object $record -Name "runtimePackageKey")
}

$runner = Get-PropertyOrNull -Object $record -Name "runner"
$packageOutput = Get-PropertyOrNull -Object $record -Name "packageOutput"
$commands = @(Get-PropertyOrNull -Object $record -Name "commands")
$runnerOwner = [string](Get-PropertyOrNull -Object $record -Name "runnerOwner")
$evidenceRationale = [string](Get-PropertyOrNull -Object $record -Name "evidenceRationale")

$runnerIsLinux = Test-Truthy (Get-PropertyOrNull -Object $runner -Name "isLinux")
$runnerIsX64 = Test-Truthy (Get-PropertyOrNull -Object $runner -Name "isX64")
$declaredProof = Test-Truthy (Get-PropertyOrNull -Object $record -Name "isRealLinuxRunnerProof")
$declaredPromotable = Test-Truthy (Get-PropertyOrNull -Object $record -Name "canPromoteLinuxPackage")

$nativeMissing = Get-PropertyOrNull -Object $packageOutput -Name "nativeAssetMissingCount"
$nativeCopied = Get-PropertyOrNull -Object $packageOutput -Name "nativeAssetCopiedCount"
$consumerStatus = [string](Get-PropertyOrNull -Object $packageOutput -Name "packageConsumerStatus")
$optionalSmokeStatus = [string](Get-PropertyOrNull -Object $packageOutput -Name "optionalSmokeStatus")
$runtimeNupkgPath = [string](Get-PropertyOrNull -Object $packageOutput -Name "runtimeNupkgPath")
$runtimeNupkgSha256 = [string](Get-PropertyOrNull -Object $packageOutput -Name "runtimeNupkgSha256")

$requiredCommandIds = @(
  "validate-linux-runtime-inputs",
  "cmake-configure",
  "cmake-build",
  "collect-runtime-assets",
  "pack-runtime-nupkg",
  "package-consumer-copy"
)

$commandById = @{}
foreach ($command in $commands) {
  $id = [string](Get-PropertyOrNull -Object $command -Name "id")
  if (-not [string]::IsNullOrWhiteSpace($id)) {
    $commandById[$id] = $command
  }
}

$missingCommandIds = @($requiredCommandIds | Where-Object { -not $commandById.ContainsKey($_) })
$failedCommandIds = New-Object System.Collections.Generic.List[string]
$pendingCommandIds = New-Object System.Collections.Generic.List[string]
$missingCommandEvidenceIds = New-Object System.Collections.Generic.List[string]

foreach ($id in $requiredCommandIds) {
  if (-not $commandById.ContainsKey($id)) {
    continue
  }

  $command = $commandById[$id]
  $status = [string](Get-PropertyOrNull -Object $command -Name "status")
  $actualExitCode = Get-PropertyOrNull -Object $command -Name "actualExitCode"
  $logPath = [string](Get-PropertyOrNull -Object $command -Name "logPath")

  if ($status -in @("pending-linux-runner-execution", "pending", "not-run", "")) {
    $pendingCommandIds.Add($id) | Out-Null
  }
  elseif ($status -notin @("succeeded", "passed", "ready")) {
    $failedCommandIds.Add($id) | Out-Null
  }

  if ($null -ne $actualExitCode -and [int]$actualExitCode -ne 0) {
    if (-not $failedCommandIds.Contains($id)) {
      $failedCommandIds.Add($id) | Out-Null
    }
  }

  if ([string]::IsNullOrWhiteSpace($logPath)) {
    $missingCommandEvidenceIds.Add($id) | Out-Null
  }
}

$nativeMissingIsZero = $false
if ($null -ne $nativeMissing) {
  $nativeMissingIsZero = [int]$nativeMissing -eq 0
}

$nativeCopiedPresent = $false
if ($null -ne $nativeCopied) {
  $nativeCopiedPresent = [int]$nativeCopied -gt 0
}

$consumerReady = $consumerStatus -in @("ready", "passed", "dependency-probe-passed", "native-copy-passed", "package-consumer-passed")
$hasRuntimeNupkg = -not [string]::IsNullOrWhiteSpace($runtimeNupkgPath)
$hasSha256 = -not [string]::IsNullOrWhiteSpace($runtimeNupkgSha256)
$isTemplateOnly = $templateOnlyDeclared -or $recordKind -eq "linux-runner-evidence-record-template" -or $recordState -eq "template-only" -or $executionState -eq "pending-linux-runner-execution"

$validationItems = @(
  New-ValidationItem -Id "record-kind" -Passed ($recordKind -in @("linux-runner-evidence-record", "linux-runner-evidence-record-template")) -Severity "blocker" -Detail "recordKind must identify a Linux runner evidence record or template."
  New-ValidationItem -Id "real-record-kind" -Passed ($recordKind -eq "linux-runner-evidence-record" -and -not $templateOnlyDeclared) -Severity "proof-required" -Detail "Real proof requires recordKind=linux-runner-evidence-record and templateOnly=false."
  New-ValidationItem -Id "runtime-key-match" -Passed ([string]::Equals($recordRuntimeKey, $RuntimePackageKey, [System.StringComparison]::Ordinal)) -Severity "blocker" -Detail "Record runtime key must match the requested runtime package key."
  New-ValidationItem -Id "runner-owner" -Passed (-not [string]::IsNullOrWhiteSpace($runnerOwner)) -Severity "proof-required" -Detail "Real proof requires runnerOwner."
  New-ValidationItem -Id "evidence-rationale" -Passed (-not [string]::IsNullOrWhiteSpace($evidenceRationale)) -Severity "proof-required" -Detail "Real proof requires evidenceRationale."
  New-ValidationItem -Id "runner-linux-x64" -Passed ($runnerIsLinux -and $runnerIsX64) -Severity "proof-required" -Detail "Real Linux proof requires a Linux x64 runner."
  New-ValidationItem -Id "required-commands-present" -Passed ($missingCommandIds.Count -eq 0) -Severity "blocker" -Detail "Required command records must be present."
  New-ValidationItem -Id "required-commands-succeeded" -Passed ($failedCommandIds.Count -eq 0 -and $pendingCommandIds.Count -eq 0) -Severity "proof-required" -Detail "Required commands must have succeeded and not be pending."
  New-ValidationItem -Id "command-evidence-paths" -Passed ($missingCommandEvidenceIds.Count -eq 0) -Severity "proof-required" -Detail "Required commands must include log paths or equivalent evidence references."
  New-ValidationItem -Id "runtime-nupkg" -Passed ($hasRuntimeNupkg -and $hasSha256) -Severity "proof-required" -Detail "Runtime package path and SHA256 must be present."
  New-ValidationItem -Id "native-assets" -Passed ($nativeCopiedPresent -and $nativeMissingIsZero) -Severity "proof-required" -Detail "Native asset copied count must be present and missing count must be zero."
  New-ValidationItem -Id "package-consumer" -Passed $consumerReady -Severity "proof-required" -Detail "Package consumer restore/build/native-copy evidence must be ready."
)

$failedBlockers = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$failedProofItems = @($validationItems | Where-Object { -not $_.passed -and $_.severity -eq "proof-required" })

$canPromoteLinuxPackage = $failedBlockers.Count -eq 0 -and $failedProofItems.Count -eq 0 -and -not $isTemplateOnly
$isRealLinuxRunnerProof = $canPromoteLinuxPackage -and $declaredProof -and $declaredPromotable

if ($isRealLinuxRunnerProof) {
  $validationState = "real-linux-runner-proof"
}
elseif ($isTemplateOnly) {
  $validationState = "template-only"
}
elseif ($failedBlockers.Count -gt 0) {
  $validationState = "invalid-record"
}
else {
  $validationState = "incomplete-linux-runner-evidence"
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\linux-dry-run\$RuntimePackageKey"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "linux-runner-evidence-validation.json"
$markdownPath = Join-Path $outputRoot "linux-runner-evidence-validation.md"

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationKind = "linux-runner-evidence-validation"
  runtimePackageKey = $RuntimePackageKey
  evidenceRecordPath = $EvidenceRecordPath
  validationState = $validationState
  isTemplateOnly = $isTemplateOnly
  inputTemplateOnly = $templateOnlyDeclared
  isRealLinuxRunnerProof = $isRealLinuxRunnerProof
  canPromoteLinuxPackage = $canPromoteLinuxPackage
  runnerOwnerPresent = -not [string]::IsNullOrWhiteSpace($runnerOwner)
  evidenceRationalePresent = -not [string]::IsNullOrWhiteSpace($evidenceRationale)
  declaredProof = $declaredProof
  declaredPromotable = $declaredPromotable
  runnerIsLinux = $runnerIsLinux
  runnerIsX64 = $runnerIsX64
  requiredCommandCount = $requiredCommandIds.Count
  missingCommandIds = @($missingCommandIds)
  pendingCommandIds = @($pendingCommandIds.ToArray())
  failedCommandIds = @($failedCommandIds.ToArray())
  missingCommandEvidenceIds = @($missingCommandEvidenceIds.ToArray())
  runtimeNupkgPath = $runtimeNupkgPath
  runtimeNupkgSha256Present = $hasSha256
  nativeAssetCopiedCount = $nativeCopied
  nativeAssetMissingCount = $nativeMissing
  packageConsumerStatus = $consumerStatus
  optionalSmokeStatus = $optionalSmokeStatus
  validationItems = @($validationItems)
  promotionRules = @(
    "Template-only records must keep isRealLinuxRunnerProof=false.",
    "Real records must set recordKind=linux-runner-evidence-record, templateOnly=false, runnerOwner, and evidenceRationale.",
    "A Linux x64 runner must execute the required commands before proof can be promoted.",
    "Package consumer native-copy evidence is required before canPromoteLinuxPackage=true.",
    "GPU smoke is separate from packaging proof and must not be inferred from dependency probes.",
    "Real callback runtime proof remains separate and requires InvocationCount>0 plus IsRealCallbackRuntimeProof=True."
  )
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Linux Runner Evidence Validation")
$lines.Add("")
$lines.Add("- runtime key: ``$RuntimePackageKey``")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- template only: ``$isTemplateOnly``")
$lines.Add("- real Linux runner proof: ``$isRealLinuxRunnerProof``")
$lines.Add("- can promote Linux package: ``$canPromoteLinuxPackage``")
$lines.Add("- evidence record: ``$EvidenceRecordPath``")
$lines.Add("")
$lines.Add("## Validation Items")
$lines.Add("")
$lines.Add("| ID | Passed | Severity | Detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $validationItems) {
  $lines.Add("| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |")
}
$lines.Add("")
$lines.Add("## Command Evidence")
$lines.Add("")
$lines.Add("- missing commands: ``$([string]::Join(', ', @($missingCommandIds)))``")
$lines.Add("- pending commands: ``$([string]::Join(', ', @($pendingCommandIds.ToArray())))``")
$lines.Add("- failed commands: ``$([string]::Join(', ', @($failedCommandIds.ToArray())))``")
$lines.Add("- missing command evidence: ``$([string]::Join(', ', @($missingCommandEvidenceIds.ToArray())))``")
$lines.Add("")
$lines.Add("## Promotion Rules")
$lines.Add("")
foreach ($rule in $summary.promotionRules) {
  $lines.Add("- $rule")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Linux runner evidence validation written to $jsonPath"
Write-Host "Linux runner evidence validation written to $markdownPath"
Write-Host "ValidationState=$validationState IsRealLinuxRunnerProof=$isRealLinuxRunnerProof CanPromoteLinuxPackage=$canPromoteLinuxPackage"

if ($FailOnNotProof.IsPresent -and -not $isRealLinuxRunnerProof) {
  Write-Error "Linux runner evidence is not real proof. ValidationState=$validationState"
  exit 1
}
