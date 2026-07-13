[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\owner-proof-input-readiness.json",
  [string]$OutputRoot,
  [string]$RepositoryRoot,
  [switch]$FailOnInvalid
)

$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "OwnerRealProofCommon.ps1")

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

function Resolve-InputPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
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
    return $Object.PSObject.Properties[$Name].Value
  }

  return $null
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

function New-ValidationItem {
  param(
    [string]$Id,
    [bool]$Passed,
    [string]$Severity,
    [string]$Detail,
    [string]$OwnerAction = "",
    [string]$Boundary = ""
  )

  [pscustomobject]@{
    id = $Id
    passed = $Passed
    severity = $Severity
    detail = $Detail
    ownerAction = $OwnerAction
    boundary = $Boundary
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Test-ArrayContainsText {
  param(
    [AllowNull()][object[]]$Values,
    [string]$Needle
  )

  if ($null -eq $Values -or $Values.Count -eq 0) {
    return $false
  }

  $joined = [string]::Join(" ", @($Values | ForEach-Object { [string]$_ }))
  return $joined.Contains($Needle, [System.StringComparison]::Ordinal)
}

$resolvedInputPath = Resolve-InputPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Owner proof input readiness record '$InputPath' was not found."
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}
elseif (-not [IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$items = New-Object System.Collections.Generic.List[object]

$recordKind = [string](Get-PropertyOrNull -Object $record -Name "recordKind")
$readinessState = [string](Get-PropertyOrNull -Object $record -Name "readinessState")
$contractCount = [int](Get-PropertyOrNull -Object $record -Name "contractCount")
$readyContractCount = [int](Get-PropertyOrNull -Object $record -Name "readyContractCount")
$blockedContractCount = [int](Get-PropertyOrNull -Object $record -Name "blockedContractCount")
$performsPublish = Test-Truthy (Get-PropertyOrNull -Object $record -Name "performsPublish")
$canPublishPublicly = Test-Truthy (Get-PropertyOrNull -Object $record -Name "canPublishPublicly")
$canCloseReleaseIssue = Test-Truthy (Get-PropertyOrNull -Object $record -Name "canCloseReleaseIssue")
$requiresHumanOwner = Test-Truthy (Get-PropertyOrNull -Object $record -Name "requiresHumanOwner")
$requiresCompatibleHost = Test-Truthy (Get-PropertyOrNull -Object $record -Name "requiresCompatibleHost")
$contracts = @(Get-PropertyOrNull -Object $record -Name "ownerInputContracts")
$sourceEvidence = @(Get-PropertyOrNull -Object $record -Name "sourceEvidence")
$nonSubstituteProofKinds = @(Get-PropertyOrNull -Object $record -Name "nonSubstituteProofKinds")

$expectedBlockers = @(
  "owner-authorization",
  "package-consumer-runtime",
  "linux-runner-proof",
  "real-model-runtime",
  "post-publish-verification"
)

$requiredContractProperties = @(
  "blockerId",
  "proofClass",
  "title",
  "currentState",
  "ready",
  "contractState",
  "requiredInputFiles",
  "templateFiles",
  "replaceOrFillInstructions",
  "validatorCommand",
  "successCriteria",
  "nonSubstitutes",
  "expectedOutputArtifacts",
  "sourceGuidance",
  "canPromoteOnPass",
  "performsPublish",
  "canPublishPublicly",
  "canCloseReleaseIssue",
  "ownerAction"
)

$requiredNonSubstitutes = @(
  "template",
  "draft",
  "checklist",
  "dashboard",
  "runbook",
  "collection package",
  "input package",
  "local feed",
  "ProjectReference",
  "DependencyProbe",
  "dependency-probe-only",
  "sidecar-only",
  "build-only",
  "parse-only",
  "Skipped=True",
  "blocked-by-cuda-driver"
)

$expectedValidatorMarkers = @{
  "owner-authorization" = @("Test-ReleaseOwnerApprovalInput.ps1", "Test-OwnerAuthorizedPublishCommandPlan.ps1")
  "package-consumer-runtime" = @("Test-ExternalRuntimeProofRecord.ps1", "-RequireExistingLog", "-FailOnNotProof")
  "linux-runner-proof" = @("Test-LinuxRunnerEvidenceRecord.ps1")
  "real-model-runtime" = @("Test-SampleAssetManifest.ps1", "Test-SampleRunEvidenceRecord.ps1", "-RequireExistingLog")
  "post-publish-verification" = @("Test-PostPublishVerificationRecord.ps1", "-RequireExistingLog", "-FailOnNotProof")
}

$items.Add((New-ValidationItem -Id "record-kind" -Passed ($recordKind -eq "owner-proof-input-readiness") -Severity "blocker" -Detail "recordKind must be owner-proof-input-readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "readiness-state" -Passed ($readinessState -eq "blocked-real-proof-input-required") -Severity "blocker" -Detail "readinessState must remain blocked-real-proof-input-required until real owner proof inputs exist.")) | Out-Null
$items.Add((New-ValidationItem -Id "contract-counts" -Passed ($contractCount -eq 5 -and $readyContractCount -eq 0 -and $blockedContractCount -eq 5 -and $contracts.Count -eq 5) -Severity "blocker" -Detail "The readiness record must expose exactly five blocked owner input contracts and zero ready contracts.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-side-effects" -Passed (-not $performsPublish -and -not $canPublishPublicly -and -not $canCloseReleaseIssue) -Severity "blocker" -Detail "The readiness record must not publish, allow public publish, or allow release close.")) | Out-Null
$items.Add((New-ValidationItem -Id "owner-host-required" -Passed ($requiresHumanOwner -and $requiresCompatibleHost) -Severity "blocker" -Detail "The readiness record must require both human owner input and compatible host proof input.")) | Out-Null

foreach ($marker in $requiredNonSubstitutes) {
  $items.Add((New-ValidationItem -Id "global-non-substitute-$($marker.Replace(' ', '-').Replace('=', '-'))" -Passed (Test-ArrayContainsText -Values $nonSubstituteProofKinds -Needle $marker) -Severity "blocker" -Detail "Global nonSubstituteProofKinds must include '$marker'.")) | Out-Null
}

foreach ($source in @(
    "artifacts/final-release/owner-release-execution-package.json",
    "artifacts/final-release/compatible-host-proof-execution-pack.json",
    "artifacts/final-release/release-proof-readiness-snapshot.json",
    "artifacts/final-release/real-model-and-package-proof-input-package.json",
    "artifacts/final-release/external-runtime-proof-collection-package.json",
    "artifacts/final-release/post-publish-verification-collection-package.json",
    "artifacts/final-release/release-owner-approval-input-validation.json",
    "artifacts/final-release/owner-authorized-publish-command-plan-validation.json",
    "artifacts/final-release/external-runtime-proof-validation.json",
    "artifacts/user-acceptance/sample-run-evidence-record-validation.json",
    "artifacts/final-release/post-publish-verification-validation.json"
  )) {
  $items.Add((New-ValidationItem -Id "source-$([IO.Path]::GetFileNameWithoutExtension($source))" -Passed ($sourceEvidence -contains $source) -Severity "blocker" -Detail "sourceEvidence must include '$source'.")) | Out-Null
}

foreach ($blocker in $expectedBlockers) {
  $contract = @($contracts | Where-Object { [string](Get-PropertyOrNull -Object $_ -Name "blockerId") -eq $blocker }) | Select-Object -First 1
  $items.Add((New-ValidationItem -Id "contract-present-$blocker" -Passed ($null -ne $contract) -Severity "blocker" -Detail "Contract '$blocker' must be present.")) | Out-Null
  if ($null -eq $contract) {
    continue
  }

  foreach ($propertyName in $requiredContractProperties) {
    $items.Add((New-ValidationItem -Id "contract-$blocker-property-$propertyName" -Passed ($contract.PSObject.Properties.Name -contains $propertyName) -Severity "blocker" -Detail "Contract '$blocker' must expose '$propertyName'.")) | Out-Null
  }

  $proofClass = [string](Get-PropertyOrNull -Object $contract -Name "proofClass")
  $ready = Test-Truthy (Get-PropertyOrNull -Object $contract -Name "ready")
  $contractState = [string](Get-PropertyOrNull -Object $contract -Name "contractState")
  $requiredInputFiles = @(Get-PropertyOrNull -Object $contract -Name "requiredInputFiles")
  $templateFiles = @(Get-PropertyOrNull -Object $contract -Name "templateFiles")
  $replaceOrFillInstructions = @(Get-PropertyOrNull -Object $contract -Name "replaceOrFillInstructions")
  $successCriteria = @(Get-PropertyOrNull -Object $contract -Name "successCriteria")
  $contractNonSubstitutes = @(Get-PropertyOrNull -Object $contract -Name "nonSubstitutes")
  $expectedOutputArtifacts = @(Get-PropertyOrNull -Object $contract -Name "expectedOutputArtifacts")
  $sourceGuidance = @(Get-PropertyOrNull -Object $contract -Name "sourceGuidance")
  $validatorCommand = [string](Get-PropertyOrNull -Object $contract -Name "validatorCommand")
  $canPromoteOnPass = Test-Truthy (Get-PropertyOrNull -Object $contract -Name "canPromoteOnPass")
  $contractPerformsPublish = Test-Truthy (Get-PropertyOrNull -Object $contract -Name "performsPublish")
  $contractCanPublish = Test-Truthy (Get-PropertyOrNull -Object $contract -Name "canPublishPublicly")
  $contractCanClose = Test-Truthy (Get-PropertyOrNull -Object $contract -Name "canCloseReleaseIssue")

  $items.Add((New-ValidationItem -Id "contract-$blocker-proof-class" -Passed ($proofClass -eq $blocker) -Severity "blocker" -Detail "Contract '$blocker' proofClass must match blockerId.")) | Out-Null
  $items.Add((New-ValidationItem -Id "contract-$blocker-blocked-state" -Passed (-not $ready -and $contractState -eq "blocked-real-owner-input-required") -Severity "blocker" -Detail "Contract '$blocker' must remain blocked until real owner input passes validators.")) | Out-Null
  $items.Add((New-ValidationItem -Id "contract-$blocker-no-publish" -Passed (-not $canPromoteOnPass -and -not $contractPerformsPublish -and -not $contractCanPublish -and -not $contractCanClose) -Severity "blocker" -Detail "Contract '$blocker' cannot promote, publish, or close by itself.")) | Out-Null
  $items.Add((New-ValidationItem -Id "contract-$blocker-input-depth" -Passed ($requiredInputFiles.Count -ge 2 -and $templateFiles.Count -ge 2 -and $replaceOrFillInstructions.Count -ge 3 -and $successCriteria.Count -ge 3 -and $expectedOutputArtifacts.Count -ge 2 -and $sourceGuidance.Count -ge 1) -Severity "blocker" -Detail "Contract '$blocker' must include actionable inputs, templates, instructions, success criteria, outputs, and guidance.")) | Out-Null
  $items.Add((New-ValidationItem -Id "contract-$blocker-validator-command" -Passed (-not [string]::IsNullOrWhiteSpace($validatorCommand)) -Severity "blocker" -Detail "Contract '$blocker' must include a validatorCommand.")) | Out-Null

  foreach ($marker in $expectedValidatorMarkers[$blocker]) {
    $items.Add((New-ValidationItem -Id "contract-$blocker-validator-$($marker.Replace('.', '-').Replace(' ', '-'))" -Passed ($validatorCommand.Contains($marker, [System.StringComparison]::Ordinal)) -Severity "blocker" -Detail "Contract '$blocker' validatorCommand must include '$marker'.")) | Out-Null
  }

  foreach ($marker in $requiredNonSubstitutes) {
    $items.Add((New-ValidationItem -Id "contract-$blocker-non-substitute-$($marker.Replace(' ', '-').Replace('=', '-'))" -Passed (Test-ArrayContainsText -Values $contractNonSubstitutes -Needle $marker) -Severity "blocker" -Detail "Contract '$blocker' nonSubstitutes must include '$marker'.")) | Out-Null
  }
}

$failedBlockers = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "valid-owner-proof-input-readiness" } else { "invalid-owner-proof-input-readiness" }
$isValidOwnerProofInputReadiness = $failedBlockers.Count -eq 0

$summary = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "owner-proof-input-readiness-validation"
  inputPath = $InputPath
  validationState = $validationState
  isValidOwnerProofInputReadiness = $isValidOwnerProofInputReadiness
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  contractCount = $contractCount
  readyContractCount = $readyContractCount
  blockedContractCount = $blockedContractCount
  failedBlockerCount = $failedBlockers.Count
  validationItems = @($items.ToArray())
  boundary = "This validator checks the owner proof input readiness contract only. It does not create proof, publish packages, upload assets, or close the release issue."
}

$jsonPath = Join-Path $OutputRoot "owner-proof-input-readiness-validation.json"
$markdownPath = Join-Path $OutputRoot "owner-proof-input-readiness-validation.md"

Write-Utf8File -LiteralPath $jsonPath -InputObject ($summary | ConvertTo-Json -Depth 10)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner Proof Input Readiness Validation")
$lines.Add("")
$lines.Add("- generated: ``$($summary.generatedAtUtc)``")
$lines.Add("- validation state: ``$validationState``")
$lines.Add("- valid owner proof input readiness: ``$isValidOwnerProofInputReadiness``")
$lines.Add("- contract count: ``$contractCount``")
$lines.Add("- ready contract count: ``$readyContractCount``")
$lines.Add("- blocked contract count: ``$blockedContractCount``")
$lines.Add("- failed blocker count: ``$($failedBlockers.Count)``")
$lines.Add("- performs publish: ``False``")
$lines.Add("- can publish publicly: ``False``")
$lines.Add("- can close release issue: ``False``")
$lines.Add("")
$lines.Add("## Validation Items")
$lines.Add("")
$lines.Add("| id | passed | severity | detail |")
$lines.Add("| --- | --- | --- | --- |")
foreach ($item in $items) {
  $lines.Add("| ``$($item.id)`` | ``$($item.passed)`` | ``$($item.severity)`` | $(ConvertTo-MarkdownCell $item.detail) |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($summary.boundary)

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner proof input readiness validation written to $jsonPath"
Write-Host "Owner proof input readiness validation written to $markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count) PerformsPublish=False CanPublishPublicly=False CanCloseReleaseIssue=False"

if ($FailOnInvalid -and -not $isValidOwnerProofInputReadiness) {
  Write-Error "Owner proof input readiness validation failed. ValidationState=$validationState"
  exit 1
}
