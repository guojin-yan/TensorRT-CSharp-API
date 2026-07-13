[CmdletBinding()]
param(
  [string]$RepositoryRoot
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

function Read-JsonOrNull {
  param([string]$RelativePath)
  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { return $null }
  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Find-PackageConsumerRuntimeProofPreflightEntry {
  param(
    [AllowNull()][object]$Matrix,
    [string]$RuntimePackageKey
  )

  if ($null -eq $Matrix) {
    return $null
  }

  foreach ($entry in @($Matrix.entries)) {
    if ([string]::Equals([string](Get-PropertyOrDefault -Object $entry -Name "runtimePackageKey" -DefaultValue ""), $RuntimePackageKey, [System.StringComparison]::OrdinalIgnoreCase)) {
      return $entry
    }
  }

  return $null
}

function ConvertTo-PreflightOption {
  param([object]$Entry)

  [pscustomobject]@{
    runtimePackageKey = [string](Get-PropertyOrDefault -Object $Entry -Name "runtimePackageKey" -DefaultValue "")
    runtimePackageId = [string](Get-PropertyOrDefault -Object $Entry -Name "runtimePackageId" -DefaultValue "")
    restoreSourceMode = [string](Get-PropertyOrDefault -Object $Entry -Name "restoreSourceMode" -DefaultValue "")
    nativeAssetCopyExpected = Get-PropertyOrDefault -Object $Entry -Name "nativeAssetCopyExpected" -DefaultValue $null
    validatorCommand = [string](Get-PropertyOrDefault -Object $Entry -Name "validatorCommand" -DefaultValue "")
    ownerActionRequired = [bool](Get-PropertyOrDefault -Object $Entry -Name "ownerActionRequired" -DefaultValue $true)
    canPromotePackageConsumerRuntimeProof = [bool](Get-PropertyOrDefault -Object $Entry -Name "canPromotePackageConsumerRuntimeProof" -DefaultValue $false)
  }
}

function New-ExecutionBundleItem {
  param(
    [object]$ResultInput,
    [object[]]$CloseDryRunItems,
    [AllowNull()][object]$PreflightMatrix
  )

  $resultInputId = [string](Get-PropertyOrDefault -Object $ResultInput -Name "resultInputId" -DefaultValue "unknown-result-input")
  $executionInputId = [string](Get-PropertyOrDefault -Object $ResultInput -Name "executionInputId" -DefaultValue "unknown-execution-input")
  $candidateId = [string](Get-PropertyOrDefault -Object $ResultInput -Name "candidateId" -DefaultValue "unknown-candidate")
  $proofLane = [string](Get-PropertyOrDefault -Object $ResultInput -Name "proofLane" -DefaultValue "unknown-proof-lane")
  $runtimePackageKey = [string](Get-PropertyOrDefault -Object $ResultInput -Name "runtimePackageKey" -DefaultValue "unknown-runtime-package-key")
  $preflightEntry = Find-PackageConsumerRuntimeProofPreflightEntry -Matrix $PreflightMatrix -RuntimePackageKey $runtimePackageKey
  $preflightOptions = @(@(Get-PropertyOrDefault -Object $PreflightMatrix -Name "entries" -DefaultValue @()) | ForEach-Object { ConvertTo-PreflightOption -Entry $_ })
  $requiresRuntimePackagePreflightSelection = [string]::Equals($proofLane, "package-consumer-runtime", [System.StringComparison]::OrdinalIgnoreCase)
  if ($requiresRuntimePackagePreflightSelection -and $null -eq $preflightEntry) {
    $preflightEntry = @(@(Get-PropertyOrDefault -Object $PreflightMatrix -Name "entries" -DefaultValue @()) |
      Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "runtimePackageKey" -DefaultValue "") -eq "win-x64-trt11.0-cuda13.2-cudnn9.22" } |
      Select-Object -First 1)[0]
  }
  $preflightRuntimePackageId = [string](Get-PropertyOrDefault -Object $preflightEntry -Name "runtimePackageId" -DefaultValue "")
  $preflightRestoreSourceMode = [string](Get-PropertyOrDefault -Object $preflightEntry -Name "restoreSourceMode" -DefaultValue "")
  $selectedRuntimePackageKey = [string](Get-PropertyOrDefault -Object $preflightEntry -Name "runtimePackageKey" -DefaultValue "")
  $preflightNativeAssetCopyExpected = Get-PropertyOrDefault -Object $preflightEntry -Name "nativeAssetCopyExpected" -DefaultValue $null
  $preflightValidatorCommand = [string](Get-PropertyOrDefault -Object $preflightEntry -Name "validatorCommand" -DefaultValue "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ExternalRuntimeProofRecord.ps1 -RequireExistingLog -FailOnNotProof")
  $preflightForbiddenSubstitutes = @(Get-PropertyOrDefault -Object $preflightEntry -Name "forbiddenProofSubstitutes" -DefaultValue @(
      "readonly summary",
      "bridge-only",
      "dependency probe",
      "local feed",
      "ProjectReference",
      "direct .nupkg",
      "build-only",
      "dry-run",
      "template",
      "blocked-by-cuda-driver"
    ))
  $closeItem = $CloseDryRunItems | Where-Object { [string](Get-PropertyOrDefault -Object $_ -Name "executionInputId" -DefaultValue "") -eq $executionInputId } | Select-Object -First 1
  $requiredResultFields = @(Get-PropertyOrDefault -Object $ResultInput -Name "requiredResultFields" -DefaultValue @())
  $nonSubstituteConfirmations = @(Get-PropertyOrDefault -Object $ResultInput -Name "nonSubstituteConfirmations" -DefaultValue @())
  $validatorCommands = @(Get-PropertyOrDefault -Object $ResultInput -Name "validatorCommands" -DefaultValue @())
  if ($validatorCommands -notcontains $preflightValidatorCommand) {
    $validatorCommands += $preflightValidatorCommand
  }
  $expectedArtifacts = @(Get-PropertyOrDefault -Object $ResultInput -Name "expectedArtifacts" -DefaultValue @())

  $laneDirectory = "artifacts/external-proof/$proofLane/$runtimePackageKey"
  $commandSequence = @(
    "New-Item -ItemType Directory -Force -Path `"$laneDirectory`"",
    "Record host metadata: OS, arch, GPU, driver, CUDA, TensorRT, .NET SDK/runtime.",
    "Select one RuntimeProofPreflight option when this lane consumes a runtime package; default selection is $selectedRuntimePackageKey. Copy runtimePackageKey, runtimePackageId, restoreSourceMode, and nativeAssetCopyExpected into the owner result input.",
    "Install package from the required public or release channel source for $runtimePackageKey.",
    "Run the lane command from a clean working directory and tee stdout/stderr to files.",
    "Merge stdout/stderr into a transcript and compute SHA256 for every captured file.",
    "Run strict validators and capture validator output plus SHA256.",
    "Copy the completed result JSON to owner-external-proof-execution-result.input.json."
  )
  $hashCommands = @(
    "Get-FileHash -Algorithm SHA256 <stdoutPath>",
    "Get-FileHash -Algorithm SHA256 <stderrPath>",
    "Get-FileHash -Algorithm SHA256 <mergedTranscriptPath>",
    "Get-FileHash -Algorithm SHA256 <validatorOutputPath>",
    "Get-FileHash -Algorithm SHA256 <nupkgPath>"
  )
  $importTargetFieldMap = $requiredResultFields | ForEach-Object {
    [pscustomobject]@{
      sourceField = $_
      resultInputId = $resultInputId
      importTarget = "artifacts/final-release/owner-external-proof-execution-result.input.json"
      required = $true
      substituteAllowed = $false
    }
  }
  $remainingGaps = @(Get-PropertyOrDefault -Object $closeItem -Name "remainingGaps" -DefaultValue @())

  [pscustomobject]@{
    executionBundleItemId = "$executionInputId-owner-external-proof-execution-bundle-item"
    resultInputId = $resultInputId
    executionInputId = $executionInputId
    candidateId = $candidateId
    proofLane = $proofLane
    runtimePackageKey = $runtimePackageKey
    runtimeProofPreflight = [pscustomobject]@{
      matrixFound = $null -ne $PreflightMatrix
      entryFound = $null -ne $preflightEntry
      requiresRuntimePackagePreflightSelection = $requiresRuntimePackagePreflightSelection
      availableRuntimePackageOptionCount = $preflightOptions.Count
      selectedRuntimePackageKey = $selectedRuntimePackageKey
      selectedRuntimePackageId = $preflightRuntimePackageId
      selectedRestoreSourceMode = $preflightRestoreSourceMode
      selectedNativeAssetCopyExpected = $preflightNativeAssetCopyExpected
      runtimePackageOptions = @($preflightOptions)
      validatorCommand = $preflightValidatorCommand
      ownerActionRequired = $true
      canPromotePackageConsumerRuntimeProof = $false
      boundary = "RuntimeProofPreflight is an owner-action-required audit contract, not a proof promotion source."
    }
    bundleItemState = "blocked-owner-external-proof-execution-required"
    cleanWorkingDirectoryPlan = $laneDirectory
    commandSequence = $commandSequence
    expectedStdoutPath = "$laneDirectory/stdout.log"
    expectedStderrPath = "$laneDirectory/stderr.log"
    expectedMergedTranscriptPath = "$laneDirectory/transcript.log"
    expectedValidatorOutputPath = "$laneDirectory/validator-output.json"
    requiredSha256Commands = $hashCommands
    packageSourceRequirements = @(
      "For package-consumer-runtime proof, choose one RuntimeProofPreflight option; default is $selectedRuntimePackageKey. Copy runtimePackageKey, runtimePackageId, restoreSourceMode, and nativeAssetCopyExpected into the owner result input.",
      "Record results.nativeAssetsExpected from the selected RuntimeProofPreflight option and results.nativeAssetsFound from the clean consumer output.",
      "Use public/release package source when the lane requires post-publish or package-consumer proof.",
      "Do not substitute ProjectReference, local feed, direct nupkg, build-only output, or DependencyProbe-only output."
    )
    hostMetadataChecklist = @("os", "arch", "gpu", "driverVersion", "cudaVersion", "tensorRtVersion", "dotnetVersion")
    runtimeProofPreflightRequiredFields = @(
      "packageSource.runtimePackageKey",
      "packageSource.runtimePackageId",
      "packageSource.restoreSourceMode",
      "results.nativeAssetsExpected",
      "results.nativeAssetsFound",
      "command.smokeCommand",
      "command.logPath",
      "command.logSha256"
    )
    forbiddenProofSubstitutes = @($preflightForbiddenSubstitutes)
    nonSubstituteChecklist = $nonSubstituteConfirmations
    validatorCommands = $validatorCommands
    expectedArtifacts = $expectedArtifacts
    importTargetFieldMap = @($importTargetFieldMap)
    remainingCloseGaps = $remainingGaps
    remainingCloseGapCount = $remainingGaps.Count
    readyForExecution = $false
    performsPublish = $false
    canPromoteRuntimeProof = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    isRuntimeExecutionProof = $false
    isReleaseCloseProof = $false
    nonProofBoundary = "This execution bundle item is owner guidance only. RuntimeProofPreflight is required alignment metadata, not proof. The item cannot substitute real external execution logs, matching SHA256 values, package source proof, owner review, post-publish proof, rollback approval, or release close approval."
  }
}

$resultTemplate = Read-JsonOrNull "artifacts\final-release\owner-runtime-proof-result-input.template.json"
$resultValidation = Read-JsonOrNull "artifacts\final-release\owner-runtime-proof-result-input-validation.json"
$closeDryRun = Read-JsonOrNull "artifacts\final-release\release-close-strict-dry-run-summary.json"
$laneDryRun = Read-JsonOrNull "artifacts\final-release\runtime-proof-lane-dry-run-summary.json"
$runtimeProofPreflightMatrix = Read-JsonOrNull "artifacts\final-release\package-consumer-runtime-proof-preflight-matrix.json"

if ($null -eq $resultTemplate) {
  throw "Missing owner runtime proof result input template. Run Export-OwnerRuntimeProofResultInputTemplate.ps1 first."
}
if ($null -eq $closeDryRun) {
  throw "Missing release close strict dry-run summary. Run Export-ReleaseCloseStrictDryRunSummary.ps1 first."
}

$resultInputs = @(Get-PropertyOrDefault -Object $resultTemplate -Name "resultInputs" -DefaultValue @())
$closeDryRunItems = @(Get-PropertyOrDefault -Object $closeDryRun -Name "closeDryRunItems" -DefaultValue @())
$items = @($resultInputs | ForEach-Object { New-ExecutionBundleItem -ResultInput $_ -CloseDryRunItems $closeDryRunItems -PreflightMatrix $runtimeProofPreflightMatrix })
$blockedItemCount = @($items | Where-Object { [string]$_.bundleItemState -eq "blocked-owner-external-proof-execution-required" }).Count
$remainingGapCount = ($items | ForEach-Object { [int]$_.remainingCloseGapCount } | Measure-Object -Sum).Sum
if ($null -eq $remainingGapCount) { $remainingGapCount = 0 }

$record = [pscustomobject]@{
  recordKind = "owner-external-proof-execution-bundle"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  bundleState = "blocked-owner-external-proof-execution-required"
  resultInputTemplateState = [string](Get-PropertyOrDefault -Object $resultTemplate -Name "templateState" -DefaultValue "missing-owner-runtime-proof-result-input-template")
  resultInputValidationState = [string](Get-PropertyOrDefault -Object $resultValidation -Name "validationState" -DefaultValue "missing-owner-runtime-proof-result-input-validation")
  laneDryRunState = [string](Get-PropertyOrDefault -Object $laneDryRun -Name "dryRunState" -DefaultValue "missing-runtime-proof-lane-dry-run-summary")
  releaseCloseStrictDryRunState = [string](Get-PropertyOrDefault -Object $closeDryRun -Name "closeDryRunState" -DefaultValue "missing-release-close-strict-dry-run-summary")
  runtimeProofPreflightMatrixFound = $null -ne $runtimeProofPreflightMatrix
  runtimeProofPreflightEntryCount = @((Get-PropertyOrDefault -Object $runtimeProofPreflightMatrix -Name "entries" -DefaultValue @())).Count
  executionBundleItemCount = $items.Count
  blockedExecutionBundleItemCount = $blockedItemCount
  readyExecutionBundleItemCount = 0
  remainingCloseGapCount = [int]$remainingGapCount
  executionBundleItems = $items
  sourceArtifacts = @(
    "artifacts/final-release/owner-runtime-proof-result-input.template.json",
    "artifacts/final-release/owner-runtime-proof-result-input-validation.json",
    "artifacts/final-release/runtime-proof-lane-dry-run-summary.json",
    "artifacts/final-release/release-close-strict-dry-run-summary.json",
    "artifacts/final-release/package-consumer-runtime-proof-preflight-matrix.json"
  )
  performsPublish = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isReleaseCloseProof = $false
  safetyBoundary = "Owner external proof execution bundle is an owner execution package only. It carries RuntimeProofPreflight alignment requirements but does not run proof, publish packages, verify post-publish state, approve rollback, or close the release issue."
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "owner-external-proof-execution-bundle.json"
$markdownPath = Join-Path $artifactRoot "owner-external-proof-execution-bundle.md"
Write-Utf8File -LiteralPath $jsonPath -InputObject ($record | ConvertTo-Json -Depth 14)
$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Owner External Proof Execution Bundle")
$lines.Add("")
$lines.Add("该 bundle 将 Owner runtime proof result input 与 release close strict dry-run 合并为可执行外部 proof 任务。它仍是 blocked/non-proof owner guidance。")
$lines.Add("")
$lines.Add("| 项目 | 当前值 |")
$lines.Add("|---|---|")
$lines.Add("| bundleState | ``$(ConvertTo-MarkdownCell $record.bundleState)`` |")
$lines.Add("| executionBundleItemCount | ``$($record.executionBundleItemCount)`` |")
$lines.Add("| blockedExecutionBundleItemCount | ``$($record.blockedExecutionBundleItemCount)`` |")
$lines.Add("| readyExecutionBundleItemCount | ``$($record.readyExecutionBundleItemCount)`` |")
$lines.Add("| remainingCloseGapCount | ``$($record.remainingCloseGapCount)`` |")
$lines.Add("| runtimeProofPreflightMatrixFound | ``$($record.runtimeProofPreflightMatrixFound)`` |")
$lines.Add("| runtimeProofPreflightEntryCount | ``$($record.runtimeProofPreflightEntryCount)`` |")
$lines.Add("| canPromoteRuntimeProof | ``$($record.canPromoteRuntimeProof)`` |")
$lines.Add("| canPublishPublicly | ``$($record.canPublishPublicly)`` |")
$lines.Add("| canCloseReleaseIssue | ``$($record.canCloseReleaseIssue)`` |")
$lines.Add("")
$lines.Add("## Bundle Items")
$lines.Add("")
$lines.Add("| Item | Lane | Runtime Package | Preflight Entry | Native Assets Expected | State | Remaining Gaps |")
$lines.Add("|---|---|---|---|---:|---|---:|")
foreach ($item in $items) {
  $lines.Add("| $(ConvertTo-MarkdownCell $item.executionBundleItemId) | $(ConvertTo-MarkdownCell $item.proofLane) | $(ConvertTo-MarkdownCell $item.runtimePackageKey) | ``$($item.runtimeProofPreflight.entryFound)`` | ``$($item.runtimeProofPreflight.selectedNativeAssetCopyExpected)`` | $(ConvertTo-MarkdownCell $item.bundleItemState) | ``$($item.remainingCloseGapCount)`` |")
}
$lines.Add("")
$lines.Add("## Boundary")
$lines.Add("")
$lines.Add($record.safetyBoundary)
$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Owner external proof execution bundle written to $jsonPath"
Write-Host "Owner external proof execution bundle markdown written to $markdownPath"
Write-Host "BundleState=$($record.bundleState) Items=$($record.executionBundleItemCount) Blocked=$($record.blockedExecutionBundleItemCount) RemainingGaps=$($record.remainingCloseGapCount)"
