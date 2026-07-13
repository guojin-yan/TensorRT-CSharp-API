[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [int]$MaxWorkItems = 60
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

function Read-Json {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required JSON artifact was not found: $path"
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param(
    [AllowNull()][object]$Object,
    [string]$Name,
    [AllowNull()][object]$DefaultValue
  )

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function Get-ArrayOrEmpty {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  $value = Get-PropertyOrDefault -Object $Object -Name $Name -DefaultValue @()
  if ($null -eq $value) {
    return @()
  }

  return @($value)
}

function Convert-ToStringArray {
  param([AllowNull()][object]$Values)

  $items = [System.Collections.Generic.List[string]]::new()
  foreach ($value in @($Values)) {
    if ($null -ne $value -and -not [string]::IsNullOrWhiteSpace([string]$value)) {
      $items.Add([string]$value)
    }
  }

  return ,$items
}

function Resolve-ImplementationPhase {
  param([object]$Candidate)

  $bucket = [string](Get-PropertyOrDefault -Object $Candidate -Name "closureBucket" -DefaultValue "")
  $className = [string](Get-PropertyOrDefault -Object $Candidate -Name "class" -DefaultValue "")
  $methodName = [string](Get-PropertyOrDefault -Object $Candidate -Name "method" -DefaultValue "")
  $interface = [string](Get-PropertyOrDefault -Object $Candidate -Name "interface" -DefaultValue "")
  $combined = "$bucket $className $methodName $interface"

  if ($bucket -eq "already-safe-alternative-proof") {
    return "phase-1-safe-alternative-proof"
  }

  if ($bucket -eq "docs-test-proof-needed") {
    return "phase-2-wrapper-docs-quality-proof"
  }

  if ($bucket -eq "runtime-smoke-backfill-needed" -or $combined -match "Plugin|Registry|Runtime|Parser|Refitter") {
    return "phase-3-smoke-backfill"
  }

  return "phase-4-manual-review-hold"
}

function Resolve-WorkItemAction {
  param([string]$Phase)

  switch ($Phase) {
    "phase-1-safe-alternative-proof" { return "Add or tighten ProjectQuality proof that the safe alternative is present, pointer-free, and backed by non-deferred manifest/source/wrapper evidence." }
    "phase-2-wrapper-docs-quality-proof" { return "Confirm high-level C# wrapper and XML docs are public, then add docs and quality assertions for the safe wrapper path." }
    "phase-3-smoke-backfill" { return "Extend an existing smoke runner or add a small package-consumer surface smoke before treating the wrapper as runtime-observed." }
    default { return "Keep as manual review; do not promote until owner lifetime, pointer-free API shape, and runtime proof are defined." }
  }
}

function Resolve-ValidationCommands {
  param([string]$Phase)

  $commands = [System.Collections.Generic.List[string]]::new()
  $commands.Add("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredReadOnlyApiCandidatePlan.ps1 -IncludeMediumRisk -MaxItems 60")
  $commands.Add("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierProofClosureDashboard.ps1")
  $commands.Add("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-DeferredBTierAliasProofClosureRecord.ps1")

  if ($Phase -eq "phase-3-smoke-backfill") {
    $commands.Add("dotnet build .\TensorRtSharp.sln -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false")
    $commands.Add("dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter `"FullyQualifiedName~PluginRegistryInventoryTests|FullyQualifiedName~BridgePackageConsumerTests`" /p:UseSharedCompilation=false /nr:false")
  }
  elseif ($Phase -eq "phase-4-manual-review-hold") {
    $commands.Add("dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter `"FullyQualifiedName~DeferredBTierImplementationWorkPackageTests|FullyQualifiedName~DeferredBTierProofClosureTests|FullyQualifiedName~DeferredBTierAliasProofClosureTests`" /p:UseSharedCompilation=false /nr:false")
  }
  else {
    $commands.Add("dotnet build .\TensorRtSharp.sln -c Debug --no-restore /m:1 /p:UseSharedCompilation=false /nr:false")
    $commands.Add("dotnet test .\tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj -c Debug --no-build --filter `"FullyQualifiedName~DeferredBTierImplementationWorkPackageTests|FullyQualifiedName~DeferredBTierProofClosureTests|FullyQualifiedName~DeferredBTierAliasProofClosureTests`" /p:UseSharedCompilation=false /nr:false")
  }

  return ,$commands
}

function Write-TextFileWithRetry {
  param(
    [string]$Path,
    [string]$Value,
    [System.Text.Encoding]$Encoding,
    [int]$RetryCount = 40,
    [int]$DelayMilliseconds = 250
  )

  $directory = Split-Path -Parent $Path
  New-Item -ItemType Directory -Force -Path $directory | Out-Null

  $tempFileName = ".{0}.{1}.tmp" -f ([System.IO.Path]::GetFileName($Path)), ([System.Guid]::NewGuid().ToString("N"))
  $tempPath = Join-Path $directory $tempFileName

  try {
    [System.IO.File]::WriteAllText($tempPath, $Value, $Encoding)

    for ($attempt = 1; $attempt -le $RetryCount; $attempt++) {
      try {
        Move-Item -LiteralPath $tempPath -Destination $Path -Force
        return
      }
      catch {
        if ($attempt -eq $RetryCount) {
          throw
        }

        Start-Sleep -Milliseconds $DelayMilliseconds
      }
    }
  }
  finally {
    if (Test-Path -LiteralPath $tempPath -PathType Leaf) {
      Remove-Item -LiteralPath $tempPath -Force -ErrorAction SilentlyContinue
    }
  }
}

$aliasRecord = Read-Json "artifacts\interface-coverage\deferred-btier-alias-proof-closure-record.json"
$dashboard = Read-Json "artifacts\interface-coverage\deferred-btier-proof-closure-dashboard.json"

$closureCandidates = @(
  Get-ArrayOrEmpty -Object $aliasRecord -Name "closureCandidates" |
    Where-Object {
      [string](Get-PropertyOrDefault -Object $_ -Name "closureProofState" -DefaultValue "") -eq "alias-proof-ready" -and
      [string](Get-PropertyOrDefault -Object $_ -Name "safetyTier" -DefaultValue "") -eq "B - safe-alternative-or-alias"
    }
)

$stableWorkItemCandidateKeys = @(
  "TRT10|IExecutionContext::getName",
  "TRT11|IProfiler::getInterfaceInfo",
  "TRT8|ICudaEngine::getProfileShape",
  "TRT8|IExecutionContext::getName",
  "TRT8|ILayer::getInput",
  "TRT10|IBuilder::getMaxDLABatchSize",
  "TRT10|IBuilder::getMaxThreads",
  "TRT10|IBuilder::isNetworkSupported",
  "TRT10|IBuilderConfig::canRunOnDLA",
  "TRT10|IBuilderConfig::getAvgTimingIterations",
  "TRT10|IBuilderConfig::getDefaultDeviceType",
  "TRT10|IBuilderConfig::getDeviceType",
  "TRT10|IBuilderConfig::getDLACore",
  "TRT10|IBuilderConfig::getL2LimitForTiling",
  "TRT10|IBuilderConfig::getMaxNbTactics",
  "TRT10|IBuilderConfig::getQuantizationFlag",
  "TRT10|IBuilderConfig::getQuantizationFlags",
  "TRT11|IBuilderConfig::getAvgTimingIterations",
  "TRT11|IParser::getError",
  "TRT11|IParser::isSubgraphSupported",
  "TRT11|IParserRefitter::getError",
  "TRT8|IBuilder::getMaxBatchSize",
  "TRT8|IBuilder::getMaxDLABatchSize",
  "TRT8|IBuilder::getMaxThreads",
  "TRT8|IBuilder::isNetworkSupported",
  "TRT8|IBuilderConfig::canRunOnDLA",
  "TRT8|IBuilderConfig::getAvgTimingIterations",
  "TRT8|IBuilderConfig::getDefaultDeviceType",
  "TRT8|IBuilderConfig::getDeviceType",
  "TRT8|IBuilderConfig::getDLACore",
  "TRT8|IBuilderConfig::getFlag",
  "TRT8|IBuilderConfig::getFlags",
  "TRT8|IBuilderConfig::getMaxWorkspaceSize",
  "TRT8|IBuilderConfig::getMinTimingIterations",
  "TRT8|IBuilderConfig::getQuantizationFlag",
  "TRT8|IBuilderConfig::getQuantizationFlags",
  "TRT8|ICudaEngine::getHardwareCompatibilityLevel",
  "TRT8|ICudaEngine::getProfileDimensions",
  "TRT8|IExecutionContext::getNvtxVerbosity",
  "TRT8|IParser::getError"
)

$stableWorkItemRanks = @{}
for ($stableIndex = 0; $stableIndex -lt $stableWorkItemCandidateKeys.Count; $stableIndex++) {
  $stableWorkItemRanks[$stableWorkItemCandidateKeys[$stableIndex]] = $stableIndex
}

$orderedClosureCandidates = @(
  $closureCandidates |
    Sort-Object `
      @{ Expression = {
          $candidateVersion = [string](Get-PropertyOrDefault -Object $_ -Name "version" -DefaultValue "")
          $candidateInterface = [string](Get-PropertyOrDefault -Object $_ -Name "interface" -DefaultValue "")
          $key = $candidateVersion + "|" + $candidateInterface
          if ($stableWorkItemRanks.ContainsKey($key)) {
            return [int]$stableWorkItemRanks[$key]
          }

          return [int]::MaxValue
        } },
      @{ Expression = { Resolve-ImplementationPhase -Candidate $_ } },
      "version",
      "class",
      "method",
      "interface"
)

$workItems = @(
  $orderedClosureCandidates |
    Select-Object -First $MaxWorkItems |
    ForEach-Object {
      $phase = Resolve-ImplementationPhase -Candidate $_
      $safeAlternativeManifestIds = Convert-ToStringArray -Values (Get-ArrayOrEmpty -Object $_ -Name "safeAlternativeManifestIds")
      $deferredHistoryManifestIds = Convert-ToStringArray -Values (Get-ArrayOrEmpty -Object $_ -Name "deferredHistoryManifestIds")
      $sourceEvidence = Convert-ToStringArray -Values (Get-ArrayOrEmpty -Object $_ -Name "sourceEvidence")

      [pscustomobject]@{
        workItemId = "btier-{0:d3}" -f ($script:workItemIndex += 1)
        phase = $phase
        interface = [string](Get-PropertyOrDefault -Object $_ -Name "interface" -DefaultValue "")
        version = [string](Get-PropertyOrDefault -Object $_ -Name "version" -DefaultValue "")
        class = [string](Get-PropertyOrDefault -Object $_ -Name "class" -DefaultValue "")
        method = [string](Get-PropertyOrDefault -Object $_ -Name "method" -DefaultValue "")
        tensorRtLine = [string](Get-PropertyOrDefault -Object $_ -Name "tensorRtLine" -DefaultValue "")
        ownershipRisk = [string](Get-PropertyOrDefault -Object $_ -Name "ownershipRisk" -DefaultValue "")
        safetyTier = [string](Get-PropertyOrDefault -Object $_ -Name "safetyTier" -DefaultValue "")
        designGroup = [string](Get-PropertyOrDefault -Object $_ -Name "designGroup" -DefaultValue "")
        closureBucket = [string](Get-PropertyOrDefault -Object $_ -Name "closureBucket" -DefaultValue "")
        evidenceKind = [string](Get-PropertyOrDefault -Object $_ -Name "evidenceKind" -DefaultValue "")
        safeAlternativeManifestIds = $safeAlternativeManifestIds
        deferredHistoryManifestIds = $deferredHistoryManifestIds
        sourceEvidence = $sourceEvidence
        requiredFilesToInspect = @(
          $sourceEvidence |
            Where-Object {
              $_ -match "^(native|src|tests|smoke|docs|samples)/" -or
              $_ -match "^artifacts/interface-coverage/"
            } |
            Select-Object -First 8
        )
        implementationAction = Resolve-WorkItemAction -Phase $phase
        acceptanceCriteria = @(
          "safe alternative manifest IDs remain present",
          "deferred history manifest IDs remain present",
          "no public API exposes borrowed IntPtr or native plugin/recorder/allocator pointers",
          "native/source, C# wrapper, docs, and tests are all updated before promotion",
          "release flags remain canPublishPublicly=false and canCloseReleaseIssue=false"
        )
        validationCommands = Resolve-ValidationCommands -Phase $phase
        canDeleteDeferredRecord = $false
        canPromoteReleaseProof = $false
        isRuntimeExecutionProof = $false
        isPackageConsumerRuntimeProof = $false
      }
    }
)

$phaseSummaries = @(
  $workItems |
    Group-Object phase |
    Sort-Object Name |
    ForEach-Object {
      [pscustomobject]@{
        phase = $_.Name
        workItemCount = $_.Count
        representativeInterfaces = @($_.Group | Select-Object -First 8 | ForEach-Object { "$($_.interface) [$($_.version)]" })
        validationCommands = Resolve-ValidationCommands -Phase $_.Name
      }
    }
)

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "deferred-btier-implementation-work-package"
  workPackageState = "ready-for-next-implementation-batch"
  sourceAliasRecord = "artifacts/interface-coverage/deferred-btier-alias-proof-closure-record.json"
  sourceDashboard = "artifacts/interface-coverage/deferred-btier-proof-closure-dashboard.json"
  sourceArtifacts = @(
    "artifacts/interface-coverage/deferred-btier-alias-proof-closure-record.json",
    "artifacts/interface-coverage/deferred-btier-proof-closure-dashboard.json",
    "artifacts/interface-coverage/deferred-candidate-safety-triage.json",
    "artifacts/interface-coverage/tensorrt-interface-comparison.csv"
  )
  sourceAliasClosureCandidateCount = [int](Get-PropertyOrDefault -Object $aliasRecord -Name "closureCandidateCount" -DefaultValue 0)
  dashboardSelectedCandidateCount = [int](Get-PropertyOrDefault -Object $dashboard -Name "selectedCandidateCount" -DefaultValue 0)
  workItemOrderingPolicy = "stable-v1-existing-40-then-deterministic-append"
  stableWorkItemKeyCount = $stableWorkItemCandidateKeys.Count
  workItemTargetCount = $MaxWorkItems
  workItemCount = $workItems.Count
  phaseSummaries = $phaseSummaries
  workItems = $workItems
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canDeleteDeferredRecords = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  boundary = "This work package converts selected B-tier alias-proof candidates into ordered engineering tasks. It is not runtime proof, release proof, owner approval, post-publish verification, or permission to delete deferred records."
  nextBatchPromptFocus = @(
    "Start from artifacts/interface-coverage/deferred-btier-implementation-work-package.json rather than rescanning broad deferred files.",
    "Pick the first 12 to 20 workItems in phase order.",
    "For each item, prove the existing safe alternative through wrapper/docs/tests before touching deferred manifests.",
    "Do not delete deferred history records; they remain audit history until an owner-approved release policy changes.",
    "Run the validationCommands emitted on each workItem and record results in the stage diary."
  )
}

$artifactRoot = Join-Path $RepositoryRoot "artifacts\interface-coverage"
New-Item -ItemType Directory -Force -Path $artifactRoot | Out-Null

$jsonPath = Join-Path $artifactRoot "deferred-btier-implementation-work-package.json"
$markdownPath = Join-Path $artifactRoot "deferred-btier-implementation-work-package.md"

$recordJson = $record | ConvertTo-Json -Depth 16
Write-TextFileWithRetry -Path $jsonPath -Value $recordJson -Encoding $utf8

$phaseRows = $phaseSummaries | ForEach-Object {
  $representatives = (@($_.representativeInterfaces) -join "<br>").Replace("|", "\|")
  "| ``$($_.phase)`` | ``$($_.workItemCount)`` | $representatives |"
}

$workItemRows = $workItems | ForEach-Object {
  $safeAlternative = (@($_.safeAlternativeManifestIds) | Select-Object -First 3) -join "<br>"
  $deferredHistory = (@($_.deferredHistoryManifestIds) | Select-Object -First 3) -join "<br>"
  $files = (@($_.requiredFilesToInspect) | Select-Object -First 4) -join "<br>"
  "| ``$($_.workItemId)`` | ``$($_.phase)`` | ``$($_.version)`` | ``$($_.interface)`` | $($safeAlternative.Replace("|", "\|")) | $($deferredHistory.Replace("|", "\|")) | $($files.Replace("|", "\|")) | $($_.implementationAction.Replace("|", "\|")) |"
}

$markdown = @"
# Deferred B-tier Implementation Work Package

生成时间：$($record.generatedAtUtc)

## 总结

``recordKind=deferred-btier-implementation-work-package``，``workPackageState=ready-for-next-implementation-batch``。该工作包把 B-tier alias-proof 候选整理成下一批工程任务；它不是 runtime proof、release proof、owner approval、post-publish verification，也不是删除 deferred 记录的许可。

| 项目 | 值 |
|---|---:|
| source alias closure candidates | ``$($record.sourceAliasClosureCandidateCount)`` |
| dashboard selected candidates | ``$($record.dashboardSelectedCandidateCount)`` |
| work item target count | ``$($record.workItemTargetCount)`` |
| work item count | ``$($record.workItemCount)`` |
| performsPublish | ``False`` |
| canPublishPublicly | ``False`` |
| canCloseReleaseIssue | ``False`` |
| canDeleteDeferredRecords | ``False`` |
| isRuntimeExecutionProof | ``False`` |
| isPackageConsumerRuntimeProof | ``False`` |

## Phases

| Phase | Work items | Representatives |
|---|---:|---|
$($phaseRows -join "`r`n")

## Work Items

| ID | Phase | Version | Interface | Safe alternative manifests | Deferred history manifests | Files to inspect first | Action |
|---|---|---|---|---|---|---|---|
$($workItemRows -join "`r`n")

## Next Batch Prompt Focus

$(@($record.nextBatchPromptFocus | ForEach-Object { "- $_" }) -join "`r`n")

## Boundary

$($record.boundary)
"@

Write-TextFileWithRetry -Path $markdownPath -Value $markdown -Encoding $utf8

Write-Output "Deferred B-tier implementation work package written:"
Write-Output "  Json=$jsonPath"
Write-Output "  Markdown=$markdownPath"
Write-Output "WorkItemCount=$($record.workItemCount)"
Write-Output "CanPublishPublicly=False"
Write-Output "CanCloseReleaseIssue=False"
Write-Output "CanDeleteDeferredRecords=False"
