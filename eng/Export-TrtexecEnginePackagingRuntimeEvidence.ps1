[CmdletBinding()]
param(
  [string]$Trt8ReportPath = "artifacts/real-case/trtexec-engine-packaging-policy/trt8-packaging-version-guards.json",
  [string]$Trt10VersionRefitReportPath = "artifacts/real-case/trtexec-engine-packaging-policy/trt10-version-compatible-refit.json",
  [string]$Trt10StripReportPath = "artifacts/real-case/trtexec-engine-packaging-policy/trt10-strip-weights.json",
  [string]$Trt10WeightStreamingReportPath = "artifacts/real-case/trtexec-engine-packaging-policy/trt10-yolox-weight-streaming-50-percent.json",
  [string]$Trt10LoadEngineReportPath = "artifacts/real-case/trtexec-engine-packaging-policy/trt10-yolox-load-engine-weight-streaming-auto.json",
  [string]$Trt11ReportPath = "artifacts/real-case/trtexec-engine-packaging-policy/trt11-packaging-dependency-probe.json",
  [string]$OutputPath = "artifacts/interface-coverage/trtexec-engine-packaging-runtime-evidence.json"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot

function Resolve-RepositoryPath {
  param([Parameter(Mandatory = $true)][string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return [IO.Path]::GetFullPath($Path)
  }

  return [IO.Path]::GetFullPath((Join-Path $repositoryRoot ($Path -replace "/", "\")))
}

function Get-RepositoryRelativePath {
  param([Parameter(Mandatory = $true)][string]$Path)

  return [IO.Path]::GetRelativePath($repositoryRoot, (Resolve-RepositoryPath $Path)).Replace("\", "/")
}

function Read-Report {
  param([Parameter(Mandatory = $true)][string]$Path)

  $fullPath = Resolve-RepositoryPath $Path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    throw "Required report was not found: $fullPath"
  }

  return Get-Content -LiteralPath $fullPath -Raw | ConvertFrom-Json
}

function Get-Artifact {
  param([Parameter(Mandatory = $true)][string]$Path)

  $fullPath = Resolve-RepositoryPath $Path
  if (-not (Test-Path -LiteralPath $fullPath -PathType Leaf)) {
    throw "Required artifact was not found: $fullPath"
  }

  $item = Get-Item -LiteralPath $fullPath
  return [pscustomobject][ordered]@{
    path = Get-RepositoryRelativePath $fullPath
    lengthBytes = [long]$item.Length
    sha256 = (Get-FileHash -LiteralPath $fullPath -Algorithm SHA256).Hash.ToLowerInvariant()
  }
}

function Find-LogLine {
  param(
    [Parameter(Mandatory = $true)]$Report,
    [Parameter(Mandatory = $true)][string]$Prefix,
    [string[]]$RequiredFragments = @()
  )

  foreach ($line in @($Report.LogLines)) {
    $text = [string]$line
    if (-not $text.StartsWith($Prefix, [StringComparison]::Ordinal)) {
      continue
    }

    $matches = $true
    foreach ($fragment in $RequiredFragments) {
      if (-not $text.Contains($fragment, [StringComparison]::Ordinal)) {
        $matches = $false
        break
      }
    }

    if ($matches) {
      return $text
    }
  }

  throw "Report did not contain required log line: $Prefix [$($RequiredFragments -join ', ')]"
}

function Get-InputArtifactPath {
  param([Parameter(Mandatory = $true)][string]$Mapping)

  $separator = $Mapping.IndexOf(":", [StringComparison]::Ordinal)
  if ($separator -lt 1 -or $separator -eq ($Mapping.Length - 1)) {
    throw "Unexpected --loadInputs mapping: $Mapping"
  }

  return $Mapping.Substring($separator + 1)
}

$trt8 = Read-Report $Trt8ReportPath
$trt10VersionRefit = Read-Report $Trt10VersionRefitReportPath
$trt10Strip = Read-Report $Trt10StripReportPath
$trt10Weight = Read-Report $Trt10WeightStreamingReportPath
$trt10Load = Read-Report $Trt10LoadEngineReportPath
$trt11 = Read-Report $Trt11ReportPath

$trt8VersionLine = Find-LogLine $trt8 "TrtexecDeploymentControl Name=VersionCompatible" @("Applied=True", "ReadbackMatch=True")
$trt8ExcludeLine = Find-LogLine $trt8 "TrtexecDeploymentControl Name=ExcludeLeanRuntime" @("Applied=True", "ReadbackMatch=True")
$trt8RefitGuardLine = Find-LogLine $trt8 "TrtexecDeploymentControl Name=Refit" @("Applied=False", "VersionGuard=TRT8", "version-compatible-refit-vendor-readback-conflict")
$trt8StripGuardLine = Find-LogLine $trt8 "TrtexecDeploymentControl Name=StripWeights" @("Applied=False", "VersionGuard=TRT8")
$trt8StreamingGuardLine = Find-LogLine $trt8 "TrtexecDeploymentControl Name=WeightStreaming" @("Applied=False", "VersionGuard=TRT8")
$trt8StrongGuardLine = Find-LogLine $trt8 "TrtexecDeploymentControl Name=StronglyTyped" @("Applied=False", "VersionGuard=TRT8")

$trt10VersionLine = Find-LogLine $trt10VersionRefit "TrtexecDeploymentControl Name=VersionCompatible" @("Applied=True", "ReadbackMatch=True")
$trt10RefitLine = Find-LogLine $trt10VersionRefit "TrtexecDeploymentControl Name=Refit" @("Applied=True", "ReadbackMatch=True")
$trt10EngineRefitLine = Find-LogLine $trt10VersionRefit "TrtexecEnginePolicy Name=Refit" @("Readback=True", "ReadbackMatch=True")
$trt10HostCodeLine = Find-LogLine $trt10VersionRefit "TrtexecRuntimePolicy Name=EngineHostCodeAllowed" @("Readback=True", "ReadbackMatch=True")
$trt10StripLine = Find-LogLine $trt10Strip "TrtexecDeploymentControl Name=StripWeights" @("Applied=True", "RefitMode=RefitIdentical", "ReadbackMatch=True")
$trt10StreamingLine = Find-LogLine $trt10Weight "TrtexecDeploymentControl Name=WeightStreaming" @("Applied=True", "ReadbackMatch=True")
$trt10StrongLine = Find-LogLine $trt10Weight "TrtexecDeploymentControl Name=StronglyTyped" @("Applied=True", "ReadbackMatch=True")
$trt10BudgetLine = Find-LogLine $trt10Weight "TrtexecDeploymentControl Name=WeightStreamingBudget" @("Applied=True", "Mode=percentage", "ReadbackMatch=True")
$trt10RuntimeLine = Find-LogLine $trt10Weight "ExternalOnnxBoundedRuntime" @("Attempted=True", "Succeeded=True")
$trt10LoadBudgetLine = Find-LogLine $trt10Load "TrtexecDeploymentControl Name=WeightStreamingBudget" @("Applied=True", "Mode=automatic", "ReadbackMatch=True")
$trt10LoadDiagnosticsLine = Find-LogLine $trt10Load "LoadEngineReadonlyDiagnostics Attempted=True" @("Succeeded=True")
$trt10LoadRuntimeLine = Find-LogLine $trt10Load "LoadEngineBoundedRuntime" @("Attempted=True", "Succeeded=True")

$budgetMatch = [regex]::Match(
  $trt10BudgetLine,
  "ResolvedBytes=(?<resolved>[0-9]+).*StreamableWeightsBytes=(?<streamable>[0-9]+).*AutomaticBudgetBytes=(?<automatic>[0-9]+).*Readback=(?<readback>[0-9]+).*ScratchBytes=(?<scratch>[0-9]+)")
if (-not $budgetMatch.Success) {
  throw "Could not parse the TRT10 percentage budget readback line."
}

$loadBudgetMatch = [regex]::Match(
  $trt10LoadBudgetLine,
  "ResolvedBytes=(?<resolved>[0-9]+).*StreamableWeightsBytes=(?<streamable>[0-9]+).*AutomaticBudgetBytes=(?<automatic>[0-9]+).*Readback=(?<readback>[0-9]+).*ScratchBytes=(?<scratch>[0-9]+)")
if (-not $loadBudgetMatch.Success) {
  throw "Could not parse the TRT10 automatic budget readback line."
}

$modelArtifact = Get-Artifact ([string]$trt10Weight.ModelSource)
$inputArtifact = Get-Artifact (Get-InputArtifactPath ([string]$trt10Weight.RuntimeOptions.LoadInputs))

$record = [pscustomobject][ordered]@{
  schemaVersion = "trtexec-engine-packaging-runtime-evidence.v1"
  generatedAt = (Get-Date).ToString("yyyy-MM-dd")
  evidenceKind = "local-project-reference-engine-packaging-and-weight-streaming-policy"
  reports = [pscustomobject][ordered]@{
    trt8 = Get-Artifact $Trt8ReportPath
    trt10VersionRefit = Get-Artifact $Trt10VersionRefitReportPath
    trt10Strip = Get-Artifact $Trt10StripReportPath
    trt10WeightStreaming = Get-Artifact $Trt10WeightStreamingReportPath
    trt10LoadEngine = Get-Artifact $Trt10LoadEngineReportPath
    trt11 = Get-Artifact $Trt11ReportPath
  }
  model = $modelArtifact
  inputTensor = $inputArtifact
  tensorRt8 = [pscustomobject][ordered]@{
    state = [string]$trt8.State
    proofClassification = [string]$trt8.ProofClassification
    versionCompatibleApplied = $true
    excludeLeanRuntimeApplied = $true
    refitVersionCompatibleConflictGuarded = $true
    stripWeightsGuarded = $true
    weightStreamingGuarded = $true
    stronglyTypedGuarded = $true
    logLines = @($trt8VersionLine, $trt8ExcludeLine, $trt8RefitGuardLine, $trt8StripGuardLine, $trt8StreamingGuardLine, $trt8StrongGuardLine)
  }
  tensorRt10 = [pscustomobject][ordered]@{
    versionCompatibleRefit = [pscustomobject][ordered]@{
      state = [string]$trt10VersionRefit.State
      proofClassification = [string]$trt10VersionRefit.ProofClassification
      inferenceRan = [bool]$trt10VersionRefit.InferenceRan
      outputMatch = [bool]$trt10VersionRefit.OutputMatch
      versionCompatibleReadbackMatch = $true
      refitConfigReadbackMatch = $true
      refittableEngineReadbackMatch = $true
      runtimeHostCodeReadbackMatch = $true
      logLines = @($trt10VersionLine, $trt10RefitLine, $trt10EngineRefitLine, $trt10HostCodeLine)
    }
    stripWeights = [pscustomobject][ordered]@{
      state = [string]$trt10Strip.State
      proofClassification = [string]$trt10Strip.ProofClassification
      stripPlanReadbackMatch = $true
      defaultRefitMode = "RefitIdentical"
      logLine = $trt10StripLine
    }
    weightStreaming = [pscustomobject][ordered]@{
      state = [string]$trt10Weight.State
      proofClassification = [string]$trt10Weight.ProofClassification
      inferenceRan = [bool]$trt10Weight.InferenceRan
      outputMatch = [bool]$trt10Weight.OutputMatch
      outputValidationState = "captured-unverified"
      stronglyTypedReadbackMatch = $true
      builderFlagReadbackMatch = $true
      budgetMode = "percentage"
      budgetArgument = "50%"
      streamableWeightsBytes = [long]$budgetMatch.Groups["streamable"].Value
      automaticBudgetBytes = [long]$budgetMatch.Groups["automatic"].Value
      resolvedBudgetBytes = [long]$budgetMatch.Groups["resolved"].Value
      readbackBudgetBytes = [long]$budgetMatch.Groups["readback"].Value
      scratchBytes = [long]$budgetMatch.Groups["scratch"].Value
      readbackMatch = $true
      contextCreatedAfterBudgetReadback = $true
      logLines = @($trt10StrongLine, $trt10StreamingLine, $trt10BudgetLine, $trt10RuntimeLine)
    }
    loadEngineAutomaticBudget = [pscustomobject][ordered]@{
      state = [string]$trt10Load.State
      diagnosticsSucceeded = [bool]$trt10Load.LoadedEngineDiagnostics.Succeeded
      inferenceRan = [bool]$trt10Load.InferenceRan
      budgetMode = "automatic"
      streamableWeightsBytes = [long]$loadBudgetMatch.Groups["streamable"].Value
      automaticBudgetBytes = [long]$loadBudgetMatch.Groups["automatic"].Value
      resolvedBudgetBytes = [long]$loadBudgetMatch.Groups["resolved"].Value
      readbackBudgetBytes = [long]$loadBudgetMatch.Groups["readback"].Value
      scratchBytes = [long]$loadBudgetMatch.Groups["scratch"].Value
      readbackMatch = $true
      logLines = @($trt10LoadDiagnosticsLine, $trt10LoadBudgetLine, $trt10LoadRuntimeLine)
    }
  }
  tensorRt11 = [pscustomobject][ordered]@{
    state = [string]$trt11.State
    proofClassification = [string]$trt11.ProofClassification
    knownVendorStructuredExceptionCode = 3228369022
    reachedBuilderPolicyApplication = $false
    requestedOptionsRemainParseOnly = $true
    skipReason = [string]$trt11.SkipReason
  }
  proofBoundary = [pscustomobject][ordered]@{
    isBuilderAndEnginePolicyEvidence = $true
    isWeightedModelEnqueueEvidence = $true
    isModelAccuracyProof = $false
    isCrossVersionLeanRuntimeProof = $false
    isStrippedPlanRefitLifecycleProof = $false
    isPackageConsumerRuntimeProof = $false
    isPostPublishProof = $false
    canPublishPublicly = $false
    publicReleaseSideEffectsExecuted = $false
  }
}

$outputFullPath = Resolve-RepositoryPath $OutputPath
$outputDirectory = Split-Path -Parent $outputFullPath
New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
$json = $record | ConvertTo-Json -Depth 20
[IO.File]::WriteAllText($outputFullPath, $json + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))

$markdownPath = [IO.Path]::ChangeExtension($outputFullPath, ".md")
$markdown = @"
# Trtexec Engine Packaging Runtime Evidence

- Evidence kind: ``$($record.evidenceKind)``
- TRT8: ``$($record.tensorRt8.state)``; version/exclude applied, refit conflict and strip/streaming guarded
- TRT10 version-compatible/refit: inference ``$($record.tensorRt10.versionCompatibleRefit.inferenceRan)``; output match ``$($record.tensorRt10.versionCompatibleRefit.outputMatch)``
- TRT10 strip mode: ``$($record.tensorRt10.stripWeights.defaultRefitMode)``; readback ``$($record.tensorRt10.stripWeights.stripPlanReadbackMatch)``
- TRT10 weighted model streamable bytes: ``$($record.tensorRt10.weightStreaming.streamableWeightsBytes)``
- TRT10 50% budget/readback/scratch: ``$($record.tensorRt10.weightStreaming.resolvedBudgetBytes)`` / ``$($record.tensorRt10.weightStreaming.readbackBudgetBytes)`` / ``$($record.tensorRt10.weightStreaming.scratchBytes)``
- TRT10 load-engine automatic budget: ``$($record.tensorRt10.loadEngineAutomaticBudget.resolvedBudgetBytes)``; readback ``$($record.tensorRt10.loadEngineAutomaticBudget.readbackMatch)``
- TRT11: ``$($record.tensorRt11.proofClassification)``; structured exception ``$($record.tensorRt11.knownVendorStructuredExceptionCode)``

This is local builder/engine policy and weighted-model enqueue evidence. It is not model accuracy, cross-version lean-runtime, stripped-plan refit lifecycle, package-consumer, post-publish, or public release proof.
"@
[IO.File]::WriteAllText($markdownPath, $markdown.TrimEnd() + [Environment]::NewLine, [Text.UTF8Encoding]::new($false))

Write-Host "Trtexec engine packaging runtime evidence written to $outputFullPath"
Write-Host "Trtexec engine packaging runtime evidence written to $markdownPath"
