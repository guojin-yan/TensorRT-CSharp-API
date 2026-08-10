[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $InputPath,

    [Parameter(Mandatory = $false)]
    [string] $OutputPath = "artifacts/final-release/tensor-rt-exec-report-validation.json",

    [Parameter(Mandatory = $false)]
    [switch] $Strict
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $repoRoot

$requiredTopLevel = @(
    "Success",
    "Skipped",
    "State",
    "TensorRtLine",
    "ModelSource",
    "EnginePath",
    "Parsed",
    "DryRun",
    "InferenceRan",
    "OutputMatch",
    "OutputValidated",
    "IdentityOutputMatch",
    "NormalizedCommandLine",
    "NormalizedCommandSha256",
    "DeploymentOptions",
    "RuntimeOptions",
    "PreflightMetadata",
    "LoadedEngineDiagnostics",
    "LayerInfoArtifact",
    "BindingMetadata",
    "TimingCacheArtifact",
    "CapabilityProbe",
    "WorkspaceBytes",
    "ParserPreflightSnapshot",
    "RefitSnapshot",
    "RefitPersistenceSnapshot",
    "OptionImplementationStatus",
    "ProofClassification",
    "BuildEvidenceOnly",
    "IsRuntimeExecutionProof",
    "IsRealModelRuntimeProof",
    "IsPackageConsumerRuntimeProof",
    "ModelEvidence",
    "Diagnostics",
    "LogLines",
    "ReportBoundary"
)

$knownProofClassifications = @(
    "build-only",
    "dependency-probe-only",
    "precheck",
    "synthetic-input-runtime",
    "real-model-runtime",
    "package-consumer-runtime"
)

$forbiddenSubstitutes = @(
    "TensorRtExec report",
    "OnnxToEngine report",
    "readonly diagnostics",
    "build-only",
    "dry-run",
    "template",
    "local feed",
    "ProjectReference",
    'direct `.nupkg`',
    "YoloVision matrix",
    "capability-probe-only",
    "ONNX Parser diagnostic snapshot",
    "ONNX ParserRefitter diagnostic snapshot",
    "copied-parser-diagnostics",
    "copied-parser-refitter-diagnostics"
)

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string] $Path)

    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }

    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

function Test-JsonProperty {
    param(
        [AllowNull()] $Object,
        [Parameter(Mandatory = $true)] [string] $Name
    )

    return $null -ne $Object -and $null -ne $Object.PSObject.Properties[$Name] -and $null -ne $Object.$Name
}

function Get-JsonString {
    param(
        [AllowNull()] $Object,
        [Parameter(Mandatory = $true)] [string] $Name
    )

    if (-not (Test-JsonProperty $Object $Name)) {
        return ""
    }

    return [string]$Object.$Name
}

function Add-ValidationItem {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [System.Collections.Generic.List[object]] $Items,
        [Parameter(Mandatory = $true)] [string] $Id,
        [Parameter(Mandatory = $true)] [bool] $Passed,
        [Parameter(Mandatory = $true)] [string] $Severity,
        [Parameter(Mandatory = $true)] [string] $Message
    )

    $Items.Add([ordered]@{
        id = $Id
        passed = $Passed
        severity = $Severity
        message = $Message
    }) | Out-Null
}

$resolvedInput = Resolve-RepoPath $InputPath
$items = New-Object System.Collections.Generic.List[object]

if (-not (Test-Path -LiteralPath $resolvedInput -PathType Leaf)) {
    Add-ValidationItem $items "file-exists" $false "blocker" "TensorRtExec report does not exist: $InputPath"
}
else {
    $text = Get-Content -LiteralPath $resolvedInput -Raw -Encoding utf8
    $report = $text | ConvertFrom-Json -Depth 64

    Add-ValidationItem $items "file-exists" $true "required" "TensorRtExec report exists."
    foreach ($name in $requiredTopLevel) {
        Add-ValidationItem $items ("required-" + $name) (Test-JsonProperty $report $name) "blocker" "Report must contain top-level property $name."
    }

    $proofClassification = Get-JsonString $report "ProofClassification"
    $normalizedCommandSha256 = Get-JsonString $report "NormalizedCommandSha256"
    $forbidden = @()
    if (Test-JsonProperty $report "ReportBoundary") {
        $forbidden = @($report.ReportBoundary.ForbiddenSubstitutes | ForEach-Object { [string]$_ })
    }

    Add-ValidationItem $items "proof-classification-known" ($knownProofClassifications -contains $proofClassification) "blocker" "ProofClassification must be known."
    Add-ValidationItem $items "normalized-command-sha256" ($normalizedCommandSha256 -cmatch "^[0-9a-f]{64}$") "blocker" "NormalizedCommandSha256 must be a lowercase SHA256 string."
    Add-ValidationItem $items "report-boundary-present" (Test-JsonProperty $report "ReportBoundary") "blocker" "ReportBoundary must be present."
    Add-ValidationItem $items "report-boundary-not-runtime-proof" ((Test-JsonProperty $report "ReportBoundary") -and -not [bool]$report.ReportBoundary.IsRuntimeProof) "blocker" "ReportBoundary.IsRuntimeProof must be false for build/report evidence."
    Add-ValidationItem $items "copied-diagnostics-boundary-present" ((Test-JsonProperty $report "ReportBoundary") -and (Test-JsonProperty $report.ReportBoundary "CopiedDiagnosticsBoundary")) "blocker" "ReportBoundary.CopiedDiagnosticsBoundary must be present."
    Add-ValidationItem $items "parser-diagnostics-kind" ((Test-JsonProperty $report "ReportBoundary") -and [string]$report.ReportBoundary.ParserDiagnosticsEvidenceKind -eq "copied-parser-diagnostics") "blocker" "ReportBoundary.ParserDiagnosticsEvidenceKind must be copied-parser-diagnostics."
    Add-ValidationItem $items "parser-refitter-diagnostics-kind" ((Test-JsonProperty $report "ReportBoundary") -and [string]$report.ReportBoundary.ParserRefitterDiagnosticsEvidenceKind -eq "copied-parser-refitter-diagnostics") "blocker" "ReportBoundary.ParserRefitterDiagnosticsEvidenceKind must be copied-parser-refitter-diagnostics."
    Add-ValidationItem $items "copied-diagnostics-not-runtime-proof" ((Test-JsonProperty $report "ReportBoundary") -and -not [bool]$report.ReportBoundary.CanPromoteCopiedDiagnosticsToRuntimeProof) "blocker" "Copied parser/refitter diagnostics cannot promote runtime proof."
    Add-ValidationItem $items "parser-diagnostics-owner-action-present" ((Test-JsonProperty $report "ReportBoundary") -and (Test-JsonProperty $report.ReportBoundary "ParserDiagnosticsOwnerAction")) "blocker" "ReportBoundary.ParserDiagnosticsOwnerAction must be present."
    Add-ValidationItem $items "forbidden-substitutes-complete" (($forbiddenSubstitutes | Where-Object { $forbidden -notcontains $_ }).Count -eq 0) "blocker" "ReportBoundary.ForbiddenSubstitutes must include every non-proof substitute."
    Add-ValidationItem $items "option-status-present" ((Test-JsonProperty $report.OptionImplementationStatus "ParsedOptions") -and (Test-JsonProperty $report.OptionImplementationStatus "AppliedOptions") -and (Test-JsonProperty $report.OptionImplementationStatus "ParseOnlyOptions")) "blocker" "OptionImplementationStatus must split parsed/applied/parse-only options."
    $appliedOptions = @($report.OptionImplementationStatus.AppliedOptions | ForEach-Object { [string]$_ })

    $refitAttempted = (Test-JsonProperty $report.RefitSnapshot "Attempted") -and [bool]$report.RefitSnapshot.Attempted
    $refitReady = $true
    $refitSourceMatches = $true
    if ($refitAttempted) {
        $refitReady = [bool]$report.RefitSnapshot.Succeeded -and
            [string]$report.RefitSnapshot.State -eq "onnx-refit-complete" -and
            [bool]$report.RefitSnapshot.EngineRefittableBefore -and
            [bool]$report.RefitSnapshot.EngineRefittableAfter -and
            [bool]$report.RefitSnapshot.ParserRefitReturned -and
            [bool]$report.RefitSnapshot.EngineRefitReturned -and
            [int]$report.RefitSnapshot.ParserErrorCount -eq 0 -and
            @($report.RefitSnapshot.MissingWeightsAfter).Count -eq 0 -and
            [bool]$report.RefitSnapshot.ContextCreationAllowed -and
            [string]$report.RefitSnapshot.SourceSha256 -cmatch "^[0-9a-f]{64}$" -and
            [long]$report.RefitSnapshot.SourceLengthBytes -gt 0 -and
            $appliedOptions -contains "--refitFromOnnx"

        $refitSourcePath = [string]$report.RefitSnapshot.SourcePath
        $refitSourceMatches = -not [string]::IsNullOrWhiteSpace($refitSourcePath) -and
            (Test-Path -LiteralPath $refitSourcePath -PathType Leaf)
        if ($refitSourceMatches) {
            $refitSourceFile = Get-Item -LiteralPath $refitSourcePath
            $refitSourceSha256 = (Get-FileHash -LiteralPath $refitSourcePath -Algorithm SHA256).Hash.ToLowerInvariant()
            $refitSourceMatches = $refitSourceFile.Length -eq [long]$report.RefitSnapshot.SourceLengthBytes -and
                $refitSourceSha256 -ceq [string]$report.RefitSnapshot.SourceSha256
        }
    }
    Add-ValidationItem $items "refit-lifecycle-consistent" $refitReady "blocker" "An attempted ONNX refit must pass parser load, engine commit, zero-error/missing, applied-option, and context gates."
    Add-ValidationItem $items "refit-source-file-match" $refitSourceMatches "blocker" "An attempted ONNX refit source must exist and match its reported length and SHA256."

    $persistenceAttempted = (Test-JsonProperty $report.RefitPersistenceSnapshot "Attempted") -and [bool]$report.RefitPersistenceSnapshot.Attempted
    $persistenceReady = $true
    $persistenceFilesMatch = $true
    if ($persistenceAttempted) {
        $excludeWeightsFlag = 1
        $persistenceReady = [bool]$report.RefitPersistenceSnapshot.Succeeded -and
            [string]$report.RefitPersistenceSnapshot.State -eq "refitted-plan-persisted-and-reloaded" -and
            (([int]$report.RefitPersistenceSnapshot.SerializationFlagsBefore -band $excludeWeightsFlag) -ne 0) -and
            (([int]$report.RefitPersistenceSnapshot.SerializationFlagsAfter -band $excludeWeightsFlag) -eq 0) -and
            [bool]$report.RefitPersistenceSnapshot.RefittableWeightsIncludedInSerialization -and
            [bool]$report.RefitPersistenceSnapshot.ArtifactDiffersFromStrippedPlan -and
            [bool]$report.RefitPersistenceSnapshot.OriginalRefittedEngineDisposedBeforeReload -and
            [bool]$report.RefitPersistenceSnapshot.ReloadAttempted -and
            [bool]$report.RefitPersistenceSnapshot.ReloadSucceeded -and
            [int]$report.RefitPersistenceSnapshot.ReloadIOTensorCount -gt 0 -and
            [int]$report.RefitPersistenceSnapshot.ReloadLayerCount -gt 0 -and
            [int]$report.RefitPersistenceSnapshot.ReloadOptimizationProfileCount -gt 0 -and
            [bool]$report.RefitPersistenceSnapshot.ReloadContextCreationAllowed -and
            $appliedOptions -contains "--saveRefittedEngine"
        if ([bool]$report.InferenceRan) {
            $persistenceReady = $persistenceReady -and
                [bool]$report.RefitPersistenceSnapshot.ReloadEngineSelectedForRuntime -and
                [bool]$report.RefitPersistenceSnapshot.InferenceRanFromReloadedEngine
        }

        $strippedPlanPath = [string]$report.RefitPersistenceSnapshot.StrippedPlanPath
        $persistedPlanPath = [string]$report.RefitPersistenceSnapshot.PersistedPlanPath
        $persistenceFilesMatch = -not [string]::IsNullOrWhiteSpace($strippedPlanPath) -and
            -not [string]::IsNullOrWhiteSpace($persistedPlanPath) -and
            -not [string]::Equals($strippedPlanPath, $persistedPlanPath, [StringComparison]::OrdinalIgnoreCase) -and
            (Test-Path -LiteralPath $strippedPlanPath -PathType Leaf) -and
            (Test-Path -LiteralPath $persistedPlanPath -PathType Leaf)
        if ($persistenceFilesMatch) {
            $strippedPlanFile = Get-Item -LiteralPath $strippedPlanPath
            $persistedPlanFile = Get-Item -LiteralPath $persistedPlanPath
            $strippedPlanSha256 = (Get-FileHash -LiteralPath $strippedPlanPath -Algorithm SHA256).Hash.ToLowerInvariant()
            $persistedPlanSha256 = (Get-FileHash -LiteralPath $persistedPlanPath -Algorithm SHA256).Hash.ToLowerInvariant()
            $persistenceFilesMatch = $strippedPlanFile.Length -eq [long]$report.RefitPersistenceSnapshot.StrippedPlanLengthBytes -and
                $persistedPlanFile.Length -eq [long]$report.RefitPersistenceSnapshot.PersistedPlanLengthBytes -and
                $strippedPlanSha256 -ceq [string]$report.RefitPersistenceSnapshot.StrippedPlanSha256 -and
                $persistedPlanSha256 -ceq [string]$report.RefitPersistenceSnapshot.PersistedPlanSha256 -and
                $strippedPlanSha256 -cne $persistedPlanSha256
        }
    }
    Add-ValidationItem $items "refit-persistence-consistent" $persistenceReady "blocker" "An attempted refitted-plan persistence lifecycle must clear ExcludeWeights, dispose the original owner, pass reload metadata/context gates, and route requested runtime through the reload."
    Add-ValidationItem $items "refit-persistence-file-match" $persistenceFilesMatch "blocker" "Attempted stripped and persisted plans must be distinct files matching their reported lengths and SHA256 values."
    Add-ValidationItem $items "preflight-boundary-present" (Test-JsonProperty $report.PreflightMetadata "EvidenceBoundary") "blocker" "PreflightMetadata.EvidenceBoundary must be present."
    Add-ValidationItem $items "loaded-engine-boundary-present" (Test-JsonProperty $report.LoadedEngineDiagnostics "EvidenceBoundary") "blocker" "LoadedEngineDiagnostics.EvidenceBoundary must be present."
    Add-ValidationItem $items "layer-info-boundary-present" (Test-JsonProperty $report.LayerInfoArtifact "EvidenceBoundary") "blocker" "LayerInfoArtifact.EvidenceBoundary must be present."
    Add-ValidationItem $items "layer-info-pointer-free" ((Test-JsonProperty $report.LayerInfoArtifact "PointerFreeCopiedSnapshot") -and [bool]$report.LayerInfoArtifact.PointerFreeCopiedSnapshot) "blocker" "LayerInfoArtifact must be a pointer-free copied snapshot."
    Add-ValidationItem $items "layer-info-not-runtime-proof" ((Test-JsonProperty $report.LayerInfoArtifact "CanPromoteRuntimeProof") -and -not [bool]$report.LayerInfoArtifact.CanPromoteRuntimeProof) "blocker" "LayerInfoArtifact cannot promote runtime proof."
    Add-ValidationItem $items "layer-info-not-release-proof" ((Test-JsonProperty $report.LayerInfoArtifact "CanPromoteReleaseProof") -and -not [bool]$report.LayerInfoArtifact.CanPromoteReleaseProof) "blocker" "LayerInfoArtifact cannot promote release proof."
    $layerInfoCollected = (Test-JsonProperty $report.LayerInfoArtifact "Collected") -and [bool]$report.LayerInfoArtifact.Collected
    $layerInfoSha256 = Get-JsonString $report.LayerInfoArtifact "Sha256"
    Add-ValidationItem $items "layer-info-collected-hash" ((-not $layerInfoCollected) -or $layerInfoSha256 -cmatch "^[0-9a-f]{64}$") "blocker" "A collected LayerInfoArtifact must record a lowercase SHA256."
    $layerInfoExportWritten = (Test-JsonProperty $report.LayerInfoArtifact "ExportWritten") -and [bool]$report.LayerInfoArtifact.ExportWritten
    $layerInfoExportRequested = (Test-JsonProperty $report.LayerInfoArtifact "ExportRequested") -and [bool]$report.LayerInfoArtifact.ExportRequested
    $layerInfoExportPath = Get-JsonString $report.LayerInfoArtifact "ExportPath"
    Add-ValidationItem $items "layer-info-export-consistent" ((-not $layerInfoExportWritten) -or ($layerInfoExportRequested -and $layerInfoCollected -and -not [string]::IsNullOrWhiteSpace($layerInfoExportPath))) "blocker" "A written LayerInfoArtifact must be requested, collected, and have an export path."
    $layerInfoFileMatches = $true
    if ($layerInfoExportWritten -and -not [string]::IsNullOrWhiteSpace($layerInfoExportPath)) {
        $resolvedLayerInfoPath = Resolve-RepoPath $layerInfoExportPath
        $layerInfoFileMatches = (Test-Path -LiteralPath $resolvedLayerInfoPath -PathType Leaf)
        if ($layerInfoFileMatches) {
            $layerInfoFile = Get-Item -LiteralPath $resolvedLayerInfoPath
            $layerInfoActualSha256 = (Get-FileHash -LiteralPath $resolvedLayerInfoPath -Algorithm SHA256).Hash.ToLowerInvariant()
            $layerInfoFileMatches = $layerInfoFile.Length -eq [long]$report.LayerInfoArtifact.LengthBytes -and $layerInfoActualSha256 -ceq $layerInfoSha256
        }
    }
    Add-ValidationItem $items "layer-info-export-file-match" $layerInfoFileMatches "blocker" "A written LayerInfoArtifact file must exist and match its reported length and SHA256."
    Add-ValidationItem $items "layer-info-json-content-kind" (([string]$report.LayerInfoArtifact.InformationFormat -ne "Json") -or ([string]$report.LayerInfoArtifact.ContentKind -eq "json-document")) "blocker" "Json layer information must use the json-document content kind."
    Add-ValidationItem $items "binding-metadata-boundary-present" (Test-JsonProperty $report.BindingMetadata "EvidenceBoundary") "blocker" "BindingMetadata.EvidenceBoundary must be present."
    Add-ValidationItem $items "binding-metadata-pointer-free" ((Test-JsonProperty $report.BindingMetadata "PointerFreeCopiedSnapshot") -and [bool]$report.BindingMetadata.PointerFreeCopiedSnapshot) "blocker" "BindingMetadata must be a pointer-free copied snapshot."
    Add-ValidationItem $items "binding-metadata-not-runtime-proof" ((Test-JsonProperty $report.BindingMetadata "CanPromoteRuntimeProof") -and -not [bool]$report.BindingMetadata.CanPromoteRuntimeProof) "blocker" "BindingMetadata cannot promote runtime proof."
    Add-ValidationItem $items "binding-metadata-not-release-proof" ((Test-JsonProperty $report.BindingMetadata "CanPromoteReleaseProof") -and -not [bool]$report.BindingMetadata.CanPromoteReleaseProof) "blocker" "BindingMetadata cannot promote release proof."
    Add-ValidationItem $items "capability-probe-boundary-present" (Test-JsonProperty $report.CapabilityProbe "EvidenceBoundary") "blocker" "CapabilityProbe.EvidenceBoundary must be present."
    Add-ValidationItem $items "non-proof-build-report" (-not [bool]$report.IsPackageConsumerRuntimeProof) "blocker" "TensorRtExec report cannot by itself promote package-consumer-runtime proof."
    Add-ValidationItem $items "no-yolodet" (-not $text.Contains("YoloDet", [System.StringComparison]::OrdinalIgnoreCase)) "blocker" "Report must not refer to the retired YoloDet sample name."
}

$failedBlockerCount = @($items | Where-Object { -not $_.passed -and $_.severity -eq "blocker" }).Count
$validationState = if ($failedBlockerCount -eq 0) { "tensor-rt-exec-report-ready" } else { "invalid" }

$result = [ordered]@{
    recordKind = "tensor-rt-exec-report-validation"
    inputPath = $InputPath
    resolvedInputPath = $resolvedInput
    validationState = $validationState
    failedBlockers = $failedBlockerCount
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
    proofBoundary = "TensorRtExec and OnnxToEngine reports are build/report evidence. They do not replace real-model-runtime or package-consumer-runtime proof."
    validationItems = [object[]]@($items.ToArray())
}

$resolvedOutput = Resolve-RepoPath $OutputPath
$outputDirectory = Split-Path -Parent $resolvedOutput
if (-not [string]::IsNullOrWhiteSpace($outputDirectory)) {
    New-Item -ItemType Directory -Force -Path $outputDirectory | Out-Null
}

$result | ConvertTo-Json -Depth 64 | Set-Content -LiteralPath $resolvedOutput -Encoding utf8

Write-Host "ValidationState=$validationState FailedBlockers=$failedBlockerCount"
Write-Host "TensorRtExec report validation written:"
Write-Host "  Json=$resolvedOutput"

if ($Strict.IsPresent -and $failedBlockerCount -gt 0) {
    exit 1
}
