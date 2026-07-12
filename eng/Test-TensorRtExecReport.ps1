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
    "NormalizedCommandLine",
    "NormalizedCommandSha256",
    "DeploymentOptions",
    "RuntimeOptions",
    "PreflightMetadata",
    "LoadedEngineDiagnostics",
    "CapabilityProbe",
    "WorkspaceBytes",
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
    "capability-probe-only"
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
    Add-ValidationItem $items "forbidden-substitutes-complete" (($forbiddenSubstitutes | Where-Object { $forbidden -notcontains $_ }).Count -eq 0) "blocker" "ReportBoundary.ForbiddenSubstitutes must include every non-proof substitute."
    Add-ValidationItem $items "option-status-present" ((Test-JsonProperty $report.OptionImplementationStatus "ParsedOptions") -and (Test-JsonProperty $report.OptionImplementationStatus "AppliedOptions") -and (Test-JsonProperty $report.OptionImplementationStatus "ParseOnlyOptions")) "blocker" "OptionImplementationStatus must split parsed/applied/parse-only options."
    Add-ValidationItem $items "preflight-boundary-present" (Test-JsonProperty $report.PreflightMetadata "EvidenceBoundary") "blocker" "PreflightMetadata.EvidenceBoundary must be present."
    Add-ValidationItem $items "loaded-engine-boundary-present" (Test-JsonProperty $report.LoadedEngineDiagnostics "EvidenceBoundary") "blocker" "LoadedEngineDiagnostics.EvidenceBoundary must be present."
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
