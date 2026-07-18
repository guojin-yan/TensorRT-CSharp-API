[CmdletBinding()]
param(
    [Parameter(Mandatory = $false)]
    [string] $InputPath = "samples/assets/yolovision-real-asset-owner-backfill-pack.json",

    [Parameter(Mandatory = $false)]
    [string] $ContractPath = "samples/YoloVision/yolovision-task-output-contract.json",

    [Parameter(Mandatory = $false)]
    [string] $ArticleCasePackPath = "samples/assets/yolovision-article-case-pack.json",

    [Parameter(Mandatory = $false)]
    [switch] $Strict,

    [Parameter(Mandatory = $false)]
    [string] $OutputPath = "artifacts/yolovision/yolovision-real-asset-owner-backfill-pack-validation.json"
)

$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
Set-Location $repoRoot

function Resolve-RepoPath {
    param([Parameter(Mandatory = $true)][string] $Path)
    if ([System.IO.Path]::IsPathRooted($Path)) {
        return [System.IO.Path]::GetFullPath($Path)
    }
    return [System.IO.Path]::GetFullPath((Join-Path $repoRoot $Path))
}

function Test-Sha256 {
    param([AllowNull()][string] $Value)
    return -not [string]::IsNullOrWhiteSpace($Value) -and $Value -cmatch "^[0-9a-fA-F]{64}$"
}

function Test-Placeholder {
    param([AllowNull()][string] $Value)
    if ([string]::IsNullOrWhiteSpace($Value)) { return $true }
    return $Value -match "owner-required|owner-action-required|template-only|owner-required-or-no-stderr"
}

function Get-JsonString {
    param(
        [AllowNull()] $Object,
        [Parameter(Mandatory = $true)][string] $Name
    )
    if ($null -eq $Object -or $null -eq $Object.PSObject.Properties[$Name] -or $null -eq $Object.$Name) {
        return $null
    }
    return [string]$Object.$Name
}

function Add-ValidationItem {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyCollection()]
        [System.Collections.Generic.List[object]] $Items,
        [Parameter(Mandatory = $true)][string] $Id,
        [Parameter(Mandatory = $true)][bool] $Passed,
        [Parameter(Mandatory = $true)][string] $Severity,
        [Parameter(Mandatory = $true)][string] $Message
    )
    $Items.Add([ordered]@{
        id = $Id
        passed = $Passed
        severity = $Severity
        message = $Message
    }) | Out-Null
}

function Test-RequiredOrHash {
    param([AllowNull()][string] $Value)
    return (Test-Placeholder $Value) -or (Test-Sha256 $Value)
}

function Test-FileHashMatches {
    param(
        [Parameter(Mandatory = $true)][string] $Path,
        [AllowNull()][string] $ExpectedHash
    )

    if (-not (Test-Sha256 $ExpectedHash)) { return $true }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    return [string]::Equals(
        (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash,
        $ExpectedHash,
        [System.StringComparison]::OrdinalIgnoreCase)
}

function Test-PreflightExecutionDisabled {
    param([AllowNull()] $Execution)

    return $null -ne $Execution -and
        -not [bool]$Execution.tensorRtRuntimeProbed -and
        -not [bool]$Execution.onnxParserInvoked -and
        -not [bool]$Execution.engineBuildInvoked -and
        -not [bool]$Execution.inferenceInvoked
}

function Test-PreflightBoundaryDisabled {
    param([AllowNull()] $Boundary)

    return $null -ne $Boundary -and
        (Get-JsonString $Boundary "proofClassification") -eq "precheck" -and
        -not [bool]$Boundary.isRuntimeProof -and
        -not [bool]$Boundary.isRealModelRuntimeProof -and
        -not [bool]$Boundary.isPackageConsumerRuntimeProof -and
        -not [bool]$Boundary.canPromoteRealModelRuntime -and
        -not [bool]$Boundary.canPromotePackageConsumerRuntime
}

function ConvertTo-StringArray {
    param([AllowNull()] $Value)
    if ($null -eq $Value) { return @() }
    return @($Value | ForEach-Object { [string]$_ })
}

function Get-CommandArgumentValue {
    param(
        [AllowNull()][string] $Command,
        [Parameter(Mandatory = $true)][string] $ArgumentName
    )

    if ([string]::IsNullOrWhiteSpace($Command)) { return $null }
    $match = [regex]::Match($Command, ("--" + [regex]::Escape($ArgumentName) + "\s+(?<value>\S+)"))
    if ($match.Success) { return $match.Groups["value"].Value.Trim() }
    return $null
}

function Normalize-YoloFamilyAlias {
    param([AllowNull()][string] $Family)

    if ([string]::IsNullOrWhiteSpace($Family)) { return "" }
    $value = $Family.Trim().ToLowerInvariant()
    if ($value -match "^v(?<number>5|6|7|8|9|10|11|26)$") {
        return "yolov" + $Matches["number"]
    }
    return $value
}

$resolvedInput = Resolve-RepoPath $InputPath
$resolvedContract = Resolve-RepoPath $ContractPath
$resolvedArticleCasePack = Resolve-RepoPath $ArticleCasePackPath
$items = New-Object System.Collections.Generic.List[object]
$records = New-Object System.Collections.Generic.List[object]

Add-ValidationItem $items "file-exists" (Test-Path -LiteralPath $resolvedInput) "blocker" "Owner backfill pack must exist."
if (-not (Test-Path -LiteralPath $resolvedInput)) {
    throw "Missing owner backfill pack: $InputPath"
}

Add-ValidationItem $items "task-output-contract-exists" (Test-Path -LiteralPath $resolvedContract) "blocker" "YoloVision task/output contract must exist."
if (-not (Test-Path -LiteralPath $resolvedContract)) {
    throw "Missing YoloVision task/output contract: $ContractPath"
}

Add-ValidationItem $items "article-case-pack-exists" (Test-Path -LiteralPath $resolvedArticleCasePack) "blocker" "YoloVision article case pack must exist."
if (-not (Test-Path -LiteralPath $resolvedArticleCasePack)) {
    throw "Missing YoloVision article case pack: $ArticleCasePackPath"
}

$pack = Get-Content -LiteralPath $resolvedInput -Raw | ConvertFrom-Json -Depth 64
$contract = Get-Content -LiteralPath $resolvedContract -Raw | ConvertFrom-Json -Depth 64
$articleCasePack = Get-Content -LiteralPath $resolvedArticleCasePack -Raw | ConvertFrom-Json -Depth 64

$contractTasks = @($contract.tasks)
$contractTaskNames = @($contractTasks | ForEach-Object { [string]$_.task })
$contractFamilyNames = @($contractTasks | ForEach-Object { ConvertTo-StringArray $_.families } | ForEach-Object { [string]$_ })
$articleCaseArray = @($articleCasePack.cases)
$articleCaseTasks = @($articleCaseArray | ForEach-Object { [string]$_.task })
$contractTaskMap = @{}
foreach ($contractTask in $contractTasks) {
    $contractTaskMap[[string]$contractTask.task] = $contractTask
}

Add-ValidationItem $items "record-kind" ((Get-JsonString $pack "recordKind") -eq "yolovision-real-asset-owner-backfill-pack") "blocker" "recordKind must be yolovision-real-asset-owner-backfill-pack."
Add-ValidationItem $items "sample-name" ((Get-JsonString $pack "sampleName") -eq "YoloVision") "blocker" "sampleName must be YoloVision."
Add-ValidationItem $items "contract-id" ((Get-JsonString $contract "contractId") -eq "yolovision-task-output-contract") "blocker" "Contract must be yolovision-task-output-contract."
Add-ValidationItem $items "contract-boundary-real-model" ((Get-JsonString $contract "proofBoundary").Contains("not real-model-runtime proof")) "blocker" "Contract boundary must block real-model-runtime proof promotion."
Add-ValidationItem $items "contract-boundary-package" ((Get-JsonString $contract "proofBoundary").Contains("not package-consumer-runtime proof")) "blocker" "Contract boundary must block package-consumer-runtime proof promotion."
Add-ValidationItem $items "contract-task-count-six" ($contractTaskNames.Count -eq 6) "blocker" "Contract must cover six YoloVision tasks."
foreach ($task in @("det", "cls", "seg", "obb", "pose", "sem")) {
    Add-ValidationItem $items ("contract-task-" + $task) ($contractTaskNames -contains $task) "blocker" "Contract must include $task."
}
foreach ($family in @("yolov5", "yolov6", "yolov7", "yolov8", "yolov9", "yolov10", "yolov11", "yolov26", "custom")) {
    Add-ValidationItem $items ("contract-family-" + $family) ($contractFamilyNames -contains $family) "blocker" "Contract must include family $family."
}
Add-ValidationItem $items "pack-state" ((Get-JsonString $pack "packState") -eq "owner-action-required") "required" "Template pack must stay owner-action-required until real owner proof is imported."
Add-ValidationItem $items "does-not-publish" (-not [bool]$pack.performsPublish) "blocker" "Validator pack must not perform publish."
Add-ValidationItem $items "no-package-consumer-promotion" (-not [bool]$pack.canPromotePackageConsumerRuntime) "blocker" "YoloVision backfill pack cannot promote package-consumer-runtime."
Add-ValidationItem $items "no-public-publish" (-not [bool]$pack.canPublishPublicly) "blocker" "Template pack cannot publish publicly."
Add-ValidationItem $items "no-template-real-promotion" (-not [bool]$pack.canPromoteRealModelRuntime) "blocker" "Template pack cannot promote real-model-runtime."
Add-ValidationItem $items "proof-boundary-real-model" ((Get-JsonString $pack "proofBoundary").Contains("not real-model-runtime proof")) "blocker" "proofBoundary must state it is not real-model-runtime proof."
$preflightContract = $pack.preflightContract
$preflightSchemaPath = Get-JsonString $preflightContract "schemaPath"
$preflightSchemaExists = -not [string]::IsNullOrWhiteSpace($preflightSchemaPath) -and (Test-Path -LiteralPath (Resolve-RepoPath $preflightSchemaPath) -PathType Leaf)
Add-ValidationItem $items "preflight-schema-version" ((Get-JsonString $preflightContract "schemaVersion") -eq "yolovision-preflight.v1") "blocker" "Pack preflightContract must declare yolovision-preflight.v1."
Add-ValidationItem $items "preflight-schema-file" $preflightSchemaExists "blocker" "Pack preflightContract schemaPath must point to the checked-in preflight schema."
Add-ValidationItem $items "preflight-proof-classification" ((Get-JsonString $preflightContract "proofClassification") -eq "precheck") "blocker" "Pack preflightContract must remain proofClassification=precheck."
Add-ValidationItem $items "preflight-no-runtime-promotion" (-not [bool]$preflightContract.canPromoteRealModelRuntime -and -not [bool]$preflightContract.canPromotePackageConsumerRuntime) "blocker" "Pack preflightContract cannot promote runtime proof."

$forbidden = @($pack.forbiddenSubstitutes | ForEach-Object { [string]$_ })
foreach ($forbiddenName in @("build-only", "dry-run", "template", "ProjectReference", "TensorRtExec report", "YoloVision matrix", "sidecar-only report", "blocked-by-cuda-driver")) {
    Add-ValidationItem $items ("forbidden-substitute-" + ($forbiddenName -replace "[^A-Za-z0-9]+", "-").Trim("-")) ($forbidden -contains $forbiddenName) "required" "forbiddenSubstitutes must include $forbiddenName."
}

$requiredPreflightEvidence = ConvertTo-StringArray $pack.requiredPreflightEvidence
foreach ($requiredPreflightField in @(
        "yoloVisionPreflight.command",
        "yoloVisionPreflight.reportPath",
        "yoloVisionPreflight.reportSha256",
        "yoloVisionPreflight.schemaVersion=yolovision-preflight.v1",
        "yoloVisionPreflight.proofClassification=precheck",
        "yoloVisionPreflight.execution.*=false",
        "yoloVisionPreflight.boundary.canPromote*=false"
    )) {
    Add-ValidationItem $items ("required-preflight-" + ($requiredPreflightField -replace "[^A-Za-z0-9]+", "-").Trim("-")) ($requiredPreflightEvidence -contains $requiredPreflightField) "blocker" "requiredPreflightEvidence must include $requiredPreflightField."
}

$caseArray = @($pack.cases)
Add-ValidationItem $items "case-count-six" ($caseArray.Count -eq 6) "blocker" "Pack must cover six YOLOv8n task cases, including semantic segmentation."
$caseTasks = @($caseArray | ForEach-Object { [string]$_.task })
foreach ($task in @("det", "seg", "pose", "obb", "cls", "sem")) {
    Add-ValidationItem $items ("case-task-" + $task) ($caseTasks -contains $task) "blocker" "Pack must include $task."
    Add-ValidationItem $items ("case-task-in-contract-" + $task) ($contractTaskNames -contains $task) "blocker" "Backfill task $task must be declared in yolovision-task-output-contract.json."
}
foreach ($task in @("det", "cls", "seg", "obb", "pose", "sem")) {
    Add-ValidationItem $items ("article-case-task-" + $task) ($articleCaseTasks -contains $task) "blocker" "Article case pack must provide owner-action path for $task."
}

foreach ($case in $caseArray) {
    $caseItems = New-Object System.Collections.Generic.List[object]
    $id = Get-JsonString $case "id"
    $task = Get-JsonString $case "task"
    $contractTask = $contractTaskMap[$task]
    $contractRequiredMetadata = ConvertTo-StringArray $contractTask.requiredMetadata
    $contractProfileHint = Get-JsonString $contractTask "tensorRtExecProfileHint"
    $articleCase = @($articleCaseArray | Where-Object { [string]$_.id -eq $id }) | Select-Object -First 1
    $caseFamily = Get-JsonString $case "family"
    $runCommand = Get-JsonString $case.yoloVision "runCommand"
    $preflightCommand = Get-JsonString $case.yoloVisionPreflight "command"
    $runFamilyToken = Get-CommandArgumentValue -Command $runCommand -ArgumentName "family"
    $preflightFamilyToken = Get-CommandArgumentValue -Command $preflightCommand -ArgumentName "family"
    $normalizedFamily = Normalize-YoloFamilyAlias $caseFamily
    $allowedFamilies = ConvertTo-StringArray $contractTask.families
    $contractArticleEntrypoints = ConvertTo-StringArray $contractTask.articleEntrypoints
    $articlePath = Get-JsonString $case "article"

    Add-ValidationItem $caseItems "case-id-present" (-not [string]::IsNullOrWhiteSpace($id)) "blocker" "Case id is required."
    Add-ValidationItem $caseItems "case-task-in-contract" ($contractTaskMap.ContainsKey($task)) "blocker" "Case task must exist in yolovision-task-output-contract.json."
    Add-ValidationItem $caseItems "case-family-present" (-not [string]::IsNullOrWhiteSpace($caseFamily)) "blocker" "Case family is required."
    Add-ValidationItem $caseItems "case-family-in-contract" ($allowedFamilies -contains $normalizedFamily) "blocker" "Case family must be declared for task $task in yolovision-task-output-contract.json."
    Add-ValidationItem $caseItems "case-family-command-match" ((Normalize-YoloFamilyAlias $runFamilyToken) -eq $normalizedFamily) "blocker" "YoloVision run command family must match the case family."
    Add-ValidationItem $caseItems "case-preflight-family-match" ((Normalize-YoloFamilyAlias $preflightFamilyToken) -eq $normalizedFamily) "blocker" "YoloVision preflight command family must match the case family."
    Add-ValidationItem $caseItems "contract-required-metadata-present" ($contractRequiredMetadata.Count -ge 4) "blocker" "Contract requiredMetadata must be present for task $task."
    Add-ValidationItem $caseItems "contract-profile-hint-present" (-not [string]::IsNullOrWhiteSpace($contractProfileHint) -and $contractProfileHint.Contains("--minShapes")) "blocker" "Contract must provide TensorRtExec profile hint for task $task."
    Add-ValidationItem $caseItems "case-state-owner-action" ((Get-JsonString $case "state") -eq "owner-action-required") "required" "Template case must remain owner-action-required."
    Add-ValidationItem $caseItems "case-no-real-promotion" (-not [bool]$case.canPromoteRealModelRuntime) "blocker" "Template case cannot promote real-model-runtime."
    Add-ValidationItem $caseItems "case-no-package-promotion" (-not [bool]$case.canPromotePackageConsumerRuntime) "blocker" "Template case cannot promote package-consumer-runtime."
    Add-ValidationItem $caseItems "article-path-present" (-not [string]::IsNullOrWhiteSpace($articlePath)) "required" "Case must link an article."
    Add-ValidationItem $caseItems "article-file-exists" (-not [string]::IsNullOrWhiteSpace($articlePath) -and (Test-Path -LiteralPath (Resolve-RepoPath $articlePath) -PathType Leaf)) "blocker" "Case article path must exist in the repository."
    Add-ValidationItem $caseItems "article-entrypoint-in-contract" ($contractArticleEntrypoints -contains $articlePath) "blocker" "Case article must be listed in the task contract articleEntrypoints."
    Add-ValidationItem $caseItems "article-case-present" ($null -ne $articleCase) "blocker" "Owner case id must exist in the article case pack."
    Add-ValidationItem $caseItems "article-case-family-task-match" ($null -ne $articleCase -and [string]$articleCase.task -eq $task -and (Normalize-YoloFamilyAlias ([string]$articleCase.family)) -eq $normalizedFamily) "blocker" "Article case task and family must match the owner case and contract."

    Add-ValidationItem $caseItems "model-source-url-present" (-not [string]::IsNullOrWhiteSpace((Get-JsonString $case.model "sourceUrl"))) "blocker" "model.sourceUrl is required."
    Add-ValidationItem $caseItems "model-license-present" (-not [string]::IsNullOrWhiteSpace((Get-JsonString $case.model "license"))) "blocker" "model.license is required."
    Add-ValidationItem $caseItems "model-sha256-required-or-real" (Test-RequiredOrHash (Get-JsonString $case.model "sha256")) "blocker" "model.sha256 must be owner-required or a real SHA256."
    Add-ValidationItem $caseItems "onnx-sha256-required-or-real" (Test-RequiredOrHash (Get-JsonString $case.model "onnxSha256")) "blocker" "model.onnxSha256 must be owner-required or a real SHA256."
    Add-ValidationItem $caseItems "export-command-present" ((Get-JsonString $case.model "exportCommand").Contains("yolo export")) "required" "model.exportCommand must record yolo export."

    Add-ValidationItem $caseItems "labels-license-present" (-not [string]::IsNullOrWhiteSpace((Get-JsonString $case.labels "license"))) "blocker" "labels.license is required."
    Add-ValidationItem $caseItems "labels-sha256-required-or-real" (Test-RequiredOrHash (Get-JsonString $case.labels "sha256")) "blocker" "labels.sha256 must be owner-required or a real SHA256."
    Add-ValidationItem $caseItems "input-image-sha256-required-or-real" (Test-RequiredOrHash (Get-JsonString $case.input "imageSha256")) "blocker" "input.imageSha256 must be owner-required or a real SHA256."
    Add-ValidationItem $caseItems "preprocessed-sha256-required-or-real" (Test-RequiredOrHash (Get-JsonString $case.input "preprocessedTensorSha256")) "blocker" "input.preprocessedTensorSha256 must be owner-required or a real SHA256."
    Add-ValidationItem $caseItems "input-shape-present" (-not [string]::IsNullOrWhiteSpace((Get-JsonString $case.input "inputShape"))) "required" "input.inputShape must be present."

    Add-ValidationItem $caseItems "tensorrtexec-build-command" ((Get-JsonString $case.tensorRtExec "buildCommand").Contains("applications\TensorRtExec")) "blocker" "TensorRtExec build command must be present."
    Add-ValidationItem $caseItems "tensorrtexec-build-only" ((Get-JsonString $case.tensorRtExec "buildCommand").Contains("--buildOnly")) "blocker" "TensorRtExec command must remain build-only."
    Add-ValidationItem $caseItems "tensorrtexec-profile-hint-aligned" ((Get-JsonString $case.tensorRtExec "buildCommand").Contains($contractProfileHint)) "blocker" "TensorRtExec build command must include the contract profile hint for task $task."
    Add-ValidationItem $caseItems "tensorrtexec-report-sha256-required-or-real" (Test-RequiredOrHash (Get-JsonString $case.tensorRtExec "reportSha256")) "blocker" "TensorRtExec reportSha256 must be owner-required or a real SHA256."
    Add-ValidationItem $caseItems "engine-sha256-required-or-real" (Test-RequiredOrHash (Get-JsonString $case.tensorRtExec "engineSha256")) "blocker" "engineSha256 must be owner-required or a real SHA256."

    Add-ValidationItem $caseItems "yolovision-run-command" ((Get-JsonString $case.yoloVision "runCommand").Contains("samples\YoloVision")) "blocker" "YoloVision run command must be present."
    Add-ValidationItem $caseItems "yolovision-task-switch" ((Get-JsonString $case.yoloVision "runCommand").Contains("--task $task")) "blocker" "YoloVision command must include the task switch."
    $evidence = @($case.yoloVision.expectedEvidenceLines | ForEach-Object { [string]$_ })
    Add-ValidationItem $caseItems "expected-yolovision-passed" (@($evidence | Where-Object { $_.Contains("YoloVision Passed=True") }).Count -gt 0) "blocker" "Expected evidence must include YoloVision Passed=True."
    Add-ValidationItem $caseItems "run-log-sha256-required-or-real" (Test-RequiredOrHash (Get-JsonString $case.yoloVision "runLogSha256")) "blocker" "runLogSha256 must be owner-required or a real SHA256."
    Add-ValidationItem $caseItems "output-json-sha256-required-or-real" (Test-RequiredOrHash (Get-JsonString $case.yoloVision "outputJsonSha256")) "blocker" "outputJsonSha256 must be owner-required or a real SHA256."
    Add-ValidationItem $caseItems "stdout-summary-present" (-not [string]::IsNullOrWhiteSpace((Get-JsonString $case.yoloVision "stdoutSummary"))) "required" "stdoutSummary must be present."
    Add-ValidationItem $caseItems "stderr-summary-present" (-not [string]::IsNullOrWhiteSpace((Get-JsonString $case.yoloVision "stderrSummary"))) "required" "stderrSummary must be present."

    $preflight = $case.yoloVisionPreflight
    $preflightCommand = Get-JsonString $preflight "command"
    $preflightReportPath = Get-JsonString $preflight "reportPath"
    $preflightReportSha256 = Get-JsonString $preflight "reportSha256"
    $preflightSchemaPath = Get-JsonString $preflight "schemaPath"
    Add-ValidationItem $caseItems "preflight-present" ($null -ne $preflight) "blocker" "yoloVisionPreflight is required for every owner case."
    Add-ValidationItem $caseItems "preflight-command" (-not [string]::IsNullOrWhiteSpace($preflightCommand) -and $preflightCommand.Contains("samples\YoloVision") -and $preflightCommand.Contains("--preflight")) "blocker" "YoloVision preflight command must be present and non-executing."
    Add-ValidationItem $caseItems "preflight-report-path" (-not [string]::IsNullOrWhiteSpace($preflightReportPath)) "blocker" "YoloVision preflight reportPath is required."
    Add-ValidationItem $caseItems "preflight-report-sha256-required-or-real" (Test-RequiredOrHash $preflightReportSha256) "blocker" "YoloVision preflight reportSha256 must be owner-required or a real SHA256."
    Add-ValidationItem $caseItems "preflight-schema-version" ((Get-JsonString $preflight "schemaVersion") -eq "yolovision-preflight.v1") "blocker" "YoloVision preflight must declare schemaVersion=yolovision-preflight.v1."
    Add-ValidationItem $caseItems "preflight-schema-path" (-not [string]::IsNullOrWhiteSpace($preflightSchemaPath) -and (Test-Path -LiteralPath (Resolve-RepoPath $preflightSchemaPath) -PathType Leaf)) "blocker" "YoloVision preflight schemaPath must point to the checked-in schema."
    Add-ValidationItem $caseItems "preflight-proof-classification" ((Get-JsonString $preflight "proofClassification") -eq "precheck") "blocker" "YoloVision preflight must remain proofClassification=precheck."
    Add-ValidationItem $caseItems "preflight-expected-state" ((Get-JsonString $preflight "expectedState") -eq "owner-action-required") "required" "Template preflight evidence must remain owner-action-required."
    Add-ValidationItem $caseItems "preflight-execution-disabled" (Test-PreflightExecutionDisabled $preflight.execution) "blocker" "YoloVision preflight execution flags must all be false."
    Add-ValidationItem $caseItems "preflight-boundary-disabled" (Test-PreflightBoundaryDisabled $preflight.boundary) "blocker" "YoloVision preflight boundary must remain precheck and non-promotable."

    $preflightReport = $null
    $preflightReportExists = $false
    $preflightReportParsed = $false
    if (-not [string]::IsNullOrWhiteSpace($preflightReportPath)) {
        $resolvedPreflightReportPath = Resolve-RepoPath $preflightReportPath
        $preflightReportExists = Test-Path -LiteralPath $resolvedPreflightReportPath -PathType Leaf
        if ($preflightReportExists) {
            try {
                $preflightReport = Get-Content -LiteralPath $resolvedPreflightReportPath -Raw | ConvertFrom-Json -Depth 64
                $preflightReportParsed = $true
            }
            catch {
                $preflightReportParsed = $false
            }
        }
    }
    $preflightHashPath = if ($preflightReportExists) { $resolvedPreflightReportPath } else { [IO.Path]::GetFullPath((Join-Path $repoRoot "__missing-preflight-report__")) }
    Add-ValidationItem $caseItems "preflight-report-hash" (Test-FileHashMatches -Path $preflightHashPath -ExpectedHash $preflightReportSha256) "blocker" "A real preflight reportSha256 must match the report file."
    Add-ValidationItem $caseItems "preflight-report-schema" (-not $preflightReportExists -or ($preflightReportParsed -and (Get-JsonString $preflightReport "schemaVersion") -eq "yolovision-preflight.v1")) "blocker" "An existing preflight report must be valid JSON with schemaVersion=yolovision-preflight.v1."
    Add-ValidationItem $caseItems "preflight-report-execution-disabled" (-not $preflightReportExists -or ($preflightReportParsed -and (Test-PreflightExecutionDisabled $preflightReport.execution))) "blocker" "An existing preflight report must show no TensorRT, parser, engine, or inference execution."
    Add-ValidationItem $caseItems "preflight-report-boundary-disabled" (-not $preflightReportExists -or ($preflightReportParsed -and (Test-PreflightBoundaryDisabled $preflightReport.boundary))) "blocker" "An existing preflight report must show proofClassification=precheck and all promotion flags false."

    $caseMetadataKeys = @()
    if ($null -ne $case.outputMetadata) {
        $caseMetadataKeys = @($case.outputMetadata.PSObject.Properties.Name | ForEach-Object { [string]$_ })
    }
    foreach ($metadataName in $contractRequiredMetadata) {
        $metadataCovered = $caseMetadataKeys -contains $metadataName
        if (-not $metadataCovered) {
            switch ($metadataName) {
                "inputShape" { $metadataCovered = -not [string]::IsNullOrWhiteSpace((Get-JsonString $case.input "inputShape")) }
                "labelsPath" { $metadataCovered = -not [string]::IsNullOrWhiteSpace((Get-JsonString $case.labels "path")) }
                "classificationOutput" { $metadataCovered = (Get-JsonString $case.yoloVision "runCommand").Contains("--classification-output") }
                "outputRoleMap" { $metadataCovered = (Get-JsonString $case.yoloVision "runCommand").Contains("--output-role-map") }
                "angleOutput" { $metadataCovered = (Get-JsonString $case.yoloVision "runCommand").Contains("--obb-angle-output") -or (Get-JsonString $case.yoloVision "runCommand").Contains("--rotated-box-layout") }
                "semanticOutput" { $metadataCovered = (Get-JsonString $case.yoloVision "runCommand").Contains("--semantic-output") }
                "mapWidth" { $metadataCovered = $caseMetadataKeys -contains "semanticMapShape" }
                "mapHeight" { $metadataCovered = $caseMetadataKeys -contains "semanticMapShape" }
                "argmaxRule" { $metadataCovered = $caseMetadataKeys -contains "postprocessMetadata" -or $caseMetadataKeys -contains "argmaxRule" }
            }
        }
        Add-ValidationItem $caseItems ("contract-required-metadata-" + $metadataName) $metadataCovered "blocker" "Task $task must map contract requiredMetadata.$metadataName in the owner backfill case."
    }

    Add-ValidationItem $caseItems "owner-reviewer-present" (-not [string]::IsNullOrWhiteSpace((Get-JsonString $case.ownerReview "reviewer"))) "blocker" "ownerReview.reviewer is required."
    Add-ValidationItem $caseItems "owner-reviewed-at-present" (-not [string]::IsNullOrWhiteSpace((Get-JsonString $case.ownerReview "reviewedAtUtc"))) "blocker" "ownerReview.reviewedAtUtc is required."
    Add-ValidationItem $caseItems "owner-review-does-not-accept-template" (-not [bool]$case.ownerReview.acceptedForRealModelRuntimeCandidate) "blocker" "Template ownerReview must not accept real-model-runtime."

    switch ($task) {
        "pose" {
            Add-ValidationItem $caseItems "pose-keypoint-count" ($case.outputMetadata.keypointCount -eq 17) "blocker" "Pose must carry keypointCount=17."
        }
        "obb" {
            Add-ValidationItem $caseItems "obb-angle-unit-present" (-not [string]::IsNullOrWhiteSpace((Get-JsonString $case.outputMetadata "angleUnit"))) "blocker" "OBB must carry angleUnit."
            Add-ValidationItem $caseItems "obb-input-shape-1024" ((Get-JsonString $case.input "inputShape") -eq "1x3x1024x1024") "required" "OBB profile must be 1x3x1024x1024."
        }
        "cls" {
            Add-ValidationItem $caseItems "cls-topk" ($case.outputMetadata.topK -eq 5) "blocker" "Classification must carry topK=5."
            Add-ValidationItem $caseItems "cls-input-shape-224" ((Get-JsonString $case.input "inputShape") -eq "1x3x224x224") "required" "Classification profile must be 1x3x224x224."
        }
    }

    foreach ($caseItem in $caseItems) { $items.Add($caseItem) | Out-Null }
    $failedBlockers = @($caseItems | Where-Object { $_.severity -eq "blocker" -and -not $_.passed }).Count
    $records.Add([ordered]@{
        id = $id
        task = $task
        family = $caseFamily
        normalizedFamily = $normalizedFamily
        article = $articlePath
        contractPath = $ContractPath
        contractRequiredMetadata = [object[]]@($contractRequiredMetadata)
        tensorRtExecProfileHint = $contractProfileHint
        validationState = if ($failedBlockers -gt 0) { "invalid" } else { "owner-action-required" }
        failedBlockerCount = $failedBlockers
        canPromoteRealModelRuntime = $false
        canPromotePackageConsumerRuntime = $false
        validationItems = [object[]]@($caseItems.ToArray())
    }) | Out-Null
}

$failedBlockerCount = @($items | Where-Object { $_.severity -eq "blocker" -and -not $_.passed }).Count
$failedActionRequiredCount = @($items | Where-Object { -not $_.passed }).Count

$result = [ordered]@{
    recordKind = "yolovision-real-asset-owner-backfill-pack-validation"
    generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
    inputPath = $InputPath
    contractPath = $ContractPath
    articleCasePackPath = $ArticleCasePackPath
    contractTaskCount = $contractTaskNames.Count
    articleCaseTaskCount = $articleCaseTasks.Count
    performsPublish = $false
    canPublishPublicly = $false
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
    validationState = if ($failedBlockerCount -gt 0) { "invalid" } else { "owner-action-required" }
    failedBlockerCount = $failedBlockerCount
    failedActionRequiredCount = $failedActionRequiredCount
    records = [object[]]@($records.ToArray())
    validationItems = [object[]]@($items.ToArray())
}

$resolvedOutput = Resolve-RepoPath $OutputPath
$outputDirectory = Split-Path -Parent $resolvedOutput
if (-not (Test-Path -LiteralPath $outputDirectory)) {
    New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
}

$result | ConvertTo-Json -Depth 64 | Set-Content -LiteralPath $resolvedOutput -Encoding UTF8
Write-Host "Wrote $resolvedOutput"

if ($Strict -and $failedBlockerCount -gt 0) {
    throw "YoloVision owner backfill pack validation failed with $failedBlockerCount blocker(s)."
}
