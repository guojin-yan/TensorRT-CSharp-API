[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$ArticleCasePackPath = "samples/assets/yolovision-article-case-pack.json",
  [string]$OwnerBackfillPackPath = "samples/assets/yolovision-real-asset-owner-backfill-pack.json",
  [string]$GeneratedOwnerBackfillPackPath = "samples/assets/yolovision-real-asset-owner-backfill-pack.generated.json",
  [string]$EvidenceTemplatePath = "artifacts/user-acceptance/yolovision-real-asset-owner-backfill-sample-run-evidence.template.json",
  [string]$EvidenceTemplateMarkdownPath = "artifacts/user-acceptance/yolovision-real-asset-owner-backfill-sample-run-evidence.template.md",
  [string]$ComparisonReportPath = "artifacts/yolovision/yolovision-real-asset-owner-backfill-pack-projection-report.json",
  [switch]$NoGeneratedPack
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

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function Get-RequiredProperty {
  param(
    [object]$Object,
    [string]$Name,
    [string]$Context
  )

  if ($null -eq $Object -or -not ($Object.PSObject.Properties.Name -contains $Name)) {
    throw "Missing '$Name' in $Context."
  }

  return $Object.$Name
}

function ConvertTo-LocalPath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return ""
  }

  if ($Path.StartsWith(".\", [System.StringComparison]::Ordinal)) {
    return $Path
  }

  return ".\" + $Path.Replace("/", "\")
}

function Get-OutputJsonPathFromCommand {
  param(
    [string]$RunCommand,
    [string]$Fallback
  )

  $match = [regex]::Match($RunCommand, "--output-json\s+(?<path>\S+)")
  if ($match.Success) {
    return $match.Groups["path"].Value.Trim()
  }

  return $Fallback
}

function Get-ReportPathFromCommand {
  param(
    [string]$BuildCommand,
    [string]$Fallback
  )

  $match = [regex]::Match($BuildCommand, "--exportReport\s+(?<path>\S+)")
  if ($match.Success) {
    return $match.Groups["path"].Value.Trim()
  }

  return $Fallback
}

function Get-SaveEnginePathFromCommand {
  param(
    [string]$BuildCommand,
    [string]$Fallback
  )

  $match = [regex]::Match($BuildCommand, "--saveEngine\s+(?<path>\S+)")
  if ($match.Success) {
    return $match.Groups["path"].Value.Trim()
  }

  return $Fallback
}

function Get-OnnxPathFromCommand {
  param(
    [string]$RunCommand,
    [string]$BuildCommand,
    [string]$Fallback
  )

  $runMatch = [regex]::Match($RunCommand, "--model\s+(?<path>\S+)")
  if ($runMatch.Success) {
    return $runMatch.Groups["path"].Value.Trim()
  }

  $buildMatch = [regex]::Match($BuildCommand, "--onnx\s+(?<path>\S+)")
  if ($buildMatch.Success) {
    return $buildMatch.Groups["path"].Value.Trim()
  }

  return $Fallback
}

function Get-LabelsPathFromCommand {
  param(
    [string]$RunCommand,
    [string]$Fallback
  )

  $match = [regex]::Match($RunCommand, "--labels\s+(?<path>\S+)")
  if ($match.Success) {
    return $match.Groups["path"].Value.Trim()
  }

  return $Fallback
}

function Get-InputTensorPathFromCommand {
  param(
    [string]$RunCommand,
    [string]$Fallback
  )

  $match = [regex]::Match($RunCommand, "--input-data\s+(?<path>\S+)")
  if ($match.Success) {
    return $match.Groups["path"].Value.Trim()
  }

  $match = [regex]::Match($RunCommand, "--preprocessed-output\s+(?<path>\S+)")
  if ($match.Success) {
    return $match.Groups["path"].Value.Trim()
  }

  return $Fallback
}

function Get-ImagePathFromCommand {
  param(
    [string]$RunCommand,
    [string]$Fallback
  )

  $match = [regex]::Match($RunCommand, "--(?:image|input-image)\s+(?<path>\S+)")
  if ($match.Success) {
    return $match.Groups["path"].Value.Trim()
  }

  return $Fallback
}

function Get-CommandArgumentValue {
  param(
    [string]$Command,
    [string]$ArgumentName,
    [string]$Fallback
  )

  $match = [regex]::Match($Command, ("--" + [regex]::Escape($ArgumentName) + "\s+(?<value>\S+)"))
  if ($match.Success) {
    return $match.Groups["value"].Value.Trim()
  }

  return $Fallback
}

function Normalize-YoloFamilyAlias {
  param([string]$Family)

  $value = $Family.Trim().ToLowerInvariant()
  if ($value -match "^v(?<number>5|6|7|8|9|10|11|26)$") {
    return "yolov" + $Matches["number"]
  }

  return $value
}

function Get-PreflightReportPathFromCommand {
  param(
    [string]$PreflightCommand,
    [string]$Fallback
  )

  $match = [regex]::Match($PreflightCommand, "--preflight-report\s+(?<path>\S+)")
  if ($match.Success) {
    return $match.Groups["path"].Value.Trim()
  }

  return $Fallback
}

function Get-PreflightCommand {
  param(
    [object]$Case,
    [string]$RunCommand,
    [string]$FallbackReportPath
  )

  $caseProperty = $Case.PSObject.Properties | Where-Object { $_.Name -eq "yoloVisionPreflightCommand" } | Select-Object -First 1
  if ($null -ne $caseProperty -and -not [string]::IsNullOrWhiteSpace([string]$caseProperty.Value)) {
    return [string]$caseProperty.Value
  }

  return ($RunCommand.Trim() + " --preflight --preflight-report " + (ConvertTo-LocalPath -Path $FallbackReportPath))
}

function Test-Sha256 {
  param([AllowNull()][string]$Value)

  return -not [string]::IsNullOrWhiteSpace($Value) -and $Value -cmatch "^[0-9a-fA-F]{64}$"
}

function Test-RequiredOrHash {
  param([AllowNull()][string]$Value)

  return [string]::IsNullOrWhiteSpace($Value) -or
    $Value -match "owner-required|owner-action-required|template-only|owner-required-or-no-stderr" -or
    (Test-Sha256 $Value)
}

function New-OwnerRequiredEvidenceObject {
  [pscustomobject]@{
    hostMetadata = [pscustomobject]@{
      hostOs = "owner-required"
      hostMachineId = "owner-required"
      osArchitecture = "owner-required"
      gpuName = "owner-required"
      gpuComputeCapability = "owner-required"
      driverVersion = "owner-required"
      cudaDriverVersion = "owner-required"
      cudaRuntimeVersion = "owner-required"
      tensorRtVersion = "owner-required"
      tensorRtLine = "owner-required"
      cudnnVersion = "owner-required"
    }
    packageMetadata = [pscustomobject]@{
      packageSource = "owner-required"
      packageChannel = "owner-required"
      runtimePackageVersion = "owner-required"
      runtimePackageKey = "owner-required"
      managedPackageSha256 = "owner-required"
      nativeBridgeSha256 = "owner-required"
      runtimePackageSha256 = "owner-required"
    }
    ownerReview = [pscustomobject]@{
      ownerReviewer = "owner-required"
      ownerReviewedAtUtc = "owner-required"
      ownerAcceptanceDecision = "owner-required"
      ownerAcceptanceNotes = "owner-required"
    }
  }
}

function New-OutputMetadataObject {
  param(
    [string]$Task,
    [object]$Case
  )

  switch ($Task) {
    "det" {
      return [ordered]@{
        classCount = "owner-required"
        outputLayout = "owner-required"
        boxFormat = "owner-required"
        scoreRule = "owner-required"
        hasObjectness = "owner-required"
        nmsMode = "class-aware"
        scoreThreshold = "owner-required"
        iouThreshold = "owner-required"
      }
    }
    "seg" {
      return [ordered]@{
        classCount = "owner-required"
        outputRoleMap = "boxes:det,proto:mask-prototypes"
        maskCoefficientCount = 32
        prototypeShape = "owner-required"
        maskResizePolicy = "owner-required"
        maskThreshold = 0.5
        maskValueKind = "probability"
        maskPixelCountScope = "prototype-grid-before-crop-resize"
        letterboxContract = "owner-required"
        maskSpatialTransform = "explicit-preprocess-inverse"
        maskCoordinateSpace = "model-input-pixels"
        maskCropToDetection = $true
        finalMaskScope = "source-image-after-explicit-preprocess-inverse-and-optional-box-crop"
        spatialTransformBoundary = "owner must validate exporter-specific mask alignment"
      }
    }
    "pose" {
      return [ordered]@{
        classCount = "owner-required"
        keypointCount = 17
        keypointStride = "owner-required"
        coordinateLayout = "owner-required"
        keypointLayout = "owner-required"
        keypointScoreField = "owner-required"
        skeletonMap = "owner-required"
        letterboxContract = "owner-required"
      }
    }
    "obb" {
      return [ordered]@{
        classCount = "owner-required"
        boxFormat = "owner-required"
        angleUnit = "owner-required"
        angleRange = "owner-required"
        rotatedBoxLayout = "owner-required"
        coordinateSpace = "owner-required"
        rotatedNmsMode = "owner-required"
      }
    }
    "cls" {
      return [ordered]@{
        classCount = "owner-required"
        labelsSha256 = "owner-required"
        topK = 5
        classScoreField = "owner-required"
        softmaxApplied = "owner-required"
      }
    }
    "sem" {
      return [ordered]@{
        classCount = "owner-required"
        semanticOutput = "semantic"
        semanticOutputRole = "semanticMap"
        semanticMapShape = "owner-required"
        mapWidth = "owner-required"
        mapHeight = "owner-required"
        classMapLayout = "NCHW-logits-or-NHW-class-index-owner-confirmed"
        argmaxRule = "argmax-or-owner-confirmed"
        paletteSha256 = "owner-required"
        voidClassPolicy = "owner-required"
      }
    }
    default {
      $metadata = [ordered]@{}
      foreach ($name in @($Case.requiredOutputMetadata)) {
        $metadata[[string]$name] = "owner-required"
      }
      return $metadata
    }
  }
}

function Convert-ArticleCaseToOwnerCase {
  param([object]$Case)

  $id = [string](Get-RequiredProperty $Case "id" "article case")
  $task = [string](Get-RequiredProperty $Case "task" $id)
  $inputShape = [string](Get-RequiredProperty $Case "inputShape" $id)
  $buildCommand = [string](Get-RequiredProperty $Case "tensorRtExecBuildCommand" $id)
  $runCommand = [string](Get-RequiredProperty $Case "yoloVisionRunCommand" $id)
  $family = if ($Case.PSObject.Properties.Name -contains "family" -and -not [string]::IsNullOrWhiteSpace([string]$Case.family)) { [string]$Case.family } else { Get-CommandArgumentValue -Command $runCommand -ArgumentName "family" -Fallback "custom" }
  $modelOnnxPath = Get-OnnxPathFromCommand -RunCommand $runCommand -BuildCommand $buildCommand -Fallback ("models/" + $id + ".onnx")
  $labelsPath = Get-LabelsPathFromCommand -RunCommand $runCommand -Fallback "models/coco.names"
  $inputTensorPath = Get-InputTensorPathFromCommand -RunCommand $runCommand -Fallback ("models/" + $id + "-fp32.bin")
  $imagePath = Get-ImagePathFromCommand -RunCommand $runCommand -Fallback ("models/" + $id + ".jpg")
  $reportPath = Get-ReportPathFromCommand -BuildCommand $buildCommand -Fallback ("models/" + $id + "-build-report.json")
  $enginePath = Get-SaveEnginePathFromCommand -BuildCommand $buildCommand -Fallback ("models/" + $id + ".plan")
  $runLogPath = "models/$id-run.log"
  $outputJsonPath = Get-OutputJsonPathFromCommand -RunCommand $runCommand -Fallback ("models/$id-output.json")
  $preflightReportFallback = "models/$id-preflight.json"
  $preflightCommand = Get-PreflightCommand -Case $Case -RunCommand $runCommand -FallbackReportPath $preflightReportFallback
  $preflightReportPath = Get-PreflightReportPathFromCommand -PreflightCommand $preflightCommand -Fallback $preflightReportFallback

  [pscustomobject]@{
    id = $id
    task = $task
    family = $family
    state = "owner-action-required"
    article = [string](Get-RequiredProperty $Case "article" $id)
    model = [pscustomobject]@{
      sourceUrl = "owner-required"
      license = "owner-required"
      sha256 = "owner-required"
      onnxPath = $modelOnnxPath.TrimStart(".\")
      onnxSha256 = "owner-required"
      exportCommand = [string](Get-RequiredProperty $Case "exportCommand" $id)
    }
    labels = [pscustomobject]@{
      path = $labelsPath.TrimStart(".\")
      license = "owner-required"
      classCount = "owner-required"
      sha256 = "owner-required"
    }
    input = [pscustomobject]@{
      imagePath = $imagePath.TrimStart(".\")
      imageLicense = "owner-required"
      imageSha256 = "owner-required"
      preprocessedTensorPath = $inputTensorPath.TrimStart(".\")
      preprocessedTensorSha256 = "owner-required"
      inputShape = $inputShape
      preprocessContract = "owner-required"
    }
    tensorRtExec = [pscustomobject]@{
      buildCommand = $buildCommand
      reportPath = $reportPath.TrimStart(".\")
      reportSha256 = "owner-required"
      stdoutLogPath = "models/$id-tensorrtexec.stdout.log"
      stdoutLogSha256 = "owner-required"
      stderrLogPath = "models/$id-tensorrtexec.stderr.log"
      stderrLogSha256 = "owner-required-or-no-stderr"
      enginePath = $enginePath.TrimStart(".\")
      engineSha256 = "owner-required"
      proofBoundary = "TensorRtExec build report, command transcript, stdout/stderr logs, and engine hash are build evidence only; they are not real-model-runtime proof and never package-consumer-runtime proof."
    }
    yoloVision = [pscustomobject]@{
      runCommand = $runCommand
      expectedEvidenceLines = @($Case.expectedEvidenceLines)
      runLogPath = $runLogPath
      runLogSha256 = "owner-required"
      stdoutLogPath = "models/$id-yolovision.stdout.log"
      stdoutLogSha256 = "owner-required"
      stderrLogPath = "models/$id-yolovision.stderr.log"
      stderrLogSha256 = "owner-required-or-no-stderr"
      stdoutSummary = "owner-required"
      stderrSummary = "owner-required-or-no-stderr"
      outputJsonPath = $outputJsonPath.TrimStart(".\")
      outputJsonSha256 = "owner-required"
    }
    yoloVisionPreflight = [pscustomobject]@{
      command = $preflightCommand
      reportPath = $preflightReportPath.TrimStart(".\").Replace("\", "/")
      reportSha256 = "owner-required"
      schemaPath = "samples/YoloVision/yolovision-preflight.schema.json"
      schemaVersion = "yolovision-preflight.v1"
      expectedState = "owner-action-required"
      proofClassification = "precheck"
      execution = [pscustomobject]@{
        tensorRtRuntimeProbed = $false
        onnxParserInvoked = $false
        engineBuildInvoked = $false
        inferenceInvoked = $false
      }
      boundary = [pscustomobject]@{
        proofClassification = "precheck"
        isRuntimeProof = $false
        isRealModelRuntimeProof = $false
        isPackageConsumerRuntimeProof = $false
        canPromoteRealModelRuntime = $false
        canPromotePackageConsumerRuntime = $false
        note = "Offline YoloVision configuration and asset preflight only; it is not runtime proof."
      }
      proofBoundary = "YoloVision preflight is offline configuration/asset evidence only; it is not real-model-runtime proof and never package-consumer-runtime proof."
    }
    outputMetadata = New-OutputMetadataObject -Task $task -Case $Case
    articleEvidence = [pscustomobject]@{
      readiness = "owner-action-required"
      articleStatus = "near-ready-owner-evidence-missing"
      screenshotPlan = "owner-required-or-not-public"
      diagramPlan = "owner-required-or-not-public"
      ownerBackfillChecklist = @(
        "model source/license/hash",
        "labels/input/preprocessed tensor hashes",
        "TensorRtExec build report plus stdout/stderr transcript hashes",
        "YoloVision run log plus stdout/stderr transcript hashes",
        "owner review and publication decision"
      )
      proofBoundary = "Article readiness, screenshots, diagrams, and walkthrough text do not promote runtime proof."
    }
    ownerReview = [pscustomobject]@{
      reviewer = "owner-required"
      reviewedAtUtc = "owner-required"
      acceptanceDecision = "owner-required"
      acceptedForRealModelRuntimeCandidate = $false
      notes = "owner-required"
    }
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
  }
}

function New-OwnerBackfillPackProjection {
  param([object]$ArticlePack)

  $cases = @($ArticlePack.cases | ForEach-Object { Convert-ArticleCaseToOwnerCase -Case $_ })

  [pscustomobject]@{
    recordKind = "yolovision-real-asset-owner-backfill-pack"
    sampleName = "YoloVision"
    samplePath = "samples/YoloVision"
    sourcePack = "samples/assets/yolovision-article-case-pack.json"
    taskOutputContract = "samples/YoloVision/yolovision-task-output-contract.json"
    preflightContract = [pscustomobject]@{
      schemaPath = "samples/YoloVision/yolovision-preflight.schema.json"
      schemaVersion = "yolovision-preflight.v1"
      proofClassification = "precheck"
      canPromoteRealModelRuntime = $false
      canPromotePackageConsumerRuntime = $false
    }
    packState = "owner-action-required"
    proofBoundary = "owner backfill contract only; not real-model-runtime proof; not package-consumer-runtime proof; not post-publish proof"
    performsPublish = $false
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
    canPublishPublicly = $false
    requiredOwnerEvidence = New-OwnerRequiredEvidenceObject
    requiredGlobalEvidence = @(
      "hostOs",
      "hostMachineId",
      "osArchitecture",
      "gpuName",
      "gpuComputeCapability",
      "driverVersion",
      "cudaDriverVersion",
      "cudaRuntimeVersion",
      "tensorRtVersion",
      "tensorRtLine",
      "cudnnVersion",
      "packageSource",
      "packageChannel",
      "runtimePackageVersion",
      "runtimePackageKey",
      "managedPackageSha256",
      "nativeBridgeSha256",
      "runtimePackageSha256",
      "ownerReviewer",
      "ownerReviewedAtUtc",
      "ownerAcceptanceDecision"
    )
    requiredPreflightEvidence = @(
      "yoloVisionPreflight.command",
      "yoloVisionPreflight.reportPath",
      "yoloVisionPreflight.reportSha256",
      "yoloVisionPreflight.schemaVersion=yolovision-preflight.v1",
      "yoloVisionPreflight.proofClassification=precheck",
      "yoloVisionPreflight.execution.*=false",
      "yoloVisionPreflight.boundary.canPromote*=false"
    )
    forbiddenSubstitutes = @(
      "build-only",
      "dry-run",
      "template",
      "local feed",
      "ProjectReference",
      "direct .nupkg",
      "TensorRtExec report",
      "YoloVision matrix",
      "OnnxToEngine report",
      "readonly diagnostics",
      "screenshot",
      "sidecar-only report",
      "skipped run",
      "blocked-by-cuda-driver"
    )
    cases = $cases
  }
}

function New-EvidenceCaseFromOwnerCase {
  param([object]$Case)

  [pscustomobject]@{
    caseId = [string]$Case.id
    task = [string]$Case.task
    family = [string]$Case.family
    state = "owner-action-required"
    proofClassification = "template-only"
    templateOnly = $true
    sampleName = "YoloVision"
    article = [string]$Case.article
    model = [pscustomobject]@{
      sourceUrl = [string]$Case.model.sourceUrl
      license = [string]$Case.model.license
      sha256 = [string]$Case.model.sha256
      onnxPath = [string]$Case.model.onnxPath
      onnxSha256 = [string]$Case.model.onnxSha256
      exportCommand = [string]$Case.model.exportCommand
    }
    labels = [pscustomobject]@{
      path = [string]$Case.labels.path
      license = [string]$Case.labels.license
      classCount = [string]$Case.labels.classCount
      sha256 = [string]$Case.labels.sha256
    }
    input = [pscustomobject]@{
      imagePath = [string]$Case.input.imagePath
      imageLicense = [string]$Case.input.imageLicense
      imageSha256 = [string]$Case.input.imageSha256
      preprocessedTensorPath = [string]$Case.input.preprocessedTensorPath
      preprocessedTensorSha256 = [string]$Case.input.preprocessedTensorSha256
      inputShape = [string]$Case.input.inputShape
      preprocessContract = [string]$Case.input.preprocessContract
    }
    tensorRtExec = [pscustomobject]@{
      buildCommand = [string]$Case.tensorRtExec.buildCommand
      reportPath = [string]$Case.tensorRtExec.reportPath
      reportSha256 = [string]$Case.tensorRtExec.reportSha256
      stdoutLogPath = [string]$Case.tensorRtExec.stdoutLogPath
      stdoutLogSha256 = [string]$Case.tensorRtExec.stdoutLogSha256
      stderrLogPath = [string]$Case.tensorRtExec.stderrLogPath
      stderrLogSha256 = [string]$Case.tensorRtExec.stderrLogSha256
      enginePath = [string]$Case.tensorRtExec.enginePath
      engineSha256 = [string]$Case.tensorRtExec.engineSha256
      proofBoundary = "TensorRtExec report is build/report evidence only and is not runtime proof."
    }
    yoloVisionPreflight = [pscustomobject]@{
      command = [string]$Case.yoloVisionPreflight.command
      reportPath = [string]$Case.yoloVisionPreflight.reportPath
      reportSha256 = [string]$Case.yoloVisionPreflight.reportSha256
      schemaPath = [string]$Case.yoloVisionPreflight.schemaPath
      schemaVersion = [string]$Case.yoloVisionPreflight.schemaVersion
      expectedState = [string]$Case.yoloVisionPreflight.expectedState
      proofClassification = [string]$Case.yoloVisionPreflight.proofClassification
      execution = $Case.yoloVisionPreflight.execution
      boundary = $Case.yoloVisionPreflight.boundary
      proofBoundary = [string]$Case.yoloVisionPreflight.proofBoundary
    }
    sampleRunCommand = [string]$Case.yoloVision.runCommand
    sampleRunLogPath = [string]$Case.yoloVision.runLogPath
    sampleRunLogSha256 = [string]$Case.yoloVision.runLogSha256
    sampleRunStdoutLogPath = [string]$Case.yoloVision.stdoutLogPath
    sampleRunStdoutLogSha256 = [string]$Case.yoloVision.stdoutLogSha256
    sampleRunStderrLogPath = [string]$Case.yoloVision.stderrLogPath
    sampleRunStderrLogSha256 = [string]$Case.yoloVision.stderrLogSha256
    outputJsonPath = [string]$Case.yoloVision.outputJsonPath
    outputJsonSha256 = [string]$Case.yoloVision.outputJsonSha256
    stdoutSummary = [string]$Case.yoloVision.stdoutSummary
    stderrSummary = [string]$Case.yoloVision.stderrSummary
    expectedEvidenceLines = @($Case.yoloVision.expectedEvidenceLines)
    ownerReview = $Case.ownerReview
    outputMetadata = $Case.outputMetadata
    articleEvidence = $Case.articleEvidence
    isSmokePassed = $false
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
    validator = "eng/Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    failureReasons = @(
      "owner must fill real model, labels, input, TensorRtExec report, engine, run log, output JSON hashes",
      "owner must capture a real YoloVision run log containing YoloVision Passed=True",
      "template cases cannot promote real-model-runtime or package-consumer-runtime proof"
    )
  }
}

function New-EvidenceTemplate {
  param([object]$OwnerPack)

  [pscustomobject]@{
    schemaVersion = 1
    recordKind = "yolovision-real-asset-owner-backfill-sample-run-evidence-template"
    templateOnly = $true
    templateName = "yolovision-real-asset-owner-backfill-six-task"
    sampleName = "YoloVision"
    sourcePack = "samples/assets/yolovision-real-asset-owner-backfill-pack.json"
    proofClassification = "template-only"
    validationState = "owner-action-required"
    proofBoundary = "six-task sample-run-evidence template only; not real-model-runtime proof; not package-consumer-runtime proof; not post-publish proof"
    performsPublish = $false
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
    requiredOwnerEvidence = $OwnerPack.requiredOwnerEvidence
    requiredGlobalEvidence = @($OwnerPack.requiredGlobalEvidence)
    requiredPreflightEvidence = @($OwnerPack.requiredPreflightEvidence)
    preflightContract = $OwnerPack.preflightContract
    forbiddenSubstitutes = @($OwnerPack.forbiddenSubstitutes)
    validator = "eng/Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog"
    cases = @($OwnerPack.cases | ForEach-Object { New-EvidenceCaseFromOwnerCase -Case $_ })
    validationRules = @(
      "Each case must preserve case id, task, model source/license/hash fields, labels hash, input image hash, preprocessed tensor hash, TensorRtExec report hash, TensorRtExec stdout/stderr transcript hashes, engine hash, YoloVision preflight command/report/schema/hash/boundary, YoloVision run command, expected evidence lines, run log hash, YoloVision stdout/stderr transcript hashes, output JSON hash, stdout/stderr summaries, article readiness, and owner review.",
      "Template cases remain owner-action-required until real owner evidence replaces owner-required placeholders.",
      "TensorRtExec report, profile dump, matrix, sidecar, screenshot, dry-run, build-only, and local package output cannot promote proof.",
      "Sample run evidence can promote only real-model-runtime after Test-SampleRunEvidenceRecord.ps1 -RequireExistingLog succeeds on a real record.",
      "Package-consumer-runtime proof belongs to release proof records, never to this YoloVision sample template."
    )
  }
}

function Compare-Projection {
  param(
    [object]$ArticlePack,
    [object]$OwnerPack,
    [object]$ProjectedPack,
    [object]$EvidenceTemplate
  )

  $items = New-Object System.Collections.Generic.List[object]

  function Add-Item {
    param(
      [string]$Id,
      [bool]$Passed,
      [string]$Detail
    )

    $items.Add([pscustomobject]@{
        id = $Id
        passed = $Passed
        detail = $Detail
      })
  }

  Add-Item "article-case-count" (@($ArticlePack.cases).Count -eq 6) "Article case pack must keep six YOLOv8n task/article cases, including semantic segmentation."
  Add-Item "article-case-sem-present" (@($ArticlePack.cases | Where-Object { [string]$_.task -eq "sem" }).Count -eq 1) "Article case pack must provide a semantic segmentation owner-action article path."
  Add-Item "owner-case-count" (@($OwnerPack.cases).Count -eq 6) "Owner backfill pack must keep six YOLOv8n task cases, including semantic segmentation."
  Add-Item "owner-case-sem-present" (@($OwnerPack.cases | Where-Object { [string]$_.task -eq "sem" }).Count -eq 1) "Owner backfill pack must contain the semantic segmentation case."
  Add-Item "evidence-case-count" (@($EvidenceTemplate.cases).Count -eq 6) "Evidence template must keep six YOLOv8n task cases, including semantic segmentation."
  Add-Item "evidence-case-sem-present" (@($EvidenceTemplate.cases | Where-Object { [string]$_.task -eq "sem" }).Count -eq 1) "Evidence template must contain the semantic segmentation case."
  Add-Item "owner-pack-boundary" (-not [bool]$OwnerPack.canPromoteRealModelRuntime -and -not [bool]$OwnerPack.canPromotePackageConsumerRuntime) "Owner pack must not promote proof."
  Add-Item "evidence-template-boundary" (-not [bool]$EvidenceTemplate.canPromoteRealModelRuntime -and -not [bool]$EvidenceTemplate.canPromotePackageConsumerRuntime) "Evidence template must not promote proof."

  foreach ($articleCase in @($ArticlePack.cases)) {
    $id = [string]$articleCase.id
    $ownerCase = @($OwnerPack.cases | Where-Object { [string]$_.id -eq $id }) | Select-Object -First 1
    $projectedCase = @($ProjectedPack.cases | Where-Object { [string]$_.id -eq $id }) | Select-Object -First 1
    $evidenceCase = @($EvidenceTemplate.cases | Where-Object { [string]$_.caseId -eq $id }) | Select-Object -First 1

    Add-Item "case-$id-owner-present" ($null -ne $ownerCase) "Owner backfill pack must contain $id."
    Add-Item "case-$id-evidence-present" ($null -ne $evidenceCase) "Evidence template must contain $id."
    if ($null -eq $ownerCase -or $null -eq $projectedCase -or $null -eq $evidenceCase) {
      continue
    }

    Add-Item "case-$id-task-match" ([string]$ownerCase.task -eq [string]$articleCase.task -and [string]$evidenceCase.task -eq [string]$articleCase.task) "Task must match article, owner pack, and evidence template."
    $articleFamily = if ($articleCase.PSObject.Properties.Name -contains "family" -and -not [string]::IsNullOrWhiteSpace([string]$articleCase.family)) { [string]$articleCase.family } else { Get-CommandArgumentValue -Command ([string]$articleCase.yoloVisionRunCommand) -ArgumentName "family" -Fallback "custom" }
    Add-Item "case-$id-family-match" ([string]$ownerCase.family -eq $articleFamily -and [string]$evidenceCase.family -eq [string]$ownerCase.family) "Family must match article, owner pack, and evidence template."
    Add-Item "case-$id-input-shape-match" ([string]$ownerCase.input.inputShape -eq [string]$articleCase.inputShape -and [string]$evidenceCase.input.inputShape -eq [string]$articleCase.inputShape) "Input shape must match article, owner pack, and evidence template."
    Add-Item "case-$id-export-command-match" ([string]$ownerCase.model.exportCommand -eq [string]$articleCase.exportCommand) "Export command must match article case pack."
    Add-Item "case-$id-tensorrtexec-command-match" ([string]$ownerCase.tensorRtExec.buildCommand -eq [string]$articleCase.tensorRtExecBuildCommand -and [string]$evidenceCase.tensorRtExec.buildCommand -eq [string]$ownerCase.tensorRtExec.buildCommand) "TensorRtExec command must match article, owner pack, and evidence template."
    Add-Item "case-$id-yolovision-command-match" ([string]$ownerCase.yoloVision.runCommand -eq [string]$articleCase.yoloVisionRunCommand -and [string]$evidenceCase.sampleRunCommand -eq [string]$ownerCase.yoloVision.runCommand) "YoloVision run command must match article, owner pack, and evidence template."
    $articlePreflightCommand = if ($articleCase.PSObject.Properties.Name -contains "yoloVisionPreflightCommand") { [string]$articleCase.yoloVisionPreflightCommand } else { "" }
    $articlePreflightReportPath = if ($articleCase.PSObject.Properties.Name -contains "yoloVisionPreflightReportPath") { [string]$articleCase.yoloVisionPreflightReportPath } else { "" }
    Add-Item "case-$id-preflight-command-match" (
      [string]$ownerCase.yoloVisionPreflight.command -match "samples\\YoloVision" -and
      [string]$ownerCase.yoloVisionPreflight.command -match "--preflight" -and
      ([string]::IsNullOrWhiteSpace($articlePreflightCommand) -or [string]$ownerCase.yoloVisionPreflight.command -eq $articlePreflightCommand)
    ) "YoloVision preflight command must be present and match the article case when declared."
    Add-Item "case-$id-preflight-report-path-match" (
      -not [string]::IsNullOrWhiteSpace([string]$ownerCase.yoloVisionPreflight.reportPath) -and
      ([string]::IsNullOrWhiteSpace($articlePreflightReportPath) -or [string]$ownerCase.yoloVisionPreflight.reportPath -eq $articlePreflightReportPath.TrimStart(".\"))
    ) "YoloVision preflight report path must be present and match the article case when declared."
    Add-Item "case-$id-expected-passed" (@($evidenceCase.expectedEvidenceLines | Where-Object { ([string]$_).Contains("YoloVision Passed=True", [System.StringComparison]::Ordinal) }).Count -eq 1) "Evidence template must include YoloVision Passed=True."
    Add-Item "case-$id-owner-required-hashes" (
      [string]$evidenceCase.model.sha256 -eq "owner-required" -and
      [string]$evidenceCase.model.onnxSha256 -eq "owner-required" -and
      [string]$evidenceCase.labels.sha256 -eq "owner-required" -and
      [string]$evidenceCase.input.imageSha256 -eq "owner-required" -and
      [string]$evidenceCase.input.preprocessedTensorSha256 -eq "owner-required" -and
      [string]$evidenceCase.tensorRtExec.reportSha256 -eq "owner-required" -and
      [string]$evidenceCase.tensorRtExec.stdoutLogSha256 -eq "owner-required" -and
      [string]$evidenceCase.tensorRtExec.stderrLogSha256 -eq "owner-required-or-no-stderr" -and
      [string]$evidenceCase.tensorRtExec.engineSha256 -eq "owner-required" -and
      [string]$evidenceCase.sampleRunLogSha256 -eq "owner-required" -and
      [string]$evidenceCase.sampleRunStdoutLogSha256 -eq "owner-required" -and
      [string]$evidenceCase.sampleRunStderrLogSha256 -eq "owner-required-or-no-stderr" -and
      [string]$evidenceCase.outputJsonSha256 -eq "owner-required"
    ) "Evidence template must preserve owner-required hash fields."
    Add-Item "case-$id-preflight-contract" (
      [string]$ownerCase.yoloVisionPreflight.schemaVersion -eq "yolovision-preflight.v1" -and
      [string]$ownerCase.yoloVisionPreflight.proofClassification -eq "precheck" -and
      [string]$evidenceCase.yoloVisionPreflight.schemaVersion -eq "yolovision-preflight.v1" -and
      [string]$evidenceCase.yoloVisionPreflight.proofClassification -eq "precheck"
    ) "Owner and evidence cases must preserve the YoloVision preflight schema and precheck classification."
    Add-Item "case-$id-preflight-owner-required-hash" (
      (Test-RequiredOrHash ([string]$ownerCase.yoloVisionPreflight.reportSha256)) -and
      (Test-RequiredOrHash ([string]$evidenceCase.yoloVisionPreflight.reportSha256))
    ) "YoloVision preflight reportSha256 must be owner-required or a real SHA256."
    Add-Item "case-$id-preflight-execution-disabled" (
      -not [bool]$ownerCase.yoloVisionPreflight.execution.tensorRtRuntimeProbed -and
      -not [bool]$ownerCase.yoloVisionPreflight.execution.onnxParserInvoked -and
      -not [bool]$ownerCase.yoloVisionPreflight.execution.engineBuildInvoked -and
      -not [bool]$ownerCase.yoloVisionPreflight.execution.inferenceInvoked -and
      -not [bool]$ownerCase.yoloVisionPreflight.boundary.isRuntimeProof -and
      -not [bool]$ownerCase.yoloVisionPreflight.boundary.isRealModelRuntimeProof -and
      -not [bool]$ownerCase.yoloVisionPreflight.boundary.isPackageConsumerRuntimeProof -and
      -not [bool]$ownerCase.yoloVisionPreflight.boundary.canPromoteRealModelRuntime -and
      -not [bool]$ownerCase.yoloVisionPreflight.boundary.canPromotePackageConsumerRuntime
    ) "YoloVision preflight must keep runtime execution and promotion flags false."
    Add-Item "case-$id-enhanced-evidence-fields" (
      -not [string]::IsNullOrWhiteSpace([string]$ownerCase.articleEvidence.articleStatus) -and
      -not [string]::IsNullOrWhiteSpace([string]$evidenceCase.articleEvidence.proofBoundary) -and
      -not [string]::IsNullOrWhiteSpace([string]$evidenceCase.tensorRtExec.stdoutLogPath) -and
      -not [string]::IsNullOrWhiteSpace([string]$evidenceCase.sampleRunStdoutLogPath) -and
      -not [string]::IsNullOrWhiteSpace([string]$evidenceCase.yoloVisionPreflight.reportPath)
    ) "Owner/evidence cases must carry article readiness and command transcript fields."
    Add-Item "case-$id-no-proof-promotion" (-not [bool]$evidenceCase.canPromoteRealModelRuntime -and -not [bool]$evidenceCase.canPromotePackageConsumerRuntime -and -not [bool]$evidenceCase.isSmokePassed) "Template case must not promote proof."
  }

  $failedItems = @($items.ToArray() | Where-Object { -not $_.passed })

  [pscustomobject]@{
    generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
    recordKind = "yolovision-real-asset-owner-backfill-pack-projection-report"
    sourceArticlePack = $ArticleCasePackPath
    ownerBackfillPack = $OwnerBackfillPackPath
    generatedOwnerBackfillPack = $GeneratedOwnerBackfillPackPath
    evidenceTemplate = $EvidenceTemplatePath
    validationState = if ($failedItems.Count -eq 0) { "projection-aligned" } else { "projection-drift-detected" }
    failedCount = $failedItems.Count
    proofBoundary = "projection/drift report only; not runtime proof; not package-consumer-runtime proof"
    canPromoteRealModelRuntime = $false
    canPromotePackageConsumerRuntime = $false
    validationItems = @($items.ToArray())
  }
}

function Write-EvidenceTemplateMarkdown {
  param(
    [object]$Template,
    [string]$Path
  )

  $lines = New-Object System.Collections.Generic.List[string]
  $lines.Add("# YoloVision Real Asset Owner Backfill Sample Run Evidence Template")
  $lines.Add("")
  $lines.Add("- record kind: ``$($Template.recordKind)``")
  $lines.Add("- template name: ``$($Template.templateName)``")
  $lines.Add("- validation state: ``$($Template.validationState)``")
  $lines.Add("- proof classification: ``$($Template.proofClassification)``")
  $lines.Add("- can promote real model runtime: ``$($Template.canPromoteRealModelRuntime)``")
  $lines.Add("- can promote package consumer runtime: ``$($Template.canPromotePackageConsumerRuntime)``")
  $lines.Add("- validator: ``$($Template.validator)``")
  $lines.Add("")
  $lines.Add("## Cases")
  $lines.Add("")
  $lines.Add("| Case | Task | Input Shape | Run Log | Output JSON |")
  $lines.Add("| --- | --- | --- | --- | --- |")
  foreach ($case in $Template.cases) {
    $lines.Add("| ``$($case.caseId)`` | ``$($case.task)`` | ``$($case.input.inputShape)`` | ``$($case.sampleRunLogPath)`` | ``$($case.outputJsonPath)`` |")
  }
  $lines.Add("")
  $lines.Add("## Proof Boundary")
  $lines.Add("")
  $lines.Add($Template.proofBoundary)
  $lines.Add("")
  $lines.Add("## Validation Rules")
  $lines.Add("")
  foreach ($rule in $Template.validationRules) {
    $lines.Add("- $rule")
  }

  $lines | Set-Content -LiteralPath $Path -Encoding utf8
}

$articlePackResolvedPath = Resolve-RepositoryPath -Path $ArticleCasePackPath
$ownerBackfillResolvedPath = Resolve-RepositoryPath -Path $OwnerBackfillPackPath
$generatedPackResolvedPath = Resolve-RepositoryPath -Path $GeneratedOwnerBackfillPackPath
$evidenceTemplateResolvedPath = Resolve-RepositoryPath -Path $EvidenceTemplatePath
$evidenceTemplateMarkdownResolvedPath = Resolve-RepositoryPath -Path $EvidenceTemplateMarkdownPath
$comparisonReportResolvedPath = Resolve-RepositoryPath -Path $ComparisonReportPath

foreach ($path in @($articlePackResolvedPath, $ownerBackfillResolvedPath)) {
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    throw "Required input was not found: $path"
  }
}

$articlePack = Get-Content -LiteralPath $articlePackResolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
$ownerPack = Get-Content -LiteralPath $ownerBackfillResolvedPath -Raw -Encoding utf8 | ConvertFrom-Json
$projectedPack = New-OwnerBackfillPackProjection -ArticlePack $articlePack
$evidenceTemplate = New-EvidenceTemplate -OwnerPack $projectedPack
$projectionReport = Compare-Projection -ArticlePack $articlePack -OwnerPack $ownerPack -ProjectedPack $projectedPack -EvidenceTemplate $evidenceTemplate

foreach ($path in @($generatedPackResolvedPath, $evidenceTemplateResolvedPath, $evidenceTemplateMarkdownResolvedPath, $comparisonReportResolvedPath)) {
  New-Item -ItemType Directory -Path (Split-Path -Parent $path) -Force | Out-Null
}

if (-not $NoGeneratedPack.IsPresent) {
  $projectedPack | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $generatedPackResolvedPath -Encoding utf8
}

$evidenceTemplate | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $evidenceTemplateResolvedPath -Encoding utf8
Write-EvidenceTemplateMarkdown -Template $evidenceTemplate -Path $evidenceTemplateMarkdownResolvedPath
$projectionReport | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $comparisonReportResolvedPath -Encoding utf8

Write-Host "YoloVision owner backfill generated projection written to $generatedPackResolvedPath"
Write-Host "YoloVision sample-run-evidence template written to $evidenceTemplateResolvedPath"
Write-Host "YoloVision sample-run-evidence template markdown written to $evidenceTemplateMarkdownResolvedPath"
Write-Host "YoloVision owner backfill projection report written to $comparisonReportResolvedPath"
Write-Host "ValidationState=$($projectionReport.validationState) FailedCount=$($projectionReport.failedCount)"

if ($projectionReport.failedCount -gt 0) {
  Write-Error "YoloVision owner backfill projection drift detected. FailedCount=$($projectionReport.failedCount)"
  exit 1
}
