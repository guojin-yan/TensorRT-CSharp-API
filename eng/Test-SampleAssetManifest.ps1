[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [switch]$WarnOnly
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

$allowedStatuses = @(
  "candidate-not-downloaded",
  "asset-required",
  "downloaded-not-verified",
  "hash-verified",
  "ready-to-run",
  "smoke-passed",
  "blocked-by-license",
  "blocked-by-download",
  "blocked-by-parser",
  "blocked-by-cuda-driver"
)

$allowedProofClassifications = @(
  "template-only",
  "build-only",
  "dependency-probe-only",
  "synthetic-input-runtime",
  "real-model-runtime",
  "package-consumer-runtime"
)

$sha256Pattern = "^[a-fA-F0-9]{64}$"

function ConvertTo-RelativePath {
  param([string]$Path)

  return $Path.Substring($RepositoryRoot.Length).TrimStart('\')
}

function New-Finding {
  param(
    [string]$Manifest,
    [string]$RuleId,
    [string]$Severity,
    [string]$Message
  )

  [pscustomobject]@{
    manifest = $Manifest
    ruleId = $RuleId
    severity = $Severity
    message = $Message
  }
}

function Get-StringValue {
  param(
    [AllowNull()][object]$Value
  )

  if ($null -eq $Value) {
    return ""
  }

  return [string]$Value
}

function Test-OptionalSha256 {
  param(
    [string]$Value
  )

  return [string]::IsNullOrWhiteSpace($Value) -or ($Value -match $sha256Pattern)
}

function Resolve-RepositoryPath {
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
    return $Object.$Name
  }

  return $null
}

function Get-StringProperty {
  param(
    [AllowNull()][object]$Object,
    [string]$Name
  )

  $value = Get-PropertyOrNull -Object $Object -Name $Name
  if ($null -eq $value) {
    return ""
  }

  return [string]$value
}

function First-NonEmpty {
  param([string[]]$Values)

  foreach ($value in $Values) {
    if (-not [string]::IsNullOrWhiteSpace($value)) {
      return $value
    }
  }

  return ""
}

function ConvertTo-CanonicalSampleAssetManifest {
  param([Parameter(Mandatory = $true)][object]$Manifest)

  $candidateId = Get-StringProperty -Object $Manifest -Name "candidateId"
  if ([string]::IsNullOrWhiteSpace($candidateId)) {
    return $Manifest
  }

  $modelLocalPath = Get-StringProperty -Object $Manifest.model -Name "localPath"
  $modelBaseName = [IO.Path]::GetFileNameWithoutExtension($modelLocalPath)
  $modelSha256 = Get-StringProperty -Object $Manifest.model -Name "sha256"
  $labelsSha256 = Get-StringProperty -Object $Manifest.labels -Name "sha256"
  $inputSha256 = Get-StringProperty -Object $Manifest.input -Name "imageSha256"
  $labelsClassCount = 0
  [void][int]::TryParse((Get-StringProperty -Object $Manifest.labels -Name "classCount"), [ref]$labelsClassCount)
  if (-not (Test-OptionalSha256 -Value $modelSha256)) { $modelSha256 = "" }
  if (-not (Test-OptionalSha256 -Value $labelsSha256)) { $labelsSha256 = "" }
  if (-not (Test-OptionalSha256 -Value $inputSha256)) { $inputSha256 = "" }

  return [pscustomobject]@{
    schemaVersion = [int]$Manifest.schemaVersion
    sampleName = Get-StringProperty -Object $Manifest -Name "sampleName"
    candidateId = $candidateId
    status = if ((Get-StringProperty -Object $Manifest -Name "status") -eq "owner-action-required") { "candidate-not-downloaded" } else { Get-StringProperty -Object $Manifest -Name "status" }
    proofClassification = Get-StringProperty -Object $Manifest -Name "proofClassification"
    isSmokePassed = $false
    isRedistributableInRepository = $false
    model = [pscustomobject]@{
      name = Get-StringProperty -Object $Manifest.model -Name "name"
      sourceUrl = Get-StringProperty -Object $Manifest.model -Name "sourceUrl"
      license = Get-StringProperty -Object $Manifest.model -Name "license"
      downloadUrl = Get-StringProperty -Object $Manifest.model -Name "downloadUrl"
      sha256 = $modelSha256
      opset = Get-StringProperty -Object $Manifest.model -Name "opset"
      exportCommand = Get-StringProperty -Object $Manifest.model -Name "onnxExportCommand"
      localPath = $modelLocalPath
    }
    labels = [pscustomobject]@{
      sourceUrl = Get-StringProperty -Object $Manifest.labels -Name "sourceUrl"
      license = Get-StringProperty -Object $Manifest.labels -Name "license"
      sha256 = $labelsSha256
      classCount = $labelsClassCount
      localPath = Get-StringProperty -Object $Manifest.labels -Name "localPath"
    }
    input = [pscustomobject]@{
      sourceUrl = Get-StringProperty -Object $Manifest.input -Name "imageSourceUrl"
      license = Get-StringProperty -Object $Manifest.input -Name "imageLicense"
      sha256 = $inputSha256
      localPath = Get-StringProperty -Object $Manifest.input -Name "imagePath"
    }
    tensor = [pscustomobject]@{
      inputName = Get-StringProperty -Object $Manifest.outputMetadata -Name "inputTensorName"
      inputShape = Get-StringProperty -Object $Manifest.input -Name "inputShape"
      layout = "NCHW"
      dtype = "float32"
      outputName = (@(Get-PropertyOrNull -Object $Manifest.outputMetadata -Name "outputTensorNames") -join ",")
      outputShape = Get-StringProperty -Object $Manifest.outputMetadata -Name "outputShape"
    }
    evidence = [pscustomobject]@{
      evidenceSidecar = if ([string]::IsNullOrWhiteSpace($modelBaseName)) { "" } else { "models/$modelBaseName-evidence.sidecar.json" }
      sampleRunEvidenceRecord = if ([string]::IsNullOrWhiteSpace($modelBaseName)) { "" } else { "artifacts/user-acceptance/$modelBaseName-sample-run-evidence.json" }
      sampleRunEvidenceValidation = "artifacts/user-acceptance/sample-run-evidence-record-validation.json"
      buildOnlyCommand = Get-StringProperty -Object $Manifest.commands -Name "tensorRtExecBuildCommand"
      runCommand = Get-StringProperty -Object $Manifest.commands -Name "yoloVisionRunCommand"
      expectedEvidenceLines = @(Get-PropertyOrNull -Object $Manifest.proofChecklist -Name "requiredEvidenceLines")
      lastRunStatus = "not-run"
      stdoutSummary = ""
      stderrSummary = ""
      lastRunLog = ""
      lastRunLogSha256 = ""
    }
  }
}

function Get-SampleProjectInfo {
  param([string]$SampleName)

  $isValidSampleName = -not [string]::IsNullOrWhiteSpace($SampleName) -and ($SampleName -match '^[A-Za-z0-9_.-]+$')
  $relativePath = ""
  $fullPath = ""
  $directoryExists = $false
  $projectExists = $false
  $projectNameMatches = $false
  $projectCount = 0

  if ($isValidSampleName) {
    # Samples and complete applications intentionally live in separate roots. Resolve the
    # project by its declared project name instead of rebuilding the retired samples/<name>
    # layout; this keeps the asset audit aligned with the public repository structure.
    $projectFiles = @()
    foreach ($projectRootName in @("samples", "applications")) {
      $projectRoot = Join-Path $RepositoryRoot $projectRootName
      if (Test-Path -LiteralPath $projectRoot -PathType Container) {
        $projectFiles += @(Get-ChildItem -LiteralPath $projectRoot -Recurse -Filter "$SampleName.csproj" -File)
      }
    }

    $projectCount = $projectFiles.Count
    $directoryExists = $projectCount -gt 0
    $projectExists = $projectCount -eq 1
    if ($projectExists) {
      $fullPath = $projectFiles[0].FullName
      $relativePath = ConvertTo-RelativePath -Path $fullPath
      $projectNameMatches = [string]::Equals(
        [IO.Path]::GetFileNameWithoutExtension($fullPath),
        $SampleName,
        [System.StringComparison]::Ordinal)
    }
  }

  return [pscustomobject]@{
    sampleName = $SampleName
    isValidSampleName = $isValidSampleName
    relativePath = $relativePath
    fullPath = $fullPath
    directoryExists = $directoryExists
    projectExists = $projectExists
    projectNameMatches = $projectNameMatches
    projectCount = $projectCount
  }
}

$manifestRoot = Join-Path $RepositoryRoot "samples\assets"
if (-not (Test-Path -LiteralPath $manifestRoot -PathType Container)) {
  throw "Sample asset manifest folder was not found: $manifestRoot"
}

$manifestFiles = @(
  Get-ChildItem -LiteralPath $manifestRoot -File |
    Where-Object { $_.Name.EndsWith(".template.json", [StringComparison]::OrdinalIgnoreCase) -or $_.Name.EndsWith("-example.json", [StringComparison]::OrdinalIgnoreCase) } |
    Sort-Object Name
)
$findings = New-Object System.Collections.Generic.List[object]
$items = New-Object System.Collections.Generic.List[object]

foreach ($file in $manifestFiles) {
  $relative = ConvertTo-RelativePath -Path $file.FullName
  $rawManifest = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8 | ConvertFrom-Json
  $manifest = ConvertTo-CanonicalSampleAssetManifest -Manifest $rawManifest
  $manifestSampleName = Get-StringValue $manifest.sampleName
  $sampleProject = Get-SampleProjectInfo -SampleName $manifestSampleName

  if ([int]$manifest.schemaVersion -ne 1) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "schema-version" -Severity "error" -Message "schemaVersion must be 1."))
  }

  if ([string]::IsNullOrWhiteSpace($manifestSampleName)) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "missing-sample-name" -Severity "error" -Message "sampleName is required."))
  }
  elseif (-not $sampleProject.isValidSampleName) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "sample-name-format" -Severity "error" -Message "sampleName must be a project-safe directory name."))
  }
  elseif (-not $sampleProject.directoryExists) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "sample-project-directory-missing" -Severity "error" -Message "sampleName must map to an existing sample or application project."))
  }
  elseif (-not $sampleProject.projectExists) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "sample-project-missing" -Severity "error" -Message "sampleName must map to exactly one sample or application project."))
  }
  elseif (-not $sampleProject.projectNameMatches) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "sample-project-name-mismatch" -Severity "error" -Message "sample project file name must match sampleName."))
  }

  if ($sampleProject.directoryExists -and $sampleProject.projectCount -ne 1) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "sample-project-count" -Severity "error" -Message "sampleName must map to exactly one sample project file."))
  }

  $status = Get-StringValue $manifest.status
  $proofClassification = Get-StringValue $manifest.proofClassification
  if (-not ($allowedStatuses -contains $status)) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "status-enum" -Severity "error" -Message "status '$status' is not allowed."))
  }

  if (-not ($allowedProofClassifications -contains $proofClassification)) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "proof-classification-enum" -Severity "error" -Message "proofClassification '$proofClassification' is not allowed."))
  }

  if ([bool]$manifest.isSmokePassed -and $status -ne "smoke-passed") {
    $findings.Add((New-Finding -Manifest $relative -RuleId "smoke-status-mismatch" -Severity "error" -Message "isSmokePassed can be true only when status is smoke-passed."))
  }

  if ($status -eq "smoke-passed" -and -not [bool]$manifest.isSmokePassed) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "smoke-claim-mismatch" -Severity "error" -Message "status smoke-passed requires isSmokePassed=true."))
  }

  if ($status -eq "smoke-passed" -and $proofClassification -ne "real-model-runtime") {
    $findings.Add((New-Finding -Manifest $relative -RuleId "smoke-proof-classification-mismatch" -Severity "error" -Message "Sample smoke-passed manifests require proofClassification=real-model-runtime. Package-consumer runtime proof is tracked separately."))
  }

  if ($proofClassification -eq "package-consumer-runtime") {
    $findings.Add((New-Finding -Manifest $relative -RuleId "sample-package-consumer-proof" -Severity "error" -Message "Sample asset manifests cannot claim proofClassification=package-consumer-runtime; package consumer proof is tracked by release proof records."))
  }

  foreach ($section in @("model", "labels", "input", "tensor", "evidence")) {
    if (-not $manifest.PSObject.Properties.Name.Contains($section)) {
      $findings.Add((New-Finding -Manifest $relative -RuleId "missing-section" -Severity "error" -Message "Missing section '$section'."))
    }
  }

  foreach ($path in @($manifest.model.localPath, $manifest.labels.localPath, $manifest.input.localPath)) {
    if ([string]::IsNullOrWhiteSpace((Get-StringValue $path))) {
      $findings.Add((New-Finding -Manifest $relative -RuleId "missing-local-path" -Severity "error" -Message "model, labels, and input localPath values are required."))
    }
  }

  foreach ($hash in @($manifest.model.sha256, $manifest.labels.sha256, $manifest.input.sha256)) {
    $hashText = Get-StringValue $hash
    if (-not (Test-OptionalSha256 -Value $hashText)) {
      $findings.Add((New-Finding -Manifest $relative -RuleId "invalid-sha256" -Severity "error" -Message "SHA256 values must be empty or 64 hexadecimal characters."))
    }
  }

  if ([string]::IsNullOrWhiteSpace((Get-StringValue $manifest.model.sourceUrl))) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "missing-model-source" -Severity "error" -Message "model.sourceUrl is required."))
  }

  if ([string]::IsNullOrWhiteSpace((Get-StringValue $manifest.model.license))) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "missing-model-license" -Severity "error" -Message "model.license is required."))
  }

  if ([string]::IsNullOrWhiteSpace((Get-StringValue $manifest.tensor.inputShape))) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "missing-input-shape" -Severity "error" -Message "tensor.inputShape is required."))
  }

  if ([string]::IsNullOrWhiteSpace((Get-StringValue $manifest.tensor.layout))) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "missing-layout" -Severity "error" -Message "tensor.layout is required."))
  }

  if ([string]::IsNullOrWhiteSpace((Get-StringValue $manifest.tensor.dtype))) {
    $findings.Add((New-Finding -Manifest $relative -RuleId "missing-dtype" -Severity "error" -Message "tensor.dtype is required."))
  }

  if (-not [bool]$manifest.isSmokePassed -and $status -in @("candidate-not-downloaded", "asset-required", "downloaded-not-verified", "hash-verified", "ready-to-run")) {
    $expectedRun = Get-StringValue $manifest.evidence.runCommand
    if ([string]::IsNullOrWhiteSpace($expectedRun)) {
      $findings.Add((New-Finding -Manifest $relative -RuleId "missing-run-command" -Severity "error" -Message "Non-smoke-passed manifests still need an expected runCommand."))
    }
  }

  if ($proofClassification -eq "real-model-runtime") {
    if (-not [bool]$manifest.isSmokePassed -or $status -ne "smoke-passed") {
      $findings.Add((New-Finding -Manifest $relative -RuleId "real-model-runtime-without-smoke" -Severity "error" -Message "proofClassification=real-model-runtime requires status=smoke-passed and isSmokePassed=true."))
    }

    foreach ($requiredHash in @($manifest.model.sha256, $manifest.labels.sha256, $manifest.input.sha256, $manifest.evidence.lastRunLogSha256)) {
      $hashText = Get-StringValue $requiredHash
      if (-not ($hashText -match $sha256Pattern)) {
        $findings.Add((New-Finding -Manifest $relative -RuleId "real-model-runtime-missing-hash" -Severity "error" -Message "real-model-runtime requires 64-character SHA256 values for model, labels, input, and run log."))
        break
      }
    }

    if ([string]::IsNullOrWhiteSpace((Get-StringValue $manifest.evidence.stdoutSummary)) -and [string]::IsNullOrWhiteSpace((Get-StringValue $manifest.evidence.stderrSummary))) {
      $findings.Add((New-Finding -Manifest $relative -RuleId "real-model-runtime-missing-output-summary" -Severity "error" -Message "real-model-runtime requires stdoutSummary or stderrSummary."))
    }
  }

  $evidenceSidecar = Get-StringValue $manifest.evidence.evidenceSidecar
  $sampleRunEvidenceRecord = Get-StringValue $manifest.evidence.sampleRunEvidenceRecord
  $sampleRunEvidenceValidation = Get-StringValue $manifest.evidence.sampleRunEvidenceValidation
  $sidecarExists = $false
  $sidecarCrossCheckState = if ([string]::IsNullOrWhiteSpace($evidenceSidecar)) { "missing-sidecar-path" } else { "owner-action-required" }
  if (-not [string]::IsNullOrWhiteSpace($evidenceSidecar)) {
    $resolvedSidecarPath = Resolve-RepositoryPath -Path $evidenceSidecar
    if (Test-Path -LiteralPath $resolvedSidecarPath -PathType Leaf) {
      $sidecarExists = $true
      $sidecarCrossCheckState = "checked"
      try {
        $sidecar = Get-Content -LiteralPath $resolvedSidecarPath -Raw -Encoding utf8 | ConvertFrom-Json
        $sidecarModelEvidence = Get-PropertyOrNull -Object $sidecar -Name "modelEvidence"
        $sidecarModelSha256 = First-NonEmpty @(
          (Get-StringProperty -Object $sidecarModelEvidence -Name "modelSha256"),
          (Get-StringProperty -Object $sidecarModelEvidence -Name "ModelSha256"),
          (Get-StringProperty -Object $sidecar -Name "modelSha256")
        )
        $sidecarInputAssetSha256 = First-NonEmpty @(
          (Get-StringProperty -Object $sidecarModelEvidence -Name "inputAssetSha256"),
          (Get-StringProperty -Object $sidecarModelEvidence -Name "InputAssetSha256"),
          (Get-StringProperty -Object $sidecar -Name "inputAssetSha256")
        )
        $sidecarModelLicense = First-NonEmpty @(
          (Get-StringProperty -Object $sidecarModelEvidence -Name "modelLicense"),
          (Get-StringProperty -Object $sidecarModelEvidence -Name "ModelLicense"),
          (Get-StringProperty -Object $sidecar -Name "modelLicense")
        )
        $sidecarStdoutSummary = Get-StringProperty -Object $sidecar -Name "stdoutSummary"
        $sidecarStderrSummary = Get-StringProperty -Object $sidecar -Name "stderrSummary"
        $manifestModelSha256 = Get-StringValue $manifest.model.sha256
        $manifestInputSha256 = Get-StringValue $manifest.input.sha256

        if (-not [string]::IsNullOrWhiteSpace($manifestModelSha256) -and
          -not [string]::IsNullOrWhiteSpace($sidecarModelSha256) -and
          -not [string]::Equals($manifestModelSha256, $sidecarModelSha256, [System.StringComparison]::OrdinalIgnoreCase)) {
          $findings.Add((New-Finding -Manifest $relative -RuleId "sidecar-model-sha256-mismatch" -Severity "error" -Message "evidence sidecar modelSha256 does not match manifest model.sha256."))
        }

        if (-not [string]::IsNullOrWhiteSpace($manifestInputSha256) -and
          -not [string]::IsNullOrWhiteSpace($sidecarInputAssetSha256) -and
          -not [string]::Equals($manifestInputSha256, $sidecarInputAssetSha256, [System.StringComparison]::OrdinalIgnoreCase)) {
          $findings.Add((New-Finding -Manifest $relative -RuleId "sidecar-input-sha256-mismatch" -Severity "error" -Message "evidence sidecar inputAssetSha256 does not match manifest input.sha256."))
        }

        if ($proofClassification -eq "real-model-runtime" -and [string]::IsNullOrWhiteSpace($sidecarModelLicense)) {
          $findings.Add((New-Finding -Manifest $relative -RuleId "sidecar-model-license-missing" -Severity "error" -Message "real-model-runtime manifests require evidence sidecar modelLicense."))
        }

        if ($proofClassification -eq "real-model-runtime" -and
          [string]::IsNullOrWhiteSpace((Get-StringValue $manifest.evidence.stdoutSummary)) -and
          [string]::IsNullOrWhiteSpace((Get-StringValue $manifest.evidence.stderrSummary)) -and
          [string]::IsNullOrWhiteSpace($sidecarStdoutSummary) -and
          [string]::IsNullOrWhiteSpace($sidecarStderrSummary)) {
          $findings.Add((New-Finding -Manifest $relative -RuleId "sidecar-output-summary-missing" -Severity "error" -Message "real-model-runtime requires stdout/stderr summary in manifest or sidecar."))
        }
      }
      catch {
        $sidecarCrossCheckState = "invalid-sidecar-json"
        $findings.Add((New-Finding -Manifest $relative -RuleId "sidecar-json-parse" -Severity "error" -Message "evidence sidecar JSON could not be parsed: $($_.Exception.Message)"))
      }
    }
  }

  $sampleRunEvidenceExists = $false
  $sampleRunEvidenceCrossCheckState = if ([string]::IsNullOrWhiteSpace($sampleRunEvidenceRecord)) { "missing-sample-run-evidence-path" } else { "owner-action-required" }
  if (-not [string]::IsNullOrWhiteSpace($sampleRunEvidenceRecord)) {
    $resolvedSampleRunEvidencePath = Resolve-RepositoryPath -Path $sampleRunEvidenceRecord
    if (Test-Path -LiteralPath $resolvedSampleRunEvidencePath -PathType Leaf) {
      $sampleRunEvidenceExists = $true
      $sampleRunEvidenceCrossCheckState = "checked"
      try {
        $sampleRunRecord = Get-Content -LiteralPath $resolvedSampleRunEvidencePath -Raw -Encoding utf8 | ConvertFrom-Json
        $sampleRunSampleName = Get-StringProperty -Object $sampleRunRecord -Name "sampleName"
        $sampleRunProofClassification = Get-StringProperty -Object $sampleRunRecord -Name "proofClassification"
        $sampleRunModelSha256 = Get-StringProperty -Object $sampleRunRecord -Name "modelSha256"
        $sampleRunLabelsSha256 = Get-StringProperty -Object $sampleRunRecord -Name "labelsSha256"
        $sampleRunInputAssetSha256 = Get-StringProperty -Object $sampleRunRecord -Name "inputAssetSha256"
        $manifestSampleName = Get-StringValue $manifest.sampleName
        $manifestModelSha256 = Get-StringValue $manifest.model.sha256
        $manifestLabelsSha256 = Get-StringValue $manifest.labels.sha256
        $manifestInputSha256 = Get-StringValue $manifest.input.sha256

        if (-not [string]::IsNullOrWhiteSpace($sampleRunSampleName) -and
          -not [string]::Equals($manifestSampleName, $sampleRunSampleName, [System.StringComparison]::Ordinal)) {
          $findings.Add((New-Finding -Manifest $relative -RuleId "sample-run-evidence-sample-name-mismatch" -Severity "error" -Message "sample run evidence sampleName does not match manifest sampleName."))
        }

        if ([string]::Equals($sampleRunProofClassification, "package-consumer-runtime", [System.StringComparison]::OrdinalIgnoreCase)) {
          $findings.Add((New-Finding -Manifest $relative -RuleId "sample-run-evidence-package-consumer-proof" -Severity "error" -Message "sample run evidence records cannot claim proofClassification=package-consumer-runtime."))
        }

        if (-not [string]::IsNullOrWhiteSpace($manifestModelSha256) -and
          -not [string]::IsNullOrWhiteSpace($sampleRunModelSha256) -and
          -not [string]::Equals($manifestModelSha256, $sampleRunModelSha256, [System.StringComparison]::OrdinalIgnoreCase)) {
          $findings.Add((New-Finding -Manifest $relative -RuleId "sample-run-evidence-model-sha256-mismatch" -Severity "error" -Message "sample run evidence modelSha256 does not match manifest model.sha256."))
        }

        if (-not [string]::IsNullOrWhiteSpace($manifestLabelsSha256) -and
          -not [string]::IsNullOrWhiteSpace($sampleRunLabelsSha256) -and
          -not [string]::Equals($manifestLabelsSha256, $sampleRunLabelsSha256, [System.StringComparison]::OrdinalIgnoreCase)) {
          $findings.Add((New-Finding -Manifest $relative -RuleId "sample-run-evidence-labels-sha256-mismatch" -Severity "error" -Message "sample run evidence labelsSha256 does not match manifest labels.sha256."))
        }

        if (-not [string]::IsNullOrWhiteSpace($manifestInputSha256) -and
          -not [string]::IsNullOrWhiteSpace($sampleRunInputAssetSha256) -and
          -not [string]::Equals($manifestInputSha256, $sampleRunInputAssetSha256, [System.StringComparison]::OrdinalIgnoreCase)) {
          $findings.Add((New-Finding -Manifest $relative -RuleId "sample-run-evidence-input-sha256-mismatch" -Severity "error" -Message "sample run evidence inputAssetSha256 does not match manifest input.sha256."))
        }
      }
      catch {
        $sampleRunEvidenceCrossCheckState = "invalid-sample-run-evidence-json"
        $findings.Add((New-Finding -Manifest $relative -RuleId "sample-run-evidence-json-parse" -Severity "error" -Message "sample run evidence JSON could not be parsed: $($_.Exception.Message)"))
      }
    }
  }

  $items.Add([pscustomobject]@{
      manifest = $relative
      sampleName = $manifestSampleName
      sampleProjectRelativePath = $sampleProject.relativePath
      sampleProjectExists = $sampleProject.projectExists
      sampleProjectNameMatches = $sampleProject.projectNameMatches
      sampleProjectDirectoryExists = $sampleProject.directoryExists
      sampleProjectCount = $sampleProject.projectCount
      status = $status
      proofClassification = $proofClassification
      isSmokePassed = [bool]$manifest.isSmokePassed
      isRedistributableInRepository = [bool]$manifest.isRedistributableInRepository
      modelName = Get-StringValue $manifest.model.name
      modelSourceUrl = Get-StringValue $manifest.model.sourceUrl
      modelLicense = Get-StringValue $manifest.model.license
      inputShape = Get-StringValue $manifest.tensor.inputShape
      layout = Get-StringValue $manifest.tensor.layout
      dtype = Get-StringValue $manifest.tensor.dtype
      runCommand = Get-StringValue $manifest.evidence.runCommand
      buildOnlyCommand = Get-StringValue $manifest.evidence.buildOnlyCommand
      evidenceSidecar = $evidenceSidecar
      evidenceSidecarExists = $sidecarExists
      sidecarCrossCheckState = $sidecarCrossCheckState
      sampleRunEvidenceRecord = $sampleRunEvidenceRecord
      sampleRunEvidenceValidation = $sampleRunEvidenceValidation
      sampleRunEvidenceExists = $sampleRunEvidenceExists
      sampleRunEvidenceCrossCheckState = $sampleRunEvidenceCrossCheckState
      stdoutSummary = Get-StringValue $manifest.evidence.stdoutSummary
      stderrSummary = Get-StringValue $manifest.evidence.stderrSummary
    })
}

$errorFindings = @($findings | Where-Object { $_.severity -eq "error" })
$outputRoot = Join-Path $RepositoryRoot "artifacts\user-acceptance"
New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "sample-asset-manifest-audit.json"
$markdownPath = Join-Path $outputRoot "sample-asset-manifest-audit.md"

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  manifestCount = $manifestFiles.Count
  findingCount = $findings.Count
  errorCount = $errorFindings.Count
  allowedStatuses = $allowedStatuses
  allowedProofClassifications = $allowedProofClassifications
  sampleProjectCrossCheckRules = @(
    "manifest.sampleName must resolve to exactly one project under samples/ or applications/.",
    "manifest.sampleName must be a project-safe project name and must not contain path separators.",
    "Each manifest sampleName must map to exactly one sample or application project file.",
    "Renamed samples such as YoloVision must not keep stale project identities such as YoloDet."
  )
  sidecarCrossCheckRules = @(
    "If evidence.evidenceSidecar is missing on disk for a template manifest, it remains owner-action-required and is not an error.",
    "If evidence.evidenceSidecar exists, modelSha256 and inputAssetSha256 are cross-checked against the manifest when both sides are populated.",
    "real-model-runtime manifests require modelLicense and stdout/stderr summary through the manifest or sidecar.",
    "The sidecar cross-check does not promote build-only reports or sample manifests to package-consumer-runtime."
  )
  sampleRunEvidenceCrossCheckRules = @(
    "If evidence.sampleRunEvidenceRecord is missing on disk for a template manifest, it remains owner-action-required and is not an error.",
    "If evidence.sampleRunEvidenceRecord exists, sampleName must match the manifest sampleName.",
    "If evidence.sampleRunEvidenceRecord exists, modelSha256, labelsSha256, and inputAssetSha256 are cross-checked against the manifest when both sides are populated.",
    "Sample run evidence records cannot claim proofClassification=package-consumer-runtime.",
    "Sample run evidence can promote only to real-model-runtime and does not replace release proof records."
  )
  items = @($items.ToArray())
  findings = @($findings.ToArray())
}

$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# Sample Asset Manifest Audit")
$lines.Add("")
$lines.Add("- generated at UTC: ``$($summary.generatedAtUtc)``")
$lines.Add("- manifest count: $($summary.manifestCount)")
$lines.Add("- finding count: $($summary.findingCount)")
$lines.Add("- error count: $($summary.errorCount)")
$lines.Add("")
$lines.Add("## Manifests")
$lines.Add("")
$lines.Add("| Manifest | Sample | Project | Status | Proof classification | Smoke passed | Model | Input |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($item in $items) {
  $projectState = if ($item.sampleProjectExists -and $item.sampleProjectNameMatches) { "matched" } else { "missing-or-mismatch" }
  $lines.Add("| ``$($item.manifest)`` | $($item.sampleName) | ``$projectState`` | ``$($item.status)`` | ``$($item.proofClassification)`` | ``$($item.isSmokePassed)`` | $($item.modelName.Replace("|", "\|")) | ``$($item.inputShape)`` |")
}
$lines.Add("")
$lines.Add("## Findings")
$lines.Add("")
if ($findings.Count -eq 0) {
  $lines.Add("- none")
}
else {
  $lines.Add("| Manifest | Rule | Severity | Message |")
  $lines.Add("| --- | --- | --- | --- |")
  foreach ($finding in $findings) {
    $lines.Add("| ``$($finding.manifest)`` | ``$($finding.ruleId)`` | ``$($finding.severity)`` | $($finding.message.Replace("|", "\|")) |")
  }
}
$lines.Add("")
$lines.Add("Candidate manifests do not make Classification or YoloVision sample smoke passed. They only record model, labels, image, preprocessing, postprocessing, proof classification, evidence sidecar, and evidence requirements.")
$lines.Add("")
$lines.Add("`template-only`, `build-only`, `dependency-probe-only`, and `synthetic-input-runtime` are not real model proof. `package-consumer-runtime` is forbidden in sample asset manifests and belongs to release proof records.")
$lines.Add("")
$lines.Add("## Sample Project Cross-Check Rules")
$lines.Add("")
foreach ($rule in $summary.sampleProjectCrossCheckRules) {
  $lines.Add("- $rule")
}
$lines.Add("")
$lines.Add("## Sidecar Cross-Check Rules")
$lines.Add("")
foreach ($rule in $summary.sidecarCrossCheckRules) {
  $lines.Add("- $rule")
}
$lines.Add("")
$lines.Add("## Sample Run Evidence Cross-Check Rules")
$lines.Add("")
foreach ($rule in $summary.sampleRunEvidenceCrossCheckRules) {
  $lines.Add("- $rule")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Sample asset manifest audit written to $jsonPath"
Write-Host "Sample asset manifest audit written to $markdownPath"

if ($errorFindings.Count -gt 0) {
  $message = "Found $($errorFindings.Count) sample asset manifest error(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
