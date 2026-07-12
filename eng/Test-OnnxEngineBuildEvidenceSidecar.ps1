[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$SidecarPath,
  [string]$OutputRoot,
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

$allowedProofClassifications = @(
  "template-only",
  "build-only",
  "dependency-probe-only",
  "synthetic-input-runtime",
  "real-model-runtime",
  "package-consumer-runtime"
)

$sha256Pattern = "^[a-fA-F0-9]{64}$"

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([IO.Path]::IsPathRooted($Path)) {
    return $Path
  }

  return Join-Path $RepositoryRoot $Path
}

function ConvertTo-RelativePath {
  param([string]$Path)

  if ([string]::IsNullOrWhiteSpace($Path)) {
    return ""
  }

  $full = [IO.Path]::GetFullPath($Path)
  $root = [IO.Path]::GetFullPath($RepositoryRoot)
  if ($full.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) {
    return $full.Substring($root.Length).TrimStart('\', '/')
  }

  return $full
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

function Test-Sha256 {
  param([string]$Value)

  return -not [string]::IsNullOrWhiteSpace($Value) -and ($Value -match $sha256Pattern)
}

function New-Finding {
  param(
    [string]$Source,
    [string]$RuleId,
    [string]$Severity,
    [string]$Message
  )

  [pscustomobject]@{
    source = $Source
    ruleId = $RuleId
    severity = $Severity
    message = $Message
  }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function Read-SidecarItem {
  param(
    [string]$Source,
    [string]$SampleName,
    [string]$Manifest,
    [string]$SidecarPathValue,
    [bool]$MissingSidecarIsError
  )

  $itemFindings = New-Object System.Collections.Generic.List[object]
  $resolvedSidecarPath = ""
  $sidecarExists = $false
  $sidecar = $null
  $state = "owner-action-required"

  if ([string]::IsNullOrWhiteSpace($SidecarPathValue)) {
    $itemFindings.Add((New-Finding -Source $Source -RuleId "sidecar-path-missing" -Severity "owner-action-required" -Message "evidence.evidenceSidecar is empty; create a sidecar before promoting real model evidence."))
  }
  else {
    $resolvedSidecarPath = Resolve-RepositoryPath -Path $SidecarPathValue
    if (Test-Path -LiteralPath $resolvedSidecarPath -PathType Leaf) {
      $sidecarExists = $true
      try {
        $sidecar = Get-Content -LiteralPath $resolvedSidecarPath -Raw -Encoding utf8 | ConvertFrom-Json
      }
      catch {
        $itemFindings.Add((New-Finding -Source $Source -RuleId "sidecar-json-parse" -Severity "error" -Message "Evidence sidecar JSON could not be parsed: $($_.Exception.Message)"))
      }
    }
    else {
      $severity = if ($MissingSidecarIsError) { "error" } else { "owner-action-required" }
      $itemFindings.Add((New-Finding -Source $Source -RuleId "sidecar-file-missing" -Severity $severity -Message "Evidence sidecar '$SidecarPathValue' was not found. Missing sidecars in templates are owner action, not proof failure."))
    }
  }

  $modelSource = ""
  $modelSha256 = ""
  $modelLicense = ""
  $inputAssetName = ""
  $inputAssetSha256 = ""
  $preprocessedInputTensorName = ""
  $preprocessedInputTensorSha256 = ""
  $stdoutSummary = ""
  $stderrSummary = ""
  $proofClassification = ""

  if ($null -ne $sidecar) {
    $modelEvidence = Get-PropertyOrNull -Object $sidecar -Name "modelEvidence"
    $modelSource = First-NonEmpty @(
      (Get-StringProperty -Object $modelEvidence -Name "modelSource"),
      (Get-StringProperty -Object $modelEvidence -Name "ModelSource"),
      (Get-StringProperty -Object $sidecar -Name "modelSource")
    )
    $modelSha256 = First-NonEmpty @(
      (Get-StringProperty -Object $modelEvidence -Name "modelSha256"),
      (Get-StringProperty -Object $modelEvidence -Name "ModelSha256"),
      (Get-StringProperty -Object $sidecar -Name "modelSha256")
    )
    $modelLicense = First-NonEmpty @(
      (Get-StringProperty -Object $modelEvidence -Name "modelLicense"),
      (Get-StringProperty -Object $modelEvidence -Name "ModelLicense"),
      (Get-StringProperty -Object $sidecar -Name "modelLicense")
    )
    $inputAssetName = First-NonEmpty @(
      (Get-StringProperty -Object $modelEvidence -Name "inputAssetName"),
      (Get-StringProperty -Object $modelEvidence -Name "InputAssetName"),
      (Get-StringProperty -Object $sidecar -Name "inputAssetName")
    )
    $inputAssetSha256 = First-NonEmpty @(
      (Get-StringProperty -Object $modelEvidence -Name "inputAssetSha256"),
      (Get-StringProperty -Object $modelEvidence -Name "InputAssetSha256"),
      (Get-StringProperty -Object $sidecar -Name "inputAssetSha256")
    )
    $preprocessedInputTensorName = First-NonEmpty @(
      (Get-StringProperty -Object $modelEvidence -Name "preprocessedInputTensorName"),
      (Get-StringProperty -Object $modelEvidence -Name "PreprocessedInputTensorName"),
      (Get-StringProperty -Object $sidecar -Name "preprocessedInputTensorName")
    )
    $preprocessedInputTensorSha256 = First-NonEmpty @(
      (Get-StringProperty -Object $modelEvidence -Name "preprocessedInputTensorSha256"),
      (Get-StringProperty -Object $modelEvidence -Name "PreprocessedInputTensorSha256"),
      (Get-StringProperty -Object $sidecar -Name "preprocessedInputTensorSha256")
    )
    $stdoutSummary = Get-StringProperty -Object $sidecar -Name "stdoutSummary"
    $stderrSummary = Get-StringProperty -Object $sidecar -Name "stderrSummary"
    $proofClassification = Get-StringProperty -Object $sidecar -Name "proofClassification"
  }

  $proofClassificationKnown = -not [string]::IsNullOrWhiteSpace($proofClassification) -and ($allowedProofClassifications -ccontains $proofClassification)
  $modelSha256Ready = Test-Sha256 -Value $modelSha256
  $modelLicenseReady = -not [string]::IsNullOrWhiteSpace($modelLicense)
  $inputAssetNameReady = -not [string]::IsNullOrWhiteSpace($inputAssetName)
  $inputAssetSha256Ready = Test-Sha256 -Value $inputAssetSha256
  $preprocessedInputTensorDeclared = -not [string]::IsNullOrWhiteSpace($preprocessedInputTensorName)
  $preprocessedInputTensorSha256Ready = Test-Sha256 -Value $preprocessedInputTensorSha256
  $stdoutStderrSummariesReady = -not [string]::IsNullOrWhiteSpace($stdoutSummary) -or -not [string]::IsNullOrWhiteSpace($stderrSummary)
  $realModelEvidenceReady = $modelSha256Ready -and $modelLicenseReady -and $inputAssetNameReady -and $inputAssetSha256Ready -and $stdoutStderrSummariesReady

  if ($sidecarExists) {
    if ([string]::IsNullOrWhiteSpace($proofClassification)) {
      $itemFindings.Add((New-Finding -Source $Source -RuleId "proof-classification-missing" -Severity "owner-action-required" -Message "proofClassification is required before the sidecar can be audited."))
    }
    elseif (-not $proofClassificationKnown) {
      $itemFindings.Add((New-Finding -Source $Source -RuleId "proof-classification-known" -Severity "error" -Message "proofClassification '$proofClassification' must be one of template-only, build-only, dependency-probe-only, synthetic-input-runtime, real-model-runtime, or package-consumer-runtime."))
    }

    if ([string]::Equals($proofClassification, "package-consumer-runtime", [System.StringComparison]::Ordinal)) {
      $itemFindings.Add((New-Finding -Source $Source -RuleId "sidecar-package-consumer-record-only" -Severity "boundary" -Message "package-consumer-runtime can be recorded in a sidecar for diagnostics, but sidecars cannot promote build reports or sample manifests to package-consumer runtime proof."))
    }

    if ([string]::Equals($proofClassification, "real-model-runtime", [System.StringComparison]::Ordinal) -and -not $realModelEvidenceReady) {
      $itemFindings.Add((New-Finding -Source $Source -RuleId "real-model-runtime-evidence-incomplete" -Severity "proof-required" -Message "real-model-runtime sidecars require modelSha256, modelLicense, inputAssetName, inputAssetSha256, and stdoutSummary or stderrSummary before they are promotable as real model evidence."))
    }

    if ([string]::Equals($proofClassification, "real-model-runtime", [System.StringComparison]::Ordinal) -and $preprocessedInputTensorDeclared -and -not $preprocessedInputTensorSha256Ready) {
      $itemFindings.Add((New-Finding -Source $Source -RuleId "preprocessed-input-tensor-sha256-missing" -Severity "proof-required" -Message "real-model-runtime sidecars with preprocessedInputTensorName require a 64-character preprocessedInputTensorSha256."))
    }
  }

  $hasError = @($itemFindings | Where-Object { $_.severity -eq "error" }).Count -gt 0
  $needsOwnerAction = @($itemFindings | Where-Object { $_.severity -in @("owner-action-required", "proof-required") }).Count -gt 0
  if ($hasError) {
    $state = "invalid-sidecar"
  }
  elseif ($needsOwnerAction) {
    $state = "owner-action-required"
  }
  elseif ([string]::Equals($proofClassification, "package-consumer-runtime", [System.StringComparison]::Ordinal)) {
    $state = "record-only"
  }
  elseif ($sidecarExists) {
    $state = "sidecar-recorded"
  }

  [pscustomobject]@{
    item = [pscustomobject]@{
      source = $Source
      sampleName = $SampleName
      manifest = $Manifest
      sidecarPath = $SidecarPathValue
      resolvedSidecarPath = $resolvedSidecarPath
      sidecarExists = $sidecarExists
      state = $state
      proofClassification = $proofClassification
      proofClassificationKnown = $proofClassificationKnown
      proofClassificationIsPackageConsumerRuntime = [string]::Equals($proofClassification, "package-consumer-runtime", [System.StringComparison]::Ordinal)
      canPromoteBuildReport = $false
      canPromotePackageConsumerRuntime = $false
      realModelEvidenceReady = $realModelEvidenceReady
      modelSource = $modelSource
      modelSha256 = $modelSha256
      modelSha256Ready = $modelSha256Ready
      modelLicense = $modelLicense
      modelLicenseReady = $modelLicenseReady
      inputAssetName = $inputAssetName
      inputAssetNameReady = $inputAssetNameReady
      inputAssetSha256 = $inputAssetSha256
      inputAssetSha256Ready = $inputAssetSha256Ready
      preprocessedInputTensorName = $preprocessedInputTensorName
      preprocessedInputTensorDeclared = $preprocessedInputTensorDeclared
      preprocessedInputTensorSha256 = $preprocessedInputTensorSha256
      preprocessedInputTensorSha256Ready = $preprocessedInputTensorSha256Ready
      stdoutSummary = $stdoutSummary
      stderrSummary = $stderrSummary
      stdoutStderrSummariesReady = $stdoutStderrSummariesReady
    }
    findings = @($itemFindings.ToArray())
  }
}

$items = New-Object System.Collections.Generic.List[object]
$findings = New-Object System.Collections.Generic.List[object]

if ([string]::IsNullOrWhiteSpace($SidecarPath)) {
  $scanMode = "manifest-scan"
  $manifestRoot = Join-Path $RepositoryRoot "samples\assets"
  if (-not (Test-Path -LiteralPath $manifestRoot -PathType Container)) {
    throw "Sample asset manifest folder was not found: $manifestRoot"
  }

  $templateFiles = @(Get-ChildItem -LiteralPath $manifestRoot -Filter "*.template.json" -File)
  $exampleFiles = @(Get-ChildItem -LiteralPath $manifestRoot -Filter "*example.json" -File)
  $manifestFiles = @($templateFiles + $exampleFiles | Sort-Object Name)

  foreach ($file in $manifestFiles) {
    $relative = ConvertTo-RelativePath -Path $file.FullName
    $manifest = Get-Content -LiteralPath $file.FullName -Raw -Encoding utf8 | ConvertFrom-Json
    $evidence = Get-PropertyOrNull -Object $manifest -Name "evidence"
    $sidecarValue = Get-StringProperty -Object $evidence -Name "evidenceSidecar"
    $result = Read-SidecarItem `
      -Source $relative `
      -SampleName (Get-StringProperty -Object $manifest -Name "sampleName") `
      -Manifest $relative `
      -SidecarPathValue $sidecarValue `
      -MissingSidecarIsError $false
    $items.Add($result.item)
    foreach ($finding in $result.findings) {
      $findings.Add($finding)
    }
  }
}
else {
  $scanMode = "single-sidecar"
  $result = Read-SidecarItem `
    -Source (ConvertTo-RelativePath -Path (Resolve-RepositoryPath -Path $SidecarPath)) `
    -SampleName "" `
    -Manifest "" `
    -SidecarPathValue $SidecarPath `
    -MissingSidecarIsError $true
  $items.Add($result.item)
  foreach ($finding in $result.findings) {
    $findings.Add($finding)
  }
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = "artifacts\user-acceptance"
}

if ([IO.Path]::IsPathRooted($OutputRoot)) {
  $outputRoot = $OutputRoot
}
else {
  $outputRoot = Join-Path $RepositoryRoot $OutputRoot
}

New-Item -ItemType Directory -Path $outputRoot -Force | Out-Null
$jsonPath = Join-Path $outputRoot "onnx-engine-build-evidence-sidecar-audit.json"
$markdownPath = Join-Path $outputRoot "onnx-engine-build-evidence-sidecar-audit.md"

$errorFindings = @($findings | Where-Object { $_.severity -eq "error" })
$ownerActionFindings = @($findings | Where-Object { $_.severity -in @("owner-action-required", "proof-required") })
$auditState = if ($errorFindings.Count -gt 0) {
  "invalid"
}
elseif ($ownerActionFindings.Count -gt 0) {
  "owner-action-required"
}
else {
  "ready"
}

$summary = [pscustomobject]@{
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  auditKind = "onnx-engine-build-evidence-sidecar-audit"
  scanMode = $scanMode
  auditState = $auditState
  itemCount = $items.Count
  findingCount = $findings.Count
  errorCount = $errorFindings.Count
  ownerActionRequiredCount = $ownerActionFindings.Count
  allowedProofClassifications = $allowedProofClassifications
  sidecarRules = @(
    "Missing sidecars in sample asset templates are owner-action-required, not errors.",
    "proofClassification must be one of template-only, build-only, dependency-probe-only, synthetic-input-runtime, real-model-runtime, or package-consumer-runtime.",
    "package-consumer-runtime can only be recorded by a sidecar; it cannot promote TensorRtExec/OnnxToEngine build reports.",
    "real-model-runtime sidecars require modelSha256, modelLicense, inputAssetName, inputAssetSha256, and stdoutSummary or stderrSummary.",
    "real-model-runtime sidecars with preprocessedInputTensorName require a 64-character preprocessedInputTensorSha256.",
    "A sidecar is not a sample smoke pass; Classification/YoloVision manifests still need real runner logs and hashes."
  )
  items = @($items.ToArray())
  findings = @($findings.ToArray())
}

$summary | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$lines = New-Object System.Collections.Generic.List[string]
$lines.Add("# ONNX Engine Build Evidence Sidecar Audit")
$lines.Add("")
$lines.Add("- generated at UTC: ``$($summary.generatedAtUtc)``")
$lines.Add("- scan mode: ``$scanMode``")
$lines.Add("- audit state: ``$auditState``")
$lines.Add("- item count: $($summary.itemCount)")
$lines.Add("- finding count: $($summary.findingCount)")
$lines.Add("- error count: $($summary.errorCount)")
$lines.Add("- owner action required count: $($summary.ownerActionRequiredCount)")
$lines.Add("")
$lines.Add("## Items")
$lines.Add("")
$lines.Add("| Source | Sample | Sidecar | Exists | State | Proof classification | Real model evidence ready |")
$lines.Add("| --- | --- | --- | --- | --- | --- | --- |")
foreach ($item in $items) {
  $lines.Add("| ``$($item.source)`` | $(ConvertTo-MarkdownCell $item.sampleName) | ``$($item.sidecarPath)`` | ``$($item.sidecarExists)`` | ``$($item.state)`` | ``$($item.proofClassification)`` | ``$($item.realModelEvidenceReady)`` |")
}
$lines.Add("")
$lines.Add("## Findings")
$lines.Add("")
if ($findings.Count -eq 0) {
  $lines.Add("- none")
}
else {
  $lines.Add("| Source | Rule | Severity | Message |")
  $lines.Add("| --- | --- | --- | --- |")
  foreach ($finding in $findings) {
    $lines.Add("| ``$($finding.source)`` | ``$($finding.ruleId)`` | ``$($finding.severity)`` | $(ConvertTo-MarkdownCell $finding.message) |")
  }
}
$lines.Add("")
$lines.Add("## Sidecar Rules")
$lines.Add("")
foreach ($rule in $summary.sidecarRules) {
  $lines.Add("- $rule")
}

$lines | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "ONNX engine build evidence sidecar audit written to $jsonPath"
Write-Host "ONNX engine build evidence sidecar audit written to $markdownPath"
Write-Host "AuditState=$auditState ErrorCount=$($errorFindings.Count) OwnerActionRequiredCount=$($ownerActionFindings.Count)"

if ($errorFindings.Count -gt 0) {
  $message = "Found $($errorFindings.Count) ONNX engine build evidence sidecar error(s)."
  if ($WarnOnly.IsPresent) {
    Write-Warning $message
  }
  else {
    Write-Error $message
    exit 1
  }
}
