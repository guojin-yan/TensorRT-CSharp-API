[CmdletBinding()]
param(
  [string]$SkeletonPath = "artifacts\final-release\final-owner-execution-input-skeleton.json",
  [string]$OutputRoot = "artifacts\final-release",
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot $OutputRoot
}

if (-not [System.IO.Path]::IsPathRooted($SkeletonPath)) {
  $SkeletonPath = Join-Path $RepositoryRoot $SkeletonPath
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Write-Utf8File {
  param([string]$LiteralPath, [AllowNull()][object]$InputObject)
  [System.IO.File]::WriteAllText($LiteralPath, ((@($InputObject) -join [Environment]::NewLine) + [Environment]::NewLine), $script:utf8)
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)
  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function Convert-ToArray {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)
  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

if (-not (Test-Path -LiteralPath $SkeletonPath -PathType Leaf)) {
  throw "Final owner execution input skeleton not found: $SkeletonPath"
}

$skeleton = Get-Content -LiteralPath $SkeletonPath -Raw -Encoding utf8 | ConvertFrom-Json
$groups = @(Convert-ToArray (Get-PropertyOrDefault -Object $skeleton -Name "fieldGroups" -DefaultValue @()))
$fields = @($groups | ForEach-Object { Convert-ToArray (Get-PropertyOrDefault -Object $_ -Name "fields" -DefaultValue @()) })

$fieldValues = [ordered]@{}
$fileEvidence = New-Object System.Collections.Generic.List[object]
$hashEvidence = New-Object System.Collections.Generic.List[object]
foreach ($field in $fields) {
  $id = [string](Get-PropertyOrDefault -Object $field -Name "id" -DefaultValue "")
  $fieldPath = [string](Get-PropertyOrDefault -Object $field -Name "fieldPath" -DefaultValue "")
  $kind = [string](Get-PropertyOrDefault -Object $field -Name "kind" -DefaultValue "")
  if ([string]::IsNullOrWhiteSpace($id)) { continue }
  $fieldValues[$id] = "<owner-real-input-required>"
  if ($kind -eq "path") {
    $fileEvidence.Add([pscustomobject]@{
        fieldId = $id
        fieldPath = $fieldPath
        path = "<owner-real-file-path-required>"
        sha256FieldId = ""
        mustExist = $true
      }) | Out-Null
  }
  if ($kind -eq "sha256") {
    $hashEvidence.Add([pscustomobject]@{
        fieldId = $id
        fieldPath = $fieldPath
        sha256 = "<owner-real-sha256-required>"
        format = "64 lowercase or uppercase hexadecimal characters"
        mustMatchFile = $true
      }) | Out-Null
  }
}

$forbiddenSubstitutes = @(
  "local feed",
  "ProjectReference",
  "direct .nupkg",
  "direct nupkg",
  "local smoke",
  "pre-publish smoke reused as post-publish proof",
  "build-only",
  "dependency-probe",
  "dashboard",
  "runbook",
  "draft",
  "candidate",
  "template",
  "Skipped=True"
)

$confirmations = @($forbiddenSubstitutes | ForEach-Object {
    [pscustomobject]@{
      marker = $_
      confirmedAbsent = $false
      ownerNote = "<owner-confirm-after-real-check>"
    }
  })

$template = [pscustomobject]@{
  recordKind = "final-owner-execution-real-input-template"
  ownerInputRecordId = "<owner-real-input-record-id>"
  createdAtUtc = "<owner-created-at-utc>"
  templateState = "blocked-final-owner-real-input-required"
  sourceSkeletonPath = "artifacts/final-release/final-owner-execution-input-skeleton.json"
  owner = [pscustomobject]@{
    name = "<owner-name>"
    machine = "<owner-machine>"
    reviewedAtUtc = "<owner-reviewed-at-utc>"
    note = "<owner-review-note>"
  }
  fieldValues = [pscustomobject]$fieldValues
  fileEvidence = @($fileEvidence.ToArray())
  hashEvidence = @($hashEvidence.ToArray())
  hostMetadata = [pscustomobject]@{
    os = "<owner-host-os>"
    arch = "<owner-host-arch>"
    rid = "<owner-host-rid>"
    gpuName = "<owner-gpu-name>"
    nvidiaDriver = "<owner-nvidia-driver>"
    cudaRuntimeToolkit = "<owner-cuda-runtime-toolkit>"
    tensorrt = "<owner-tensorrt-version>"
    cudnn = "<owner-cudnn-version>"
  }
  packageMetadata = [pscustomobject]@{
    packageSourceUrl = "<owner-package-source-url>"
    managedPackageId = "<owner-managed-package-id>"
    managedPackageVersion = "<owner-managed-package-version>"
    managedPackageSha256 = "<owner-managed-package-sha256>"
    runtimePackageId = "<owner-runtime-package-id>"
    runtimePackageVersion = "<owner-runtime-package-version>"
    runtimePackageKey = "<owner-runtime-package-key>"
    runtimePackageSha256 = "<owner-runtime-package-sha256>"
  }
  postPublishEvidence = [pscustomobject]@{
    downloadedPackageSha256 = "<owner-post-publish-downloaded-package-sha256>"
    proofLogPath = "<owner-post-publish-proof-log-path>"
    proofLogSha256 = "<owner-post-publish-proof-log-sha256>"
    confirmsNotPrePublishSmoke = $false
  }
  dualPackageRouteProof = [pscustomobject]@{
    nugetSmallBridgeCore = [pscustomobject]@{
      ownerAuthorizationUrl = "<owner-dual-package-nuget-owner-authorization-url>"
      publicPackageDownloadUrl = "<owner-dual-package-nuget-public-download-url>"
      cleanConsumerProofLogPath = "<owner-dual-package-nuget-clean-consumer-log-path>"
      postPublishProofLogSha256 = "<owner-dual-package-nuget-post-publish-proof-log-sha256>"
      confirmsNoSubstituteProof = $false
    }
    githubPackagesFullRuntime = [pscustomobject]@{
      ownerAuthorizationUrl = "<owner-dual-package-github-owner-authorization-url>"
      restoreSourceUrl = "<owner-dual-package-github-restore-source-url>"
      runtimeDllResolutionReportPath = "<owner-dual-package-github-runtime-dll-resolution-report-path>"
      cleanRuntimeSmokeLogSha256 = "<owner-dual-package-github-clean-runtime-smoke-log-sha256>"
      confirmsNoSubstituteProof = $false
    }
    routeIds = @("nuget-small-bridge-core", "github-packages-full-runtime")
    sourceValidators = @(
      "eng/Test-DualPackagePublishPreflightMatrix.ps1 -Strict",
      "eng/Test-FinalCloseGateConvergence.ps1 -Strict"
    )
    boundary = "Dual-package route proof fields are Owner-fillable placeholders only. They are not publish approval, not package push, not post-publish proof, and cannot close release lanes until real public publish, public download/restore, clean consumer, and strict validation evidence is supplied."
  }
  rollbackReview = [pscustomobject]@{
    decision = "<owner-rollback-decision>"
    rationale = "<owner-rollback-rationale>"
    reviewedAtUtc = "<owner-rollback-reviewed-at-utc>"
  }
  finalCloseDecision = [pscustomobject]@{
    decision = "<owner-final-close-decision>"
    releaseIssueUrl = "<owner-release-issue-url>"
    rationale = "<owner-final-close-rationale>"
    decidedAtUtc = "<owner-final-close-decided-at-utc>"
  }
  strictValidatorOutputs = [pscustomobject]@{
    outputPath = "<owner-strict-validator-output-path>"
    outputSha256 = "<owner-strict-validator-output-sha256>"
    chainState = "<owner-strict-validator-chain-state>"
    validators = @(
      "eng/Test-FinalOwnerExecutionRealInputImport.ps1 -Strict",
      "eng/Test-FinalOwnerExecutionRealInputStrictPreflight.ps1 -Strict",
      "eng/Test-ReleaseEvidenceClassificationAudit.ps1 -Strict"
    )
  }
  nonSubstituteConfirmations = @($confirmations)
  ownerActionRequired = $true
  readyForImport = $false
  isExample = $false
  isTemplate = $true
  performsPublish = $false
  performsRuntimeExecution = $false
  canPromoteRuntimeProof = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  isRuntimeExecutionProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  boundary = "Final Owner execution real input template is owner-fillable input only. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
}

$example = $template.PSObject.Copy()
$example.recordKind = "final-owner-execution-real-input-example"
$example.ownerInputRecordId = "example-not-real-proof"
$example.createdAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
$example.templateState = "example-blocked-not-real-proof"
$example.isExample = $true
$example.isTemplate = $false
$example.owner = [pscustomobject]@{
  name = "Example Owner"
  machine = "EXAMPLE-MACHINE"
  reviewedAtUtc = "2000-01-01T00:00:00Z"
  note = "Example only; not real proof and not importable."
}
$example.boundary = "Example Owner input is illustrative only and intentionally not importable. It is not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."

$templateJsonPath = Join-Path $OutputRoot "final-owner-execution-real-input.template.json"
$templateMarkdownPath = Join-Path $OutputRoot "final-owner-execution-real-input.template.md"
$exampleJsonPath = Join-Path $OutputRoot "final-owner-execution-real-input.example.json"
$exampleMarkdownPath = Join-Path $OutputRoot "final-owner-execution-real-input.example.md"

$template | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $templateJsonPath -Encoding utf8
$example | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $exampleJsonPath -Encoding utf8

$fieldRows = foreach ($field in $fields) {
  $fieldId = ConvertTo-MarkdownCell (Get-PropertyOrDefault -Object $field -Name "id" -DefaultValue "")
  $fieldPath = ConvertTo-MarkdownCell (Get-PropertyOrDefault -Object $field -Name "fieldPath" -DefaultValue "")
  $fieldKind = ConvertTo-MarkdownCell (Get-PropertyOrDefault -Object $field -Name "kind" -DefaultValue "")
  "| ``$fieldId`` | ``$fieldPath`` | ``$fieldKind`` |"
}

$templateMarkdown = @"
# Final Owner Execution Real Input Template

| Field | Value |
|---|---|
| recordKind | ``$($template.recordKind)`` |
| templateState | ``$($template.templateState)`` |
| fieldCount | ``$($fields.Count)`` |
| fileEvidenceCount | ``$($fileEvidence.Count)`` |
| hashEvidenceCount | ``$($hashEvidence.Count)`` |
| readyForImport | ``$($template.readyForImport)`` |
| canPromoteRuntimeProof | ``$($template.canPromoteRuntimeProof)`` |
| canPublishPublicly | ``$($template.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($template.canCloseReleaseIssue)`` |

## Fields

| ID | Field Path | Kind |
|---|---|---|
$($fieldRows -join "`r`n")

## Boundary

$($template.boundary)
"@
Write-Utf8File -LiteralPath $templateMarkdownPath -InputObject $templateMarkdown

$exampleMarkdown = @"
# Final Owner Execution Real Input Example

This file is an example only. It is intentionally not real proof and not release-ready.

| Field | Value |
|---|---|
| recordKind | ``$($example.recordKind)`` |
| ownerInputRecordId | ``$($example.ownerInputRecordId)`` |
| isExample | ``$($example.isExample)`` |
| readyForImport | ``$($example.readyForImport)`` |
| canPublishPublicly | ``$($example.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($example.canCloseReleaseIssue)`` |

## Boundary

$($example.boundary)
"@
Write-Utf8File -LiteralPath $exampleMarkdownPath -InputObject $exampleMarkdown

Write-Host "Final owner execution real input template written:"
Write-Host "  Template=$templateJsonPath"
Write-Host "  Example=$exampleJsonPath"
Write-Host "Fields=$($fields.Count) FileEvidence=$($fileEvidence.Count) HashEvidence=$($hashEvidence.Count) Ready=False"
