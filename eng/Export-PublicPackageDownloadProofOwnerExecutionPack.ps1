[CmdletBinding()]
param(
  [string]$RuntimePackageKey = "win-x64-trt11.0-cuda13.2-cudnn9.22",
  [string]$OutputRoot,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
  $OutputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
}

New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Read-JsonOrNull {
  param([string]$RelativePath)

  $path = Join-Path $RepositoryRoot $RelativePath
  if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
    return $null
  }

  return Get-Content -LiteralPath $path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) {
    return $DefaultValue
  }

  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.PSObject.Properties[$Name].Value
  }

  return $DefaultValue
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

function New-OwnerStep {
  param(
    [string]$Id,
    [string]$Title,
    [string]$Command,
    [string]$ExpectedArtifact,
    [string]$OwnerAction,
    [bool]$Ready,
    [string]$Boundary
  )

  [pscustomobject]@{
    stepId = $Id
    title = $Title
    command = $Command
    expectedArtifact = $ExpectedArtifact
    ownerAction = $OwnerAction
    ready = $Ready
    stepState = if ($Ready) { "ready-non-proof" } else { "blocked-owner-execution-required" }
    boundary = $Boundary
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    canPromotePublicProof = $false
    canPromotePostPublishProof = $false
    isRuntimeExecutionProof = $false
    isPackageConsumerRuntimeProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
  }
}

function New-OwnerInputField {
  param([string]$Name, [string]$Category, [string]$ExpectedSource)

  [pscustomobject]@{
    fieldName = $Name
    category = $Category
    expectedSource = $ExpectedSource
    state = "blocked-owner-real-input-required"
    ownerMustProvide = $true
    acceptsPlaceholder = $false
    acceptsLocalFeed = $false
    acceptsProjectReference = $false
    acceptsDirectNupkg = $false
    acceptsDryRun = $false
    acceptsTemplate = $false
    performsPublish = $false
    usesPublishToken = $false
    canPublishPublicly = $false
    canCloseReleaseIssue = $false
    canPromoteRuntimeProof = $false
    canPromotePublicProof = $false
    canPromotePostPublishProof = $false
    isRuntimeExecutionProof = $false
    isPackageConsumerRuntimeProof = $false
    isPostPublishProof = $false
    isReleaseCloseProof = $false
    boundary = "Owner real public download field only; not runtime proof, not post-publish proof, not publish approval, not release close approval, and not package push."
  }
}

$docsAuditValidation = Read-JsonOrNull "artifacts\final-release\release-docs-and-nuget-metadata-audit-validation.json"
$postPublishUserVerificationPackValidation = Read-JsonOrNull "artifacts\final-release\post-publish-user-verification-pack-validation.json"
$publicDownloadTemplate = Read-JsonOrNull "artifacts\final-release\public-package-download-proof-input.template.json"

$docsAuditReady = [string](Get-PropertyOrDefault -Object $docsAuditValidation -Name "validationState" -DefaultValue "") -eq "release-docs-and-nuget-metadata-audit-ready-non-proof"
$verificationPackReadyForHandoff = [string](Get-PropertyOrDefault -Object $postPublishUserVerificationPackValidation -Name "validationState" -DefaultValue "") -eq "blocked-post-publish-user-verification-required"
$templatePresent = $null -ne $publicDownloadTemplate
$managedPackageId = [string](Get-PropertyOrDefault -Object $publicDownloadTemplate -Name "managedPackageId" -DefaultValue "JYPPX.TensorRT.CSharp.API")
$runtimePackageId = [string](Get-PropertyOrDefault -Object $publicDownloadTemplate -Name "runtimePackageId" -DefaultValue "JYPPX.TensorRT.CSharp.API.runtime.$RuntimePackageKey")

$ownerInputFields = @(
  New-OwnerInputField -Name "publicFeedKind" -Category "public-channel" -ExpectedSource "Owner-selected public NuGet/GitHub Packages/GitHub Release route"
  New-OwnerInputField -Name "packageSourceUrl" -Category "public-channel" -ExpectedSource "Public package source URL visible after publication"
  New-OwnerInputField -Name "managedPackageIdentity" -Category "package-identity" -ExpectedSource "Published managed package id/version"
  New-OwnerInputField -Name "runtimePackageIdentity" -Category "package-identity" -ExpectedSource "Published runtime package id/version"
  New-OwnerInputField -Name "managedPackageSha256" -Category "package-hash" -ExpectedSource "SHA256 of the downloaded managed .nupkg"
  New-OwnerInputField -Name "runtimePackageSha256" -Category "package-hash" -ExpectedSource "SHA256 of the downloaded runtime .nupkg or GitHub release asset"
  New-OwnerInputField -Name "managedPackageDownloadedPath" -Category "download" -ExpectedSource "Repository-local owner-downloads path created from public URL"
  New-OwnerInputField -Name "runtimePackageDownloadedPath" -Category "download" -ExpectedSource "Repository-local owner-downloads path created from public URL"
  New-OwnerInputField -Name "downloadedAtUtc" -Category "download" -ExpectedSource "UTC timestamp from the real Owner download session"
  New-OwnerInputField -Name "cleanTempDirectory" -Category "clean-consumer" -ExpectedSource "Fresh directory used for public restore validation"
  New-OwnerInputField -Name "restoreSource" -Category "clean-consumer" -ExpectedSource "Public restore source, not local feed or direct nupkg"
  New-OwnerInputField -Name "consumerProjectPath" -Category "clean-consumer" -ExpectedSource "Repository-external consumer project path"
  New-OwnerInputField -Name "restoreTranscriptPath" -Category "transcript" -ExpectedSource "dotnet restore transcript from public source"
  New-OwnerInputField -Name "restoreTranscriptSha256" -Category "transcript" -ExpectedSource "SHA256 of restore transcript"
  New-OwnerInputField -Name "buildTranscriptPath" -Category "transcript" -ExpectedSource "dotnet build transcript from public package consumer"
  New-OwnerInputField -Name "buildTranscriptSha256" -Category "transcript" -ExpectedSource "SHA256 of build transcript"
  New-OwnerInputField -Name "testTranscriptPath" -Category "transcript" -ExpectedSource "dotnet test/smoke transcript from public package consumer"
  New-OwnerInputField -Name "testTranscriptSha256" -Category "transcript" -ExpectedSource "SHA256 of test/smoke transcript"
  New-OwnerInputField -Name "reviewer" -Category "owner-review" -ExpectedSource "Owner reviewer identity"
  New-OwnerInputField -Name "reviewedAtUtc" -Category "owner-review" -ExpectedSource "UTC timestamp of owner review"
)

$steps = @(
  New-OwnerStep `
    -Id "preflight-claim-safety" `
    -Title "Run claim-safety preflight" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicDocsAndPackageMetadataGate.ps1 -Strict; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseDocsAndNuGetMetadataAudit.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseDocsAndNuGetMetadataAudit.ps1 -Strict" `
    -ExpectedArtifact "artifacts/final-release/release-docs-and-nuget-metadata-audit-validation.json" `
    -OwnerAction "Keep README/docs/NuGet metadata claim-safe before public package evidence is collected." `
    -Ready $docsAuditReady `
    -Boundary "Claim-safety preflight only; not runtime proof, not post-publish proof, not package publish, and not release close approval."

  New-OwnerStep `
    -Id "confirm-public-publish-result-exists" `
    -Title "Confirm real public publish result exists" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-OwnerPublicPublishExecutionResultCandidate.ps1 -Strict" `
    -ExpectedArtifact "artifacts/final-release/owner-public-publish-execution-result-candidate-validation.json" `
    -OwnerAction "Owner must provide the already-completed public publish result before download proof is meaningful." `
    -Ready $false `
    -Boundary "Validation of owner-supplied result only; this pack does not publish packages or approve release close."

  New-OwnerStep `
    -Id "generate-public-download-input-template" `
    -Title "Generate public download input template" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PublicPackageDownloadProofInputTemplate.ps1 -RuntimePackageKey $RuntimePackageKey" `
    -ExpectedArtifact "artifacts/final-release/public-package-download-proof-input.template.json" `
    -OwnerAction "Generate the owner-fill template that will receive public package URLs, downloaded paths, sizes, and hashes." `
    -Ready $templatePresent `
    -Boundary "Template generation only; placeholders are not public package download proof."

  New-OwnerStep `
    -Id "download-managed-package" `
    -Title "Download managed package from public source" `
    -Command "Invoke-WebRequest -Uri https://www.nuget.org/api/v2/package/$managedPackageId/<owner-fill-version> -OutFile .\artifacts\final-release\owner-downloads\$managedPackageId.<owner-fill-version>.nupkg" `
    -ExpectedArtifact "artifacts/final-release/owner-downloads/$managedPackageId.<owner-fill-version>.nupkg" `
    -OwnerAction "Owner must replace the version placeholder and download from the public NuGet endpoint or another public package source URL." `
    -Ready $false `
    -Boundary "Manual download instruction only; local feed, direct .nupkg, artifacts folder, and dry-run package output cannot substitute."

  New-OwnerStep `
    -Id "download-runtime-package" `
    -Title "Download runtime package from public source" `
    -Command "Invoke-WebRequest -Uri https://www.nuget.org/api/v2/package/$runtimePackageId/<owner-fill-version> -OutFile .\artifacts\final-release\owner-downloads\$runtimePackageId.<owner-fill-version>.nupkg" `
    -ExpectedArtifact "artifacts/final-release/owner-downloads/$runtimePackageId.<owner-fill-version>.nupkg" `
    -OwnerAction "Owner must download the matching runtime split package or document the GitHub release asset route with SHA256 and size." `
    -Ready $false `
    -Boundary "Manual runtime-package download instruction only; it is not runtime execution proof or post-publish proof."

  New-OwnerStep `
    -Id "hash-downloaded-packages" `
    -Title "Hash downloaded packages" `
    -Command "Get-FileHash -Algorithm SHA256 .\artifacts\final-release\owner-downloads\*.nupkg" `
    -ExpectedArtifact "owner-filled SHA256 fields in public-package-download-proof-input.template.json" `
    -OwnerAction "Record SHA256 and file sizes from the files downloaded from public sources." `
    -Ready $false `
    -Boundary "Hash command guidance only; hash slots are not proof until tied to public URLs and validator output."

  New-OwnerStep `
    -Id "fill-public-download-input" `
    -Title "Fill public download proof input" `
    -Command "Owner edits artifacts\final-release\public-package-download-proof-input.template.json with public URLs, downloaded paths, SHA256, sizes, timestamps, and reviewer fields." `
    -ExpectedArtifact "artifacts/final-release/public-package-download-proof-input.template.json" `
    -OwnerAction "Replace every owner-fill placeholder with real public package evidence." `
    -Ready $false `
    -Boundary "Owner-fill operation only; placeholders, local paths, and dry-run artifacts remain forbidden substitutes."

  New-OwnerStep `
    -Id "validate-public-download-input" `
    -Title "Validate public download proof input" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPackageDownloadProofInput.ps1 -Strict" `
    -ExpectedArtifact "artifacts/final-release/public-package-download-proof-input-validation.json" `
    -OwnerAction "Run the strict validator against the owner-filled public package download proof input." `
    -Ready $false `
    -Boundary "Input validation only; it cannot publish, promote runtime proof, or close release issue."

  New-OwnerStep `
    -Id "import-public-download-candidate" `
    -Title "Import public download proof candidate" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Import-PublicPackageDownloadProofCandidate.ps1" `
    -ExpectedArtifact "artifacts/final-release/public-package-download-proof-candidate.json" `
    -OwnerAction "Convert the validated input into a public download proof candidate for downstream gates." `
    -Ready $false `
    -Boundary "Candidate import only; downstream gates still decide whether evidence is usable."

  New-OwnerStep `
    -Id "validate-public-download-candidate" `
    -Title "Validate public download proof candidate" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PublicPackageDownloadProofCandidate.ps1 -Strict" `
    -ExpectedArtifact "artifacts/final-release/public-package-download-proof-candidate-validation.json" `
    -OwnerAction "Run the strict candidate validator and keep all public-source and forbidden-substitute checks visible." `
    -Ready $false `
    -Boundary "Candidate validation only; public download proof still does not equal clean external runtime proof."

  New-OwnerStep `
    -Id "refresh-post-publish-verification" `
    -Title "Refresh post-publish user verification pack" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-PostPublishUserVerificationPack.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-PostPublishUserVerificationPack.ps1 -Strict" `
    -ExpectedArtifact "artifacts/final-release/post-publish-user-verification-pack-validation.json" `
    -OwnerAction "Refresh the user-facing verification aggregator after public download proof candidate validation." `
    -Ready $verificationPackReadyForHandoff `
    -Boundary "Aggregator refresh only; it does not run clean consumer smoke or close release issue."

  New-OwnerStep `
    -Id "refresh-release-evidence" `
    -Title "Refresh release evidence and classification audit" `
    -Command "pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseEvidenceBundle.ps1; pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Test-ReleaseEvidenceClassificationAudit.ps1 -Strict" `
    -ExpectedArtifact "artifacts/final-release/release-evidence-classification-audit.json" `
    -OwnerAction "Refresh the release evidence bundle after owner evidence is imported." `
    -Ready $false `
    -Boundary "Evidence refresh only; it is not runtime proof, post-publish proof, publish approval, or release close approval."
)

$blockedSteps = @($steps | Where-Object { -not [bool]$_.ready })
$readySteps = @($steps | Where-Object { [bool]$_.ready })
$manualCommands = @($steps | Where-Object { -not [string]::IsNullOrWhiteSpace([string]$_.command) } | ForEach-Object { [string]$_.command })

$record = [ordered]@{
  recordKind = "public-package-download-proof-owner-execution-pack"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  packState = "blocked-public-package-download-proof-owner-execution-required"
  runtimePackageKey = $RuntimePackageKey
  ownerStepCount = $steps.Count
  readyOwnerStepCount = $readySteps.Count
  blockedOwnerStepCount = $blockedSteps.Count
  manualCommandCount = $manualCommands.Count
  requiredOwnerFieldCount = $ownerInputFields.Count
  blockedRequiredOwnerFieldCount = @($ownerInputFields | Where-Object { [string]$_.state -eq "blocked-owner-real-input-required" }).Count
  rejectedSubstituteCount = 8
  sourceReadinessSignalCount = 6
  ownerInputFields = $ownerInputFields
  ownerSteps = $steps
  manualCommands = $manualCommands
  requiredOwnerActions = @($blockedSteps | ForEach-Object {
      [pscustomobject]@{
        stepId = $_.stepId
        ownerAction = $_.ownerAction
        expectedArtifact = $_.expectedArtifact
      }
    })
  sourceArtifacts = @(
    "artifacts/final-release/release-docs-and-nuget-metadata-audit-validation.json",
    "artifacts/final-release/post-publish-user-verification-pack-validation.json",
    "artifacts/final-release/public-package-download-proof-input.template.json",
    "artifacts/final-release/public-package-download-proof-input-validation.json",
    "artifacts/final-release/public-package-download-proof-candidate-validation.json"
  )
  performsPublish = $false
  usesPublishToken = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  canPromotePublicProof = $false
  canPromotePostPublishProof = $false
  isRuntimeExecutionProof = $false
  isPackageConsumerRuntimeProof = $false
  isPostPublishProof = $false
  isReleaseCloseProof = $false
  safetyBoundary = "Public package download proof owner execution pack is manual owner guidance only. It does not publish packages, use tokens, download packages by itself, run clean consumer smoke, promote proof, or close release issues. Real public package download proof requires owner-filled public URLs, downloaded files, SHA256 values, sizes, and strict validator output."
}

$jsonPath = Join-Path $OutputRoot "public-package-download-proof-owner-execution-pack.json"
$markdownPath = Join-Path $OutputRoot "public-package-download-proof-owner-execution-pack.md"
$record | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$stepRows = foreach ($step in $steps) {
  "| ``$(ConvertTo-MarkdownCell $step.stepId)`` | ``$(ConvertTo-MarkdownCell $step.stepState)`` | ``$($step.ready)`` | $(ConvertTo-MarkdownCell $step.expectedArtifact) | $(ConvertTo-MarkdownCell $step.ownerAction) |"
}

$commandRows = foreach ($command in $manualCommands) {
  "- ``$(ConvertTo-MarkdownCell $command)``"
}

$markdown = @"
# Public Package Download Proof Owner Execution Pack

Generated at: ``$($record.generatedAtUtc)``

## Summary

- recordKind: ``$($record.recordKind)``
- packState: ``$($record.packState)``
- runtimePackageKey: ``$($record.runtimePackageKey)``
- ownerStepCount: ``$($record.ownerStepCount)``
- readyOwnerStepCount: ``$($record.readyOwnerStepCount)``
- blockedOwnerStepCount: ``$($record.blockedOwnerStepCount)``
- manualCommandCount: ``$($record.manualCommandCount)``
- requiredOwnerFieldCount: ``$($record.requiredOwnerFieldCount)``
- blockedRequiredOwnerFieldCount: ``$($record.blockedRequiredOwnerFieldCount)``
- rejectedSubstituteCount: ``$($record.rejectedSubstituteCount)``
- sourceReadinessSignalCount: ``$($record.sourceReadinessSignalCount)``
- performsPublish: ``False``
- usesPublishToken: ``False``
- canPublishPublicly: ``False``
- canCloseReleaseIssue: ``False``
- isPostPublishProof: ``False``

## Owner Steps

| Step | State | Ready | Expected Artifact | Owner Action |
|---|---:|---:|---|---|
$($stepRows -join "`r`n")

## Owner Input Fields

| Field | Category | State | Expected Source |
|---|---|---|---|
$(@($ownerInputFields | ForEach-Object { "| ``$(ConvertTo-MarkdownCell $_.fieldName)`` | ``$(ConvertTo-MarkdownCell $_.category)`` | ``$(ConvertTo-MarkdownCell $_.state)`` | $(ConvertTo-MarkdownCell $_.expectedSource) |" }) -join "`r`n")

## Manual Commands

$($commandRows -join "`r`n")

## Boundary

$($record.safetyBoundary)
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Output "Public package download proof owner execution pack written to $jsonPath"
Write-Output "PackState=$($record.packState) Steps=$($record.ownerStepCount) Blocked=$($record.blockedOwnerStepCount) Commands=$($record.manualCommandCount)"
