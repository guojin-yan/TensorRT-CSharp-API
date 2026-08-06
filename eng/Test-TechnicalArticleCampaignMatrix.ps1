[CmdletBinding()]
param(
  [string]$InputPath = "artifacts\final-release\technical-article-campaign-matrix.json",
  [string]$OutputRoot = "artifacts\final-release",
  [switch]$Strict,
  [string]$RepositoryRoot
)

$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

function Resolve-RepositoryPath {
  param([string]$Path)

  if ([System.IO.Path]::IsPathRooted($Path)) { return $Path }
  return Join-Path $RepositoryRoot $Path
}

function Get-PropertyOrDefault {
  param([AllowNull()][object]$Object, [string]$Name, [AllowNull()][object]$DefaultValue)

  if ($null -eq $Object) { return $DefaultValue }
  if ($Object.PSObject.Properties.Name -contains $Name) { return $Object.PSObject.Properties[$Name].Value }
  return $DefaultValue
}

function ConvertTo-Array {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return @() }
  if ($Value -is [System.Array]) { return @($Value) }
  return @($Value)
}

function New-ValidationItem {
  param([string]$Id, [bool]$Passed, [string]$Severity, [string]$Detail)

  [pscustomobject]@{ id = $Id; passed = $Passed; severity = $Severity; detail = $Detail }
}

function ConvertTo-MarkdownCell {
  param([AllowNull()][object]$Value)

  if ($null -eq $Value) { return "" }
  return ([string]$Value).Replace("|", "\|").Replace("`r", " ").Replace("`n", " ")
}

$resolvedInputPath = Resolve-RepositoryPath -Path $InputPath
if (-not (Test-Path -LiteralPath $resolvedInputPath -PathType Leaf)) {
  throw "Technical article campaign matrix not found: $resolvedInputPath"
}

if (-not [System.IO.Path]::IsPathRooted($OutputRoot)) { $OutputRoot = Join-Path $RepositoryRoot $OutputRoot }
New-Item -ItemType Directory -Path $OutputRoot -Force | Out-Null

$record = Get-Content -LiteralPath $resolvedInputPath -Raw -Encoding utf8 | ConvertFrom-Json
$articles = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "articles" -DefaultValue @())
$tracks = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "articleTracks" -DefaultValue @())
$executionStages = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "articleExecutionStages" -DefaultValue @())
$assetRequirements = ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "assetRequirements" -DefaultValue @())
$sourceCodeSampleLinks = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "sourceCodeSampleLinks" -DefaultValue @())) | ForEach-Object { [string]$_ })
$yoloFamilies = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "yoloFamilies" -DefaultValue @())) | ForEach-Object { [string]$_ })
$yoloTasks = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "yoloTasks" -DefaultValue @())) | ForEach-Object { [string]$_ })
$titles = @($articles | ForEach-Object { [string](Get-PropertyOrDefault -Object $_ -Name "title" -DefaultValue "") })
$forbiddenSubstituteMarkers = @((ConvertTo-Array (Get-PropertyOrDefault -Object $record -Name "forbiddenSubstituteMarkers" -DefaultValue @())) | ForEach-Object { [string]$_ })
$proofPromotionBoundary = Get-PropertyOrDefault -Object $record -Name "proofPromotionBoundary" -DefaultValue $null
$contentReadinessDefinition = Get-PropertyOrDefault -Object $record -Name "contentReadinessDefinition" -DefaultValue $null
$publishingReadiness = Get-PropertyOrDefault -Object $record -Name "publishingReadiness" -DefaultValue $null
$allText = (($record | ConvertTo-Json -Depth 16) -join "`n")

$items = New-Object System.Collections.Generic.List[object]
$items.Add((New-ValidationItem -Id "record-kind" -Passed ([string](Get-PropertyOrDefault -Object $record -Name "recordKind" -DefaultValue "") -eq "technical-article-campaign-matrix") -Severity "blocker" -Detail "recordKind must be technical-article-campaign-matrix.")) | Out-Null
$items.Add((New-ValidationItem -Id "article-count" -Passed ([int](Get-PropertyOrDefault -Object $record -Name "articleCount" -DefaultValue 0) -ge 30 -and $articles.Count -ge 30) -Severity "blocker" -Detail "Campaign must plan at least 30 articles.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-publish-close-or-proof" -Passed (-not [bool](Get-PropertyOrDefault -Object $record -Name "performsPublish" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPublishPublicly" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canCloseReleaseIssue" -DefaultValue $true) -and -not [bool](Get-PropertyOrDefault -Object $record -Name "canPromoteRuntimeProof" -DefaultValue $true)) -Severity "blocker" -Detail "Campaign matrix must not publish, close, or promote proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "tracks" -Passed ($tracks.Count -ge 6) -Severity "blocker" -Detail "Campaign must group articles into multiple publication tracks.")) | Out-Null
$items.Add((New-ValidationItem -Id "execution-stages" -Passed ($executionStages.Count -ge 4 -and @($executionStages | Where-Object { [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "ownerAction" -DefaultValue "")) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "status" -DefaultValue "")) }).Count -eq 0) -Severity "blocker" -Detail "Campaign must expose staged article execution with status and owner action.")) | Out-Null
$items.Add((New-ValidationItem -Id "source-links" -Passed ($sourceCodeSampleLinks -contains "applications/YoloVision" -and $sourceCodeSampleLinks -contains "applications/OnnxToEngine" -and $sourceCodeSampleLinks -contains "applications/TensorRtExec" -and $sourceCodeSampleLinks -contains "src/JYPPX.TensorRtSharp") -Severity "blocker" -Detail "Campaign must link YoloVision, OnnxToEngine, TensorRtExec, and core wrapper source paths.")) | Out-Null
$items.Add((New-ValidationItem -Id "asset-requirements" -Passed ($assetRequirements.Count -ge 5 -and [int](Get-PropertyOrDefault -Object $record -Name "missingAssetCount" -DefaultValue 0) -ge 1 -and [bool](Get-PropertyOrDefault -Object $record -Name "requiresRealModelAssets" -DefaultValue $false)) -Severity "blocker" -Detail "Campaign must expose missing asset requirements and real model asset dependency.")) | Out-Null
$items.Add((New-ValidationItem -Id "yolo-families" -Passed ((@("yolov5","yolov6","yolov7","yolov8","yolov9","yolov10","yolov11","yolov26","custom") | Where-Object { $yoloFamilies -notcontains $_ } | Measure-Object).Count -eq 0) -Severity "blocker" -Detail "Campaign must cover YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom.")) | Out-Null
$items.Add((New-ValidationItem -Id "yolo-tasks" -Passed ((@("det","cls","seg","obb","pose","sem") | Where-Object { $yoloTasks -notcontains $_ } | Measure-Object).Count -eq 0) -Severity "blocker" -Detail "Campaign must cover det/cls/seg/obb/pose/sem.")) | Out-Null
$items.Add((New-ValidationItem -Id "topic-coverage" -Passed ($allText.Contains("项目总览", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("ABI", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("CUDA / TensorRT", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("NuGet", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Windows", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Linux", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("OnnxToEngine", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("TensorRtExec", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("YoloVision", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Plugin Registry", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Engine Inspector", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Package Consumer Runtime Proof", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("Post-Publish Verification", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("FAQ", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("性能调优", [StringComparison]::OrdinalIgnoreCase) -and $allText.Contains("多平台部署", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Campaign must cover required project, install, tool, YoloVision, proof, FAQ, tuning, and deployment topics.")) | Out-Null
$items.Add((New-ValidationItem -Id "article-fields" -Passed (@($articles | Where-Object { [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "title" -DefaultValue "")) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "targetAudience" -DefaultValue "")) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "articleType" -DefaultValue "")) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "modelAcquisitionPlaceholder" -DefaultValue "")) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "readiness" -DefaultValue "")) -or [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $_ -Name "proofBoundary" -DefaultValue "")) }).Count -eq 0) -Severity "blocker" -Detail "Every article must include title, audience, type, model placeholder, readiness, and proof boundary.")) | Out-Null
$items.Add((New-ValidationItem -Id "screenshots" -Passed (@($articles | Where-Object { (ConvertTo-Array (Get-PropertyOrDefault -Object $_ -Name "screenshotsOrImagesNeeded" -DefaultValue @())).Count -eq 0 }).Count -eq 0) -Severity "blocker" -Detail "Every article must include screenshot/image requirements.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-retired-sample-identity" -Passed (-not $allText.Contains("YoloDet", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Campaign must not revive the retired detection-only sample identity.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-push-command" -Passed (-not $allText.Contains("dotnet nuget push", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Campaign must not instruct package push.")) | Out-Null
$items.Add((New-ValidationItem -Id "no-overclaim" -Passed (-not $allText.Contains("canPublishPublicly=true", [StringComparison]::OrdinalIgnoreCase) -and -not $allText.Contains("canCloseReleaseIssue=true", [StringComparison]::OrdinalIgnoreCase) -and -not $allText.Contains("canPromoteRuntimeProof=true", [StringComparison]::OrdinalIgnoreCase)) -Severity "blocker" -Detail "Campaign must not overclaim publication, release close, or proof promotion readiness.")) | Out-Null
$items.Add((New-ValidationItem -Id "forbidden-substitute-markers" -Passed ((@("candidate","draft","dashboard","dry-run","local feed","ProjectReference","direct .nupkg","template","build-only","blocked-by-cuda-driver") | Where-Object { $forbiddenSubstituteMarkers -notcontains $_ }).Count -eq 0) -Severity "blocker" -Detail "Campaign must list forbidden substitutes that cannot become proof.")) | Out-Null
$items.Add((New-ValidationItem -Id "proof-promotion-boundary" -Passed ([bool](Get-PropertyOrDefault -Object $proofPromotionBoundary -Name "articlesAreNotProof" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $proofPromotionBoundary -Name "screenshotsAreNotProof" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $proofPromotionBoundary -Name "matricesAreNotProof" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $proofPromotionBoundary -Name "strictValidatorRequiredForProofClaims" -DefaultValue $false) -and [bool](Get-PropertyOrDefault -Object $proofPromotionBoundary -Name "publicReleaseClaimsBlocked" -DefaultValue $false)) -Severity "blocker" -Detail "Campaign must state that articles, screenshots, and matrices are not proof, and strict validators are required for proof claims.")) | Out-Null
$items.Add((New-ValidationItem -Id "content-readiness-definition" -Passed (-not [string]::IsNullOrWhiteSpace([string](Get-PropertyOrDefault -Object $contentReadinessDefinition -Name "ownerProofRequired" -DefaultValue "")) -and [int](Get-PropertyOrDefault -Object $publishingReadiness -Name "executionStageCount" -DefaultValue 0) -ge 4 -and [int](Get-PropertyOrDefault -Object $publishingReadiness -Name "p0DraftCandidateCount" -DefaultValue 0) -ge 8) -Severity "blocker" -Detail "Campaign must define content readiness stages and enough P0 draft candidates.")) | Out-Null

$validationItems = @($items.ToArray())
$failedBlockers = @($validationItems | Where-Object { -not [bool]$_.passed -and [string]$_.severity -eq "blocker" })
$validationState = if ($failedBlockers.Count -eq 0) { "technical-article-campaign-matrix-ready" } else { "blocked-technical-article-campaign-matrix-invalid" }

$validation = [pscustomobject]@{
  recordKind = "technical-article-campaign-matrix-validation"
  generatedAtUtc = [DateTimeOffset]::UtcNow.ToString("O")
  validationState = $validationState
  inputPath = $resolvedInputPath
  articleCount = $articles.Count
  trackCount = $tracks.Count
  executionStageCount = $executionStages.Count
  assetRequirementCount = $assetRequirements.Count
  validationItemCount = $validationItems.Count
  failedBlockerCount = $failedBlockers.Count
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  canPromoteRuntimeProof = $false
  validationItems = $validationItems
  boundary = "Validation confirms campaign planning shape only; it is not publication, post-publish verification, or package-consumer-runtime proof."
}

$jsonPath = Join-Path $OutputRoot "technical-article-campaign-matrix-validation.json"
$markdownPath = Join-Path $OutputRoot "technical-article-campaign-matrix-validation.md"
$validation | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$rows = $validationItems | ForEach-Object {
  "| ``$($_.id)`` | ``$($_.passed)`` | ``$($_.severity)`` | $(ConvertTo-MarkdownCell $_.detail) |"
}

$markdown = @"
# Technical Article Campaign Matrix Validation

| Field | Value |
| --- | --- |
| validationState | ``$($validation.validationState)`` |
| articleCount | ``$($validation.articleCount)`` |
| trackCount | ``$($validation.trackCount)`` |
| executionStageCount | ``$($validation.executionStageCount)`` |
| assetRequirementCount | ``$($validation.assetRequirementCount)`` |
| validationItemCount | ``$($validation.validationItemCount)`` |
| failedBlockerCount | ``$($validation.failedBlockerCount)`` |
| performsPublish | ``$($validation.performsPublish)`` |
| canPublishPublicly | ``$($validation.canPublishPublicly)`` |
| canCloseReleaseIssue | ``$($validation.canCloseReleaseIssue)`` |
| canPromoteRuntimeProof | ``$($validation.canPromoteRuntimeProof)`` |

## Validation Items

| ID | Passed | Severity | Detail |
|---|---:|---|---|
$($rows -join "`r`n")

## Boundary

$($validation.boundary)
"@

$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "Technical article campaign matrix validation written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ValidationState=$validationState FailedBlockers=$($failedBlockers.Count)"

if ($Strict.IsPresent -and $failedBlockers.Count -gt 0) {
  foreach ($failure in $failedBlockers) {
    Write-Error "$($failure.id): $($failure.detail)"
  }

  exit 1
}
