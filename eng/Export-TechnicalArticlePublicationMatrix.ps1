param(
  [string]$RepositoryRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function ConvertFrom-MarkdownCellText {
  param([string]$Value)

  if ($null -eq $Value) {
    return ""
  }

  return ($Value -replace "<br\s*/?>", " " -replace "`r", " " -replace "`n", " ").Trim()
}

function ConvertTo-PathList {
  param([string]$Value)

  if ([string]::IsNullOrWhiteSpace($Value)) {
    return @()
  }

  $matches = [System.Text.RegularExpressions.Regex]::Matches($Value, '`([^`]+)`')
  if ($matches.Count -eq 0) {
    return @()
  }

  return @($matches | ForEach-Object { $_.Groups[1].Value } | Where-Object {
    $_ -match "^(docs|samples|applications|smoke|src|artifacts|eng|tests|README)"
  } | Select-Object -Unique)
}

function ConvertTo-CommandList {
  param(
    [string]$Title,
    [string]$Content,
    [string[]]$RepoCodePaths
  )

  $commands = New-Object System.Collections.Generic.List[string]

  if ($Title -match "TensorRtExec" -or @($RepoCodePaths | Where-Object { $_ -like "applications/TensorRtExec*" }).Count -gt 0) {
    $commands.Add("dotnet run --project .\applications\TensorRtExec -- --help")
    $commands.Add("dotnet run --project .\applications\TensorRtExec -- --onnx .\models\model.onnx --saveEngine .\models\model.plan --buildOnly --exportReport .\artifacts\model-build-report.json")
  }

  if ($Title -match "OnnxToEngine|ONNX|trtexec" -or @($RepoCodePaths | Where-Object { $_ -like "samples/OnnxToEngine*" }).Count -gt 0) {
    $commands.Add("dotnet run --project .\samples\OnnxToEngine -- --onnx .\models\model.onnx --engine .\models\model.plan --min-shapes input:1x3x224x224 --opt-shapes input:1x3x224x224 --max-shapes input:1x3x224x224")
  }

  if ($Title -match "Yolo|YOLO|YoloVision" -or $Content -match "YOLO|YoloVision" -or @($RepoCodePaths | Where-Object { $_ -like "samples/YoloVision*" }).Count -gt 0) {
    $commands.Add("dotnet run --project .\samples\YoloVision -- --model .\models\yolo.onnx --labels .\models\coco.names --input-data .\models\yolo-preprocessed-fp32.bin --input-shape 1x3x640x640 --family v8 --task det --layout auto --confidence 0.25")
  }

  if ($Title -match "Classification|分类" -or @($RepoCodePaths | Where-Object { $_ -like "samples/Classification*" }).Count -gt 0) {
    $commands.Add("dotnet run --project .\samples\Classification -- --model .\models\classifier.onnx --labels .\models\labels.txt --input .\models\image.jpg --input-shape 1x3x224x224 --top-k 5")
  }

  if ($Content -match "package-consumer-runtime|post-publish|release proof|owner|proof") {
    $commands.Add("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseRuntimeProofExecutionMatrix.ps1")
    $commands.Add("pwsh -NoProfile -ExecutionPolicy Bypass -File .\eng\Export-ReleaseFreezeFinalVerification.ps1")
  }

  if ($commands.Count -eq 0) {
    $commands.Add("owner-action-required: article needs narrative expansion or real proof execution before publication claim.")
  }

  return @($commands | Select-Object -Unique)
}

function Get-SampleProject {
  param([string[]]$RepoCodePaths)

  $sample = $RepoCodePaths | Where-Object { $_ -like "samples/*" } | Select-Object -First 1
  if (-not [string]::IsNullOrWhiteSpace($sample)) {
    return $sample
  }

  $application = $RepoCodePaths | Where-Object { $_ -like "applications/*" } | Select-Object -First 1
  if (-not [string]::IsNullOrWhiteSpace($application)) {
    return $application
  }

  $smoke = $RepoCodePaths | Where-Object { $_ -like "smoke/*" } | Select-Object -First 1
  if (-not [string]::IsNullOrWhiteSpace($smoke)) {
    return $smoke
  }

  return "none"
}

function Get-ScreenshotNeeds {
  param(
    [string]$Title,
    [string]$Content,
    [string[]]$RepoCodePaths
  )

  $needs = New-Object System.Collections.Generic.List[string]

  if ($Title -match "GUI|WinForms|TensorRtExec" -or @($RepoCodePaths | Where-Object { $_ -like "applications/TensorRtExec*" }).Count -gt 0) {
    $needs.Add("TensorRtExec WinForms main screen")
    $needs.Add("build/report output screenshot")
  }

  if ($Title -match "Yolo|YOLO|Classification|分类|检测|分割|姿态|OBB" -or $Content -match "YoloVision|Classification") {
    $needs.Add("sample command output")
    $needs.Add("model asset manifest or evidence sidecar screenshot")
  }

  if ($Content -match "runtime package|NuGet|post-publish|package-consumer|owner|proof") {
    $needs.Add("validator output or final audit table screenshot")
  }

  if ($needs.Count -eq 0) {
    $needs.Add("diagram or table optional")
  }

  return @($needs | Select-Object -Unique)
}

function Get-ProofDependencies {
  param([string]$Text)

  $dependencies = New-Object System.Collections.Generic.List[string]

  foreach ($marker in @(
      "owner-authorization",
      "package-consumer-runtime",
      "linux-runner-proof",
      "real-model-runtime",
      "post-publish verification",
      "post-publish-verification",
      "sample-run-evidence",
      "build-only",
      "sidecar-only",
      "blocked-by-cuda-driver")) {
    if ($Text -match [System.Text.RegularExpressions.Regex]::Escape($marker)) {
      $dependencies.Add($marker)
    }
  }

  if ($dependencies.Count -eq 0) {
    $dependencies.Add("documentation-only")
  }

  return @($dependencies | Select-Object -Unique)
}

$roadmapPath = Join-Path $RepositoryRoot "docs\articles\zh-cn\technical-article-roadmap.md"
if (-not (Test-Path -LiteralPath $roadmapPath)) {
  throw "Missing roadmap: $roadmapPath"
}

$lines = Get-Content -LiteralPath $roadmapPath -Encoding utf8
$articleRows = @($lines | Where-Object { $_ -match "^\|\s*\d+\s*\|" })
$articles = New-Object System.Collections.Generic.List[object]

foreach ($line in $articleRows) {
  $cells = @($line.Trim().Trim("|").Split("|") | ForEach-Object { ConvertFrom-MarkdownCellText $_ })
  if ($cells.Count -lt 7) {
    continue
  }

  $articleId = [int]$cells[0]
  $category = $cells[1]
  $title = $cells[2]
  $coreScenario = $cells[3]
  $evidence = $cells[4]
  $assetRequirement = $cells[5]
  $status = $cells[6]
  $repoCodePaths = @(ConvertTo-PathList $evidence)
  $sampleProject = Get-SampleProject -RepoCodePaths $repoCodePaths
  $text = "$title $category $coreScenario $evidence $assetRequirement $status"
  $commands = @(ConvertTo-CommandList -Title $title -Content $coreScenario -RepoCodePaths $repoCodePaths)
  $proofDependencies = @(Get-ProofDependencies -Text $text)
  $screenshotsOrImagesNeeded = @(Get-ScreenshotNeeds -Title $title -Content $coreScenario -RepoCodePaths $repoCodePaths)
  $publishChannelFit = if ($category -match "宣发|项目总览|发布总结|README|文章矩阵") {
    @("微信公众号", "博客", "docs")
  }
  elseif ($category -match "案例|应用教程|安装部署") {
    @("博客", "docs")
  }
  else {
    @("docs", "博客")
  }

  $qualityNotes = New-Object System.Collections.Generic.List[string]
  $qualityNotes.Add("release freeze boundary: canPublishPublicly=false until all real proof validators pass.")
  $qualityNotes.Add("文章不得把 template/draft/runbook/local feed/ProjectReference/dependency probe/build-only/sidecar 当成真实 proof。")
  if ($title -match "Yolo|YOLO|YoloVision" -or $coreScenario -match "YoloVision|YOLO") {
    $qualityNotes.Add("YoloVision scope: YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom; tasks det/cls/seg/obb/pose/sem.")
  }
  if ($title -match "TensorRtExec|OnnxToEngine|trtexec" -or $coreScenario -match "TensorRtExec|OnnxToEngine|trtexec") {
    $qualityNotes.Add("TensorRtExec/OnnxToEngine build-only/report/sidecar evidence is not runtime proof.")
  }
  if ($assetRequirement -match "用户自备|owner|兼容|真实") {
    $qualityNotes.Add("owner-action-required until real assets, logs, hashes, licenses, and host metadata are supplied.")
  }

  $article = [ordered]@{
      articleId = $articleId
      title = $title
      category = $category
      targetAudience = if ($category -match "宣发|项目总览") { "项目评估者、技术负责人、潜在用户" } elseif ($category -match "案例|应用教程") { "C#/.NET TensorRT 使用者" } elseif ($category -match "发布|Owner|证据") { "release owner 与维护者" } else { "项目使用者与维护者" }
      coreScenario = $coreScenario
      repoCodePaths = @($repoCodePaths)
      sampleProject = $sampleProject
      modelAssetsRequired = $assetRequirement
      commands = @($commands)
      screenshotsOrImagesNeeded = @($screenshotsOrImagesNeeded)
      proofDependencies = @($proofDependencies)
      publishChannelFit = @($publishChannelFit)
      qualityNotes = @($qualityNotes.ToArray())
      completionStatus = $status
    }

  $articles.Add([pscustomobject]$article)
}

$articleArray = @($articles.ToArray())
$categories = @($articleArray | ForEach-Object { $_.category } | Sort-Object -Unique)
$yoloArticles = @($articleArray | Where-Object { $_.title -match "Yolo|YOLO|YoloVision" -or $_.coreScenario -match "YoloVision|YOLO" })
$tensorRtExecArticles = @($articleArray | Where-Object { $_.title -match "TensorRtExec|trtexec" -or $_.coreScenario -match "TensorRtExec|trtexec" })
$onnxToEngineArticles = @($articleArray | Where-Object { $_.title -match "OnnxToEngine|ONNX" -or $_.coreScenario -match "OnnxToEngine|ONNX" })
$ownerActionArticles = @($articleArray | Where-Object { @($_.commands | Where-Object { $_ -match "owner-action-required" }).Count -gt 0 -or $_.modelAssetsRequired -match "owner|用户自备|真实|兼容" })

$record = [ordered]@{
  generatedAtUtc = (Get-Date).ToUniversalTime().ToString("o")
  recordKind = "technical-article-publication-matrix"
  matrixState = "publication-planning-release-frozen"
  sourceRoadmap = "docs/articles/zh-cn/technical-article-roadmap.md"
  performsPublish = $false
  canPublishPublicly = $false
  canCloseReleaseIssue = $false
  releaseProofBoundary = "Article planning, tutorials, screenshots, build-only reports, sidecars, local feeds, ProjectReference consumers, and owner-action-required records cannot substitute real owner authorization, package-consumer-runtime, Linux runner, real-model-runtime, or post-publish verification proof."
  articleCount = $articleArray.Count
  categoryCount = $categories.Count
  categories = $categories
  yoloVisionArticleCount = $yoloArticles.Count
  tensorRtExecArticleCount = $tensorRtExecArticles.Count
  onnxToEngineArticleCount = $onnxToEngineArticles.Count
  ownerActionRequiredArticleCount = $ownerActionArticles.Count
  requiredFields = @(
    "articleId",
    "title",
    "category",
    "targetAudience",
    "coreScenario",
    "repoCodePaths",
    "sampleProject",
    "modelAssetsRequired",
    "commands",
    "screenshotsOrImagesNeeded",
    "proofDependencies",
    "publishChannelFit",
    "qualityNotes",
    "completionStatus"
  )
  yoloVisionScope = "YOLO v5/v6/v7/v8/v9/v10/v11/v26/custom"
  yoloVisionTaskScope = "det/cls/seg/obb/pose/sem; det、cls、seg、obb、pose、sem"
  articles = $articleArray
}

$outputRoot = Join-Path $RepositoryRoot "artifacts\final-release"
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null

$jsonPath = Join-Path $outputRoot "technical-article-publication-matrix.json"
$markdownPath = Join-Path $outputRoot "technical-article-publication-matrix.md"

$record | ConvertTo-Json -Depth 16 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$articleRowsMarkdown = $articleArray | ForEach-Object {
  $repoCodePathValues = @($_.repoCodePaths)
  $paths = if ($repoCodePathValues.Count -gt 0) { ($repoCodePathValues -join "<br>") } else { "owner-action-required" }
  $commandsCell = ($_.commands | Select-Object -First 2) -join "<br>"
  $proofs = ($_.proofDependencies -join "<br>").Replace("|", "\|")
  "| $($_.articleId) | $($_.category.Replace("|", "\|")) | $($_.title.Replace("|", "\|")) | $($_.sampleProject.Replace("|", "\|")) | $paths | $commandsCell | $proofs | $($_.completionStatus.Replace("|", "\|")) |"
}

$markdown = @"
# Technical Article Publication Matrix

生成时间：$($record.generatedAtUtc)

## Summary

- record kind: ``$($record.recordKind)``
- matrix state: ``$($record.matrixState)``
- article count: ``$($record.articleCount)``
- category count: ``$($record.categoryCount)``
- YoloVision article count: ``$($record.yoloVisionArticleCount)``
- TensorRtExec article count: ``$($record.tensorRtExecArticleCount)``
- OnnxToEngine article count: ``$($record.onnxToEngineArticleCount)``
- owner-action-required article count: ``$($record.ownerActionRequiredArticleCount)``
- performsPublish=false
- canPublishPublicly=false
- canCloseReleaseIssue=false

## Proof Boundary

$($record.releaseProofBoundary)

## YoloVision Scope

- family scope: ``$($record.yoloVisionScope)``
- task scope: ``$($record.yoloVisionTaskScope)``

## Articles

| ID | Category | Title | Sample/Application | Repo code paths | Commands | Proof dependencies | Status |
|---:|---|---|---|---|---|---|---|
$($articleRowsMarkdown -join "`r`n")
"@

Set-Content -LiteralPath $markdownPath -Value $markdown -Encoding utf8

Write-Host "Technical article publication matrix written:"
Write-Host "  Json=$jsonPath"
Write-Host "  Markdown=$markdownPath"
Write-Host "ArticleCount=$($record.articleCount)"
Write-Host "CanPublishPublicly=False"
