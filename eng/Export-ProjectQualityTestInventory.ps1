[CmdletBinding()]
param(
  [string]$Configuration = "Debug",
  [string]$TestProject = "",
  [string]$OutputDirectory = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repositoryRoot = Split-Path -Parent $PSScriptRoot
if ([string]::IsNullOrWhiteSpace($TestProject)) {
  $TestProject = Join-Path $repositoryRoot "tests\JYPPX.ProjectQuality.Tests\JYPPX.ProjectQuality.Tests.csproj"
}
elseif (-not [IO.Path]::IsPathRooted($TestProject)) {
  $TestProject = Join-Path $repositoryRoot $TestProject
}

if ([string]::IsNullOrWhiteSpace($OutputDirectory)) {
  $OutputDirectory = Join-Path $repositoryRoot "artifacts\test-analysis"
}
elseif (-not [IO.Path]::IsPathRooted($OutputDirectory)) {
  $OutputDirectory = Join-Path $repositoryRoot $OutputDirectory
}

New-Item -ItemType Directory -Force -Path $OutputDirectory | Out-Null

$logPath = Join-Path $OutputDirectory "project-quality-list-tests.log"
$arguments = @(
  "test",
  $TestProject,
  "-c",
  $Configuration,
  "--no-build",
  "--list-tests"
)

$lines = & dotnet @arguments 2>&1
$exitCode = $LASTEXITCODE
$lines | Set-Content -LiteralPath $logPath -Encoding utf8
if ($exitCode -ne 0) {
  $lines | ForEach-Object { Write-Host $_ }
  throw "dotnet test --list-tests failed with exit code $exitCode."
}

$tests = @(
  $lines |
    ForEach-Object { $_.ToString().Trim() } |
    Where-Object { $_ -match "^JYPPX\.ProjectQuality\.Tests\.[A-Za-z0-9_]+\." }
)

$classes = @(
  $tests |
    ForEach-Object {
      if ($_ -match "^(JYPPX\.ProjectQuality\.Tests\.[A-Za-z0-9_]+)\.") {
        $Matches[1]
      }
    } |
    Sort-Object -Unique
)

if ($tests.Count -eq 0 -or $classes.Count -eq 0) {
  throw "No ProjectQuality tests were discovered from '$TestProject'."
}

$definitions = @(
  [pscustomobject]@{ id = "A-F"; pattern = "^[A-F]" },
  [pscustomobject]@{ id = "G-M"; pattern = "^[G-M]" },
  [pscustomobject]@{ id = "N-S"; pattern = "^[N-S]" },
  [pscustomobject]@{ id = "T-Z"; pattern = "^[T-Z]" }
)

$shards = @(
  foreach ($definition in $definitions) {
    $shardClasses = @(
      $classes |
        Where-Object { ($_.Split(".")[-1]) -match $definition.pattern }
    )
    $filterExpression = ($shardClasses | ForEach-Object { "FullyQualifiedName~$_" }) -join "|"
    [pscustomobject][ordered]@{
      id = $definition.id
      classNamePattern = $definition.pattern
      classCount = $shardClasses.Count
      classes = $shardClasses
      filterExpression = $filterExpression
    }
  }
)

$unassignedClasses = @(
  $classes |
    Where-Object {
      $simpleName = $_.Split(".")[-1]
      -not ($definitions | Where-Object { $simpleName -match $_.pattern })
    }
)

$record = [pscustomobject][ordered]@{
  recordKind = "project-quality-test-inventory"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  configuration = $Configuration
  testProject = [IO.Path]::GetRelativePath($repositoryRoot, $TestProject).Replace("\", "/")
  testCount = $tests.Count
  classCount = $classes.Count
  listTestsLogPath = [IO.Path]::GetRelativePath($repositoryRoot, $logPath).Replace("\", "/")
  listTestsLogSha256 = (Get-FileHash -LiteralPath $logPath -Algorithm SHA256).Hash.ToLowerInvariant()
  shards = $shards
  unassignedClassCount = $unassignedClasses.Count
  unassignedClasses = $unassignedClasses
  boundary = "Inventory only; list-tests does not execute or pass the test suite. Each shard requires bounded execution with process-tree cleanup and preserved logs."
}

$jsonPath = Join-Path $OutputDirectory "project-quality-test-inventory.json"
$markdownPath = Join-Path $OutputDirectory "project-quality-test-inventory.md"
$record | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $jsonPath -Encoding utf8

$markdown = [Collections.Generic.List[string]]::new()
$markdown.Add("# ProjectQuality 测试清单与分片")
$markdown.Add("")
$markdown.Add("- 生成时间（UTC）：``$($record.generatedAtUtc)``")
$markdown.Add("- 配置：``$Configuration``")
$markdown.Add("- 测试：``$($record.testCount)``")
$markdown.Add("- 测试类：``$($record.classCount)``")
$markdown.Add("- 未分配测试类：``$($record.unassignedClassCount)``")
$markdown.Add("- 原始清单：``$($record.listTestsLogPath)``")
$markdown.Add("- 原始清单 SHA256：``$($record.listTestsLogSha256)``")
$markdown.Add("")
$markdown.Add("| 分片 | 类数量 | 过滤表达式长度 |")
$markdown.Add("| --- | ---: | ---: |")
foreach ($shard in $shards) {
  $markdown.Add("| ``$($shard.id)`` | $($shard.classCount) | $($shard.filterExpression.Length) |")
}
$markdown.Add("")
$markdown.Add("## 执行边界")
$markdown.Add("")
$markdown.Add($record.boundary)
$markdown | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "ProjectQuality test inventory written."
Write-Host "JSON=$jsonPath"
Write-Host "Markdown=$markdownPath"
Write-Host "Tests=$($record.testCount) Classes=$($record.classCount) Unassigned=$($record.unassignedClassCount)"
Write-Host "Shards=$((@($record.shards | ForEach-Object { "$($_.id):$($_.classCount)" })) -join ', ')"
