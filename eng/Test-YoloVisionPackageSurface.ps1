[CmdletBinding()]
param(
  [string]$RepositoryRoot,
  [string]$PackagePath,
  [string]$BuiltOutputDirectory,
  [string]$ReportDirectory,
  [string]$PackageVersion = "4.0.0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

$RepositoryRoot = [IO.Path]::GetFullPath($RepositoryRoot)
$utf8 = [Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8
Add-Type -AssemblyName System.IO.Compression.FileSystem

if ($PSVersionTable.PSEdition -ne "Core") {
  throw "YoloVision net8.0 package surface validation requires PowerShell 7 or later (pwsh)."
}

if ([string]::IsNullOrWhiteSpace($PackagePath)) {
  $PackagePath = Join-Path $RepositoryRoot "artifacts\yolovision-nupkg\JYPPX.TensorRT.CSharp.API.YoloVision.$PackageVersion.nupkg"
}
if ([string]::IsNullOrWhiteSpace($BuiltOutputDirectory)) {
  $BuiltOutputDirectory = Join-Path $RepositoryRoot "applications\YoloVision\bin\Release\net8.0"
}
if ([string]::IsNullOrWhiteSpace($ReportDirectory)) {
  $ReportDirectory = Join-Path $RepositoryRoot "artifacts\yolovision\package-surface-audit"
}

$PackagePath = [IO.Path]::GetFullPath($PackagePath)
$BuiltOutputDirectory = [IO.Path]::GetFullPath($BuiltOutputDirectory)
$ReportDirectory = [IO.Path]::GetFullPath($ReportDirectory)
$builtAssemblyPath = Join-Path $BuiltOutputDirectory "YoloVision.dll"
$builtXmlPath = Join-Path $BuiltOutputDirectory "YoloVision.xml"
foreach ($requiredPath in @($PackagePath, $builtAssemblyPath, $builtXmlPath)) {
  if (-not (Test-Path -LiteralPath $requiredPath -PathType Leaf)) {
    throw "Required YoloVision package-surface input does not exist: $requiredPath"
  }
}

function Get-ZipEntrySha256 {
  param([IO.Compression.ZipArchiveEntry]$Entry)

  $stream = $Entry.Open()
  try {
    $sha = [Security.Cryptography.SHA256]::Create()
    try {
      return ([BitConverter]::ToString($sha.ComputeHash($stream))).Replace("-", "").ToLowerInvariant()
    }
    finally {
      $sha.Dispose()
    }
  }
  finally {
    $stream.Dispose()
  }
}

function Get-SignatureTypes {
  param([Reflection.MemberInfo]$Member)

  if ($Member -is [Reflection.MethodInfo]) {
    @($Member.ReturnType) + @($Member.GetParameters() | ForEach-Object { $_.ParameterType })
    return
  }
  if ($Member -is [Reflection.ConstructorInfo]) {
    @($Member.GetParameters() | ForEach-Object { $_.ParameterType })
    return
  }
  if ($Member -is [Reflection.PropertyInfo]) {
    @($Member.PropertyType) + @($Member.GetIndexParameters() | ForEach-Object { $_.ParameterType })
    return
  }
  if ($Member -is [Reflection.FieldInfo]) {
    @($Member.FieldType)
    return
  }
  if ($Member -is [Reflection.EventInfo] -and $null -ne $Member.EventHandlerType) {
    @($Member.EventHandlerType)
  }
}

function Add-TypeFindings {
  param(
    [Type]$Type,
    [string]$Surface,
    [Collections.Generic.List[object]]$Findings
  )

  $candidate = $Type
  while ($candidate.HasElementType -and $null -ne $candidate.GetElementType()) {
    $candidate = $candidate.GetElementType()
  }

  if ($candidate.IsPointer -or
      $candidate -eq [IntPtr] -or
      $candidate -eq [UIntPtr] -or
      [Runtime.InteropServices.SafeHandle].IsAssignableFrom($candidate)) {
    $Findings.Add([pscustomobject]@{ category = "forbidden-pointer-or-handle"; surface = $Surface; type = [string]$Type })
  }
  $candidateFullName = if ($null -eq $candidate.FullName) { "" } else { [string]$candidate.FullName }
  if ($candidateFullName.Contains("OnnxSampleOptions", [StringComparison]::Ordinal) -or
      [string]::Equals($candidate.Namespace, "JYPPX.SampleSupport", [StringComparison]::Ordinal)) {
    $Findings.Add([pscustomobject]@{ category = "sample-internal-type-leak"; surface = $Surface; type = [string]$Type })
  }
  if ($candidate.IsGenericType) {
    foreach ($argument in $candidate.GetGenericArguments()) {
      Add-TypeFindings -Type $argument -Surface $Surface -Findings $Findings
    }
  }
}

$zip = [IO.Compression.ZipFile]::OpenRead($PackagePath)
try {
  $entries = @($zip.Entries)
  $assemblyEntry = @($entries | Where-Object { [string]::Equals($_.FullName, "lib/net8.0/YoloVision.dll", [StringComparison]::OrdinalIgnoreCase) })
  $xmlEntry = @($entries | Where-Object { [string]::Equals($_.FullName, "lib/net8.0/YoloVision.xml", [StringComparison]::OrdinalIgnoreCase) })
  $nuspecEntry = @($entries | Where-Object { $_.FullName.EndsWith(".nuspec", [StringComparison]::OrdinalIgnoreCase) })
  if ($assemblyEntry.Count -ne 1 -or $xmlEntry.Count -ne 1 -or $nuspecEntry.Count -ne 1) {
    throw "YoloVision package must contain exactly one net8.0 DLL, XML document, and nuspec."
  }
  $packageAssemblySha256 = Get-ZipEntrySha256 -Entry $assemblyEntry[0]
  $packageXmlSha256 = Get-ZipEntrySha256 -Entry $xmlEntry[0]

  $nuspecStream = $nuspecEntry[0].Open()
  try {
    $reader = [IO.StreamReader]::new($nuspecStream, [Text.Encoding]::UTF8)
    try { [xml]$nuspec = $reader.ReadToEnd() } finally { $reader.Dispose() }
  }
  finally {
    $nuspecStream.Dispose()
  }
}
finally {
  $zip.Dispose()
}

$builtAssemblySha256 = (Get-FileHash -LiteralPath $builtAssemblyPath -Algorithm SHA256).Hash.ToLowerInvariant()
$builtXmlSha256 = (Get-FileHash -LiteralPath $builtXmlPath -Algorithm SHA256).Hash.ToLowerInvariant()
foreach ($dependency in Get-ChildItem -LiteralPath $BuiltOutputDirectory -File -Filter *.dll) {
  try {
    [Runtime.Loader.AssemblyLoadContext]::Default.LoadFromAssemblyPath($dependency.FullName) | Out-Null
  }
  catch {
  }
}
$assembly = [Runtime.Loader.AssemblyLoadContext]::Default.LoadFromAssemblyPath($builtAssemblyPath)
$exportedTypes = @($assembly.GetExportedTypes() | Sort-Object FullName)
$publicMembers = @($exportedTypes | ForEach-Object {
  $_.GetMembers([Reflection.BindingFlags]'Public,Instance,Static,DeclaredOnly')
})
$findings = [Collections.Generic.List[object]]::new()
foreach ($type in $exportedTypes) {
  Add-TypeFindings -Type $type -Surface ([string]$type.FullName) -Findings $findings
  foreach ($member in $type.GetMembers([Reflection.BindingFlags]'Public,Instance,Static,DeclaredOnly')) {
    foreach ($signatureType in @(Get-SignatureTypes -Member $member)) {
      Add-TypeFindings -Type $signatureType -Surface "$($type.FullName).$($member.Name)" -Findings $findings
    }
  }
}

[xml]$xmlDocumentation = Get-Content -LiteralPath $builtXmlPath -Raw -Encoding utf8
$documentedMembers = @($xmlDocumentation.doc.members.member)
$documentedNames = @($documentedMembers | ForEach-Object { [string]$_.name })
$requiredCommandDocs = @(
  "T:YoloVisionSample.YoloVisionCommand",
  "M:YoloVisionSample.YoloVisionCommand.Run(System.String[])"
)
$missingCommandDocs = @($requiredCommandDocs | Where-Object { $documentedNames -notcontains $_ })
$packageId = [string]$nuspec.package.metadata.id
$packageVersionValue = [string]$nuspec.package.metadata.version
$packageAssemblyMatchesBuild = [string]::Equals($packageAssemblySha256, $builtAssemblySha256, [StringComparison]::Ordinal)
$packageXmlMatchesBuild = [string]::Equals($packageXmlSha256, $builtXmlSha256, [StringComparison]::Ordinal)
$valid = $packageId -eq "JYPPX.TensorRT.CSharp.API.YoloVision" -and
  $packageVersionValue -eq $PackageVersion -and
  $packageAssemblyMatchesBuild -and
  $packageXmlMatchesBuild -and
  $findings.Count -eq 0 -and
  $missingCommandDocs.Count -eq 0

New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null
$report = [pscustomobject][ordered]@{
  schemaVersion = 1
  recordKind = "yolovision-nuget-public-surface-audit"
  generatedAtUtc = [DateTime]::UtcNow.ToString("O")
  validationState = if ($valid) { "passed-pointer-free-package-surface-audit" } else { "failed-package-surface-audit" }
  valid = $valid
  package = [pscustomobject][ordered]@{
    id = $packageId
    version = $packageVersionValue
    length = (Get-Item -LiteralPath $PackagePath).Length
    sha256 = (Get-FileHash -LiteralPath $PackagePath -Algorithm SHA256).Hash.ToLowerInvariant()
    assemblySha256 = $packageAssemblySha256
    xmlSha256 = $packageXmlSha256
    assemblyMatchesBuiltOutput = $packageAssemblyMatchesBuild
    xmlMatchesBuiltOutput = $packageXmlMatchesBuild
  }
  surface = [pscustomobject][ordered]@{
    exportedTypeCount = $exportedTypes.Count
    publicDeclaredMemberCount = $publicMembers.Count
    xmlDocumentedMemberCount = $documentedMembers.Count
    requiredCommandDocumentationCount = $requiredCommandDocs.Count
    missingCommandDocumentation = $missingCommandDocs
    forbiddenPointerOrHandleFindingCount = @($findings | Where-Object { $_.category -eq "forbidden-pointer-or-handle" }).Count
    sampleInternalTypeLeakFindingCount = @($findings | Where-Object { $_.category -eq "sample-internal-type-leak" }).Count
    findings = @($findings)
  }
  boundary = [pscustomobject][ordered]@{
    pointerFreePackageSurfaceAudit = $true
    runtimeExecutionProof = $false
    packageConsumerRuntimeProof = $false
    publicRedistributionApproval = $false
    performsPublish = $false
    canPublishPublicly = $false
  }
}

$reportPath = Join-Path $ReportDirectory "yolovision-package-surface-audit.json"
$report | ConvertTo-Json -Depth 10 | Set-Content -LiteralPath $reportPath -Encoding utf8
$markdownPath = [IO.Path]::ChangeExtension($reportPath, ".md")
@(
  "# YoloVision Package Surface Audit",
  "",
  "- state: ``$($report.validationState)``",
  "- package: ``$packageId $packageVersionValue``",
  "- exported types: ``$($exportedTypes.Count)``",
  "- public declared members: ``$($publicMembers.Count)``",
  "- XML documented members: ``$($documentedMembers.Count)``",
  "- pointer/handle findings: ``$($report.surface.forbiddenPointerOrHandleFindingCount)``",
  "- sample-internal leaks: ``$($report.surface.sampleInternalTypeLeakFindingCount)``",
  "- package DLL/XML match built output: ``$packageAssemblyMatchesBuild`` / ``$packageXmlMatchesBuild``",
  "- performs publish: ``False``",
  "",
  "This audit checks the actual DLL and XML stored in the local YoloVision NuGet package. It is a source/package quality record, not runtime, public-download, redistribution, or publication proof."
) | Set-Content -LiteralPath $markdownPath -Encoding utf8

Write-Host "ValidationState=$($report.validationState) ExportedTypeCount=$($exportedTypes.Count) PublicDeclaredMemberCount=$($publicMembers.Count) FindingCount=$($findings.Count)"
Write-Host "PackageAssemblyMatchesBuild=$packageAssemblyMatchesBuild PackageXmlMatchesBuild=$packageXmlMatchesBuild PerformsPublish=False"
Write-Host "Report=$reportPath"

if (-not $valid) {
  throw "YoloVision package surface audit failed. See $reportPath"
}
