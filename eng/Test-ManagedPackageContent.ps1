[CmdletBinding()]
param(
  [string]$PackagePath,
  [string]$RepositoryRoot
)

if ([string]::IsNullOrWhiteSpace($RepositoryRoot)) {
  $scriptRoot = if ([string]::IsNullOrWhiteSpace($PSScriptRoot)) { (Get-Location).Path } else { $PSScriptRoot }
  $RepositoryRoot = (Resolve-Path (Join-Path $scriptRoot "..")).Path
}

if ([string]::IsNullOrWhiteSpace($PackagePath)) {
  $PackagePath = (Get-ChildItem -Path (Join-Path $RepositoryRoot "artifacts\managed") -Filter "JYPPX.TensorRT.CSharp.API.*.nupkg" -ErrorAction SilentlyContinue |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1).FullName
}

if ([string]::IsNullOrWhiteSpace($PackagePath) -or -not (Test-Path -LiteralPath $PackagePath)) {
  throw "Managed package was not found. Provide -PackagePath or pack into artifacts/managed first."
}

Add-Type -AssemblyName System.IO.Compression.FileSystem
$package = [System.IO.Compression.ZipFile]::OpenRead((Resolve-Path -LiteralPath $PackagePath).Path)
try {
  $entryNames = @($package.Entries | ForEach-Object { $_.FullName })
  $requiredAssemblies = @(
    "JYPPX.Shared.dll",
    "JYPPX.TensorRtSharp.dll",
    "JYPPX.CudaSharp.dll"
  )

  $requiredFrameworks = @("net47", "net481", "netstandard2.0", "netstandard2.1", "net8.0", "net10.0")
  $errors = New-Object System.Collections.Generic.List[string]

  foreach ($framework in $requiredFrameworks) {
    foreach ($assembly in $requiredAssemblies) {
      $expected = "lib/$framework/$assembly"
      if ($entryNames -notcontains $expected) {
        $errors.Add("Missing package entry: $expected")
      }
    }
  }

  if ($errors.Count -gt 0) {
    $errors | ForEach-Object { Write-Error $_ }
    exit 1
  }
}
finally {
  $package.Dispose()
}

Write-Host "Managed package content is valid: $PackagePath"
