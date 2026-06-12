[CmdletBinding()]
param(
  [string]$RequestedVersion,
  [string]$DefaultVersion = "4.0.0"
)

$utf8 = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = $utf8
$OutputEncoding = $utf8

$version = if ([string]::IsNullOrWhiteSpace($RequestedVersion)) {
  $DefaultVersion
}
else {
  $RequestedVersion.Trim()
}

if ($version -notmatch '^4\.0\.\d+([\-+][0-9A-Za-z][0-9A-Za-z\.-]*)?$') {
  throw "Package version must start with 4.0.x. Current value: '$version'."
}

Write-Output $version
