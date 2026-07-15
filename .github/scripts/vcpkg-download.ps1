# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

param(
  [Parameter(Mandatory = $true)] [string] $Url,
  [Parameter(Mandatory = $true)] [string] $Sha512,
  [Parameter(Mandatory = $true)] [string] $Destination
)

$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
$partial = "$Destination.partial"

New-Item -ItemType Directory -Force (Split-Path -Parent $Destination) | Out-Null
foreach ($attempt in 1..3) {
  $errors = @()
  try {
    Remove-Item $partial -Force -ErrorAction SilentlyContinue
    Invoke-WebRequest -UseBasicParsing -Uri $Url -OutFile $partial
    $actual = (Get-FileHash $partial -Algorithm SHA512).Hash.ToLowerInvariant()
    if ($actual -ne $Sha512.ToLowerInvariant()) {
      throw "SHA512 mismatch: expected $Sha512, got $actual"
    }
    Move-Item $partial $Destination -Force
    exit 0
  } catch {
    $errors += "Invoke-WebRequest: $($_.Exception.Message)"
  }

  # Some runner endpoints reject WinHTTP while the system curl succeeds; GitHub
  # release assets exhibit the inverse. Trying both avoids domain-specific policy.
  $curl = Join-Path $env:SystemRoot 'System32\curl.exe'
  if (Test-Path $curl) {
    try {
      Remove-Item $partial -Force -ErrorAction SilentlyContinue
      & $curl --fail --location --silent --show-error --output $partial $Url
      if ($LASTEXITCODE -ne 0) {
        throw "curl exited with status $LASTEXITCODE"
      }
      $actual = (Get-FileHash $partial -Algorithm SHA512).Hash.ToLowerInvariant()
      if ($actual -ne $Sha512.ToLowerInvariant()) {
        throw "SHA512 mismatch: expected $Sha512, got $actual"
      }
      Move-Item $partial $Destination -Force
      exit 0
    } catch {
      $errors += "curl: $($_.Exception.Message)"
    }
  }
  Write-Warning "Asset download failed (attempt $attempt/3): $($errors -join '; ')"
}

Remove-Item $partial -Force -ErrorAction SilentlyContinue
throw "Could not download and verify $Url"
