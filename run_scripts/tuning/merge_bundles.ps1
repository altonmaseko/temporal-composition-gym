param([Parameter(Mandatory)][string]$ResultsRoot, [Parameter(Mandatory)][string[]]$Bundles)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression.FileSystem
$targetRoot = [IO.Path]::GetFullPath($ResultsRoot)
foreach ($bundle in $Bundles) {
    $expected = (Get-Content -LiteralPath "$bundle.sha256").Split(' ')[0].ToLower()
    if ((Get-FileHash -LiteralPath $bundle -Algorithm SHA256).Hash.ToLower() -ne $expected) { throw "Hash mismatch: $bundle" }
    $zip = [IO.Compression.ZipFile]::OpenRead($bundle)
    try {
        foreach ($entry in $zip.Entries) {
            $dest = [IO.Path]::GetFullPath((Join-Path $targetRoot $entry.FullName))
            if (-not $dest.StartsWith($targetRoot + [IO.Path]::DirectorySeparatorChar) -or $dest.Length -ge 240) { throw "Unsafe/long path: $dest" }
            if ([IO.Path]::GetExtension($dest) -notin '.csv','.json') { throw "Unexpected member: $dest" }
            $inputStream = $entry.Open()
            $memory = [IO.MemoryStream]::new()
            try { $inputStream.CopyTo($memory); $bytes = $memory.ToArray() } finally { $inputStream.Dispose(); $memory.Dispose() }
            if (Test-Path -LiteralPath $dest) {
                $old = [IO.File]::ReadAllBytes($dest)
                if ([Convert]::ToBase64String($old) -ne [Convert]::ToBase64String($bytes)) { throw "Conflicting evidence: $dest" }
            } else {
                [IO.Directory]::CreateDirectory([IO.Path]::GetDirectoryName($dest)) | Out-Null
                [IO.File]::WriteAllBytes($dest,$bytes)
            }
        }
    } finally { $zip.Dispose() }
}
Write-Output 'BUNDLES VERIFIED AND MERGED'
