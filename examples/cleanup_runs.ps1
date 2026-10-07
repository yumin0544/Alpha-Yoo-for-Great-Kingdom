# Explicit, recoverable run cleanup. Without -Apply this only prints a plan.
[CmdletBinding()]
param(
    [string]$RunsPath = (Join-Path $PSScriptRoot '..\runs'),
    [string[]]$ArchiveRun = @(),
    [switch]$CompactLogs,
    [switch]$Apply,
    [string]$ArchiveName = ('cleanup-' + (Get-Date -Format 'yyyyMMdd-HHmmss') + '.zip')
)

$ErrorActionPreference = 'Stop'
$workspaceRoot = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
$runsRoot = (Resolve-Path -LiteralPath $RunsPath).Path.TrimEnd('\')
if (-not $runsRoot.StartsWith($workspaceRoot + '\', [StringComparison]::OrdinalIgnoreCase) -or
        [IO.Path]::GetFileName($runsRoot) -ne 'runs') {
    throw 'RunsPath must be a runs directory inside this repository.'
}

function Assert-RunPath([string]$Path) {
    $absolute = [IO.Path]::GetFullPath($Path)
    if (-not $absolute.StartsWith($runsRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
        throw "Target escapes runs: $absolute"
    }
    $relative = [IO.Path]::GetRelativePath($runsRoot, $absolute)
    if (($relative -split '[\\/]') -contains 'past-version') {
        throw "Protected past-version target: $absolute"
    }
    # Reject real links/junctions, including in ancestors. OneDrive cloud
    # placeholders have no LinkType and are not treated as symbolic links.
    $cursor = $absolute
    while ($cursor -ne $runsRoot) {
        if (Test-Path -LiteralPath $cursor) {
            if ((Get-Item -LiteralPath $cursor -Force).LinkType) {
                throw "Linked target cannot be cleaned: $cursor"
            }
        }
        $cursor = [IO.Path]::GetDirectoryName($cursor)
    }
    return $absolute
}

function Get-ProtectedSnapshot {
    $protected = Join-Path $runsRoot 'past-version'
    if (-not (Test-Path -LiteralPath $protected)) { return @() }
    return @(Get-ChildItem -LiteralPath $protected -File -Recurse -Force |
        Sort-Object FullName | ForEach-Object {
            [ordered]@{ path = [IO.Path]::GetRelativePath($runsRoot, $_.FullName);
                bytes = $_.Length; modified = $_.LastWriteTimeUtc.Ticks;
                sha256 = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash }
        })
}

$folders = @()
foreach ($name in ($ArchiveRun | Select-Object -Unique)) {
    if ($name -ne [IO.Path]::GetFileName($name) -or $name -in @('matches', 'archive')) {
        throw "ArchiveRun must name an explicit training folder: $name"
    }
    $folder = Assert-RunPath (Join-Path $runsRoot $name)
    if (-not (Test-Path -LiteralPath $folder -PathType Container)) {
        throw "Training folder does not exist: $folder"
    }
    foreach ($child in (Get-ChildItem -LiteralPath $folder -Directory -Recurse -Force)) {
        $null = Assert-RunPath $child.FullName
    }
    $folders += $folder
}

$terminalFiles = @()
$matches = Join-Path $runsRoot 'matches'
if (Test-Path -LiteralPath $matches) {
    foreach ($folder in (Get-ChildItem -LiteralPath $matches -Directory)) {
        $result = Join-Path $folder.FullName 'results.jsonl'
        if (-not (Test-Path -LiteralPath $result)) { continue }
        $last = Get-Content -LiteralPath $result -Tail 1 | ConvertFrom-Json
        if ($last.type -notin @('summary', 'aborted') -or
                $last.status -notin @('complete', 'cancelled', 'failed')) { continue }
        $names = @('progress.json', 'stop.request')
        # Failure logs can contain the only explanation of an error.
        if ($last.status -ne 'failed') { $names += 'worker.log' }
        foreach ($name in $names) {
            $file = Join-Path $folder.FullName $name
            if (Test-Path -LiteralPath $file) { $terminalFiles += (Assert-RunPath $file) }
        }
    }
}

$metricFiles = @()
if ($CompactLogs) {
    foreach ($folder in (Get-ChildItem -LiteralPath $runsRoot -Directory)) {
        if ($folder.Name -in @('past-version', 'archive', 'matches') -or
                $folder.FullName -in $folders) { continue }
        $file = Join-Path $folder.FullName 'metrics.jsonl'
        if (Test-Path -LiteralPath $file) { $metricFiles += (Assert-RunPath $file) }
    }
}

$files = @($terminalFiles) + @($metricFiles)
foreach ($folder in $folders) {
    $files += @(Get-ChildItem -LiteralPath $folder -File -Recurse -Force |
        ForEach-Object { Assert-RunPath $_.FullName })
}
$files = @($files | Sort-Object -Unique)
$plan = [ordered]@{ runs_root = $runsRoot; archive_runs = @($folders | ForEach-Object {
        [IO.Path]::GetFileName($_) }); terminal_sidecars = $terminalFiles;
    compact_logs = $metricFiles; backup_files = $files.Count;
    original_bytes = ($files | ForEach-Object { (Get-Item -LiteralPath $_).Length } |
        Measure-Object -Sum).Sum; apply = [bool]$Apply }
if (-not $Apply) { $plan | ConvertTo-Json -Depth 8; return }
if (-not $files.Count) { throw 'No files selected; nothing to clean.' }
if ($ArchiveName -ne [IO.Path]::GetFileName($ArchiveName) -or
        -not $ArchiveName.EndsWith('.zip')) { throw 'ArchiveName must be a ZIP filename.' }

$archiveRoot = Assert-RunPath (Join-Path $runsRoot 'archive')
$archivePath = Assert-RunPath (Join-Path $archiveRoot $ArchiveName)
$partialPath = Assert-RunPath ($archivePath + '.partial')
$reportPath = Assert-RunPath (Join-Path $archiveRoot ($ArchiveName + '.json'))
foreach ($path in @($archivePath, $partialPath, $reportPath)) {
    if (Test-Path -LiteralPath $path) { throw "Refusing to overwrite: $path" }
}
$protectedBefore = Get-ProtectedSnapshot
$snapshots = @($files | ForEach-Object {
    $item = Get-Item -LiteralPath $_
    [ordered]@{ path = [IO.Path]::GetRelativePath($runsRoot, $_).Replace('\', '/');
        bytes = $item.Length; modified = $item.LastWriteTimeUtc.Ticks;
        sha256 = (Get-FileHash -LiteralPath $_ -Algorithm SHA256).Hash }
})
$manifest = [ordered]@{ created_utc = [DateTime]::UtcNow.ToString('o');
    plan = $plan; files = $snapshots; protected_before = $protectedBefore }
$null = New-Item -ItemType Directory -Path $archiveRoot -Force
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem
$zip = [IO.Compression.ZipFile]::Open($partialPath, [IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($file in $snapshots) {
        $source = Assert-RunPath (Join-Path $runsRoot $file.path)
        Write-Host "Backing up $($file.path)"
        $null = [IO.Compression.ZipFileExtensions]::CreateEntryFromFile(
            $zip, $source, $file.path, [IO.Compression.CompressionLevel]::Optimal)
    }
    $entry = $zip.CreateEntry('cleanup-manifest.json')
    $writer = [IO.StreamWriter]::new($entry.Open(), [Text.UTF8Encoding]::new($false))
    try { $writer.Write(($manifest | ConvertTo-Json -Depth 12)) } finally { $writer.Dispose() }
} finally { $zip.Dispose() }

# Read every compressed entry back and verify SHA-256 before deleting anything.
$zip = [IO.Compression.ZipFile]::OpenRead($partialPath)
try {
    foreach ($file in $snapshots) {
        $entry = $zip.GetEntry($file.path)
        if ($null -eq $entry -or $entry.Length -ne $file.bytes) { throw 'Incomplete archive.' }
        $stream = $entry.Open()
        $algorithm = [Security.Cryptography.SHA256]::Create()
        try { $hash = [Convert]::ToHexString($algorithm.ComputeHash($stream)) }
        finally { $stream.Dispose(); $algorithm.Dispose() }
        if ($hash -ne $file.sha256) { throw "Archive checksum mismatch: $($file.path)" }
    }
} finally { $zip.Dispose() }
foreach ($file in $snapshots) {
    $source = Assert-RunPath (Join-Path $runsRoot $file.path)
    $item = Get-Item -LiteralPath $source
    if ($item.Length -ne $file.bytes -or $item.LastWriteTimeUtc.Ticks -ne $file.modified) {
        throw "Source changed during backup; nothing deleted: $source"
    }
}
foreach ($folder in $folders) {
    $expected = @($files | Where-Object {
        $_.StartsWith($folder + '\', [StringComparison]::OrdinalIgnoreCase)
    } | Sort-Object)
    $current = @(Get-ChildItem -LiteralPath $folder -File -Recurse -Force |
        ForEach-Object { Assert-RunPath $_.FullName } | Sort-Object)
    if (($expected -join "`n") -ne ($current -join "`n")) {
        throw "Folder contents changed during backup; nothing deleted: $folder"
    }
}
Move-Item -LiteralPath $partialPath -Destination $archivePath
Write-Host "Verified $($files.Count) backup entries. Archive: $archivePath"

if ($metricFiles.Count) {
    $python = Join-Path $workspaceRoot '.venv\Scripts\python.exe'
    $program = Join-Path $PSScriptRoot 'compact_training_logs.py'
    & $python -X utf8 -B $program --runs-root $runsRoot --apply @metricFiles
    if ($LASTEXITCODE -ne 0) { throw 'Log compaction failed; backup retained; no folders deleted.' }
}
foreach ($file in $terminalFiles) {
    $target = Assert-RunPath $file
    Remove-Item -LiteralPath $target -Force
}
foreach ($folder in $folders) {
    # Validate the resolved absolute target immediately before recursive deletion.
    $target = Assert-RunPath ((Resolve-Path -LiteralPath $folder).Path)
    Remove-Item -LiteralPath $target -Recurse -Force
}
$protectedAfter = Get-ProtectedSnapshot
if (($protectedBefore | ConvertTo-Json -Depth 8 -Compress) -ne
        ($protectedAfter | ConvertTo-Json -Depth 8 -Compress)) {
    throw 'past-version changed externally during cleanup; inspect protected files.'
}
$manifest['archive_bytes'] = (Get-Item -LiteralPath $archivePath).Length
$manifest['protected_unchanged'] = $true
$manifest['completed_utc'] = [DateTime]::UtcNow.ToString('o')
[IO.File]::WriteAllText($reportPath, ($manifest | ConvertTo-Json -Depth 12),
    [Text.UTF8Encoding]::new($false))
Write-Host "Completed. past-version content and timestamps unchanged. Report: $reportPath"
