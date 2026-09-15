Add-Type -AssemblyName presentationCore

$player = New-Object System.Windows.Media.MediaPlayer

$outputDir = Join-Path $PSScriptRoot "test_outputs"
$files = Get-ChildItem -Path $outputDir -Filter "*.wav" | Sort-Object Name

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  RACHANA XTTS - VOICE OUTPUT PLAYER" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Total files : $($files.Count)" -ForegroundColor Yellow
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

$i = 1
foreach ($file in $files) {
    Write-Host "------------------------------------------------------------" -ForegroundColor DarkGray
    Write-Host "  [$i/$($files.Count)] Playing : $($file.Name)" -ForegroundColor Green
    Write-Host ""

    $player.Open([uri]$file.FullName)
    $player.Play()

    Start-Sleep -Milliseconds 800

    $dur = $player.NaturalDuration
    if ($dur.HasTimeSpan) {
        $secs = [int]$dur.TimeSpan.TotalSeconds + 1
    } else {
        $secs = 12
    }

    Write-Host "  Duration : ~$($secs - 1) seconds" -ForegroundColor DarkYellow
    Write-Host ""

    Start-Sleep -Seconds $secs

    $player.Stop()
    Start-Sleep -Milliseconds 400
    $i++
}

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  All outputs played! Rachana voice test complete." -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
