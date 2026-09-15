$sourceDir   = Join-Path $PSScriptRoot "test_outputs"
$destination = Join-Path $env:USERPROFILE "Downloads\Rachana_XTTS_Voice_Outputs.zip"

Write-Host ""
Write-Host "Packing audio files..." -ForegroundColor Cyan

Compress-Archive -Path "$sourceDir\*.wav" -DestinationPath $destination -Force

Write-Host ""
Write-Host "Done! Zip saved to:" -ForegroundColor Green
Write-Host $destination -ForegroundColor Yellow
Write-Host ""

# Open the Downloads folder
Start-Process explorer.exe -ArgumentList (Split-Path $destination)
