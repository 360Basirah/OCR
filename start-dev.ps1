param(
    [int]$RestartDelaySeconds = 10
)

$Host.UI.RawUI.WindowTitle = "PaddleOCR Service (auto-restart)"

while ($true) {
    Write-Host "`n========================================" -ForegroundColor Cyan
    Write-Host "  Starting PaddleOCR service..." -ForegroundColor Cyan
    Write-Host "  $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')" -ForegroundColor DarkGray
    Write-Host "========================================`n" -ForegroundColor Cyan

    uvicorn app.main:app --host 0.0.0.0 --port 8090

    $code = $LASTEXITCODE
    if ($code -eq 0) {
        Write-Host "`nService exited cleanly (code 0). Stopping." -ForegroundColor Green
        break
    }

    Write-Host "`n========================================" -ForegroundColor Yellow
    Write-Host "  Service crashed (exit code $code)" -ForegroundColor Red
    Write-Host "  Restarting in $RestartDelaySeconds seconds..." -ForegroundColor Yellow
    Write-Host "  Press Ctrl+C to stop." -ForegroundColor DarkGray
    Write-Host "========================================`n" -ForegroundColor Yellow

    Start-Sleep -Seconds $RestartDelaySeconds
}
