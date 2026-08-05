#Requires -Version 5.1
$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "JoseCast Analyzer baslatiliyor..."
Write-Host "Tum ciktilar josecast.log, josecast_fault.log ve josecast_console.log dosyalarina yazilacak."

$proc = Start-Process -FilePath "python" -ArgumentList "main.py" -NoNewWindow -PassThru -RedirectStandardOutput "josecast_console.log" -RedirectStandardError "josecast_console_err.log" -Wait
$exit = $proc.ExitCode
Write-Host ""
Write-Host "Program sonlandi. Hata kodu (EXIT CODE): $exit"
Write-Host "josecast_console.log, josecast_console_err.log ve josecast_fault.log dosyalarindaki icerikleri gonder."
Pause
