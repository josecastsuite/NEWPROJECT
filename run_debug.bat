@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo JoseCast Analyzer baslatiliyor...
echo Tum ciktilar josecast.log, josecast_fault.log ve josecast_console.log dosyalarina yazilacak.
echo Lutfen pencere kapanmadan once konsoldaki bilgileri not edin.
echo.
python main.py > "josecast_console.log" 2>&1
set EXITCODE=%ERRORLEVEL%
echo.
echo Program sonlandi. Hata kodu (EXIT CODE): %EXITCODE%
echo.
echo josecast_console.log dosyasinda yazi varsa veya josecast_fault.log bos degilse bana gonder.
pause
