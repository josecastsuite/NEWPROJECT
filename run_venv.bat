@echo off
chcp 65001 >nul
cd /d "%~dp0"
call .venv\Scripts\activate.bat
set PATH=%~dp0\win_dlls;%PATH%
echo JoseCast Analyzer (venv) baslatiliyor...
python main.py > "josecast_console.log" 2>&1
set EXITCODE=%ERRORLEVEL%
echo.
echo Program sonlandi. Hata kodu: %EXITCODE%
echo josecast_console.log dosyasinda yazanlar varsa gonder.
pause
