@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo JoseCast icin izole Python ortami (venv) hazirlaniyor...
echo Bu, bilgisayarindaki diger Python paketlerini etkilemeden calisir.
python -m venv .venv
call .venv\Scripts\activate.bat
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Kurulum tamamlandi. Programi 'run_venv.bat' ile calistirin.
pause
