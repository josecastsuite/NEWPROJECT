@echo off
chcp 65001 >nul
echo JoseCast bagimliliklari yukleniyor...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt --force-reinstall
echo.
echo Tamamlandi. Programi 'run_debug.bat' veya 'python main.py' ile calistirin.
pause
