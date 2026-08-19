@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo JoseCast bagimliliklari yukleniyor (numpy/numba uyumsuzlugunu onlemek icin once eski surumler kaldirilacak)...
python -m pip install --upgrade pip
python -m pip uninstall -y numpy numba llvmlite 2>nul
python -m pip install -r requirements.txt --force-reinstall
echo.
echo Tamamlandi. Asagidaki versiyonlar kontrol ediliyor:
python -c "import numpy, numba; print('NumPy', numpy.__version__); print('Numba', numba.__version__)"
echo.
echo Programi 'run_debug.bat' veya 'python main.py' ile calistirin.
pause
