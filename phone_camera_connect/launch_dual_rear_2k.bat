@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ========================================================
echo  ЗАПУСК ДВУХПОТОЧНОЙ ТЫЛОВОЙ КАМЕРЫ (SAMSUNG S25 ULTRA)
echo  Разрешение: 2K QHD (2560x1440 @ 60 FPS)
echo ========================================================
"C:\Users\jack_\MAIN\Programms\Python 3.11.9\python.exe" dual_rear_camera_view.py --preset 2k
pause
