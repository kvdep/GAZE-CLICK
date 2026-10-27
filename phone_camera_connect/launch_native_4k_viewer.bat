@echo off
chcp 65001 > nul
cd /d "%~dp0"
echo ========================================================
echo  ЗАПУСК АППАРАТНОГО 4K UHD ОКНА (DIRECT3D11 GPU)
echo  Разрешение: 3840x2160 @ 60 FPS, Аппаратная задержка ~25 мс
echo ========================================================
"tools\scrcpy.exe" --video-source=camera --camera-id=0 --camera-size=3840x2160 --camera-fps=60 --video-bit-rate=32M --window-title="Samsung S25 Ultra 4K Native Direct3D11 (Zero Latency)"
pause
