# setup_remote_wsl.ps1 — Автоматическая настройка изолированного окружения WSL 2
$ErrorActionPreference = "Stop"
$TailscaleIP = "100.69.31.98"
$VkrHostPath = "D:\ПРОФ\ВУЗ\ВКР"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host " Настройка удаленного доступа WSL 2 (Изоляция папки ВКР)" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Проверка и установка дистрибутива Ubuntu
$distros = wsl.exe -l -v 2>$null
if ($LASTEXITCODE -ne 0 -or ($distros -notmatch "Ubuntu")) {
    Write-Host "`n[1/5] Установка Ubuntu в WSL 2..." -ForegroundColor Yellow
    wsl.exe --install -d Ubuntu --web-download --no-launch
    Write-Host "Установка дистрибутива завершена." -ForegroundColor Green
} else {
    Write-Host "`n[1/5] Ubuntu уже установлена в WSL 2." -ForegroundColor Green
}

# 2. Настройка изоляции дисков в /etc/wsl.conf
Write-Host "`n[2/5] Применение политик изоляции в /etc/wsl.conf..." -ForegroundColor Yellow

$wslConfScript = @'
cat << 'EOF' > /etc/wsl.conf
[automount]
enabled = false
mountFsTab = true

[interop]
enabled = false
appendWindowsPath = false
EOF
'@
wsl.exe -d Ubuntu -u root -- bash -c "$wslConfScript"

# Перезапуск экземпляра WSL для применения wsl.conf
wsl.exe --terminate Ubuntu 2>$null

# 3. Создание пользователя teammate, каталога ВКР и настройка прав
Write-Host "`n[3/5] Создание пользователя teammate и точки монтирования..." -ForegroundColor Yellow

$userScript = @"
id -u teammate &>/dev/null || useradd -m -s /bin/bash -u 1001 teammate
usermod -aG video,render teammate

mkdir -p /home/teammate/vkr
grep -q "/home/teammate/vkr" /etc/fstab || echo "$VkrHostPath /home/teammate/vkr drvfs rw,noatime,uid=1001,gid=1001,umask=022 0 0" >> /etc/fstab
mount -a 2>/dev/null

mkdir -p /home/teammate/.ssh
touch /home/teammate/.ssh/authorized_keys
chown -R teammate:teammate /home/teammate/.ssh
chmod 700 /home/teammate/.ssh
chmod 600 /home/teammate/.ssh/authorized_keys
"@
wsl.exe -d Ubuntu -u root -- bash -c "$userScript"

# 4. Установка и запуск OpenSSH сервера в Ubuntu
Write-Host "`n[4/5] Установка и конфигурация OpenSSH сервера..." -ForegroundColor Yellow

$sshScript = @'
apt-get update -qq && apt-get install -y -qq openssh-server python3 python3-pip

sed -i 's/#*Port 22/Port 22/' /etc/ssh/sshd_config
sed -i 's/#*PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sed -i 's/#*PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config
grep -q "AllowUsers teammate" /etc/ssh/sshd_config || echo "AllowUsers teammate" >> /etc/ssh/sshd_config

service ssh restart
'@
wsl.exe -d Ubuntu -u root -- bash -c "$sshScript"

# 5. Настройка сетевой маршрутизации и брандмауэра Windows
Write-Host "`n[5/5] Настройка проброса порта через Tailscale ($TailscaleIP)..." -ForegroundColor Yellow

$wslIp = (wsl.exe -d Ubuntu -u root -- hostname -I).Trim().Split(' ')[0]
Write-Host "Внутренний IP адрес WSL: $wslIp" -ForegroundColor Cyan

netsh interface portproxy delete v4tov4 listenport=22 listenaddress=$TailscaleIP 2>$null
netsh interface portproxy add v4tov4 listenport=22 listenaddress=$TailscaleIP connectport=22 connectaddress=$wslIp

New-NetFirewallRule -DisplayName "Tailscale WSL SSH 22" -Direction Inbound -LocalPort 22 -Protocol TCP -Action Allow -ErrorAction SilentlyContinue | Out-Null

Write-Host "`n==========================================================" -ForegroundColor Green
Write-Host " ГОТОВО! Окружение полностью настроено." -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
Write-Host "1. Добавь публичный SSH-ключ сокомандника в файл:"
Write-Host "   \\wsl$\Ubuntu\home\teammate\.ssh\authorized_keys" -ForegroundColor Yellow
Write-Host "`n2. Команда для подключения сокомандника с его ноутбука:"
Write-Host "   ssh teammate@$TailscaleIP" -ForegroundColor Cyan
Write-Host "`n3. Проверка видеокарты внутри сессии сокомандника:"
Write-Host "   python3 -c 'import torch; print(torch.cuda.is_available())'" -ForegroundColor Gray
