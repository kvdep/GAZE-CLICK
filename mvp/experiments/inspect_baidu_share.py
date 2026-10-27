# -*- coding: utf-8 -*-
import sys, requests, json

if sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

surl = 'sP8FaZe4IfA6zb40r8gr0Q'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
    'Referer': f'https://pan.baidu.com/s/1{surl}'
}

# 1. Verify
r_verify = requests.post(f'https://pan.baidu.com/share/verify?surl={surl}', data={'pwd': '4wgk'}, headers=headers)
randsk = r_verify.json().get('randsk')
cookies = {'BDCLND': randsk}

# 2. List root
list_url = f'https://pan.baidu.com/share/list?shorturl={surl}&dir=%2F&root=1&num=100&page=1'
r_list = requests.get(list_url, headers=headers, cookies=cookies)
data = r_list.json()
print('Root list errno:', data.get('errno'))
file_list = data.get('list', [])
print(f'Total items: {len(file_list)}')

for item in file_list:
    isdir = 'DIR' if item.get('isdir') else 'FILE'
    name = item.get('server_filename')
    path = item.get('path')
    sz = float(item.get('size') or 0) / 1024 / 1024
    fs_id = item.get('fs_id')
    print(f"{isdir:<4} | {name} | {sz:>8.2f} MB")

    if item.get('isdir'):
        sub_url = f'https://pan.baidu.com/share/list?shorturl={surl}&dir={requests.utils.quote(path)}&num=100&page=1'
        r_sub = requests.get(sub_url, headers=headers, cookies=cookies)
        sub_items = r_sub.json().get('list', [])
        print(f"  --- Subfolder '{name}' has {len(sub_items)} items ---")
        for sub in sub_items:
            s_isdir = 'DIR' if sub.get('isdir') else 'FILE'
            s_name = sub.get('server_filename')
            s_sz = float(sub.get('size') or 0) / 1024 / 1024
            s_path = sub.get('path')
            print(f"    {s_isdir:<4} | {s_name} | {s_sz:>8.2f} MB | {s_path}")
            if sub.get('isdir'):
                sub2_url = f'https://pan.baidu.com/share/list?shorturl={surl}&dir={requests.utils.quote(s_path)}&num=100&page=1'
                r_sub2 = requests.get(sub2_url, headers=headers, cookies=cookies)
                for s2 in r_sub2.json().get('list', []):
                    s2_name = s2.get('server_filename')
                    s2_sz = float(s2.get('size') or 0) / 1024 / 1024
                    print(f"      {s2_name} | {s2_sz:>8.2f} MB")
