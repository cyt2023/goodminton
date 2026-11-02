"""Index user-provided videos and extract sparse visual inspection frames."""
from pathlib import Path
import json
import re
import shutil
import sqlite3
import subprocess
import sys
import uuid

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
sys.path.insert(0, str(ROOT / '.deps'))
import imageio_ffmpeg
from PIL import Image, ImageDraw, ImageFont

DATA = PROJECT / '03_数据'
PREVIEW = DATA / '画面预览'
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

def metadata(file):
    result = subprocess.run([FFMPEG, '-hide_banner', '-i', str(file), '-map', '0:v:0',
                             '-frames:v', '1', '-f', 'null', '-'], capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr[-1200:])
    log = result.stderr
    duration = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', log)
    stream = next((line for line in log.splitlines() if 'Video:' in line and 'Stream #' in line), '')
    dimensions = re.search(r'\b(\d{2,5})x(\d{2,5})\b', stream)
    fps = re.search(r'([\d.]+) fps', stream)
    codec = re.search(r'Video: ([^ ,]+)', stream)
    if not duration or not dimensions:
        raise RuntimeError('无法读取时长或分辨率')
    h, m, s = duration.groups()
    return {'duration_seconds': int(h) * 3600 + int(m) * 60 + float(s),
            'width': int(dimensions[1]), 'height': int(dimensions[2]),
            'fps': float(fps[1]) if fps else None, 'codec': codec[1] if codec else None,
            'size_bytes': file.stat().st_size}

def main():
    PREVIEW.mkdir(parents=True, exist_ok=True)
    (DATA / '个人录像').mkdir(exist_ok=True)
    db = sqlite3.connect(DATA / '研究数据.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS videos (id TEXT PRIMARY KEY, name TEXT, kind TEXT, path TEXT)')
    items = []
    for index, file in enumerate(sorted((PROJECT / '单打视频').glob('*.mp4')), 1):
        info = metadata(file)
        video_id = uuid.uuid5(uuid.NAMESPACE_URL, str(file.resolve())).hex
        destination = DATA / '个人录像' / file.name
        if not destination.exists():
            shutil.copy2(file, destination)
        db.execute('INSERT OR IGNORE INTO videos VALUES (?,?,?,?)',
                   (video_id, file.name, 'personal', str(destination.relative_to(DATA))))
        db.commit()
        frames = []
        for slot, fraction in enumerate([0.08, 0.4, 0.75]):
            time = round(info['duration_seconds'] * fraction, 2)
            image_path = PREVIEW / f'{index:02d}_{slot + 1}.jpg'
            result = subprocess.run([FFMPEG, '-hide_banner', '-loglevel', 'error', '-y', '-ss', str(time),
                                     '-i', str(file), '-frames:v', '1', '-vf', 'scale=640:-2',
                                     str(image_path)], capture_output=True, text=True)
            if result.returncode or not image_path.exists():
                raise RuntimeError(f'{file.name} 预览提取失败: {result.stderr}')
            frames.append({'timestamp_seconds': time, 'path': str(image_path.relative_to(PROJECT))})
        items.append({'number': index, 'video_id': video_id, 'name': file.name,
                      'original_path': str(file.resolve()), **info, 'preview_frames': frames,
                      'target_player': 'unconfirmed', 'rally_boundaries': 'not_annotated'})
        print(index, file.name, info['duration_seconds'], f"{info['width']}x{info['height']}", info['fps'], flush=True)
    db.close()
    (DATA / '录像清单.json').write_text(json.dumps(items, ensure_ascii=False, indent=2))
    font = ImageFont.truetype('/System/Library/Fonts/Menlo.ttc', 18)
    for page, start in enumerate(range(0, len(items), 5), 1):
        selected = items[start:start + 5]
        canvas = Image.new('RGB', (1536, len(selected) * 330), '#eef2ee')
        draw = ImageDraw.Draw(canvas)
        for row, item in enumerate(selected):
            y = row * 330
            draw.text((14, y + 8), f"{item['number']:02d} | {item['name']} | {item['duration_seconds']:.2f}s | {item['width']}x{item['height']} | {item['fps']}fps", font=font, fill='#1b3c2c')
            for col, frame in enumerate(item['preview_frames']):
                image = Image.open(PROJECT / frame['path'])
                image.thumbnail((500, 260))
                x = col * 512 + (512 - image.width) // 2
                canvas.paste(image, (x, y + 38))
                draw.text((col * 512 + 15, y + 303), f"t = {frame['timestamp_seconds']:.2f}s", font=font, fill='#354b3e')
        canvas.save(PREVIEW / f'contact_sheet_{page}.jpg', quality=90)

if __name__ == '__main__':
    main()
