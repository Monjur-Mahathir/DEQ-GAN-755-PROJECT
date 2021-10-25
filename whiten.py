from PIL import Image, UnidentifiedImageError
import os
folder = '../sprites/pokemon'

files = os.listdir(folder)
for f in files:
    try:
        png = Image.open(os.path.join(folder, f))
        png.load() # required for png.split()
        png=png.convert('RGBA')

        background = Image.new("RGB", png.size, (255, 255, 255))
        background.paste(png, mask=png.split()[3]) # 3 is the alpha channel

        background.save(os.path.join('../sprites/pokemon-white/pokemon', 
        f'{f.split(".")[0]}.jpg'),
        'JPEG', quality=80)
    except UnidentifiedImageError:
        print(f)