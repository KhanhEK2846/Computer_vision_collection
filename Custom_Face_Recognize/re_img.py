import os,uuid


folder = "images"

for count, filename in enumerate(os.listdir(folder)):
    dst = f'{str(uuid.uuid1())}.jpg'
    src =f'{folder}/{filename}'
    dst =f'{folder}/{dst}'

    os.rename(src, dst)
