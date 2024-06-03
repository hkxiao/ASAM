import os
import shutil

for file in os.listdir("."):
    if '.png' in file:
        shutil.move(file, file.replace('_control.png','.png')) 