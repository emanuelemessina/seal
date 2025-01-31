import os
from PIL import Image

images_path = 'images'

for filename in os.listdir(images_path):
    if filename.endswith('.png'):
        file_path = os.path.join(images_path, filename)
        try:
            with Image.open(file_path) as img:
                img.verify()  # Verify that it is, in fact, an image
            print(f"{filename} is valid.")
        except (IOError, SyntaxError) as e:
            print(f"{filename} is corrupted: {e}")
            quit()

print("All images are valid")
