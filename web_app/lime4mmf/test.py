from PIL import Image
import numpy as np
img_try = Image.open("01245.png")
img_try = np.array(img_try, dtype=np.uint8)
print(img_try)

PIL_image = Image.fromarray(np.uint8(img_try)).convert('RGB')

PIL_image.save("xxx.png")