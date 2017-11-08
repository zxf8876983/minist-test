from PIL import Image
import numpy as np

def MatrixToImage(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8).reshape(28,28))
    return new_im

data=np.loadtxt("x.txt")
new_im = MatrixToImage(data[2])
new_im.show()

