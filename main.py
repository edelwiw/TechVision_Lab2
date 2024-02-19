import cv2 as cv2
import matplotlib.pyplot as plt
import numpy as np

# show the source image and transformed image
def show_images(source, transformed, title="Image"):
    img_src_rgb = cv2.cvtColor(source, cv2.COLOR_BGR2RGB) # normal colors to display
    img_trf_rgb = cv2.cvtColor(transformed, cv2.COLOR_BGR2RGB) 

    plt.figure(figsize=(13, 5))
    plt.suptitle(title, fontsize=16)

    plt.subplot(1, 2, 1)
    plt.imshow(img_src_rgb)
    plt.title('Source Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_trf_rgb)
    plt.title('Transformed Image')
    plt.axis('off')

    # adjust space between images
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.subplots_adjust(left=0.05, right=0.95)



def apply_matrix_transformations(img, matrix):
    # apply the matrix to the image
    return cv2.warpAffine(img, matrix, (img.shape[1], img.shape[0]))

# open the image
img = cv2.imread("test.jpeg")
assert img is not None, "File could not be read"

# move the image 
move_matrix = np.float32([[1, 0, 50], [0, 1, 100]])

moved_img = apply_matrix_transformations(img, move_matrix)
show_images(img, moved_img, "Move Image")

plt.show()