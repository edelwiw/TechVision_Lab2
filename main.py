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
    plt.text(10, 10, f'Resolution: {source.shape[1]}x{source.shape[0]}', color='white', backgroundcolor='black', fontsize=8, ha='left', va='top')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(img_trf_rgb)
    plt.title('Transformed Image')
    plt.text(10, 10, f'Resolution: {transformed.shape[1]}x{transformed.shape[0]}', color='white', backgroundcolor='black', fontsize=8, ha='left', va='top')
    plt.axis('off')

    # adjust space between images
    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    plt.subplots_adjust(left=0.05, right=0.95)



def apply_matrix_transformations(img, matrix, shape=(0, 0)):
    # apply the matrix to the image
    if shape == (0, 0):
        shape = (img.shape[1], img.shape[0])
    return cv2.warpAffine(img, matrix, shape)


def apply_perspective_transformations(img, matrix, shape=(0, 0)):
    # apply the matrix to the image
    if shape == (0, 0):
        shape = (img.shape[1], img.shape[0])
    return cv2.warpPerspective(img, matrix, shape)

# open the image
img = cv2.imread("test.jpeg")
assert img is not None, "File could not be read"

# move the image 
move_matrix = np.float32([[1, 0, 50], [0, 1, 100]])

moved_img = apply_matrix_transformations(img, move_matrix)
show_images(img, moved_img, "Move Image")

# mirror the image on the x-axis
mirror_matrix_x = np.float32([[1, 0, 0], [0, -1, img.shape[0] - 1]])

mirrored_img = apply_matrix_transformations(img, mirror_matrix_x)
show_images(img, mirrored_img, "Mirror Image (x-axis)")

# mirror the image on the y-axis
mirror_matrix_y = np.float32([[-1, 0, img.shape[1] - 1], [0, 1, 0]])

mirrored_img = apply_matrix_transformations(img, mirror_matrix_y)
show_images(img, mirrored_img, "Mirror Image (y-axis)")

# scale the image (zoom in)
zoom_matrix = np.float32([[1.5, 0, 0], [0, 1.5, 0]])

zoomed_img = apply_matrix_transformations(img, zoom_matrix, (int(img.shape[1] * 1.5), int(img.shape[0] * 1.5)))
show_images(img, zoomed_img, "Zoom Image")

# and using opencv function
zoomed_img_cv = cv2.resize(img, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
# None is the size of the output image (None ti use scaling factors), fx and fy are the scale factors
show_images(img, zoomed_img_cv, "Zoom Image (using OpenCV)")

# rotate the image
angle = np.deg2rad(10)
rotation_matrix = np.float32([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0]])

rotated_img = apply_matrix_transformations(img, rotation_matrix)
show_images(img, rotated_img, "Rotate Image")

# rotate about the center of the image
angle = np.deg2rad(10)
move_to_center_matrix = np.float32([[1, 0, -img.shape[1] / 2], [0, 1, -img.shape[0] / 2], [0, 0, 1]])
rotation_matrix = np.float32([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
move_back_matrix = np.float32([[1, 0, img.shape[1] / 2], [0, 1, img.shape[0] / 2], [0, 0, 1]])

transformation_matrix = move_back_matrix @ rotation_matrix @ move_to_center_matrix 
# remove the last row 
transformation_matrix = transformation_matrix[:2, :]

rotated_img = apply_matrix_transformations(img, transformation_matrix)
show_images(img, rotated_img, "Rotate Image (about the center)")

# and using opencv function
angle = 10
rotation_matrix_cv = cv2.getRotationMatrix2D((img.shape[1] / 2, img.shape[0] / 2), angle, 1)

rotated_img_cv = apply_matrix_transformations(img, rotation_matrix_cv)
show_images(img, rotated_img_cv, "Rotate Image (using OpenCV)")

# affine transformation
# define the points of the source image
src_points = np.float32([[0, 0], [img.shape[1] - 1, 0], [0, img.shape[0] - 1]])
# define the points of the transformed image
dst_points = np.float32([[50, 50], [img.shape[1] - 1, 0], [0, img.shape[0] - 1]])

affine_matrix = cv2.getAffineTransform(src_points, dst_points)

affine_img = apply_matrix_transformations(img, affine_matrix)
show_images(img, affine_img, "Affine Transformation")

# shearing transformation
shearing_matrix = np.float32([[1, 0, 0], [0.2, 1, 0]])

sheared_img = apply_matrix_transformations(img, shearing_matrix)
show_images(img, sheared_img, "Shearing Transformation")

# picewise linear transformation
stratch = 4
piecewise_linear_matrix = np.float32([[stratch, 0, 0], [0, 1, 0]])

piecewise_linear_img = img.copy()
piecewise_linear_img[:, img.shape[1] // 2:] = apply_matrix_transformations(img[:, img.shape[1] // 2:], piecewise_linear_matrix)

show_images(img, piecewise_linear_img, "Piecewise Linear Transformation")

### non-linear transformations

# projective transformation
projective_matrix = np.float32([[1.1, 0.35, 0], [0.2, 1.1, 0], [0.00075, 0.0005, 1]])

projective_img = apply_perspective_transformations(img, projective_matrix)
show_images(img, projective_img, "Projective Transformation")


plt.show()