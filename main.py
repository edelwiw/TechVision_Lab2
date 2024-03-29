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

    # save image
    plt.savefig(f"results/{title}.png")



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


def distortion_correction(img, map_func):
    xi, yi = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

    # shift and normalize grid 
    xmid, xmid = img.shape[1] / 2.0, img.shape[0] / 2.0
    xi = xi - xmid
    yi = yi - xmid

    # convert to polar coordinates
    r, theta = cv2.cartToPolar(xi / xmid, yi / xmid)
    r = map_func(r)

    # convert back to cartesian coordinates
    u, v = cv2.polarToCart(r, theta)

    u, v = u * xmid + xmid, v * xmid + xmid

    # remap the image
    return cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)


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

# piecewise linear transformation
stretch = 4
piecewise_linear_matrix = np.float32([[stretch, 0, 0], [0, 1, 0]])

piecewise_linear_img = img.copy()
piecewise_linear_img[:, img.shape[1] // 2:] = apply_matrix_transformations(img[:, img.shape[1] // 2:], piecewise_linear_matrix)

show_images(img, piecewise_linear_img, "Piecewise Linear Transformation")

### non-linear transformations

# projective transformation
projective_matrix = np.float32([[1.1, 0.35, 0], [0.2, 1.1, 0], [0.00075, 0.0005, 1]])

projective_img = apply_perspective_transformations(img, projective_matrix)
show_images(img, projective_img, "Projective Transformation")

# polynomial transformation
polynomial_matrix = np.float32([[0, 0], [1, 0], [0, 1], [0.00001, 0], [0.002, 0], [0.001, 0]])

polynomial_img = np.zeros_like(img)
x, y = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))

# calculate the new coordinates
xnew = np.round(polynomial_matrix[0, 0] + polynomial_matrix[1, 0] * x + polynomial_matrix[2, 0] * y + polynomial_matrix[3, 0] * x**2 + polynomial_matrix[4, 0] * x * y + polynomial_matrix[5, 0] * y**2).astype(np.float32)
ynew = np.round(polynomial_matrix[0, 1] + polynomial_matrix[1, 1] * x + polynomial_matrix[2, 1] * y + polynomial_matrix[3, 1] * x**2 + polynomial_matrix[4, 1] * x * y + polynomial_matrix[5, 1] * y**2).astype(np.float32)

# calculate mask for valid coordinates
mask = np.logical_and(np.logical_and(xnew >= 0, xnew < img.shape[1]), np.logical_and(ynew >= 0, ynew < img.shape[0]))

# apply the transformation
if img.ndim == 2:
    polynomial_img[ynew[mask].astype(int), xnew[mask].astype(int)] = img[y[mask], x[mask]]
else:
    polynomial_img[ynew[mask].astype(int), xnew[mask].astype(int), :] = img[y[mask], x[mask], :]

show_images(img, polynomial_img, "Polynomial Transformation")

# sinusoidal transformation
u, v = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
v = v + 15 * np.sin(2 * np.pi * u / 90)

img_sinusoidal = cv2.remap(img, u.astype(np.float32), v.astype(np.float32), cv2.INTER_LINEAR)

show_images(img, img_sinusoidal, "Sinusoidal Transformation")

# distortion correction

# barrel distortion
img_distortion_corrected = distortion_correction(img, lambda r: r + 0.16 * r ** 3 + 0.1 * r ** 5)
show_images(img, img_distortion_corrected, "Distortion Correction (barrel)")

# pincushion distortion
img_distortion_corrected = distortion_correction(img, lambda r: r - 0.3 * r ** 2) # dont know if it is correct
show_images(img, img_distortion_corrected, "Distortion Correction (pincushion)")

### merging images
# open the images 
img_top = cv2.imread("zaebushek_top.png")
img_bttm = cv2.imread("zaebushek_bttm.png")
img_source = cv2.imread("zaebushek.png")

assert img_top is not None and img_bttm is not None and img_source is not None, "File could not be read"

# match template
template_size = 20

template = img_top[-template_size:, :, :]

res = cv2.matchTemplate(img_bttm, template, cv2.TM_CCOEFF)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

result_img = np.zeros((img_bttm.shape[0] + img_top.shape[0] - max_loc[1] - template_size, img_top.shape[1], img_top.shape[2]), dtype=np.uint8)
result_img[:img_top.shape[0], :, :] = img_top
result_img[img_top.shape[0]:, :, :] = img_bttm[max_loc[1] + template_size:, :, :]

show_images(img_source, result_img, "Merged Images")

# stitching images using opencv 
stitcher = cv2.Stitcher.create(cv2.Stitcher_SCANS)
status, result_img = stitcher.stitch((img_top, img_bttm))

if status == 0:
    show_images(img_source, result_img, "Stitched Images (using OpenCV)")
else:
    print("Images could not be stitched")


plt.show()