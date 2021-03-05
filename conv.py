import numpy as np
import cv2
import os


def convolve2D_1Channel(image, kernel, kernel_len, padding=0):
    # Gather Shapes of Kernel + Image + Padding
    xImgShape = image.shape[1]
    yImgShape = image.shape[0]

    # Shape of Output Convolution
    xOutput = xImgShape - kernel_len + 2 * padding + 1
    yOutput = yImgShape - kernel_len + 2 * padding + 1
    output = np.zeros((xOutput, yOutput))

    # Apply Equal Padding to All Sides
    if padding != 0:
        imagePadded = np.zeros((image.shape[0] + padding*2, image.shape[1] + padding*2))
        imagePadded[padding:-padding, padding:-padding] = image
    else:
        imagePadded = image

    # Iterate through image
    for y in range(yImgShape):
        # Exit Convolution
        if y > yImgShape - kernel_len:
            break
        for x in range(xImgShape):
            # Go to next row once kernel is out of bounds
            if x > xImgShape - kernel_len:
                break
            output[y, x] = (kernel * imagePadded[y: y + kernel_len, x: x + kernel_len]).sum()

    return output


def convolve2D(image, kernel, kernel_len, padding=0):
    if len(image.shape) == 2:  # grey image
        out_im = convolve2D_1Channel(image, kernel, kernel_len, padding)
    else:  # color image
        b = convolve2D_1Channel(image[:, :, 0], kernel, kernel_len, padding)
        g = convolve2D_1Channel(image[:, :, 1], kernel, kernel_len, padding)
        r = convolve2D_1Channel(image[:, :, 2], kernel, kernel_len, padding)
        out_im = np.stack([b, g, r], axis=2)

    # image normalization as (0, 255)
    min_el = np.min(out_im)
    max_el = np.max(out_im)
    out_im = (out_im - min_el) * 255 / max_el
    out_im = out_im.astype(np.uint8)
    return out_im


if __name__ == '__main__':
    # image path
    input_path = '/Users/puttatida/Desktop/Looggaew/Qmul/Computer Vision/Dataset/DatasetA'
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith('.jpg'):
                file_path = os.path.join(path, file)
                image = cv2.imread(file_path)

                # b) averaging kernel (3, 3)
                kernel = np.ones((3, 3), np.float32) / 9
                output = convolve2D(image, kernel, 3)
                out_path = 'output/' + file.replace('.jpg', '_b.jpg')
                cv2.imwrite(out_path, output)

                # c) kernel A
                kernelA = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], np.float32)
                output = convolve2D(image, kernelA, 3)
                out_path = 'output/' + file.replace('.jpg', '_c_kernelA.jpg')
                cv2.imwrite(out_path, output)

                # c) kernel B
                kernelB = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], np.float32)
                output = convolve2D(image, kernelB, 3)
                out_path = 'output/' + file.replace('.jpg', '_c_kernelB.jpg')
                cv2.imwrite(out_path, output)

                # d) (i)
                output = convolve2D(image, kernelA, 3)
                output = convolve2D(output, kernelA, 3)
                out_path = 'output/' + file.replace('.jpg', '_d_(i).jpg')
                cv2.imwrite(out_path, output)

                # d) (ii)
                output = convolve2D(image, kernelA, 3)
                output = convolve2D(output, kernelB, 3)
                out_path = 'output/' + file.replace('.jpg', '_d_(ii).jpg')
                cv2.imwrite(out_path, output)

                # d) (iii)
                output = convolve2D(image, kernelB, 3)
                output = convolve2D(output, kernelA, 3)
                out_path = 'output/' + file.replace('.jpg', '_d_(iii).jpg')
                cv2.imwrite(out_path, output)
