# This is a first exercise for Computer Vision program
from tkinter import *
from tkinter import filedialog
from matplotlib import pyplot as plt
from PIL import Image, ImageStat, ImageChops
from math import sqrt, pow, log
from functools import reduce
from skimage import metrics
import imutils
import cv2 as cv
import numpy as np
import random

# global variables
filter_image = None
segmentation_image = None
canny_image = None
morphological_image = None
characteristics_image = None
show_images = True
low_threshold = 255 / 3
high_threshold = 255

# edge detection kernels
# Sobel kernels
sobel_horizontal_kernel = np.array([[-1, 0, 1],
                                    [-2, 0, 2],
                                    [-1, 0, 1]
                                    ])
sobel_vertical_kernel = np.array([[1, 2, 1],
                                  [0, 0, 0],
                                  [-1, -2, -1]
                                  ])
# Prewitt kernels
prewitt_horizontal_kernel = np.array([[-1, 0, 1],
                                      [-1, 0, 1],
                                      [-1, 0, 1]
                                      ])
prewitt_vertical_kernel = np.array([[-1, -1, -1],
                                    [0, 0, 0],
                                    [1, 1, 1]
                                    ])
# Morphological kernels
empty_square = np.array([[1, 1, 1],
                         [1, 0, 1],
                         [1, 1, 1]
                         ])
full_square_5 = np.array([[1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1],
                          [1, 1, 1, 1, 1]
                          ])
plus_sign = np.array([[0, 1, 0],
                      [1, 1, 1],
                      [0, 1, 0]
                      ])
number_of_repetitions = 2


# key down functions
def click_load_image():
    path = filedialog.askopenfilename()
    img = cv.imread(path, 1)
    if img is None:
        print("Image was not selected")
    else:
        cv.imshow("Image preview", img)
        cv.waitKey(0)
        cv.destroyAllWindows()


# Didn't define the buttons and their functionalities, instead play and pause
# commence with a press of 'P'
def click_load_video():
    # x, y = 0, 0
    path = filedialog.askopenfilename()
    video = cv.VideoCapture(path)
    # play = True
    if video is None:
        print("Video was not selected")
    else:
        while video.isOpened():
            ret, frame = video.read()
            # cv.setMouseCallback("Video", click_image, param=[x, y])
            # if x >= 270 & x <= 310 & y >= 300 & y <= 340:
            #     play = False
            # if x >= 340 & x <= 380 & y >= 300 & y <= 340:
            #     play = True
            # back button
            cv.rectangle(frame, (20, 160), (60, 200), (211, 211, 211), -1)

            # pause button
            cv.rectangle(frame, (270, 300), (310, 340), (211, 211, 211), -1)
            cv.line(frame, (285, 310), (285, 330), (0, 0, 0), 3)
            cv.line(frame, (295, 310), (295, 330), (0, 0, 0), 3)

            # play button
            cv.rectangle(frame, (340, 300), (380, 340), (211, 211, 211), -1)
            cv.line(frame, (352, 310), (352, 330), (0, 0, 0), 3)
            cv.line(frame, (352, 310), (370, 320), (0, 0, 0), 3)
            cv.line(frame, (352, 330), (370, 320), (0, 0, 0), 3)
            # forward button
            cv.rectangle(frame, (580, 160), (620, 200), (211, 211, 211), -1)
            cv.imshow('Video', frame)
            key = cv.waitKey(25)
            # Close the video
            if key == ord('q'):
                break
            # Pause/Play video
            if key == ord('p'):
                cv.waitKey(0)
        video.release()
        cv.destroyAllWindows()


# Didn't define the buttons and their functionalities, instead saving starts and stops
# with a press of 'S'
def click_load_camera():
    camera = cv.VideoCapture(0, cv.CAP_DSHOW)
    width = int(camera.get(cv.CAP_PROP_FRAME_WIDTH) + 0.5)
    height = int(camera.get(cv.CAP_PROP_FRAME_HEIGHT) + 0.5)
    size = (width, height)
    fourcc = cv.VideoWriter_fourcc(*'XVID')
    saving = False
    out = cv.VideoWriter('camera_output.avi', fourcc, 20.0, size)
    if camera is None:
        print("Camera was not opened")
    if not camera.isOpened():
        print("Camera was not able to open")
    while camera.isOpened():
        ret, frame = camera.read()
        cv.imshow("Camera feed", frame)
        key = cv.waitKey(1)
        # Close the camera feed
        if key == ord('q'):
            break
        # Start/Stop saving
        if key == ord('s'):
            saving = not saving
            print("Saving" if saving else "Stopping saving")
        if saving:
            out.write(frame)
    camera.release()
    cv.destroyAllWindows()


# Checking if the image is indeed grayscale
def is_greyscale(img_path):
    image = Image.open(img_path).convert("RGB")
    stat = ImageStat.Stat(image)
    if sum(stat.sum) / 3 == stat.sum[0]:
        return True
    else:
        return False


# Opening images in grayscale mode
def open_image_in_grayscale():
    global filter_image
    cv.destroyAllWindows()
    camera = cv.VideoCapture(1, cv.CAP_DSHOW)
    ret, filter_image = camera.read()
    camera.release()
    filter_image = cv.cvtColor(filter_image, cv.COLOR_BGR2GRAY)
    cv.imshow("Original picture", filter_image)
    open_edge_detection_filters()


# Edge detection algorithms for sobel and prewitt
def apply_edge_detection(image, horizontal_kernel, vertical_kernel):
    image_pixels = np.zeros(image.shape, np.uint8)
    image_edge_direction = np.zeros(image.shape, np.float32)

    horizontal_filtering = np.zeros(image.shape, np.uint8)
    vertical_filtering = np.zeros(image.shape, np.uint8)

    height, width = image.shape

    for x in range(1, height - 1):
        for y in range(1, width - 1):
            horizontal_filtering[x][y] = np.absolute((
                    horizontal_kernel[0][0] * image[x - 1][y - 1] +
                    horizontal_kernel[1][0] * image[x][y - 1] +
                    horizontal_kernel[2][0] * image[x + 1][y - 1] +
                    horizontal_kernel[0][2] * image[x - 1][y + 1] +
                    horizontal_kernel[1][2] * image[x][y + 1] +
                    horizontal_kernel[2][2] * image[x + 1][y + 1]
            ))
            vertical_filtering[x][y] = np.absolute((
                    vertical_kernel[0][0] * image[x - 1][y - 1] +
                    vertical_kernel[0][1] * image[x - 1][y] +
                    vertical_kernel[0][2] * image[x - 1][y + 1] +
                    vertical_kernel[2][0] * image[x + 1][y - 1] +
                    vertical_kernel[2][1] * image[x + 1][y] +
                    vertical_kernel[2][2] * image[x + 1][y + 1]
            ))

            image_pixels[x][y] = sqrt(pow(vertical_filtering[x][y], 2) + pow(horizontal_filtering[x][y], 2))
            if horizontal_filtering[x][y] == 0:
                image_edge_direction[x][y] = 0
            else:
                image_edge_direction[x][y] = np.arctan(vertical_filtering[x][y] / horizontal_filtering[x][y])

    # Display the images with the filters
    if show_images:
        cv.imshow("Horizontal filtering", horizontal_filtering)
        cv.imshow("Vertical filtering", vertical_filtering)
        cv.imshow("Filtered image", image_pixels)
    return image_pixels, image_edge_direction


# Edge detection algorithms
def open_edge_detection_filters():
    filters = Tk()
    filters.title("Filtering")
    filters.configure(background="black", padx=20, pady=20)
    # Buttons for edge detection
    Label(filters, text="Filters", bg="black", fg="white").grid(row=0, column=0)
    Button(filters, text="Sobel", width=30, command=sobel_filter) \
        .grid(row=1, column=0, sticky=N)
    Label(window, bg="black") \
        .grid(row=2, column=0)
    Button(filters, text="Prewitt", width=30, command=prewitt_filter) \
        .grid(row=3, column=0, sticky=N)
    Label(window, bg="black") \
        .grid(row=4, column=0)
    Button(filters, text="Canny", width=30, command=canny) \
        .grid(row=5, column=0, sticky=N)
    filters.mainloop()


# Sobel algorithm for edge detection
def sobel_filter():
    global filter_image, sobel_horizontal_kernel, sobel_vertical_kernel
    apply_edge_detection(filter_image, sobel_horizontal_kernel, sobel_vertical_kernel)


# Prewitt algorithm for edge detection
def prewitt_filter():
    global filter_image, prewitt_horizontal_kernel, prewitt_vertical_kernel
    apply_edge_detection(filter_image, prewitt_horizontal_kernel, prewitt_vertical_kernel)


# Canny algorithm for edge detection
def canny():
    global filter_image, sobel_horizontal_kernel, sobel_vertical_kernel
    reduced_image = noise_reduction(filter_image)
    sobel_power, sobel_direction = apply_edge_detection(reduced_image, sobel_horizontal_kernel, sobel_vertical_kernel)
    shrunk_edges = edge_shrinking(sobel_power, sobel_direction)
    modified_edges = modify_weak_edges_to_strong(shrunk_edges)


# Noise reduction algorithm
def noise_reduction(image):
    noise_reduction_kernel = np.ones((7, 7), np.float32) / 49
    image_pixels = np.zeros(image.shape, np.uint8)

    for x in range(3, image.shape[0] - 3):
        for y in range(3, image.shape[1] - 3):
            sum_kernels = 0
            for i in range(0, 5):
                for j in range(0, 5):
                    offset_i = i - 3
                    offset_j = j - 3
                    sum_kernels = sum_kernels + noise_reduction_kernel[i][j] * image[x - offset_i][y - offset_j]
            image_pixels[x][y] = sum_kernels
    if show_images:
        cv.imshow('Noise reduction', image_pixels)

    return image_pixels


# Edge shrinking and checking the correct directions
def edge_shrinking(sobel_power, sobel_direction):
    shrunk_edges = sobel_power

    for i in range(1, sobel_power.shape[0] - 1):
        for j in range(1, sobel_power.shape[1] - 1):
            if 0 <= sobel_direction[i, j] < 22.5 or 157.5 <= sobel_direction[i, j] <= 180:
                if sobel_power[i, j] <= sobel_power[i, j - 1] or sobel_power[i, j] <= sobel_power[i, j + 1]:
                    shrunk_edges[i, j] = 0
            if 22.5 <= sobel_direction[i, j] < 67.5:
                if sobel_power[i, j] <= sobel_power[i - 1, j + 1] or sobel_power[i, j] <= sobel_power[i + 1, j - 1]:
                    shrunk_edges[i, j] = 0
            if 67.5 <= sobel_direction[i, j] < 112.5:
                if sobel_power[i, j] <= sobel_power[i - 1, j] or sobel_power[i, j] <= sobel_power[i + 1, j]:
                    shrunk_edges[i, j] = 0
            if 112.5 <= sobel_direction[i, j] < 157.5:
                if sobel_power[i, j] <= sobel_power[i - 1, j - 1] or sobel_power[i, j] <= sobel_power[i + 1, j + 1]:
                    shrunk_edges[i, j] = 0
    if show_images:
        cv.imshow('Shrunk edges', shrunk_edges)
    return shrunk_edges


# Change the weak edges to strong ones dependent on thresholds
def modify_weak_edges_to_strong(shrunk_edges):
    global low_threshold, high_threshold
    for i in range(1, shrunk_edges.shape[0] - 1):
        for j in range(1, shrunk_edges.shape[1] - 1):
            if shrunk_edges[i, j] > high_threshold:
                shrunk_edges[i, j] = 255
            elif high_threshold > shrunk_edges[i, j] > low_threshold:
                shrunk_edges[i, j] = 50
    if show_images:
        cv.imshow('Modified edges', shrunk_edges)
    return shrunk_edges


# Graphical interface for image segmentations
def segmentation_graphical_interface():
    global segmentation_image
    # Open image in grayscale
    path = filedialog.askopenfilename()
    segmentation_image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    cv.imshow("Grayscale image", segmentation_image)
    segmentation_gui = Tk()
    segmentation_gui.title("Segmentation")
    segmentation_gui.configure(background="black", padx=20, pady=20)
    # Buttons for edge detection
    Label(segmentation_gui, text="Image segmentation", bg="black", fg="white") \
        .grid(row=0, column=0, columnspan=3, pady=(0, 20))
    Button(segmentation_gui, text="Otsu segmentation", width=20, command=otsu_segmentation) \
        .grid(row=1, column=0, sticky=N)
    Button(segmentation_gui, text="Kapur segmentation", width=20, command=kapur_segmentation) \
        .grid(row=1, column=2, sticky=N, padx=(20, 0))
    segmentation_gui.mainloop()


# Calculate the histogram
def make_histogram(image):
    image_histogram_array = [0] * 256

    for row in image:
        for pixel in row:
            image_histogram_array[pixel] = image_histogram_array[pixel] + 1

    return image_histogram_array


# Calculate the normalized histogram
def make_normalized_histogram(histogram, pixel_count):
    normalized_histogram_array = [0] * 256
    counter = 0

    for value in histogram:
        normalized_histogram_array[counter] = (1 / pixel_count) * value
        counter = counter + 1

    return normalized_histogram_array


# Calculate the array mean
def calculate_array_mean(array):
    array_sum = 0

    for count in range(0, len(array)):
        array_sum += array[count]

    return array_sum / len(array)


# Calculate the variance in array
def calculate_variance(array, pixel_count):
    histogram_sum = 0
    mean = calculate_array_mean(array)

    for count in range(0, len(array)):
        histogram_sum += pow(array[count] - mean, 2)

    return histogram_sum / pixel_count


# Calculate the sum of the threshold
def calculate_sum_threshold(histogram, threshold):
    threshold_sum = 0

    for count in range(0, threshold):
        threshold_sum += histogram[count]

    return threshold_sum


# Calculate variances in thresholds
def calculate_variances_thresholds(histogram, threshold, pixel_count):
    left_threshold_list = list()
    for count in range(0, threshold):
        left_threshold_list.append(histogram[count])

    right_threshold_list = list()
    for count in range(threshold, len(histogram)):
        right_threshold_list.append(histogram[count])

    left_percentage = sum(left_threshold_list) / pixel_count
    # this comment is necessary to suppress an unnecessary PyCharm warning
    # noinspection PyTypeChecker
    left_variance = calculate_variance(left_threshold_list, pixel_count)

    right_percentage = sum(right_threshold_list) / pixel_count
    # this comment is necessary to suppress an unnecessary PyCharm warning
    # noinspection PyTypeChecker
    right_variance = calculate_variance(right_threshold_list, pixel_count)

    return left_percentage * left_variance + right_percentage * right_variance


# Histogram background calculation based on threshold
def histogram_background(histogram, threshold):
    background_sum = 0
    threshold_sum = calculate_sum_threshold(histogram, threshold)

    for count in range(0, threshold):
        if 0 < threshold_sum < 1 and (histogram[count] / (1 - threshold_sum)) != 0:
            background_sum += (histogram[count] / threshold_sum) * log(histogram[count] / threshold_sum)

    return background_sum * (-1)


# Histogram foreground calculation based on threshold
def histogram_foreground(histogram, threshold):
    foreground_sum = 0
    threshold_sum = calculate_sum_threshold(histogram, threshold)

    for count in range(threshold, len(histogram)):
        if 0 < threshold_sum < 1 and (histogram[count] / (1 - threshold_sum)) != 0:
            foreground_sum += (histogram[count] / (1 - threshold_sum)) * log(histogram[count] / (1 - threshold_sum))

    return foreground_sum * (-1)


# Different thresholds for segmentations
def different_thresholds_segmentation_otsu(normalized_histogram, pixel_count, low, high):
    variance_list = list()
    for count in range(low, high):
        variance_list.append(calculate_variances_thresholds(normalized_histogram, count, pixel_count))
    minimal_variance = min(variance_list)
    print("Minimal variance: ", minimal_variance)

    optimal_threshold = 0
    for count in range(0, len(variance_list)):
        if minimal_variance == variance_list[count]:
            optimal_threshold = count
    print("Optimal threshold: ", optimal_threshold)

    return optimal_threshold


# Different thresholds for segmentation
def different_threshold_segmentation_kapur(normalized_histogram, low, high):
    threshold_list = list()
    for count in range(low, high):
        threshold_list.append((histogram_background(normalized_histogram, count))
                              + (histogram_foreground(normalized_histogram, count)))
    maximum_threshold = max(threshold_list)
    print("Maximum variance: ", maximum_threshold)

    optimal_threshold = 0
    for count in range(0, len(threshold_list)):
        if maximum_threshold == threshold_list[count]:
            optimal_threshold = count
    print("Optimal threshold: ", optimal_threshold)
    return optimal_threshold


# Modify image for segmentation
def modify_image_segmentation(name, optimal_threshold):
    binary_image = np.zeros(segmentation_image.shape, np.uint8)

    x = 0
    for row in segmentation_image:
        y = 0
        for pixel in row:
            if pixel >= optimal_threshold:
                binary_image[x][y] = 255
            y += 1
        x += 1
    cv.imshow(name + str(optimal_threshold), binary_image)
    return binary_image


# Otsu segmentation function
def otsu_segmentation():
    global segmentation_image
    pixel_count = segmentation_image.shape[0] * segmentation_image.shape[1]

    histogram = make_histogram(segmentation_image)
    normalized_histogram = make_normalized_histogram(histogram, pixel_count)

    optimal_threshold_all = different_thresholds_segmentation_otsu(normalized_histogram, pixel_count, 1, 256)
    optimal_threshold_small = different_thresholds_segmentation_otsu(normalized_histogram, pixel_count, 10, 100)
    optimal_threshold_middle = different_thresholds_segmentation_otsu(normalized_histogram, pixel_count, 50, 200)

    modify_image_segmentation("Otsu segmentation", optimal_threshold_all)
    modify_image_segmentation("Otsu segmentation", optimal_threshold_middle)
    modify_image_segmentation("Otsu segmentation", optimal_threshold_small)


# Kapur segmentation function
def kapur_segmentation():
    global segmentation_image
    pixel_count = segmentation_image.shape[0] * segmentation_image.shape[1]

    histogram = make_histogram(segmentation_image)
    normalized_histogram = make_normalized_histogram(histogram, pixel_count)

    optimal_threshold_all = different_threshold_segmentation_kapur(normalized_histogram, 1, 256)
    optimal_threshold_small = different_threshold_segmentation_kapur(normalized_histogram, 10, 100)
    optimal_threshold_middle = different_threshold_segmentation_kapur(normalized_histogram, 50, 200)

    modify_image_segmentation("Kapur segmentation", optimal_threshold_all)
    modify_image_segmentation("Kapur segmentation", optimal_threshold_middle)
    modify_image_segmentation("Kapur segmentation", optimal_threshold_small)


# Morphological operators
def morphological_gui():
    global morphological_image
    morphological = Tk()
    morphological.title("Morphological operators")
    morphological.configure(background="black", padx=20, pady=20)

    path = filedialog.askopenfilename()
    morphological_image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    cv.imshow("Grayscale morphological image", morphological_image)

    Label(morphological, text="Morphological GUI", bg="black", fg="white") \
        .grid(row=0, column=0, pady=(0, 10))
    Button(morphological, text="Dilation", width=30, command=dilation_calls) \
        .grid(row=1, column=0, pady=(0, 10))
    Button(morphological, text="Erosion", width=30, command=erosion_calls) \
        .grid(row=2, column=0, pady=(0, 10))
    Button(morphological, text="Closing", width=30, command=closing_calls) \
        .grid(row=3, column=0, pady=(0, 10))
    Button(morphological, text="Opening", width=30, command=opening_calls) \
        .grid(row=4, column=0, pady=(0, 10))
    morphological.mainloop()


# Dilation calls for different kernels
def dilation_calls():
    global morphological_image, number_of_repetitions
    # Copying images for dilation
    dilate_zero = morphological_image.copy()
    dilate_one = morphological_image.copy()
    dilate_two = morphological_image.copy()
    for repetition in range(0, number_of_repetitions):
        # Empty square kernel
        dilate_zero = dilation(0, dilate_zero.copy())
        cv.imshow("Dilate 0, rep " + str(repetition), dilate_zero)
        # Full square 5x5 kernel
        dilate_one = dilation(1, dilate_one.copy())
        cv.imshow("Dilate 1, rep " + str(repetition), dilate_one)
        # Plus sign kernel
        dilate_two = dilation(2, dilate_two.copy())
        cv.imshow("Dilate 2, rep " + str(repetition), dilate_two)


# Erosion calls for different kernels
def erosion_calls():
    global morphological_image, number_of_repetitions
    # Copying images for erosion
    erode_zero = morphological_image.copy()
    erode_one = morphological_image.copy()
    erode_two = morphological_image.copy()
    for repetition in range(0, number_of_repetitions):
        # Empty square kernel
        erode_zero = erosion(0, erode_zero.copy())
        cv.imshow("Erode 0, rep " + str(repetition), erode_zero)
        # Full square 5x5 kernel
        erode_one = erosion(1, erode_one.copy())
        cv.imshow("Erode 1, rep " + str(repetition), erode_one)
        # Plus sign kernel
        erode_two = erosion(2, erode_two.copy())
        cv.imshow("Erode 2, rep " + str(repetition), erode_two)


# Closing calls for different kernels
def closing_calls():
    global morphological_image, number_of_repetitions
    # Copying images for closing
    closing_zero = morphological_image.copy()
    closing_one = morphological_image.copy()
    closing_two = morphological_image.copy()
    for repetition in range(0, number_of_repetitions):
        # Empty square kernel
        closing_zero = closing(0, closing_zero.copy())
        cv.imshow("Close 0, rep " + str(repetition), closing_zero)
        # Full square 5x5 kernel
        closing_one = closing(1, closing_one.copy())
        cv.imshow("Close 1, rep " + str(repetition), closing_one)
        # Plus sign kernel
        closing_two = closing(2, closing_two.copy())
        cv.imshow("Close 2, rep " + str(repetition), closing_two)


# Closing calls for different kernels
def opening_calls():
    global morphological_image, number_of_repetitions
    # Copying images for opening
    opening_zero = morphological_image.copy()
    opening_one = morphological_image.copy()
    opening_two = morphological_image.copy()
    for repetition in range(0, number_of_repetitions):
        # Empty square kernel
        # opening_zero = opening(0, opening_zero.copy())
        cv.imshow("Open 0, rep " + str(repetition), opening_zero)
        # Full square 5x5 kernel
        opening_one = opening(1, opening_one.copy())
        cv.imshow("Open 1, rep " + str(repetition), opening_one)
        # Plus sign kernel
        # opening_two = opening(2, opening_two.copy())
        cv.imshow("Open 2, rep " + str(repetition), opening_two)


# Morphological operator for dilation
def dilation(input_kernel, input_image):
    global full_square_5, plus_sign, empty_square
    output_image = input_image.copy()
    kernel = None
    offset = 0
    if input_kernel == 0:
        kernel = empty_square
        offset = 1
    elif input_kernel == 1:
        kernel = full_square_5
        offset = 2
    elif input_kernel == 2:
        kernel = plus_sign
        offset = 1

    if kernel is not None:
        for i in range(offset, input_image.shape[0] - offset):
            for j in range(offset, input_image.shape[1] - offset):
                max_pixel = -5
                for offset_i in range(-offset, offset + 1):
                    for offset_j in range(-offset, offset + 1):
                        if kernel[offset_i + offset, offset_j + offset] == 1 and \
                                input_image[i + offset_i, j + offset_j] > max_pixel:
                            max_pixel = input_image[i + offset_i, j + offset_j]
                if max_pixel != -5:
                    output_image[i, j] = max_pixel
                else:
                    output_image[i, j] = input_image[i, j]
        return output_image

    else:
        print("Kernel is not correct")


# Morphological operator for erosion
def erosion(input_kernel, input_image):
    global full_square_5, plus_sign, empty_square
    output_image = input_image.copy()
    kernel = None
    offset = 0
    if input_kernel == 0:
        kernel = empty_square
        offset = 1
    elif input_kernel == 1:
        kernel = full_square_5
        offset = 2
    elif input_kernel == 2:
        kernel = plus_sign
        offset = 1

    if kernel is not None:
        for i in range(offset, input_image.shape[0] - offset):
            for j in range(offset, input_image.shape[1] - offset):
                min_pixel = 270
                for offset_i in range(-offset, offset + 1):
                    for offset_j in range(-offset, offset + 1):
                        if kernel[offset_i + offset, offset_j + offset] == 1 and \
                                input_image[i + offset_i, j + offset_j] < min_pixel:
                            min_pixel = input_image[i + offset_i, j + offset_j]
                if min_pixel != 270:
                    output_image[i, j] = min_pixel
                else:
                    output_image[i, j] = input_image[i, j]
        return output_image
    else:
        print("Kernel is not correct")


# Morphological operator for closing
def closing(input_kernel, input_image):
    dilated_image = dilation(input_kernel, input_image.copy())
    return erosion(input_kernel, dilated_image.copy())


# Morphological operator for opening
def opening(input_kernel, input_image):
    eroded_image = erosion(input_kernel, input_image.copy())
    return dilation(input_kernel, eroded_image.copy())


# Temporary
def motion_track():
    motion_tracking(100, 250, 50)


# Find the moving object based on movement
def detect_moving_object(image):
    white_image = np.empty(image.shape, np.uint8)
    white_image.fill(255)
    gray_original = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    gray_white = cv.cvtColor(white_image, cv.COLOR_BGR2GRAY)

    difference = cv.subtract(gray_white, gray_original)
    ret, mask = cv.threshold(difference, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU)
    # difference[mask != 255] = [0, 0, 255]
    cv.imshow("Bounding boxes", difference)


# Create particles
def create_particle_array(image, particle_array, particle_count):
    height, width = image.shape[:2]
    for i in range(0, particle_count):
        x, y = random.randint(0, width - 1), random.randint(0, height - 1)
        particle_array[i] = (x, y)


# Weight particles
def weight_particle_array(particle_weight_array, particle_count):
    for i in range(0, len(particle_weight_array)):
        particle_weight_array[i] = 1 / particle_count


# Draw the particles on the frame
def draw_particles(image, particle_array, particle_weight_array, threshold):
    for i in range(0, len(particle_array)):
        color = (0, 0, 255)
        if particle_weight_array[i] > threshold:
            color = (0, 255, 0)
        cv.circle(image, particle_array[i], 3, color, 2)


# Grade particles based on the image
def grade_particles(image, particle_array, particle_weight_array):
    for i in range(0, len(particle_array)):
        particle_weight_array[i] = image[particle_array[i][1]][particle_array[i][0]]


# Back projection for the histogram of region of interest on the image
def draw_back_projection(image, region_of_interest):
    roi_hsv = cv.cvtColor(region_of_interest, cv.COLOR_BGR2HSV)
    image_hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    ch = (0, 0)
    hist_size = max(256, 2)
    roi_hue = np.empty(roi_hsv.shape, roi_hsv.dtype)
    image_hue = np.empty(image_hsv.shape, image_hsv.dtype)
    cv.mixChannels([roi_hsv], [roi_hue], ch)
    cv.mixChannels([image_hsv], [image_hue], ch)
    hist = cv.calcHist([roi_hue], [0], None, [hist_size], [0, 180], accumulate=False)
    cv.normalize(hist, hist, alpha=0, beta=255, norm_type=cv.NORM_MINMAX)
    back_projection = cv.calcBackProject([image_hue], [0], hist, [0, 180], scale=1)
    cv.imshow("Back projection", back_projection)
    return back_projection


# Calculate the center of gravity for the good particles
def calculate_center_of_gravity(particle_array, particle_weight_array, threshold):
    combined_x = 0
    combined_y = 0
    weight_sum = 0
    for i in range(0, len(particle_array)):
        if particle_weight_array[i] > threshold:
            combined_x = combined_x + particle_array[i][0] * particle_weight_array[i]
            combined_y = combined_y + particle_array[i][1] * particle_weight_array[i]
            weight_sum = weight_sum + particle_weight_array[i]
    if weight_sum == 0:
        weight_sum = 1
    return int(combined_x / weight_sum), int(combined_y / weight_sum)


# Move particles based on motion model
def move_particles(image, particle_array, motion_model):
    height, width = image.shape[:2]
    for i in range(0, len(particle_array)):
        move_x = particle_array[i][0] + random.randint(-motion_model, motion_model)
        move_y = particle_array[i][1] + random.randint(-motion_model, motion_model)
        if 0 < move_x < width and 0 < move_y < height:
            particle_array[i] = (move_x, move_y)


# Move bad particles close to the center of gravity
def move_particles_to_proximity_center_of_gravity(image, center_of_gravity, particle_array, particle_weight_array, threshold, offset):
    height, width = image.shape[:2]
    for i in range(0, len(particle_array)):
        if particle_weight_array[i] <= threshold:
            x = center_of_gravity[0] + random.randint(-offset, offset)
            y = center_of_gravity[1] + random.randint(-offset, offset)
            if x < 0:
                x = 0
            elif x >= width:
                x = width - 1
            if y < 0:
                y = 0
            elif y >= height:
                y = height - 1
            particle_array[i] = (x, y)
            particle_weight_array[i] = image[particle_array[i][1]][particle_array[i][0]]


# Move bad particles towards the center of gravity (My idea)
def move_particles_towards_center_of_gravity(image, center_of_gravity, particle_array, particle_weight_array, threshold, offset):
    height, width = image.shape[:2]
    good_particles_count = 0
    for i in range(0, len(particle_weight_array)):
        if particle_weight_array[i] > threshold:
            good_particles_count = good_particles_count + 1
            break
    if good_particles_count == 0:
        return
    for i in range(0, len(particle_array)):
        if particle_weight_array[i] <= threshold:
            center_of_gravity_x = center_of_gravity[0]
            center_of_gravity_y = center_of_gravity[1]
            x = particle_array[i][0]
            y = particle_array[i][1]
            if x <= center_of_gravity_x:
                if x + offset < center_of_gravity_x < width:
                    x = x + offset
                else:
                    x = center_of_gravity_x
            else:
                if x - offset > center_of_gravity_x > 0:
                    x = x - offset
                else:
                    x = center_of_gravity_x
            if y <= center_of_gravity_y:
                if y + offset < center_of_gravity_y < height:
                    y = y + offset
                else:
                    y = center_of_gravity_y
            else:
                if y - offset > center_of_gravity_y > 0:
                    y = y - offset
                else:
                    y = center_of_gravity_y
            particle_array[i] = (x, y)
            particle_weight_array[i] = image[particle_array[i][1]][particle_array[i][0]]


# Motion tracking function with particle filter
def motion_tracking(particle_grade, particle_count, motion_model):
    # Recognize the moving object
    path = filedialog.askopenfilename()
    video = cv.VideoCapture(path)
    if video is None:
        print("Video was not selected")
    else:
        ret, first_frame = video.read()
        roi = first_frame[185:330, 150:300].copy()
        color = ('b', 'g', 'r')
        cv.imshow("ROI", roi)
        for i, col in enumerate(color):
            hist = cv.calcHist([first_frame], [i], None, [256], [0, 256])
            plt.plot(hist, color=col)
            plt.xlim([0, 256])
        # plt.show()
        particle_array = [0] * particle_count
        particle_weight_array = [0] * particle_count
        create_particle_array(first_frame, particle_array, particle_count)
        weight_particle_array(particle_weight_array, particle_count)
        detect_moving_object(first_frame)
        # draw_particles(first_frame, particle_array)
        cv.imshow('First frame', first_frame)
        key = cv.waitKey(25)
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            back_projection = draw_back_projection(frame, roi)
            move_particles(frame, particle_array, motion_model)
            grade_particles(back_projection, particle_array, particle_weight_array)
            center_of_gravity = calculate_center_of_gravity(particle_array, particle_weight_array, particle_grade)
            # move_particles_to_proximity_center_of_gravity(back_projection, center_of_gravity, particle_array, particle_weight_array, particle_grade, 50)
            move_particles_towards_center_of_gravity(back_projection, center_of_gravity, particle_array, particle_weight_array, particle_grade, motion_model)
            draw_particles(frame, particle_array, particle_weight_array, particle_grade)
            cv.circle(frame, center_of_gravity, 1, (0, 0, 0), 3)
            cv.imshow('Video', frame)
            key = cv.waitKey(50)
            # Close the video
            if key == ord('q'):
                break
            # Pause/Play video
            if key == ord('p'):
                cv.waitKey(0)
        video.release()
        cv.destroyAllWindows()


# Graphical interface for characteristics
def characteristics_gui():
    global characteristics_image
    path = filedialog.askopenfilename()
    characteristics_image = cv.imread(path, cv.IMREAD_GRAYSCALE)
    if characteristics_image is None:
        print("Image was not selected")
        return

    characteristics = Tk()
    characteristics.title("Characteristics")
    characteristics.configure(background="black", padx=20, pady=20)

    Button(characteristics, text="Local binary pattern", width=20, command=calculate_image_lbp) \
        .grid(row=0, column=0, sticky=N)
    Label(characteristics, bg="black").grid(row=1, column=0)
    Button(characteristics, text="Local binary pattern distance", width=20, command=click_load_image) \
        .grid(row=2, column=0, sticky=N)


# Calculate the new image for local binary pattern
def calculate_image_lbp():
    global characteristics_image
    characteristics_calculated = characteristics_image.copy()
    height, width = characteristics_image.shape[:2]

    for j in range(2, width - 2):
        for i in range(2, height - 2):
            binary_value = [characteristics_image[i, j] < characteristics_image[i - 2, j + 1],
                            characteristics_image[i, j] < characteristics_image[i - 1, j + 2],
                            characteristics_image[i, j] < characteristics_image[i + 1, j + 2],
                            characteristics_image[i, j] < characteristics_image[i + 2, j + 2],
                            characteristics_image[i, j] < characteristics_image[i + 2, j - 1],
                            characteristics_image[i, j] < characteristics_image[i + 2, j - 2],
                            characteristics_image[i, j] < characteristics_image[i - 1, j - 2],
                            characteristics_image[i, j] < characteristics_image[i - 2, j - 2],
                            ]
            characteristics_calculated[i, j] = reduce(lambda a, b: 2*a+b, binary_value)
    list_of_histograms = cv.calcHist(characteristics_calculated[0:15, 0:15].copy(), [0], None, [256], (0, 256))
    for j in range(10, width, 10):
        for i in range(0, height, 10):
            region_of_interest = characteristics_calculated[i:i+15, j:j+15].copy()
            cv.imshow("ROI", region_of_interest)
            calculated_region_hist = cv.calcHist(region_of_interest, [0], None, [256], (0, 256))
            np.append(list_of_histograms, calculated_region_hist)
    array_of_histograms = np.asarray(list_of_histograms)
    np.savetxt('test.txt', array_of_histograms, delimiter=',')
    cv.imshow("Old", characteristics_image)
    cv.imshow("New", characteristics_calculated)


window = Tk()
window.title("My first computer vision program")
window.configure(background="black", padx=20, pady=20)

# Creating the interface for the main program with buttons
photo1 = PhotoImage(file="images/computer_vision.png")
Label(window, image=photo1, bg="black").grid(row=0, column=0, columnspan=3)
Button(window, text="Load an image from disk", width=20, command=click_load_image) \
    .grid(row=1, column=0, sticky=N)
Button(window, text="Load a video from disk", width=20, command=click_load_video) \
    .grid(row=1, column=1, sticky=N)
Button(window, text="Open camera", width=20, command=click_load_camera) \
    .grid(row=1, column=2, sticky=N)
Label(window, bg="black").grid(row=2, column=0, columnspan=3)
Button(window, text="Open grayscale image filter", width=20, command=open_image_in_grayscale) \
    .grid(row=3, column=0, sticky=N)
Button(window, text="Segmentations", width=20, command=segmentation_graphical_interface) \
    .grid(row=3, column=1, sticky=N)
Button(window, text="Morphological operators", width=20, command=morphological_gui) \
    .grid(row=3, column=2, sticky=N)
Label(window, bg="black").grid(row=4, column=0, columnspan=3)
Button(window, text="Motion Tracking", width=20, command=motion_track) \
    .grid(row=5, column=0, sticky=N)
Button(window, text="Characteristics", width=20, command=characteristics_gui) \
    .grid(row=5, column=1, sticky=N)
window.mainloop()

# Comments for the exercises
# Convolution, image filtering and edge detection

# 1. What happens if we change the noise reduction kernel size?
# By changing the size of the kernel, we are changing the intensity of the blurriness,
# bigger kernel size, bigger the blur and vise versa. By blurring the image we reduce the
# number of edges the canny algorithm can find and make it less effective in my eyes.

# 2. How does the neighborhood apply on the result in the last step?
# This part of the exercise i have not implemented, but i would guess that with more
# neighbors to choose from, we are giving the edges more of a chance to improve, meaning
# if we check 8 neighbors vs 2 neighbors, there is a bigger chance to find weak edges in that
# area

# 3. Based on a user Geerten answer on StackOverflow, there are a couple of points, which are
# limitations. When the result is binary and we need a measure of "how much" the edge qualifies
# as an edge. The amount of parameters leads to tweaking the algorithm infinitely for the "little
# better result". Also due to the smoothing, the location of the edges might be a bit off, dependent
# on the size of the smoothing kernel (gaussian, mean, ...). The Canny algorithm also has problems with
# corners and jurisdictions (smoothing blurs them out, harder to detect)
