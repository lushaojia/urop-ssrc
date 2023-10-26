from PIL import Image
from collections import Counter

def load_color_image(filename):
    """
    Loads a color image from the given file and returns a dictionary
    representing that image.

    Invoked as, for example:
       i = load_color_image('test_images/cat.png')
    """
    with open(filename, "rb") as img_handle:
        img = Image.open(img_handle)
        img = img.convert("RGB")  # in case we were given a greyscale image
        img_data = img.getdata()
        pixels = list(img_data)
        width, height = img.size
        return {"height": height, "width": width, "pixels": pixels}

def save_color_image(image, filename, mode="PNG"):
    """
    Saves the given color image to disk or to a file-like object.  If filename
    is given as a string, the file type will be inferred from the given name.
    If filename is given as a file-like object, the file type will be
    determined by the 'mode' parameter.
    """
    out = Image.new(mode="RGB", size=(image["width"], image["height"]))
    out.putdata(image["pixels"])
    if isinstance(filename, str):
        out.save(filename)
    else:
        out.save(filename, mode)
    out.close()


def get_index(image, row, col):
    """
    Computes the index of a pixel in a list of pixels
    stored in row-major order by its row and column.

    Parameters:
      * image (dict): image represented internally
      * row (int): row of target pixel
      * col (int): column of target pixel
    Returns:
        The index of target pixel in list of pixels.
    """
    return row * image["width"] + col


def get_pixel(image, row, col):
    """
    Gets target pixel by row and column.

    Parameters:
      * image (dict): image with internal representation
      * row (int): row of target pixel
      * col (int): column of target pixel
    Returns:
        Value of target pixel (rgb tuple).
    """
    i = get_index(image, row, col)
    return image["pixels"][i]

def get_pixel_extended(image, row, col):
    """
    Retrieves an out-of-bounds pixel by getting its corresponding pixel
    if the given image were extended (i.e. pixels at the edge of the image 
    extend in their respective corners/rows/columns)
    
    Parameters:
        * image (dict): image represented internally
        * row (int): row of out-of-bounds pixel
        * col (int): col of out-of-bounds pixel
    
    Returns:
        A pixel value.
    """
    width = image["width"]
    height = image["height"]

    if (col <= 0) and (row <=0):
        # OOB pixel extends from top left corner
        # print(get_pixel(image, 0, 0))
        return get_pixel(image, 0, 0)
    elif row < 0 < col < width - 1:
        # OOB pixel extends from a non-corner top edge
        # print(get_pixel (image, 0, col))
        return get_pixel (image, 0, col)
    elif (col >= width - 1) and (row <= 0):
        # OOB pixel extends from top right corner
        # print(get_pixel(image, 0, width - 1))
        return get_pixel(image, 0, width - 1)
    elif (0 < row < height - 1) and (col > width - 1):
        # OOB pixel extends from non-corner right edge
        # print(get_pixel(image, row, width - 1))
        return get_pixel(image, row, width - 1)
    elif (col >= width - 1) and (row >= height - 1):
        # OOB pixel extends from bottom right corner
        # print(get_pixel(image, height - 1, width - 1))
        return get_pixel(image, height - 1, width - 1)
    elif (0 < col < width - 1) and (row > height - 1):
        # OOB pixel extends from non-corner bottom edge
        # print(get_pixel(image, height - 1, col))
        return get_pixel(image, height - 1, col)
    elif (col <= 0) and (row >= height - 1):
        # OOB pixel extends from bottom left corner
        # print(get_pixel(image, height - 1, 0))
        return get_pixel(image, height - 1, 0)
    elif col < 0 < row < height - 1:
        # OOB pixel extends from non-corner left edge
        # print(get_pixel(image, row, 0))
        return get_pixel(image, row, 0)

def get_pixel_deluxe(image, row, col):
    """
    Given an image represented internally and
    (row, col) of desired pixel,
    retrieves pixel normally if (row, col) in-bounds and
    retrieves pixel as if image were extended by the sides otherwise.
    """
    # if pixel is in-bounds, get it normally
    if (0 <= row < image["height"]) \
        and (0 <= col < image["width"]):
        return  get_pixel(image, row, col) 
    else:
        return get_pixel_extended(image, row, col)

def set_pixel(image, row, col, color):
    """
    Sets target pixel identified by location in terms of 
    row and column to a new rgb value.

    Parameters:
      * image (dict): image represented internally
      * row (int): row of target pixel
      * col (int): column of target pixel
      * color (tuple): new color of pixel in rgb as 3-element tuple
    """
    i = get_index(image, row, col)
    image["pixels"][i] = color

def identify_majority_rgb(image, curr_row, curr_col, offset):
    """
    Given image represented internally and
    (row, col) locations of pixels in 
    a neighborhood centered at some pixel,
    returns the most frequent pixel (i.e. rgb value)
    in neighborhood.
    """
    pixels = [get_pixel_deluxe(image, row, col) 
              for row in range(curr_row - offset, curr_row + offset + 1)
                for col in range(curr_col - offset, curr_col + offset + 1)] # get pixels in neighborhood
    pixel_counter = Counter(pixels)
    return pixel_counter.most_common(1)[0][0]
            
# may need to implement function that applies some function to every pixel of image for later

def apply_majority_rgb(image, neighborhood_size=3):
    """
    Given an image with internal representation, 
    sets each pixel to the majority rgb value in
    neighborhood of given size.
    """
    for row in range(image["height"]):
        for col in range(image["width"]):
            offset =  neighborhood_size // 2
            majority_rgb = identify_majority_rgb(image, row, col, offset)
            set_pixel(image, row, col, majority_rgb)
            print(f"Done with ({row}, {col}).")
            

if __name__ == "__main__":
    fp0 = "CHHA000101/material_0.png"
    material0 = load_color_image(fp0)
    apply_majority_rgb(material0, neighborhood_size=99)
    save_color_image(material0, "test/m0_n=99.png")