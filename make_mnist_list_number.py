import numpy as np
from PIL import Image, ImageDraw, ImageFont

def create_custom_number():
    # Create a blank 28x28 pixel array (same size as MNIST images)
    image_array = np.zeros((28, 28), dtype=np.uint8)

    # Create an image and drawing context
    image = Image.fromarray(image_array, 'L')  # 'L' mode for grayscale
    draw = ImageDraw.Draw(image)

    # Set a font (PIL's default font is used here as a basic option)
    font_size = 20
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except IOError:
        font = ImageFont.load_default()  # Fallback if no TTF is available

    # Draw the number "6" in the center of the image
    text = "6"
    text_bbox = draw.textbbox((0, 0), text, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    text_x = (28 - text_width) // 2
    text_y = (28 - text_height) // 2
    draw.text((text_x, text_y), text, fill=255, font=font)  # White color (255) on black background

    # Save the image as PNG
    image.save("outside_number.png")
    print("Image saved as 'outside_number.png'")

create_custom_number()
