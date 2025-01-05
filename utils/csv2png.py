from PIL import Image, ImageDraw, ImageFont
import csv
import sys

def csv_to_png(csv_filename, output_filename):
    """Converts a CSV table to a PNG image."""

    with open(csv_filename, 'r') as file:
        reader = csv.reader(file)
        table_data = list(reader)

    if not table_data:
        return

    font_size = 14
    font = ImageFont.truetype("Arial.ttf", font_size)
    padding = 10


    max_cell_widths = [0] * len(table_data[0])
    cell_height = font_size + 2 * padding

    for row in table_data:
        for i, cell in enumerate(row):
            text_width, _ = font.getbbox(cell)[2:4]
            max_cell_widths[i] = max(max_cell_widths[i], text_width + 2 * padding)

    table_width = sum(max_cell_widths)
    table_height = len(table_data) * cell_height

    image = Image.new("RGB", (table_width, table_height), "white")
    draw = ImageDraw.Draw(image)


    y_offset = 0
    for row in table_data:
        x_offset = 0
        for i, cell in enumerate(row):
            # Draw cell borders
            draw.rectangle(
                [(x_offset, y_offset), (x_offset + max_cell_widths[i], y_offset + cell_height)],
                outline="black"
            )

            # Draw cell text
            text_x = x_offset + padding
            text_y = y_offset + padding
            draw.text((text_x, text_y), cell, fill="black", font=font)

            x_offset += max_cell_widths[i]
        y_offset += cell_height


    image.save(output_filename)

if __name__ == "__main__":
    #print(sys.argv)
    if len(sys.argv) < 3:
        print("Usage: python script.py <input_csv_file> [output_markdown_file]")
    elif len(sys.argv) == 3:
        csv_to_png(sys.argv[1], sys.argv[2])
    else:
        #csv_to_png(sys.argv[1], sys.argv[2])
        pass

