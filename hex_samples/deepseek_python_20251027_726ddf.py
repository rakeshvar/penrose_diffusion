import os
import glob
from PIL import Image
import cairosvg

def convert_svg_to_png(svg_path, png_path, size=(360, 360), rotate_degrees=90):
    """Convert SVG to PNG with fixed size and rotation"""
    # Convert SVG to PNG
    cairosvg.svg2png(url=svg_path, write_to=png_path, output_width=size[0], output_height=size[1])
    
    # Open the PNG and rotate
    with Image.open(png_path) as img:
        # Rotate clockwise by 90 degrees (counter-clockwise by -90)
        rotated_img = img.rotate(-rotate_degrees, expand=True)
        rotated_img.save(png_path)

def create_collage(image_paths, output_path, grid_size=(4, 4), image_size=(360, 360)):
    """Create a 4x4 collage from PNG images"""
    collage_width = grid_size[0] * image_size[0]
    collage_height = grid_size[1] * image_size[1]
    
    collage = Image.new('RGB', (collage_width, collage_height), 'white')
    
    for index, image_path in enumerate(image_paths):
        if index >= grid_size[0] * grid_size[1]:
            break
            
        with Image.open(image_path) as img:
            # Calculate position in grid
            row = index // grid_size[0]
            col = index % grid_size[0]
            
            x = col * image_size[0]
            y = row * image_size[1]
            
            # Paste image onto collage
            collage.paste(img, (x, y))
    
    collage.save(output_path)
    print(f"Collage saved as: {output_path}")

def main():
    # Configuration
    input_folder = "."  # Current directory, change if needed
    output_folder = "converted_images"
    collage_output = "collage_4x4.png"
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Get all SVG files in the folder
    svg_files = glob.glob(os.path.join(input_folder, "*.svg"))
    
    if not svg_files:
        print("No SVG files found in the current directory!")
        return
    
    print(f"Found {len(svg_files)} SVG files")
    
    # Convert each SVG to PNG
    png_paths = []
    for svg_file in svg_files:
        filename = os.path.splitext(os.path.basename(svg_file))[0]
        png_filename = f"{filename}.png"
        png_path = os.path.join(output_folder, png_filename)
        
        print(f"Converting: {svg_file} -> {png_path}")
        convert_svg_to_png(svg_file, png_path)
        png_paths.append(png_path)
    
    print("All SVG files converted to PNG!")
    
    # Create collage
    if len(png_paths) >= 16:
        create_collage(png_paths[:16], collage_output)
    else:
        print(f"Warning: Only {len(png_paths)} images available. Need 16 for a 4x4 collage.")
        create_collage(png_paths, collage_output)

if __name__ == "__main__":
    main()