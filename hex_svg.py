from hex_utils import hex_corners_xy, hex_centers_xy

#--------------------------------------------------------------
# Hex SVG Generation and Display
#--------------------------------------------------------------

def generate_hex_svg(hexes, side, show_coords):
    """
    Generate an SVG visualization of hexagonal grid.
    
    Args:
        hexes: List of tuples (q, r, s, color) or (q, r, s)
        side: Size of each hexagon
    
    Returns:
        SVG string that can be displayed in Jupyter
    """
    
    # Color palette
    colors = {
        0: "#D8D388",  
        1: "#4ee055",  
        2: "#A34444",  
    }
    
    def hex_path(center_x, center_y):
        """Generate SVG path for hexagon"""
        points = hex_corners_xy(center_x, center_y, side)
        path = f"M {points[0][0]},{points[0][1]}"
        for x, y in points[1:]:
            path += f" L {x},{y}"
        path += " Z"
        return path

    
    xy_data = hex_centers_xy(hexes, side)
    

    # Determine SVG size
    margin = side * 2
    xs = [x for x, y, q, r, s, color in xy_data]
    ys = [y for x, y, q, r, s, color in xy_data]
    min_x, max_x = min(xs) - margin, max(xs) + margin
    min_y, max_y = min(ys) - margin, max(ys) + margin
    size = int(max(max_x - min_x, max_y - min_y))   

    # Start building SVG
    svg_parts = [f'<svg width="{size}" height="{size}" viewBox="-{size//2} -{size//2} {size} {size}" xmlns="http://www.w3.org/2000/svg">']
    
    # Background
    svg_parts.append(f'<rect x="-{size//2}" y="-{size//2}" width="{size}" height="{size}" fill="#1e293b"/>')

    # Draw hexagons
    for x, y, q, r, s, color in xy_data:
        fill_color = colors.get(color, '#94a3b8')
        
        # Draw hexagon
        path = hex_path(x, y)
        svg_parts.append(f'<path d="{path}" fill="{fill_color}" stroke="#0f172a" stroke-width="2"/>')
        
        # Add coordinate text
        if show_coords:
            svg_parts.append(f'<text x="{x}" y="{y-5}" text-anchor="middle" font-family="monospace" font-size="9" fill="white">{q},{r},{s} {color}</text>')
        
    # Add axis labels
    svg_parts.append(f'<text x="0" y="-{size//2 - 20}" text-anchor="middle" font-size="16" font-weight="bold" fill="#86efac">+q (East)</text>')
    svg_parts.append(f'<text x="{size//2 - 40}" y="{size//2 - 20}" text-anchor="middle" font-size="16" font-weight="bold" fill="#93c5fd">+r (SE)</text>')
    svg_parts.append(f'<text x="-{size//2 - 40}" y="{size//2 - 20}" text-anchor="middle" font-size="16" font-weight="bold" fill="#d8b4fe">+s (SW)</text>')
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def display_hex_svg(hexes, side, show_coords):
    """
    Display SVG in Jupyter notebook.
    
    Args:
        hexes: List of tuples (q, r, s, color) or (q, r, s)
        size: Size of the SVG canvas
        side: Size of each hexagon
    """
    from IPython.display import SVG, display
    svg_string = generate_hex_svg(hexes, side, show_coords)
    display(SVG(svg_string))