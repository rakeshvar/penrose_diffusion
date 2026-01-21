from code.utils import detect_duplicates, svg_path
from textwrap import dedent

def save_svg(hexgrid, filename, ndigits=3, print_ok=True):
    # Color palette
    config = {
    "colors": {
        0: "#D8D388",
        1: "#4ee055",
        },
        "margin": 0.05,
        "stroke-colour": "#ffffff",
        "opacity": 0.7,
        "background": "black",
    }

    # Determine viewbox size
    xmin = ymin = float('inf')
    xmax = ymax = float('-inf')
    for t in hexgrid:
        for y, x in t.vertices:
            xmin = min(xmin, x)
            xmax = max(xmax, x)
            ymin = min(ymin, y)
            ymax = max(ymax, y)

    wd, ht = xmax-xmin, ymax-ymin

    m = config['margin']
    xmin -= m*wd
    ymin -= m*ht
    wd += 2*m*wd
    ht += 2*m*ht
    viewbox = f'{xmin:.2f} {ymin:.2f} {wd:.2f} {ht:.2f}'

    duplicates = detect_duplicates(hexgrid, ndigits=ndigits+1)

    # Build SVG header
    svg_header = dedent(f"""\
        <?xml version="1.0" encoding="utf-8"?>
        <svg preserveAspectRatio="xMidYMid meet" version="1.1" baseProfile="full" xmlns="http://www.w3.org/2000/svg"
             width="{int(1080*wd/ht)}" height="1080"
             viewBox="{viewbox}">
          
        <style>
          .colored {{
            fill: {config["colors"][1]};
            fill-opacity: {config["opacity"]};
          }}
          .uncolor {{
            fill: {config["colors"][0]};
            fill-opacity: {config["opacity"]};
          }}
          .duplicate {{
            fill: #0000c0;
            fill-opacity: 1.0;
          }}
        </style>
        <rect x="{xmin:.3f}" y="{ymin:.3f}" width="{wd:.3f}" height="{ht:.3f}" fill="{config["background"]}"/>
        <g style="stroke:{config["stroke-colour"]}; stroke-width: {hexgrid.side/20:.4f};
           stroke-linejoin: round; vector-effect: non-scaling-stroke;">
        # {len(hexgrid)} hexagons 
        # x {xmin:.2f} to {xmax:.2f} ({wd:.2f})
        # y {ymin:.2f} to {ymax:.2f} ({ht:.2f})
        """)

    # Draw hexagons
    hexagon_paths = []
    for h in hexgrid:
        css_class = "colored" if h.color else "uncolor"
        path = svg_path(h, ndigits=ndigits)
        hexagon_paths.append(f'<path class="{css_class}" d="{path}" />')
    
    hexagons = '\n'.join(hexagon_paths)

    # SVG footer
    svg_footer = "\n</g>\n</svg>"

    # Mark duplicates with red circles at their centers
    dup_markers = [f'<circle class="duplicate" cx="{h.y:.{ndigits}f}" cy="{h.x:.{ndigits}f}" r="{hexgrid.side/3:.{ndigits}f}" />' for h in duplicates]
    dup_markers = '\n'.join(dup_markers)

    # Combine all parts
    svg = svg_header + hexagons + dup_markers + svg_footer

    if filename is not None:
        with open(filename, 'w') as f:
            f.write(svg)

    if print_ok:
        print("Saved svg file  :", filename)

    if duplicates:
        print(f"DUPLICATES FOUND: {len(duplicates)}")

    return svg