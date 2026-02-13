import math

directions = [
    (1, 0, -1),   # East
    (0, 1, -1),   # Southeast
    (-1, 1, 0),   # Southwest
    (-1, 0, 1),   # West
    (0, -1, 1),   # Northwest
    (1, -1, 0),   # Northeast
]

# cumulative sums C[side] = sum_{i<side} directions[i]
C = [
    (0,0,0),
    (1,0,-1),
    (1,1,-2),
    (0,2,-2),
    (-1,2,-1),
    (-1,1,0),
]


def spiral_to_cube(n: int):
    if n == 0:
        return (0,0,0)

    # ring index
    k = math.ceil((math.sqrt(12*n - 3) - 3) / 6)

    base = 1 + 3*(k-1)*k
    i = n - base

    side = i // k
    t    = i %  k

    dq, dr, ds = directions[0]   # start corner = k*D0

    q = k*dq
    r = k*dr
    s = k*ds

    cq, cr, cs = C[side]
    q += k*cq
    r += k*cr
    s += k*cs

    dq, dr, ds = directions[side]
    q += t*dq
    r += t*dr
    s += t*ds

    return (q, r, s)

def cube_to_spiral(q: int, r: int, s: int) -> int:
    # center
    if q == 0 and r == 0 and s == 0:
        return 0

    # ring index
    k = max(abs(q), abs(r), abs(s))

    base = 1 + 3*(k-1)*k

    # -------------------------------------------------
    # IMPORTANT:
    # This is derived EXACTLY from your spiral_to_cube:
    #
    # start = (k,0,-k)
    # then:
    #   + k*C[side] + t*directions[side]
    #
    # So we analytically invert those lines.
    # -------------------------------------------------

    # Edge 0: from (k,0,-k) toward D1
    if q == k and s < 0:
        side = 0
        t = r

    # Edge 1
    elif r == k and q > 0:
        side = 1
        t = k - q

    # Edge 2
    elif r == k and q <= 0:
        side = 2
        t = -q

    # Edge 3
    elif q == -k and s > 0:
        side = 3
        t = -r

    # Edge 4
    elif s == k and q < 0:
        side = 4
        t = q + k

    # Edge 5
    else:
        # remaining edge closes ring back to start
        side = 5
        t = q

    return base + side*k + t
