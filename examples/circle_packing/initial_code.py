def pack_circles(n=23):
    import random
    import math
    if n <= 0:
        return [], 0.0
    # EVOLVE-BLOCK-START
    circles = []
    total_sum_radii = 0.0
    for i in range(n):
        radius = 0.01
        x = random.uniform(radius, 1.0 - radius)
        y = random.uniform(radius, 1.0 - radius)
        circles.append((x, y, radius))
        total_sum_radii += radius
    # EVOLVE-BLOCK-END
    return circles, total_sum_radii
