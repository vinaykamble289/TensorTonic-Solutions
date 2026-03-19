def generate_anchors(feature_size, image_size, scales, aspect_ratios):
    """
    Generate anchor boxes for object detection.
    """
    # Write code here
    
    stride = image_size / feature_size
    start = stride / 2
    list = []
    for y in range(feature_size):
        for x in range(feature_size):
            cx = start + (x * stride)
            cy = start + (y * stride)
            for s in scales:
                for r in aspect_ratios:
                    w = s* math.sqrt(r)
                    h = s / math.sqrt(r)
                    list.append([cx - w/2,
                            cy - h/2, 
                            cx + w/2, 
                            cy + h/2 ])
    return list