# overlap checking

def check_single(bbox, bboxes):
    for existing_bbox in bboxes:
        if (bbox[0] < existing_bbox[2] and bbox[2] > existing_bbox[0] and
                bbox[1] < existing_bbox[3] and bbox[3] > existing_bbox[1]):
            return True
    return False


def check_group(group_bboxes, bboxes):
    """Check if a group of bounding boxes overlaps with any existing boxes."""
    for bbox in group_bboxes:
        if check_single(bbox, bboxes):
            return True
    return False
