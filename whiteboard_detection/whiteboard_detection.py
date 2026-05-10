from whiteboard_detection.board_detection.board_detection import detect_whiteboard


def whiteboard_detection(path, dbg_dir, name, debug=False):
    warped_path = detect_whiteboard(path, dbg_dir, name, debug)
    if warped_path is None:
        print("Board detection failed for:", path)
        return None
    
    if debug:
        print("SUCCESS:", path)
    return warped_path