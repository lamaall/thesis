from graph_construction.binarization.binarization import binarize_whiteboard
from graph_construction.graph_creation.skeletonization_vectorization.skeletonize import skeletonize_binary_image
from graph_construction.graph_creation.skeletonization_vectorization.vectorization import vectorize_skeleton
from graph_construction.graph_creation.skeletonization_vectorization.rdp_vectorization import vectorize_skeleton_rdp

from graph_construction.graph_creation.dcs.dcs import run_dcs_on_image

def graph_construction(path, dbg_dir, out_dir, name, debug=False):
    binarized = binarize_whiteboard(path, dbg_dir, out_dir, name, debug)
    if binarized is None:
        print("Binarization failed for:", path)
        return
    
    #if debug:
    #    run_dcs_on_image(binarized, dbg_folder=dbg_dir)
    
    skeleton = skeletonize_binary_image(binarized, name, dbg_dir, debug)
    if skeleton is None:
        print("Skeletonization failed for:", binarized)
        return
    
    lineG = vectorize_skeleton_rdp(skeleton, name, dbg_dir, debug, epsilon=2.5)
    if lineG is None:
        print("Vectorization failed for:", skeleton)
        return
    
    print("SUCCESS:", path)
    return lineG