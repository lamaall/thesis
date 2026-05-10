from path_planning.preproccess.preproccess import remove_degree_2_nodes, preprocess_graph
from path_planning.stroke_solver.construct_strokes import construct_strokes
from path_planning.stroke_solver.solver import plan_stroke_path
from path_planning.utils.plot_strokes import plot_strokes
from path_planning.utils.plot_strokes_with_numbers import plot_strokes_with_numbers
from path_planning.utils.plot_graph import draw_graph
from path_planning.structural_solver.solver import build_human_like_path
from path_planning.utils.animate_traversal import animate_traversal


def plan_path(G, dbg_folder, name, debug=False):
    if debug:
        draw_graph(G, filename=f"{dbg_folder}/{name}_graph.png", show=False)

    removed = remove_degree_2_nodes(preprocess_graph(G))

    if debug:
        draw_graph(removed, filename=f"{dbg_folder}/{name}_graph_preprocessed.png", show=False)

    strokes = construct_strokes(removed)

    if debug:
        plot_strokes(strokes, removed, title="Constructed Strokes", dbg_folder=dbg_folder, name=name)
        
        #pos = {node: node for node in G.nodes()}
        
        #build_human_like_path(G, pos, dbg_folder)


    planned_path = plan_stroke_path(strokes)

    if debug:
        plot_strokes_with_numbers(strokes, planned_path, removed, title="Constructed Strokes", dbg_folder=dbg_folder, name=name)
        

    #animate_traversal(planned_path, strokes)

    return planned_path, strokes

