from dijkstar import Graph, find_path
import numpy as np

def find_min_path(thetalists_frames):
    graph = Graph()
    for i in range(len(thetalists_frames[0])):
        graph.add_edge("s", str(i), 0)
    for i in range(len(thetalists_frames)-1):
        for j in range(len(thetalists_frames[i])):
            for k in range(len(thetalists_frames[i+1])):
                theta_diff = np.abs(np.array(thetalists_frames[i][j]) - np.array(thetalists_frames[i+1][k])).sum()
                graph.add_edge(str(i*1000+j), str((i+1)*1000+k), theta_diff)
    for i in range(len(thetalists_frames[-1])):
        l = len(thetalists_frames)-1
        graph.add_edge(str(l*1000+i), "e", 0)
    result = find_path(graph, "s", "e")
    print(result.nodes)
    chosen_thetalists_frames = []
    for i in range(1, len(result.nodes)-1):
        idx = int(result.nodes[i]) % 1000
        frame_idx = i-1
        chosen_thetalists_frames.append(thetalists_frames[frame_idx][idx])
    return chosen_thetalists_frames

    