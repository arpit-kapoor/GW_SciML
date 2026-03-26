import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import matplotlib.colors as mcolors
import matplotlib.animation as animation
import pickle
from scipy.spatial import KDTree

def main():
    print("Loading data...")
    # Define results directory and load data
    results_dir = os.path.expanduser('~/OneDrive - The University of Sydney (Staff)/Shared/Projects/01_PhD/03_Physics_Informed/05_groundwater/groundwater_main/results/')

    # # Using the high-res updated formulation as per notebook
    # results_dir = os.path.join(results_dir, 'updated_lowres', 'resolution_1.0/gino_predictions_20260326_161714')
    results_dir = os.path.join(results_dir, 'updated_lowres', 'resolution_0.167/gino_predictions_20260326_162706')
    
    val_preds = np.load(os.path.join(results_dir, 'val_predictions.npy'))
    coords = pickle.load(open(os.path.join(results_dir, 'train_coords.pkl'), 'rb'))[0]

    # Apply inverse coordinate transformation
    coord_mean = np.array([357225.66528974305, 6457743.243307921, -9.277822477621577])
    coord_std = np.array([569.1699998979727, 566.3579737855055, 15.2656561762675])
    unnorm_coords = coords * coord_std + coord_mean

    # Load Targets
    val_targets = np.load(os.path.join(results_dir, 'val_targets.npy'))

    # Reshape predictions and targets to (N_nodes, T, N_vars)
    if len(val_preds.shape) == 4:
        preds = np.squeeze(val_preds, axis=2).transpose(1, 0, 2)
        targets = np.squeeze(val_targets, axis=2).transpose(1, 0, 2)
    else:
        preds = val_preds
        targets = val_targets

    # Index 0 is mass concentration
    mass_conc_preds = preds[:, :, 0]    # Shape: (N_nodes, T)
    mass_conc_targets = targets[:, :, 0] # Shape: (N_nodes, T)
    
    T = mass_conc_preds.shape[1]
    print(f"Number of time steps: {T}")

    # Define Points P1 and P2
    p1 = np.array([356835.28, 6459067.43])
    p2 = np.array([357877.28, 6457548.86])

    # Calculate transect vector
    dx_L = p2[0] - p1[0]
    dy_L = p2[1] - p1[1]
    length = np.hypot(dx_L, dy_L)

    print("Reading FEFLOW ASCII Mesh to extract exact geometry...")
    file_path = os.path.expanduser('~/OneDrive - The University of Sydney (Staff)/Shared/Data/FEFLOW/simulation_files/ascii_mesh.fem')

    n2d = 2360
    elements_2d = []
    coor_vals = []
    in_node = False
    in_coor = False

    with open(file_path, 'r', errors='ignore') as f:
        for l in f:
            l = l.strip()
            if l == 'NODE':
                in_node = True
                continue
            if in_node and (len(l)>0 and l[0].isalpha() and l[0].isupper()):
                in_node = False 
            elif in_node:
                nodes = [int(n) - 1 for n in l.split()]
                if nodes[0] < n2d: 
                    elements_2d.append(nodes[:3])
            
            if l == 'COOR':
                in_coor = True
                continue
            if in_coor and (len(l)>0 and l[0].isalpha() and l[0].isupper()):
                in_coor = False
            elif in_coor:
                parts = l.split(',')
                for p in parts:
                    if p.strip(): coor_vals.append(float(p))

    x_mesh = np.array(coor_vals[:n2d])
    y_mesh = np.array(coor_vals[n2d:2*n2d])

    shift_x = unnorm_coords[:, 0].min() - x_mesh.min()
    shift_y = unnorm_coords[:, 1].min() - y_mesh.min()
    x_utm = x_mesh + shift_x
    y_utm = y_mesh + shift_y
    xy_2d = np.column_stack((x_utm, y_utm))

    num_slices = 26
    tree_2d = KDTree(unnorm_coords[:, :2])
    dists_all, indices_all = tree_2d.query(xy_2d, k=num_slices)

    mapped_indices = np.zeros((n2d, num_slices), dtype=int)
    for i in range(n2d):
        vertical_indices = indices_all[i]
        z_coords_node = unnorm_coords[vertical_indices, 2]
        sorted_order = np.argsort(z_coords_node)
        mapped_indices[i] = vertical_indices[sorted_order]

    intersecting_panels = []
    for elem in elements_2d:
        pts = xy_2d[elem]
        hits = []
        for i, j in [(0,1), (1,2), (2,0)]:
            A, B = pts[i], pts[j]
            dx_E = B[0] - A[0]
            dy_E = B[1] - A[1]
            det = dx_L * dy_E - dy_L * dx_E
            if abs(det) < 1e-10: continue
                
            dx_AP = A[0] - p1[0]
            dy_AP = A[1] - p1[1]
            t_L = (dx_AP * dy_E - dy_AP * dx_E) / det
            t_E = (dx_AP * dy_L - dy_AP * dx_L) / det
            
            if 0 <= t_L <= 1 and -1e-5 <= t_E <= 1+1e-5:
                hits.append((t_L, t_E, elem[i], elem[j]))
                
        if len(hits) >= 2:
            hits.sort(key=lambda x: x[0])
            h1, h2 = hits[0], hits[-1]
            if h2[0] - h1[0] > 1e-5:
                intersecting_panels.append({
                    's1': h1[0] * length, 'w1': h1[1], 'e1': (h1[2], h1[3]),
                    's2': h2[0] * length, 'w2': h2[1], 'e2': (h2[2], h2[3])
                })

    print("Interpolating data over time...")
    s_coords = []
    z_coords = []
    triangles = []
    
    # Pre-allocate for all time steps
    # We will compute the values for all nodes across all time steps
    node_predictions = []
    node_targets = []
    
    node_counter = 0
    for panel in intersecting_panels:
        s1, w1, e1 = panel['s1'], panel['w1'], panel['e1']
        s2, w2, e2 = panel['s2'], panel['w2'], panel['e2']
        
        for k in range(num_slices):
            idx_n1a, idx_n1b = mapped_indices[e1[0], k], mapped_indices[e1[1], k]
            idx_n2a, idx_n2b = mapped_indices[e2[0], k], mapped_indices[e2[1], k]
            
            z1 = unnorm_coords[idx_n1a, 2] + w1 * (unnorm_coords[idx_n1b, 2] - unnorm_coords[idx_n1a, 2])
            z2 = unnorm_coords[idx_n2a, 2] + w2 * (unnorm_coords[idx_n2b, 2] - unnorm_coords[idx_n2a, 2])
            
            p1_val_t = mass_conc_preds[idx_n1a, :] + w1 * (mass_conc_preds[idx_n1b, :] - mass_conc_preds[idx_n1a, :])
            p2_val_t = mass_conc_preds[idx_n2a, :] + w2 * (mass_conc_preds[idx_n2b, :] - mass_conc_preds[idx_n2a, :])
            
            t1_val_t = mass_conc_targets[idx_n1a, :] + w1 * (mass_conc_targets[idx_n1b, :] - mass_conc_targets[idx_n1a, :])
            t2_val_t = mass_conc_targets[idx_n2a, :] + w2 * (mass_conc_targets[idx_n2b, :] - mass_conc_targets[idx_n2a, :])

            s_coords.extend([s1, s2])
            z_coords.extend([z1, z2])
            
            node_predictions.append(p1_val_t)
            node_predictions.append(p2_val_t)
            node_targets.append(t1_val_t)
            node_targets.append(t2_val_t)
            
            if k > 0:
                idx_bot1 = node_counter - 2
                idx_bot2 = node_counter - 1
                idx_top1 = node_counter
                idx_top2 = node_counter + 1
                
                triangles.append([idx_bot1, idx_bot2, idx_top1])
                triangles.append([idx_bot2, idx_top2, idx_top1])
                
            node_counter += 2

    s_coords = np.array(s_coords)
    z_coords = np.array(z_coords)
    triangles = np.array(triangles)
    
    # Shape: (N_interpolated_nodes, T)
    preds_vals_all = np.array(node_predictions)
    targets_vals_all = np.array(node_targets)

    print("Setting up the animation...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharex=True, sharey=True)

    levels = [-30000, 0, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 34500, 55000]
    feflow_colors = [
        '#5E4FA2', '#3973B5', '#4CB2D4', '#32C2A3', '#45D86A', 
        '#8DEB45', '#C4FA4B', '#E7FE57', '#FEE44F', '#FB9736'
    ]
    cmap = mcolors.ListedColormap(feflow_colors)
    norm = mcolors.BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    triangulation = mtri.Triangulation(s_coords, z_coords, triangles)

    def init():
        axes[0].clear()
        axes[1].clear()
        return []

    def update(frame):
        axes[0].clear()
        axes[1].clear()
        
        preds_frame = preds_vals_all[:, frame]
        targets_frame = targets_vals_all[:, frame]
        
        cf_target = axes[0].tricontourf(triangulation, targets_frame, levels=levels, cmap=cmap, norm=norm, extend='both')
        axes[0].set_title(f'GROUND TRUTH - Time Step: {frame}')
        axes[0].set_ylabel('Elevation Z (m)')
        axes[0].set_facecolor('white')

        cf_pred = axes[1].tricontourf(triangulation, preds_frame, levels=levels, cmap=cmap, norm=norm, extend='both')
        axes[1].set_title(f'PREDICTIONS - Time Step: {frame}')
        axes[1].set_xlabel(f'Distance along transect from P1 (m)')
        axes[1].set_facecolor('white')
        
        axes[0].set_xlim(s_coords.min(), s_coords.max())
        axes[0].set_ylim(z_coords.min(), z_coords.max())
        axes[1].set_xlim(s_coords.min(), s_coords.max())
        axes[1].set_ylim(z_coords.min(), z_coords.max())
        
        return []

    # Format the colorbar carefully on the right
    fig.subplots_adjust(right=0.88, left=0.08, bottom=0.15, top=0.85)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    
    # We create a dummy contour to generate the colorbar
    cf_dummy = axes[0].tricontourf(triangulation, targets_vals_all[:, 0], levels=levels, cmap=cmap, norm=norm, extend='both')
    cbar = fig.colorbar(cf_dummy, cax=cbar_ax, label='Mass Concentration (mg/l)', extend='both')
    cbar.set_ticks([0, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 34500])

    print("Generating video... This might take a few moments.")
    # Calculate time interval for 5 fps
    fps = 5
    
    ani = animation.FuncAnimation(fig, update, frames=T, init_func=init, blit=False)
    
    out_file = os.path.join(results_dir, 'cross_section_comparison.mp4')
    
    # Using ffmpeg writer, specifying h264 for compression and compatibility
    writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='SciML'), bitrate=1800, codec='h264')
    ani.save(out_file, writer=writer)
    
    print(f"Video saved to: {out_file}")

if __name__ == '__main__':
    main()
