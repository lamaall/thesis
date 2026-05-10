import numpy as np
import matplotlib.pyplot as plt
import os

# =========================================================
# IMAGE → POINT CLOUD
# =========================================================
def binary_image_to_input_data(image):
    coords = np.column_stack(np.where(image > 0))
    return coords.astype(np.float64)


def normalize_coords(coords, shape, scale=100):
    coords = coords.astype(np.float64)
    coords[:, 0] /= shape[0]
    coords[:, 1] /= shape[1]
    coords *= scale
    return coords


# =========================================================
# PLOT
# =========================================================
def plot_som(input_data, final_weights, A, dbg_folder="dbg", grid_size=(100, 100)):
    os.makedirs(dbg_folder, exist_ok=True)

    plt.figure(figsize=(8, 8))
    ax = plt.gca()

    # X normal, Y flipped (image coordinate system)
    ax.scatter(input_data[:, 1],
               grid_size[1] - input_data[:, 0],
               c='blue', marker='x', s=30, label='Input')

    ax.scatter(final_weights[:, 1],
               grid_size[1] - final_weights[:, 0],
               c='red', marker='o', s=80, label='Neurons')

    # connections
    for i in range(len(final_weights)):
        for j in range(i + 1, len(final_weights)):
            if A[i, j] > 0:
                ax.plot([final_weights[i, 1], final_weights[j, 1]],
                        [grid_size[1] - final_weights[i, 0],
                         grid_size[1] - final_weights[j, 0]],
                        'k-', lw=0.5)

    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_aspect('equal', 'box')
    ax.legend()

    plt.title("DCS / SOM on Binary Image (y-flipped)")

    filepath = os.path.join(dbg_folder, "som_result.png")
    plt.savefig(filepath, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"[Saved] SOM visualization → {filepath}")

import os
import matplotlib.pyplot as plt


def plot_som(input_data, final_weights, A, dbg_folder="dbg", grid_size=(100, 100)):
    os.makedirs(dbg_folder, exist_ok=True)

    fig, ax = plt.subplots(figsize=(6, 6))

    # White background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Flip Y only (image-style coordinates)
    x_w = final_weights[:, 1]
    y_w = grid_size[1] - final_weights[:, 0]

    # Draw connections (black lines)
    for i in range(len(final_weights)):
        for j in range(i + 1, len(final_weights)):
            if A[i, j] > 0:
                ax.plot([x_w[i], x_w[j]],
                        [y_w[i], y_w[j]],
                        color='black', lw=1)

    # Draw nodes (black points)
    ax.scatter(x_w, y_w, color='black', s=10)

    # Remove EVERYTHING visual
    ax.set_xlim(0, grid_size[0])
    ax.set_ylim(0, grid_size[1])
    ax.set_aspect('equal')

    ax.axis('off')  # <-- key line

    # Remove padding / margins
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Save
    filepath = os.path.join(dbg_folder, "som_clean.png")
    plt.savefig(filepath, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    print(f"[Saved] Clean SOM graph → {filepath}")


# =========================================================
# CORE ALGORITHMIC FUNCTIONS
# =========================================================
def initialize_network(input_data, input_dim, num_initial_units=2):
    idx = np.random.choice(len(input_data), num_initial_units, replace=False)
    w = input_data[idx].astype(np.float64)

    A = np.zeros((num_initial_units, num_initial_units))
    A[0, 1] = A[1, 0] = 1

    resource_values = np.zeros(num_initial_units)
    return w, A, resource_values


def get_next_example(input_data):
    return input_data[np.random.randint(len(input_data))]


def find_closest_units(v, w):
    d = np.linalg.norm(w - v, axis=1)
    idx = np.argsort(d)
    return idx[0], idx[1]


def update_connection_strengths(wB, wN, A, forgetting_factor, threshold):
    A[wB, wN] = A[wN, wB] = 1

    A *= forgetting_factor
    A[A < threshold] = 0


def kohonen(wB, v, w, lr_b, lr_n, A):
    w[wB] += lr_b * (v - w[wB])

    for j in range(len(w)):
        if A[wB, j] > 0:
            w[j] += lr_n * (v - w[j])


def update_resource(wB, v, w, resource_values):
    resource_values[wB] += np.linalg.norm(v - w[wB]) ** 2


def decrement_resource_values(resource_values, forgetting_factor):
    return resource_values * forgetting_factor


def add_new_neuron(w, A, resource_values):
    max_unit = np.argmax(resource_values)

    neighbors = np.where(A[max_unit] > 0)[0]
    if len(neighbors) == 0:
        return w, A, resource_values

    neighbor = neighbors[0] if len(neighbors) == 1 else neighbors[np.argmax(resource_values[neighbors])]

    new_w = (w[max_unit] + w[neighbor]) / 2.0
    w = np.vstack([w, new_w])

    A = np.pad(A, ((0, 1), (0, 1)), mode='constant')
    A[max_unit, -1] = A[neighbor, -1] = 1
    A[-1, max_unit] = A[-1, neighbor] = 1

    resource_values = np.append(resource_values, 0)

    return w, A, resource_values


# =========================================================
# MAIN FUNCTION (NOW CLEAN API)
# =========================================================
def run_dcs_on_image(binary_image,
                     dbg_folder="dbg",
                     max_iterations=100,
                     learning_rate_b=0.2,
                     learning_rate_n=0.01,
                     forgetting_factor=0.99,
                     threshold=0.001,
                     stopping_error=1):

    coords = binary_image_to_input_data(binary_image)
    input_data = normalize_coords(coords, binary_image.shape)

    w, A, r = initialize_network(input_data, input_dim=2)

    iteration = 0

    while iteration < max_iterations:

        for _ in range(len(input_data) // 2):
            v = get_next_example(input_data)
            wB, wN = find_closest_units(v, w)

            update_connection_strengths(wB, wN, A, forgetting_factor, threshold)
            kohonen(wB, v, w, learning_rate_b, learning_rate_n, A)
            update_resource(wB, v, w, r)

        errors = [np.linalg.norm(v - w[find_closest_units(v, w)[0]]) ** 2
                  for v in input_data]

        error = np.mean(errors)

        if error < stopping_error:
            break

        alive = np.sum(A, axis=1) > 0
        w = w[alive]
        A = A[alive][:, alive]
        r = r[alive]

        w, A, r = add_new_neuron(w, A, r)
        r = decrement_resource_values(r, forgetting_factor)

        iteration += 1
        print(f"Iteration {iteration} | error={error:.4f}")

    plot_som(input_data, w, A, dbg_folder)

    return w, A, r