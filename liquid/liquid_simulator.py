from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

app = FastAPI()

# Simulation parameters
BOUNDARY = 10
COLLISION_DAMPING = 0.9
FPS = 30
CORE_REPELLING_CONSTANT = 0.7
net_forces = []


# Color mapping based on speed
def calculate_color(speeds):
    # Normalize speeds to the range [0, 1]
    normalized_speeds = np.clip(speeds / np.max(speeds), 0, 1)

    # Define RGB values for violet and red
    violet = np.array([148, 0, 211]) / 255  # RGB for violet
    red = np.array([255, 0, 0]) / 255  # RGB for red
    blue = np.array([0, 0, 255]) / 255
    # Interpolate between violet and red
    colors = (normalized_speeds[:, None]) * blue + (1 - normalized_speeds[:, None]) * red * (4/5) + (1 - normalized_speeds[:, None]) * violet * (1/5)
    return colors
# Generate initial positions and velocities
def generate_points(num_points):
    np.random.seed(42)
    positions = np.random.uniform(-BOUNDARY, BOUNDARY, (num_points, 2))
    velocities = np.random.normal(0, 0.5, (num_points, 2)) * np.linalg.norm(positions, axis=1).reshape(-1, 1)
    return positions, velocities


def update_positions(positions, velocities):
    positions += velocities / FPS
    # Wall collision detection and velocity adjustment
    for i in range(len(positions)):
        if abs(positions[i, 0]) >= BOUNDARY:
            velocities[i, 0] *= -COLLISION_DAMPING
        if abs(positions[i, 1]) >= BOUNDARY:
            velocities[i, 1] *= -COLLISION_DAMPING

    # Point collision detection and response (tangential bounce)
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            # Calculate the distance vector between points i and j
            delta_pos = positions[j] - positions[i]
            dist = np.linalg.norm(delta_pos)

            # Check if points are close enough to collide
            if dist < 2 and dist != 0:
                # Normalize the distance vector to get the collision normal
                collision_normal = (delta_pos / dist) / 4

                if 0.09 <= dist < 2:
                    repelling_force = CORE_REPELLING_CONSTANT / (dist**2 + 0.1)
                    force_vector = repelling_force * collision_normal
                    net_forces[i] -= force_vector
                    net_forces[j] += force_vector


                # Project velocities onto the collision normal
                v_i_normal = np.dot(velocities[i], collision_normal)
                v_j_normal = np.dot(velocities[j], collision_normal)

                # Exchange the normal components of the velocities (elastic collision)
                velocities[i] += (v_j_normal - v_i_normal) * collision_normal
                velocities[j] += (v_i_normal - v_j_normal) * collision_normal

                net_forces[i] += (v_j_normal - v_i_normal) * collision_normal * CORE_REPELLING_CONSTANT
                net_forces[j] += (v_i_normal - v_j_normal) * collision_normal * CORE_REPELLING_CONSTANT

                velocities = net_forces
# Function to create and save the animated GIF
def plot_simulation(num_points):
    # Generate points and initialize the plot
    positions, velocities = generate_points(num_points)
    fig, ax = plt.subplots()
    ax.set_xlim(-BOUNDARY, BOUNDARY)
    ax.set_ylim(-BOUNDARY, BOUNDARY)
    scat = ax.scatter([], [], s=20)

    # Define the init function
    def init():
        scat.set_offsets(np.empty((0, 2)))
        return scat,

    # Update function for animation
    def update(frame):
        nonlocal positions, velocities
        update_positions(positions, velocities)

        # Calculate color based on speed
        speeds = np.linalg.norm(velocities, axis=1)
        colors = calculate_color(speeds)

        scat.set_offsets(positions)
        scat.set_color(colors)
        return scat,

    # Create and save the animation as GIF
    ani = animation.FuncAnimation(fig, update, frames=200, init_func=init, interval=1000 / FPS, blit=True)

    filename = "simulate.gif"
    ani.save(filename, writer="pillow", fps=FPS)
    plt.close(fig)
    return filename
# FastAPI endpoint to return the animated plot
@app.get("/simulate")
async def get_plot(num_points: int = 100):
    gif_file = plot_simulation(num_points)
    return StreamingResponse(open(gif_file, "rb"), media_type="image/gif")