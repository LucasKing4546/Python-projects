from fastapi import FastAPI
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from fastapi.responses import StreamingResponse
import numpy as np
from io import BytesIO
import tempfile
import os

app = FastAPI()

# Simulation parameters
BOUNDARY = 10
COLLISION_DAMPING = 0.9
FPS = 30
FIXED_Z = 7.5  # Fixed Z coordinate for all points
MIN_SEPARATION = 0.7  # Minimum separation distance between points
RAINDROP_FALL_SPEED = 0.1  # Speed at which raindrops fall in each frame

# Color range for raindrops (from dark blue to light blue)
DARK_BLUE = np.array([0, 0, 139]) / 255  # RGB for dark blue
LIGHT_BLUE = np.array([173, 216, 230]) / 255  # RGB for light blue

def calculate_color(speeds):
    max_speed = np.max(speeds)
    normalized_speeds = np.clip(speeds / max_speed, 0, 1) if max_speed > 0 else np.zeros_like(speeds)
    gray = np.array([90, 90, 90]) / 255
    white = np.array([255, 255, 255]) / 255
    return (normalized_speeds[:, None]) * gray + (1 - normalized_speeds[:, None]) * white

def generate_points(num_points):
    np.random.seed(42)
    positions = np.random.uniform(-BOUNDARY, BOUNDARY, (num_points, 3))
    velocities = np.random.normal(0, 0.2, (num_points, 3)) * np.linalg.norm(positions, axis=1).reshape(-1, 1)
    positions[:, 2] = FIXED_Z
    velocities[:, 2] = 0       # Ensure no movement along the z-axis
    return positions, velocities

def interpolate_color(z_pos, z_min, z_max):
    t = np.clip((z_pos - z_min) / (z_max - z_min), 0, 1)
    return t * DARK_BLUE + (1 - t) * LIGHT_BLUE

def update_positions(positions, velocities, raindrops, lightning_strikes):
    positions[:, :2] += velocities[:, :2] / FPS  # Update only x and y components

    # Handle boundary collisions for x and y only
    for i in range(len(positions)):
        for dim in range(2):  # Only update x and y, skip z
            if abs(positions[i, dim]) >= BOUNDARY:
                velocities[i, dim] *= -COLLISION_DAMPING

    import random

    # Handle collisions between points
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            delta_pos = positions[j, :2] - positions[i, :2]  # Only consider x and y for distance
            dist = np.linalg.norm(delta_pos)
            if dist < MIN_SEPARATION and dist != 0:
                overlap = MIN_SEPARATION - dist

                correction_vector = (delta_pos / dist) * (overlap / 2)
                positions[i, :2] -= correction_vector
                positions[j, :2] += correction_vector

                # Adjust velocities for collision, with slight randomness to prevent exact oscillation
                collision_normal = delta_pos / dist
                v_i_normal = np.dot(velocities[i, :2], collision_normal)
                v_j_normal = np.dot(velocities[j, :2], collision_normal)
                random_offset = np.random.uniform(-0.1, 0.1, size=2)  # Small random factor
                velocities[i, :2] += (v_j_normal - v_i_normal) * collision_normal + random_offset
                velocities[j, :2] += (v_i_normal - v_j_normal) * collision_normal - random_offset

                # Register raindrop at the collision position
                raindrops.append({"position": positions[i].copy()})
                if random.randint(0, 9) % 4 == 0:
                    # Trigger lightning effect by storing start and end points
                    lightning_strikes.append({"start": positions[i].copy(), "end": positions[j].copy()})

def plot_simulation(num_points):
    positions, velocities = generate_points(num_points)
    raindrops = []
    lightning_strikes = []
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-BOUNDARY, BOUNDARY)
    ax.set_ylim(-BOUNDARY, BOUNDARY)
    ax.set_zlim(-BOUNDARY, BOUNDARY)
    ax.set_box_aspect([1, 1, 1])

    # Initialize the scatter plot for cloud positions and raindrops
    scat = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=100)
    rain_scat = ax.scatter([], [], [], s=50)  # Empty scatter for raindrops, color updated dynamically

    def init():
        scat._offsets3d = ([], [], [])
        rain_scat._offsets3d = ([], [], [])
        return scat, rain_scat

    import random

    def generate_lightning_path_to_base(start, num_segments=10, max_offset=1):
        """
        Generate a zig-zag lightning path from the start point to the base (z = -BOUNDARY).
        """
        end = np.array([start[0], start[1], -BOUNDARY])  # Set the end point at the base (z = -BOUNDARY)
        points = [start]
        for i in range(1, num_segments):
            t = i / num_segments
            intermediate = (1 - t) * start + t * end
            offset = np.array([
                random.uniform(-max_offset, max_offset),
                random.uniform(-max_offset, max_offset),
                0
            ])
            points.append(intermediate + offset)
        points.append(end)
        return np.array(points)

    def update(frame):
        nonlocal positions, velocities, raindrops, lightning_strikes

        # Clear previous lightning lines (if any exist)
        for line in ax.lines:
            line.remove()

        update_positions(positions, velocities, raindrops, lightning_strikes)
        speeds = np.linalg.norm(velocities, axis=1)
        colors = calculate_color(speeds)

        scat._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])
        scat.set_color(colors)

        # Handle lightning strikes (from collision point to base)
        if lightning_strikes:
            for strike in lightning_strikes:
                path = generate_lightning_path_to_base(strike["start"])
                ax.plot(path[:, 0], path[:, 1], path[:, 2], color="yellow", linewidth=2)
            # Clear strikes after drawing them
            lightning_strikes.clear()

        # Update raindrop positions, colors, and make them fall
        rain_x, rain_y, rain_z = [], [], []
        rain_colors = []
        active_raindrops = []
        for drop in raindrops:
            drop["position"][1] -= RAINDROP_FALL_SPEED # Move raindrop to the side to simulate wind effect
            drop["position"][2] -= RAINDROP_FALL_SPEED  # Move raindrop downwards
            if drop["position"][2] > -BOUNDARY:  # Only keep if still in bounds
                rain_x.append(drop["position"][0])
                rain_y.append(drop["position"][1])
                rain_z.append(drop["position"][2])
                rain_colors.append(interpolate_color(drop["position"][2], -BOUNDARY, FIXED_Z))
                active_raindrops.append(drop)
        raindrops = active_raindrops  # Update list to only include active raindrops
        rain_scat._offsets3d = (rain_x, rain_y, rain_z)
        rain_scat.set_color(rain_colors)  # Set the colors of the raindrops

        return scat, rain_scat

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=200,
        init_func=init,
        interval=1000 / FPS,
        blit=True
    )

    with tempfile.NamedTemporaryFile(delete=False, suffix=".gif") as temp_file:
        ani.save(temp_file.name, writer="pillow", fps=FPS)
        temp_file.seek(0)
        gif_data = temp_file.read()

    os.unlink(temp_file.name)
    plt.close(fig)

    return BytesIO(gif_data)

@app.get("/simulate")
async def simulate(num_points: int = 100):
    gif = plot_simulation(num_points)
    return StreamingResponse(gif, media_type="image/gif")
