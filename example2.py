from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import os,json,shutil,viser,time,random,numpy as np

def copy_images():
    # Load and preprocess example images (replace with your own image paths)
    folder = "/cpfs/user/guowenqi/dataset/co3dv2/apple/110_13051_23361/images"
    image_path = [os.path.join(folder,file) for file in os.listdir(folder) if file.endswith('.jpg')]
    image_path.sort()
    image_path = image_path[::10]
    # json_file = "/cpfs/user/guowenqi/dataset/co3dv2/apple/selected_seqs_test.json"
    # with open(json_file, 'r') as f:
    #     data = json.load(f)
    #     data = data["110_13051_23361"]
    # image_path = []
    # for id in data: 
    #     fname = f"frame{id:06d}.jpg"
    #     image_path.append(os.path.join(folder, fname))
    new_folder = "/cpfs/user/guowenqi/dataset/co3dv2/apple/110_13051_23361/images2/"
    os.makedirs(new_folder, exist_ok=True)
    for path in image_path:
        shutil.copy(path, new_folder)
        print(f"Copied {path} to {new_folder}")

def viser_example1():

    server = viser.ViserServer()
    server.scene.add_icosphere(
        name="/hello_sphere",
        radius=0.5,
        color=(255, 0, 0),  # Red
        position=(0.0, 0.0, 0.0),
    )

    print("Open your browser to http://localhost:8080")
    print("Press Ctrl+C to exit")

    while True:
        time.sleep(10.0)

def viser_example2():

    server = viser.ViserServer()

    # Add 3D objects to the scene
    sphere = server.scene.add_icosphere(
        name="/sphere",
        radius=0.3,
        color=(255, 100, 100),
        position=(0.0, 0.0, 0.0),
    )
    box = server.scene.add_box(
        name="/box",
        dimensions=(0.4, 0.4, 0.4),
        color=(100, 255, 100),
        position=(1.0, 0.0, 0.0),
    )

    # Create GUI controls
    sphere_visible = server.gui.add_checkbox("Show sphere", initial_value=True)
    sphere_color = server.gui.add_rgb("Sphere color", initial_value=(255, 100, 100))
    box_height = server.gui.add_slider(
        "Box height", min=-1.0, max=1.0, step=0.1, initial_value=0.0
    )

    # Connect GUI controls to scene objects
    @sphere_visible.on_update
    def _(_):
        sphere.visible = sphere_visible.value

    @sphere_color.on_update
    def _(_):
        sphere.color = sphere_color.value

    @box_height.on_update
    def _(_):
        box.position = (1.0, 0.0, box_height.value)

    print("Server running")
    while True:
        time.sleep(10.0)

def viser_example3():
    
    server = viser.ViserServer()

    while True:
        # Add some coordinate frames to the scene. These will be visualized in the viewer.
        server.scene.add_frame(
            "/tree",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        )
        server.scene.add_frame(
            "/tree/branch",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        )
        leaf = server.scene.add_frame(
            "/tree/branch/leaf",
            wxyz=(1.0, 0.0, 0.0, 0.0),
            position=(random.random() * 2.0, 2.0, 0.2),
        )

        # Move the leaf randomly. Assigned properties are automatically updated in
        # the visualizer.
        for i in range(10):
            leaf.position = (random.random() * 2.0, 2.0, 0.2)
            time.sleep(0.5)

        # Remove the leaf node from the scene.
        leaf.remove()
        time.sleep(0.5)

def viser_example4():
    server = viser.ViserServer()

    # Generate a spiral point cloud.
    num_points = 200
    t = np.linspace(0, 10, num_points)
    spiral_positions = np.column_stack(
        [
            np.sin(t) * (1 + t / 10),
            np.cos(t) * (1 + t / 10),
            t / 5,
        ]
    )

    # Create colors based on height (z-coordinate).
    z_min, z_max = spiral_positions[:, 2].min(), spiral_positions[:, 2].max()
    normalized_z = (spiral_positions[:, 2] - z_min) / (z_max - z_min)

    # Color gradient from blue (bottom) to red (top).
    colors = np.zeros((num_points, 3), dtype=np.uint8)
    colors[:, 0] = (normalized_z * 255).astype(np.uint8)  # Red channel.
    colors[:, 2] = ((1 - normalized_z) * 255).astype(np.uint8)  # Blue channel.

    # Add the point cloud to the scene.
    server.scene.add_point_cloud(
        name="/spiral_cloud",
        points=spiral_positions,
        colors=colors,
        point_size=0.05,
    )

    # Add a second point cloud - random noise points.
    num_noise_points = 500
    noise_positions = np.random.normal(0, 1, (num_noise_points, 3))
    noise_colors = np.random.randint(0, 255, (num_noise_points, 3), dtype=np.uint8)

    server.scene.add_point_cloud(
        name="/noise_cloud",
        points=noise_positions,
        colors=noise_colors,
        point_size=0.03,
    )

    print("Point cloud visualization loaded!")
    print("- Spiral point cloud with height-based colors")
    print("- Random noise point cloud with random colors")
    print("Visit: http://localhost:8080")

    while True:
        pass

def viser_example5():
    server = viser.ViserServer()
    server.scene.world_axes.visible = True


    @server.on_client_connect
    def _(client: viser.ClientHandle) -> None:
        print("new client!")

        # This will run whenever we get a new camera!
        @client.camera.on_update
        def _(_: viser.CameraHandle) -> None:
            print(f"New camera on client {client.client_id}!")

        # Show the client ID in the GUI.
        gui_info = client.gui.add_text("Client ID", initial_value=str(client.client_id))
        gui_info.disabled = True


    while True:
        # Get all currently connected clients.
        clients = server.get_clients()
        print("Connected client IDs", clients.keys())

        for id, client in clients.items():
            print(f"Camera pose for client {id}")
            print(f"\twxyz: {client.camera.wxyz}")
            print(f"\tposition: {client.camera.position}")
            print(f"\tfov: {client.camera.fov}")
            print(f"\taspect: {client.camera.aspect}")
            print(f"\tlast update: {client.camera.update_timestamp}")
            print(
                f"\tcanvas size: {client.camera.image_width}x{client.camera.image_height}"
            )

        time.sleep(2.0)

def video2img(path,frame_interval=5,start=None,end=None):
    import cv2,os
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file {p}")
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    start = max(0,start) if start is not None else 0
    end = min(end,total_frames) if end is not None else total_frames
    if video_fps == 0:
        cap.release()
        raise ValueError(f"Error: Video FPS is 0 for {p}")
    
    framerate = video_fps / frame_interval
    
    frame_indices = list(range(start, end, frame_interval))
    print(
        f" - Video FPS: {video_fps}, Frame Interval: {frame_interval}, Total Frames to Read: {len(frame_indices)}, Processed Framerate: {framerate}"
    )
    img_paths = []
    scene = os.path.basename(path).split('.')[0]
    new_folder=f'./examples/{scene}'
    os.makedirs(new_folder)
    for index,i in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(new_folder, f"frame_{index}.jpg")
        cv2.imwrite(frame_path, frame)
        img_paths.append(frame_path)
    cap.release()

def sample_image(folder,frame_interval=15):
    import re
    paths = os.listdir(folder)
    paths = [path for path in paths if path.endswith((".jpg","png"))]
    paths = sorted(paths, key=lambda s: int(re.search(r"(\d+)", s).group(1)))
    #paths.sort()
    sampled_paths = paths[::frame_interval]
    new_folder = os.path.join(os.path.dirname(os.path.normpath(folder)) , f"_sampled_{frame_interval}")
    os.makedirs(new_folder, exist_ok=True)
    for path in sampled_paths:
        shutil.copy(os.path.join(folder, path), os.path.join(new_folder, path))
        print(f"Copied {path} to {new_folder}")

if __name__ == "__main__":
    video2img("/cpfs/user/guowenqi/TTT3R/examples/taylor.mp4",frame_interval=10,start=400,end=1000)
