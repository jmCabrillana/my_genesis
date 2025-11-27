import genesis as gs
import numpy as np
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

gs.init(backend=gs.gpu)

scene = gs.Scene(
    show_viewer=False,
    viewer_options=gs.options.ViewerOptions(
        res=(1280, 960),
        camera_pos=(3.5, 0.0, 2.5),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        world_frame_size=1.0,
        show_link_frame=False,
        show_cameras=False,
        plane_reflection=True,
        ambient_light=(0.1, 0.1, 0.1),
    ),
    renderer=gs.renderers.BatchRenderer(use_rasterizer=False), # gs.renderers.Rasterizer() gs.renderers.BatchRenderer(use_rasterizer=True/False)
)

plane = scene.add_entity(gs.morphs.Plane())
franka = scene.add_entity(
    gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"),
)

cam = scene.add_camera(
    res=(640, 480),
    pos=(3.5, 0.0, 2.5),
    lookat=(0, 0, 0.5),
    fov=30,
    GUI=False,
)

scene.build()

frames = []

for i in range(120):
    scene.step()
    cam.set_pose(
        pos=(3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
        lookat=(0, 0, 0.5),
    )

    rgb, _, _, _ = cam.render(rgb=True)  # BatchRenderer output
    rgb = np.asarray(rgb.cpu())

    # If BatchRenderer gives batched frames (N, H, W, C), pick env 0
    if rgb.ndim == 4:
        rgb = rgb[0]

    # If ever grayscale (H, W), expand to 3 channels
    if rgb.ndim == 2:
        rgb = np.stack([rgb] * 3, axis=-1)

    frames.append(rgb)

# Save video with MoviePy
clip = ImageSequenceClip(frames, fps=60)
clip.write_videofile("runs/video.mp4", codec="libx264")
