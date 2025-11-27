import math
import genesis as gs
import torch


# ---------- Differentiable pinhole camera ----------

def look_at(eye, target, up):
    """
    Build a rotation matrix R and translation t such that:
        X_cam = R @ X_world + t
    eye, target, up are 3D torch tensors.
    """
    forward = (target - eye)
    forward = forward / forward.norm()

    right = torch.cross(forward, up)
    right = right / right.norm()

    up_cam = torch.cross(right, forward)

    # Camera basis in world coordinates
    R = torch.stack([right, up_cam, -forward], dim=0)  # 3x3
    t = -R @ eye
    return R, t


def project_points(X_world, R, t, fx, fy, cx, cy):
    """
    X_world: (..., 3) points in world coords (torch or genesis.Tensor).
    R: 3x3, t: 3, intrinsics: fx, fy, cx, cy scalars.
    Returns: (u, v) in pixel coords, same leading dims as X_world[..., 0]
    """
    # flatten
    orig_shape = X_world.shape[:-1]
    Xw = X_world.reshape(-1, 3).T  # 3 x N

    Xc = R @ Xw + t.unsqueeze(1)   # 3 x N
    x = Xc[0]
    y = Xc[1]
    z = Xc[2].clamp(min=1e-4)      # avoid div by zero

    u = fx * x / z + cx
    v = fy * y / z + cy

    u = u.reshape(orig_shape)
    v = v.reshape(orig_shape)
    return u, v


# ---------- Genesis setup ----------

gs.init(
    seed=0,
    backend=gs.gpu,       
    precision="32",
    logging_level="info",
)

scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=2e-3,
        substeps=10,
        gravity=(0.0, 0.0, -9.81),
        requires_grad=True,   # <- crucial for differentiable physics
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-1.0, -1.0, 0.0),
        upper_bound=(1.0, 1.0, 1.5),
        grid_density=80,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, 0.0, 1.0),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        visualize_mpm_boundary=False,
    ),
    renderer=gs.renderers.Rasterizer()  , # gs.renderers.BatchRenderer(use_rasterizer=True),
    show_viewer=False,
)

# Ground
scene.add_entity(morph=gs.morphs.Plane())

# ---------- Soft "humanoid" agent (body) ----------

agent_height = 0.8
agent_half_width = 0.12
agent_depth = 0.12

agent = scene.add_entity(
    material=gs.materials.MPM.Elastic(
        E=8e5,      # fairly stiff to approximate "rigid-ish"
        nu=0.3,
        rho=500.0,
    ),
    morph=gs.morphs.Box(
        lower=(-agent_half_width, -agent_depth * 0.5, 0.0),
        upper=(+agent_half_width, +agent_depth * 0.5, agent_height),
    ),
    surface=gs.surfaces.Default(
        color=(0.8, 0.8, 1.0, 1.0),
    ),
    vis_mode="particle",
)

# ---------- Ball to find & grasp ----------

ball = scene.add_entity(
    material=gs.materials.MPM.Elastic(
        E=1e5,
        nu=0.3,
        rho=300.0,
    ),
    morph=gs.morphs.Sphere(
        pos=(0.6, 0.0, 0.1),   # in front of the agent
        radius=0.08,
    ),
    surface=gs.surfaces.Default(
        color=(1.0, 0.5, 0.5, 1.0),
    ),
    vis_mode="particle",
)

# ---------- Genesis camera (for non-diff video only) ----------

cam = scene.add_camera(
    res=(640, 480),
    pos=(1.6, 0.0, 0.9),
    lookat=(0.3, 0.0, 0.4),
    fov=40,
    GUI=False,
)

scene.build()

# ---------- Identify "foot" and "hand" regions ----------

scene.reset()
with torch.no_grad():
    s0 = agent.get_state()
    pos0 = s0.pos  # [N_particles, 3], genesis.Tensor (torch subclass)
    z_min = pos0[:, 2].min()
    z_max = pos0[:, 2].max()

    # bottom ~20% height = "feet"
    foot_thresh = z_min + 0.2 * (z_max - z_min)
    foot_mask = pos0[:, 2] < foot_thresh
    foot_idx = torch.nonzero(foot_mask).squeeze(1)

    # top ~20% height = "hand"
    hand_thresh = z_min + 0.8 * (z_max - z_min)
    hand_mask = pos0[:, 2] > hand_thresh
    hand_idx = torch.nonzero(hand_mask).squeeze(1)

print(f"Foot particles: {foot_idx.numel()}, Hand particles: {hand_idx.numel()}")

device = pos0.device

# ---------- Differentiable camera parameters (for loss) ----------

img_h, img_w = 128, 128
fov_deg = 40.0
fov_rad = math.radians(fov_deg)

# Simple pinhole intrinsics
fx = fy = (img_w / 2.0) / math.tan(fov_rad / 2.0)
cx = img_w / 2.0
cy = img_h / 2.0

fx = torch.tensor(fx, device=device)
fy = torch.tensor(fy, device=device)
cx = torch.tensor(cx, device=device)
cy = torch.tensor(cy, device=device)

# Camera pose in world (eye/target/up) — roughly matches the Genesis cam
eye = torch.tensor([1.6, 0.0, 0.9], device=device)
target = torch.tensor([0.3, 0.0, 0.4], device=device)
up = torch.tensor([0.0, 0.0, 1.0], device=device)

R_cam, t_cam = look_at(eye, target, up)  # 3x3, 3

# ---------- Differentiable control: walk + reach ----------

horizon = 200
n_iters = 30

# Two controls per step:
#  u_walk[t]: pushes "feet" in +x
#  u_reach[t]: stretches "hand" region in +x
u_walk_seq = [gs.tensor([0.0], requires_grad=True, device=device) for _ in range(horizon)]
u_reach_seq = [gs.tensor([0.0], requires_grad=True, device=device) for _ in range(horizon)]

optimizer = torch.optim.Adam(u_walk_seq + u_reach_seq, lr=0.3)

for it in range(n_iters):
    scene.reset()

    # rollout
    for t in range(horizon):
        u_walk = u_walk_seq[t]
        u_reach = u_reach_seq[t]

        vel = torch.zeros((agent.n_particles, 3), device=device)
        vel[foot_idx, 0] = u_walk     # walk forward
        vel[hand_idx, 0] += u_reach   # reach forward

        agent.set_velocity(vel)
        scene.step()

    # final state
    agent_state = agent.get_state()
    ball_state = ball.get_state()

    agent_pos = agent_state.pos       # [N_agent, 3]
    ball_pos = ball_state.pos         # [N_ball, 3]

    hand_pos = agent_pos[hand_idx]    # [N_hand, 3]
    hand_com = hand_pos.mean(dim=0)   # [3]

    ball_com = ball_pos.mean(dim=0)   # [3]

    # project both to image plane (differentiably)
    hand_u, hand_v = project_points(hand_com, R_cam, t_cam, fx, fy, cx, cy)
    ball_u, ball_v = project_points(ball_com, R_cam, t_cam, fx, fy, cx, cy)

    # loss 1: bring hand and ball together in image space
    img_dist = (hand_u - ball_u) ** 2 + (hand_v - ball_v) ** 2

    # loss 2: keep hand & ball near image center (so agent "finds" ball in its view)
    center_u = cx
    center_v = cy
    center_dist_ball = (ball_u - center_u) ** 2 + (ball_v - center_v) ** 2
    center_dist_hand = (hand_u - center_u) ** 2 + (hand_v - center_v) ** 2

    # small regularization: keep body COM near x=0
    body_com = agent_pos.mean(dim=0)
    body_x = body_com[0]
    reg_body = 0.01 * body_x**2

    loss = img_dist + 0.1 * (center_dist_ball + center_dist_hand) + reg_body

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f"[iter {it:02d}] loss={loss.item():.4f}, "
        f"img_dist={img_dist.item():.4f}, body_x={body_x.item():.3f}"
    )

print("Optimization done. Running evaluation + recording...")

# ---------- Evaluation rollout + RGB video (non-diff) ----------

scene.reset()
cam.start_recording()

for t in range(horizon):
    u_walk = u_walk_seq[t].detach()
    u_reach = u_reach_seq[t].detach()

    vel = torch.zeros((agent.n_particles, 3), device=device)
    vel[foot_idx, 0] = u_walk
    vel[hand_idx, 0] += u_reach

    agent.set_velocity(vel)
    scene.step()
    cam.render(rgb=True)

cam.stop_recording(save_to_filename="humanoid_find_grasp_ball.mp4", fps=int(1.0 / scene.dt))

print("Saved video to humanoid_find_grasp_ball.mp4")
