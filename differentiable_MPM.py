import genesis as gs
import torch
import math

########################## init ##########################
gs.init(
    seed=0,
    precision="32",
    logging_level="info",
    backend=gs.gpu,  
)

########################## scene ##########################
scene = gs.Scene(
    sim_options=gs.options.SimOptions(
        dt=2e-3,
        substeps=10,
        gravity=(0.0, 0.0, -9.81),
        requires_grad=True,   # <- turn on differentiable mode
    ),
    mpm_options=gs.options.MPMOptions(
        lower_bound=(-0.5, -0.5, -0.1),
        upper_bound=(0.5, 0.5, 1.5),
        grid_density=64,
    ),
    viewer_options=gs.options.ViewerOptions(
        camera_pos=(2.0, 0.0, 1.2),
        camera_lookat=(0.0, 0.0, 0.5),
        camera_fov=40,
        max_FPS=60,
    ),
    vis_options=gs.options.VisOptions(
        show_world_frame=True,
        visualize_mpm_boundary=False,
    ),
    renderer=gs.renderers.Rasterizer(),  #gs.renderers.BatchRenderer(use_rasterizer=True),
    show_viewer=False,
)

########################## ground ##########################
scene.add_entity(morph=gs.morphs.Plane())

########################## “cartpole” as stiff MPM rod ##########################
# The idea: a tall slender box standing on the plane.
# Big stiffness to approximate rigidity.
pole_height = 1.0
cart_half_width = 0.15
depth = 0.1   # thin in y so dynamics are mostly in x–z plane

pole = scene.add_entity(
    material=gs.materials.MPM.Elastic(
        E=5e5,     # Young’s modulus – crank this up for “more rigid”
        nu=0.3,    # Poisson’s ratio
        rho=800.0, # density
    ),
    morph=gs.morphs.Box(
        lower=(-cart_half_width, -depth * 0.5, 0.0),
        upper=(+cart_half_width, +depth * 0.5, pole_height),
    ),
    surface=gs.surfaces.Default(
        color=(0.9, 0.9, 0.9, 1.0),
    ),
    vis_mode="particle",
)

cam = scene.add_camera(
    res=(640, 480),
    pos=(2.0, 0.0, 1.2),
    lookat=(0.0, 0.0, 0.5),
    fov=40,
    GUI=False,
)

########################## build ##########################
scene.build()

########################## identify “cart” particles (bottom slab) ##########################
# We’ll treat the lowest ~10% of particles (in z) as the cart region
# and directly set their horizontal velocity as control.
scene.reset()
with torch.no_grad():
    s0 = pole.get_state()
    pos0 = s0.pos    # genesis tensor ~ torch.Tensor
    z_min = pos0[:, 2].min()
    z_max = pos0[:, 2].max()
    base_thresh = z_min + 0.1 * (z_max - z_min)
    base_mask = pos0[:, 2] < base_thresh
    base_idx = torch.nonzero(base_mask).squeeze(1)

print(f"Using {base_idx.numel()} particles as 'cart' region.")

########################## differentiable control setup ##########################
horizon = 200         # time steps
n_iters = 20          # optimization iterations
device = pos0.device

# control sequence: one scalar horizontal velocity for the cart region at each step
v_seq = [gs.tensor([0.0]) for _ in range(horizon)]

optimizer = torch.optim.Adam(v_seq, lr=0.5)

# target upright: angle = 0, cart near x=0 at final time
for it in range(n_iters):
    # reset sim
    scene.reset()

    # rollout
    for t in range(horizon):
        v_t = v_seq[t]  # shape [1], scalar vel in x

        # build velocity field for this step: zero everywhere,
        # except cart region gets v_t in x direction.
        vel = torch.zeros((pole.n_particles, 3), device=device)
        vel[base_idx, 0] = v_t
        pole.set_velocity(vel)   # this writes into the MPM solver field
        scene.step()

    # compute loss at the end of rollout
    state = pole.get_state()
    pos = state.pos  # [B, N, 3]

    # center of mass per environment
    com = pos.mean(dim=1)  # [B, 3]

    # angle in x–z plane per environment
    angle = torch.atan2(com[:, 0], com[:, 2])  # [B]

    # "cart position" per environment
    cart_x = com[:, 0]  # [B]

    # scalar loss: average over batch
    loss = (angle**2 + 0.1 * cart_x**2).mean()  # shape []

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print(
        f"[iter {it:02d}] loss={loss.item():.6f}, "
        f"angle={angle.item():.3f} rad, cart_x={cart_x.item():.3f}"
    )

print("Optimization done.")

########################## optional: evaluate & record a video ##########################
# Run one rollout with the learned v_seq and record as MP4.
# This uses the same cam pattern you already have working.

scene.reset()
cam.start_recording()

for t in range(horizon):
    v_t = v_seq[t].detach()

    vel = torch.zeros((pole.n_particles, 3), device=device)
    vel[base_idx, 0] = v_t
    pole.set_velocity(vel)

    scene.step()
    cam.render(rgb=True)

cam.stop_recording(save_to_filename="runs/mpm_cartpole.mp4", fps=int(1.0 / scene.dt))

print("Saved video to mpm_cartpole.mp4")
