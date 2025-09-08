import torch
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from scipy.spatial.transform import Rotation

def quaternion_to_euler(quaternion: torch.Tensor) -> torch.Tensor:

    w, x, y, z = torch.unbind(quaternion, dim=quaternion.dim() - 1)

    euler_angles: torch.Tensor = torch.stack(
        (
            torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)),
            torch.asin(2.0 * (w * y - z * x)),
            torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)),
        ),
        dim=-1,
    )

    return euler_angles


def euler_to_quaternion(euler: torch.Tensor) -> torch.Tensor:
    euler = torch.as_tensor(euler)
    r, p, y = torch.unbind(euler, dim=-1)
    cy = torch.cos(y * 0.5)
    sy = torch.sin(y * 0.5)
    cp = torch.cos(p * 0.5)
    sp = torch.sin(p * 0.5)
    cr = torch.cos(r * 0.5)
    sr = torch.sin(r * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy

    quaternion = torch.stack([qw, qx, qy, qz], dim=-1)

    return quaternion



def lemniscate(t, c):
    sin_t = torch.sin(t)
    cos_t = torch.cos(t)
    sin2p1 = torch.square(sin_t) + 1

    x = torch.stack([
        cos_t, sin_t * cos_t, c * sin_t
    ], dim=-1) / sin2p1.unsqueeze(-1)

    return x

def scale_time(t, a: float=1.0):
    return t / (1 + 1/(a*torch.abs(t)))

def quat_rotate(q: torch.Tensor, v: torch.Tensor):
    shape = q.shape
    q_w = q[:, 0]
    q_vec = q[:, 1:]
    a = v * (2.0 * q_w ** 2 - 1.0).unsqueeze(-1)
    b = torch.cross(q_vec, v, dim=-1) * q_w.unsqueeze(-1) * 2.0
    c = q_vec * torch.bmm(q_vec.view(shape[0], 1, 3), v.view(shape[0], 3, 1)).squeeze(-1) * 2.0
    return a + b + c

def rotation_matrix_to_quaternion(m: torch.Tensor) -> torch.Tensor:
    """
    Convert batch of 3x3 rotation matrices to quaternions [w, x, y, z]
    """
    r = m
    q = torch.zeros((r.shape[0], 4), device=r.device)
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    positive = trace > 0

    # Trace positive
    s = torch.sqrt(trace[positive] + 1.0) * 2
    q[positive, 0] = 0.25 * s
    q[positive, 1] = (r[positive, 2, 1] - r[positive, 1, 2]) / s
    q[positive, 2] = (r[positive, 0, 2] - r[positive, 2, 0]) / s
    q[positive, 3] = (r[positive, 1, 0] - r[positive, 0, 1]) / s

    # Trace non-positive
    negative = ~positive
    r_neg = r[negative]
    diagonals = torch.stack([r_neg[:, 0, 0], r_neg[:, 1, 1], r_neg[:, 2, 2]], dim=1)
    i = torch.argmax(diagonals, dim=1)

    for j in range(3):
        idx = (i == j).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        selected = r_neg[idx]
        s = torch.sqrt(1.0 + selected[:, j, j] - selected[:, (j+1)%3, (j+1)%3] - selected[:, (j+2)%3, (j+2)%3]) * 2
        q_idx = torch.where(negative)[0][idx]
        q[q_idx, (j+1)%4] = 0.25 * s
        q[q_idx, 0] = (selected[:, (j+2)%3, (j+1)%3] - selected[:, (j+1)%3, (j+2)%3]) / s
        q[q_idx, (j+2)%4] = (selected[:, j, (j+1)%3] + selected[:, (j+1)%3, j]) / s
        q[q_idx, (j+3)%4] = (selected[:, j, (j+2)%3] + selected[:, (j+2)%3, j]) / s

    return q


def spawn_racing_gate(
    prim_path: str,
    center=(0.0, 0.0, 1.5),
    inner_size=(1.5, 1.2),
    bar_thickness=0.05,
    depth=0.05,
    color=(0.0, 0.5, 1.0),
    kinematic=True,
    collision=True,
    # This now correctly accepts Euler angles in radians (roll, pitch, yaw)
    rotation_euler_xyz=(0.0, 0.0, 0.0),
):
    # 1. Convert Euler angles to a quaternion
    # Scipy creates a quaternion in (x, y, z, w) format
    rot_quat_xyzw = Rotation.from_euler('xyz', rotation_euler_xyz, degrees=False).as_quat()
    # Isaac Sim expects quaternion in (w, x, y, z) format, so we reorder it
    rot_quat_wxyz = (rot_quat_xyzw[3], rot_quat_xyzw[0], rot_quat_xyzw[1], rot_quat_xyzw[2])

    # 2. Create the parent Xform with the correct 'orientation' parameter
    prim_utils.create_prim(prim_path, "Xform", translation=center, orientation=rot_quat_wxyz)

    w, h = inner_size
    
    # materials / physics
    mat = sim_utils.PreviewSurfaceCfg(diffuse_color=color)
    rigid = sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=kinematic) if collision else None
    coll  = sim_utils.CollisionPropertiesCfg() if collision else None

    def bar(name, size_xyz, pos_xyz):
        cfg = sim_utils.CuboidCfg(
            size=size_xyz,
            visual_material=mat,
            rigid_props=rigid,
            collision_props=coll,
        )
        cfg.func(f"{prim_path}/{name}", cfg, translation=pos_xyz)

    half_w = w * 0.5
    half_h = h * 0.5

    # The rest of the function is the same...
    # vertical bars
    bar("left",  (depth, bar_thickness, h), (0.0, -half_w, 0.0))
    bar("right", (depth, bar_thickness, h), (0.0, half_w, 0.0))

    # horizontal bars
    bar("top",    (depth, w, bar_thickness), (0.0, 0.0, half_h))
    bar("bottom", (depth, w, bar_thickness), (0.0, 0.0, -half_h))
