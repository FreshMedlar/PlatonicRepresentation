"""
GPU-Batched 3D Ray-Cast Tabletop Environment for Baby-JEPA.

Supports:
- Exact separation of intrinsic world state (shape, color, texture, 3D pose)
  from nuisance variables (directional light angle/color, camera orbit/zoom, table shade).
- Deliberate confounded initial views ("apple stays red" & silhouette ambiguity)
  that require active disambiguating actions (turning/flipping objects, shifting lighting,
  orbiting camera) to decorrelate.
- Sparse grounded teacher emitting object concept words only when attended.
"""

import math
from typing import Dict, List, Optional, Tuple
import torch

from configs.baby_config import ACTIONS, COLORS, SHAPES, TEXTURES, TabletopEnvConfig


COLOR_RGB = torch.tensor(
    [
        [0.88, 0.18, 0.18],  # red
        [0.18, 0.78, 0.28],  # green
        [0.20, 0.42, 0.90],  # blue
        [0.92, 0.80, 0.18],  # yellow
    ],
    dtype=torch.float32,
)

LIGHT_TINTS = torch.tensor(
    [
        [1.00, 1.00, 1.00],  # neutral white
        [1.25, 0.65, 0.55],  # warm red-tinted spotlight
        [0.65, 1.15, 0.70],  # cool green-tinted spotlight
        [0.65, 0.80, 1.25],  # cool blue-tinted spotlight
    ],
    dtype=torch.float32,
)


def build_word_vocabulary() -> Tuple[Dict[str, int], List[str]]:
    """Builds a compact grounded vocabulary for shapes, colors, and textures."""
    words = ["<none>"] + list(SHAPES) + list(COLORS) + list(TEXTURES)
    word_to_id = {w: i for i, w in enumerate(words)}
    return word_to_id, words


def _rotation_matrix_yx(yaw: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    """Builds (B, 3, 3) rotation matrices R = R_y(yaw) @ R_x(pitch)."""
    cy, sy = torch.cos(yaw), torch.sin(yaw)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    zeros = torch.zeros_like(cy)
    ones = torch.ones_like(cy)

    ry = torch.stack(
        [
            torch.stack([cy, zeros, sy], dim=-1),
            torch.stack([zeros, ones, zeros], dim=-1),
            torch.stack([-sy, zeros, cy], dim=-1),
        ],
        dim=-2,
    )
    rx = torch.stack(
        [
            torch.stack([ones, zeros, zeros], dim=-1),
            torch.stack([zeros, cp, -sp], dim=-1),
            torch.stack([zeros, sp, cp], dim=-1),
        ],
        dim=-2,
    )
    return ry @ rx


class TabletopWorld:
    """
    Pure-PyTorch batched 3D ray-cast tabletop simulator.
    Renders (B, 3, H, W) RGB observations in [0, 1] directly on CPU or ROCm/CUDA GPU.
    """

    def __init__(
        self,
        config: TabletopEnvConfig,
        batch_size: int = 16,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.config = config
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.rng = torch.Generator(device="cpu").manual_seed(seed)

        self.word_to_id, self.id_to_word = build_word_vocabulary()
        self.color_palette = COLOR_RGB.to(self.device)
        self.light_tints = LIGHT_TINTS.to(self.device)

        # Precompute normalized image plane grid (H, W)
        H = W = config.image_size
        half_fov = math.tan(math.radians(config.fov_deg) * 0.5)
        ys = torch.linspace(half_fov, -half_fov, H, device=self.device)
        xs = torch.linspace(-half_fov, half_fov, W, device=self.device)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        self.grid_x = grid_x  # (H, W)
        self.grid_y = grid_y  # (H, W)

        # State tensors initialized on reset()
        self.shape_id = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.color_id = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.texture_id = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.obj_yaw = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.obj_pitch = torch.zeros(batch_size, dtype=torch.float32, device=self.device)

        self.cam_az = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.cam_el = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.cam_dist = torch.full((batch_size,), 2.35, dtype=torch.float32, device=self.device)

        self.light_az = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        self.light_tint_id = torch.zeros(batch_size, dtype=torch.long, device=self.device)
        self.table_shade = torch.full((batch_size,), 0.45, dtype=torch.float32, device=self.device)
        self.step_count = 0
        self.max_shapes = config.num_shapes

    def set_num_shapes(self, num_shapes: int):
        """Allows curriculum expansion of available 3D shapes."""
        self.max_shapes = max(1, min(num_shapes, len(SHAPES)))

    def _rand(self, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.rand(shape, generator=self.rng, dtype=torch.float32).to(self.device)

    def _randint(self, low: int, high: int, shape: Tuple[int, ...]) -> torch.Tensor:
        return torch.randint(low, high, shape, generator=self.rng, dtype=torch.long).to(self.device)

    def reset(
        self,
        confound_prob: Optional[float] = None,
        biased_view: Optional[bool] = None,
    ) -> Dict[str, torch.Tensor]:
        """Resets all B tabletop scenes, optionally with confounded priors and biased initial views."""
        B = self.batch_size
        c_prob = self.config.confound_prob if confound_prob is None else confound_prob
        use_bias = self.config.biased_initial_view if biased_view is None else biased_view

        self.shape_id = self._randint(0, self.max_shapes, (B,))
        self.texture_id = self._randint(0, self.config.num_textures, (B,))

        # Confounder: color initially correlates with shape unless broken by random draw
        random_color = self._randint(0, self.config.num_colors, (B,))
        coupled_color = self.shape_id % self.config.num_colors
        use_coupled = self._rand((B,)) < c_prob
        self.color_id = torch.where(use_coupled, coupled_color, random_color)

        if use_bias:
            # Front-biased view where cylinders look boxy, cones look round from high pitch,
            # and lighting tint reinforces the apparent color confounder.
            self.obj_yaw = (self._rand((B,)) - 0.5) * 0.20
            self.obj_pitch = (self._rand((B,)) - 0.5) * 0.16
            self.cam_az = (self._rand((B,)) - 0.5) * 0.25
            self.cam_el = torch.full((B,), 0.35, device=self.device) + (self._rand((B,)) - 0.5) * 0.10
            self.cam_dist = torch.full((B,), 2.35, device=self.device)
            self.light_az = torch.full((B,), 0.25, device=self.device)
            # Biased spotlight tint: matches shape-coupled tint with c_prob
            coupled_tint = ((self.shape_id % 3) + 1).clamp(max=len(LIGHT_TINTS) - 1)
            neutral_tint = torch.zeros(B, dtype=torch.long, device=self.device)
            self.light_tint_id = torch.where(use_coupled, coupled_tint, neutral_tint)
        else:
            self.obj_yaw = (self._rand((B,)) * 2.0 - 1.0) * math.pi
            self.obj_pitch = (self._rand((B,)) - 0.5) * 1.4
            self.cam_az = (self._rand((B,)) * 2.0 - 1.0) * math.pi
            self.cam_el = 0.22 + self._rand((B,)) * 0.95
            self.cam_dist = 1.85 + self._rand((B,)) * 1.25
            self.light_az = (self._rand((B,)) * 2.0 - 1.0) * math.pi
            self.light_tint_id = self._randint(0, len(LIGHT_TINTS), (B,))

        self.table_shade = 0.35 + self._rand((B,)) * 0.25
        self.step_count = 0
        return self.observe(emit_word=True)

    def step(self, actions: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Applies discrete motor/lighting/camera actions of shape (B,) and returns new observation.
        Action indices correspond to configs.baby_config.ACTIONS.
        """
        actions = actions.to(self.device).long()
        B = self.batch_size
        if actions.shape != (B,):
            raise ValueError(f"Expected actions shape ({B},), got {tuple(actions.shape)}")

        turn_delta = 0.65
        orbit_delta = 0.55

        self.obj_yaw = self.obj_yaw + torch.where(actions == 0, -turn_delta, 0.0)
        self.obj_yaw = self.obj_yaw + torch.where(actions == 1, turn_delta, 0.0)
        self.obj_pitch = (self.obj_pitch + torch.where(actions == 2, 0.85, 0.0)).clamp(-1.2, 1.2)

        self.cam_az = self.cam_az + torch.where(actions == 3, -orbit_delta, 0.0)
        self.cam_az = self.cam_az + torch.where(actions == 4, orbit_delta, 0.0)
        self.cam_el = (self.cam_el + torch.where(actions == 5, 0.25, 0.0)).clamp(0.18, 1.35)
        self.cam_el = (self.cam_el - torch.where(actions == 6, 0.25, 0.0)).clamp(0.18, 1.35)

        self.light_az = self.light_az + torch.where(actions == 7, 0.95, 0.0)
        shift_tint = actions == 8
        self.light_tint_id = torch.where(
            shift_tint,
            (self.light_tint_id + 1) % len(LIGHT_TINTS),
            self.light_tint_id,
        )

        self.cam_dist = (self.cam_dist - torch.where(actions == 9, 0.25, 0.0)).clamp(1.75, 3.25)
        self.cam_dist = (self.cam_dist + torch.where(actions == 10, 0.25, 0.0)).clamp(1.75, 3.25)

        # Wrap yaw/az into [-pi, pi]
        self.obj_yaw = torch.atan2(torch.sin(self.obj_yaw), torch.cos(self.obj_yaw))
        self.cam_az = torch.atan2(torch.sin(self.cam_az), torch.cos(self.cam_az))
        self.light_az = torch.atan2(torch.sin(self.light_az), torch.cos(self.light_az))

        self.step_count += 1
        return self.observe(emit_word=True)

    def _sample_teacher_words(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sparse grounded teacher: emits one concept word (shape, color, or texture)
        with probability teacher_word_prob; otherwise <none> (id=0).
        """
        B = self.batch_size
        emit_mask = self._rand((B,)) < self.config.teacher_word_prob

        # Choose concept category: 0=shape, 1=color, 2=texture
        cat = self._randint(0, 3, (B,))
        shape_offset = 1
        color_offset = 1 + len(SHAPES)
        tex_offset = 1 + len(SHAPES) + len(COLORS)

        word_id = torch.where(
            cat == 0,
            shape_offset + self.shape_id,
            torch.where(
                cat == 1,
                color_offset + self.color_id,
                tex_offset + self.texture_id,
            ),
        )
        word_id = torch.where(emit_mask, word_id, torch.zeros_like(word_id))
        return word_id, emit_mask

    @torch.no_grad()
    def render(self) -> torch.Tensor:
        """
        Renders current B scenes via analytical ray-casting on GPU/CPU.
        Returns tensor of shape (B, 3, H, W) in [0, 1].
        """
        B = self.batch_size
        H = W = self.config.image_size
        INF = 1e6

        # Camera position looking at origin (0, 0, 0)
        ce, se = torch.cos(self.cam_el), torch.sin(self.cam_el)
        ca, sa = torch.cos(self.cam_az), torch.sin(self.cam_az)
        cam_pos = torch.stack(
            [self.cam_dist * ce * sa, self.cam_dist * se, self.cam_dist * ce * ca],
            dim=-1,
        )  # (B, 3)

        forward = torch.nn.functional.normalize(-cam_pos, dim=-1)  # (B, 3)
        world_up = torch.tensor([0.0, 1.0, 0.0], device=self.device).expand(B, 3)
        right = torch.nn.functional.normalize(torch.linalg.cross(forward, world_up, dim=-1), dim=-1)
        up = torch.linalg.cross(right, forward, dim=-1)

        # Ray directions (B, H, W, 3)
        gx = self.grid_x.view(1, H, W, 1)
        gy = self.grid_y.view(1, H, W, 1)
        ray_dir = (
            forward.view(B, 1, 1, 3)
            + gx * right.view(B, 1, 1, 3)
            + gy * up.view(B, 1, 1, 3)
        )
        ray_dir = torch.nn.functional.normalize(ray_dir, dim=-1)

        # Transform rays into object local frame: p_local = R^T p_world
        rot = _rotation_matrix_yx(self.obj_yaw, self.obj_pitch)  # (B, 3, 3)
        rot_t = rot.transpose(-1, -2)  # (B, 3, 3)
        ro_local = torch.einsum("bij,bj->bi", rot_t, cam_pos).view(B, 1, 1, 3).expand(B, H, W, 3)
        rd_local = torch.einsum("bij,bhwj->bhwi", rot_t, ray_dir)

        t_obj = torch.full((B, H, W), INF, device=self.device)
        n_local = torch.zeros((B, H, W, 3), device=self.device)

        # 0: Sphere (r = 0.42)
        m_sph = (self.shape_id == 0).view(B, 1, 1)
        if m_sph.any():
            t_s, n_s = self._intersect_sphere(ro_local, rd_local, radius=0.42)
            t_obj = torch.where(m_sph, t_s, t_obj)
            n_local = torch.where(m_sph.unsqueeze(-1), n_s, n_local)

        # 1: Cube (half_extent = 0.36)
        m_box = (self.shape_id == 1).view(B, 1, 1)
        if m_box.any():
            t_b, n_b = self._intersect_box(ro_local, rd_local, half_ext=0.36)
            t_obj = torch.where(m_box, t_b, t_obj)
            n_local = torch.where(m_box.unsqueeze(-1), n_b, n_local)

        # 2: Cylinder (r = 0.34, half_h = 0.40)
        m_cyl = (self.shape_id == 2).view(B, 1, 1)
        if m_cyl.any():
            t_c, n_c = self._intersect_cylinder(ro_local, rd_local, radius=0.34, half_h=0.40)
            t_obj = torch.where(m_cyl, t_c, t_obj)
            n_local = torch.where(m_cyl.unsqueeze(-1), n_c, n_local)

        # 3: Cone (base_r = 0.42, y in [-0.40, 0.42])
        m_cone = (self.shape_id == 3).view(B, 1, 1)
        if m_cone.any():
            t_k, n_k = self._intersect_cone(ro_local, rd_local, base_r=0.42, y_min=-0.40, y_max=0.42)
            t_obj = torch.where(m_cone, t_k, t_obj)
            n_local = torch.where(m_cone.unsqueeze(-1), n_k, n_local)

        # 4: Capsule (r = 0.28, half_seg = 0.24)
        m_cap = (self.shape_id == 4).view(B, 1, 1)
        if m_cap.any():
            t_p, n_p = self._intersect_capsule(ro_local, rd_local, radius=0.28, half_seg=0.24)
            t_obj = torch.where(m_cap, t_p, t_obj)
            n_local = torch.where(m_cap.unsqueeze(-1), n_p, n_local)

        # World-space hit position and normal for object
        p_local_hit = ro_local + t_obj.unsqueeze(-1) * rd_local
        n_world = torch.einsum("bij,bhwj->bhwi", rot, n_local)
        n_world = torch.nn.functional.normalize(n_world, dim=-1)

        # Tabletop plane at y = -0.42
        cam_y = cam_pos[:, 1].view(B, 1, 1)
        rd_y = ray_dir[..., 1]
        t_plane = (-0.42 - cam_y) / rd_y.clamp(max=-1e-5)
        plane_valid = (rd_y < -1e-4) & (t_plane > 0.0) & (t_plane < 12.0)
        t_plane = torch.where(plane_valid, t_plane, torch.full_like(t_plane, INF))

        hit_obj = t_obj < t_plane
        hit_table = (~hit_obj) & (t_plane < INF)

        # Lighting setup
        light_dir = torch.stack(
            [
                0.70 * torch.sin(self.light_az),
                torch.full((B,), 0.85, device=self.device),
                0.70 * torch.cos(self.light_az),
            ],
            dim=-1,
        )
        light_dir = torch.nn.functional.normalize(light_dir, dim=-1).view(B, 1, 1, 3)
        light_rgb = self.light_tints[self.light_tint_id].view(B, 1, 1, 3)

        # Object albedo + procedural texture + subtle backside orientation marker
        base_col = self.color_palette[self.color_id].view(B, 1, 1, 3)
        tex_mod = self._eval_texture(p_local_hit, self.texture_id)  # (B, H, W, 1)
        # Distinct marker on negative local z face so turning reveals orientation
        back_marker = ((p_local_hit[..., 2] < -0.15) & (p_local_hit[..., 1].abs() < 0.14)).unsqueeze(-1)
        obj_albedo = base_col * tex_mod
        obj_albedo = torch.where(back_marker, obj_albedo * 0.55 + 0.35, obj_albedo)

        # Object shading (Lambertian + Blinn-Phong specular)
        ndotl = (n_world * light_dir).sum(dim=-1, keepdim=True).clamp(min=0.0)
        half_vec = torch.nn.functional.normalize(light_dir - ray_dir, dim=-1)
        spec = (n_world * half_vec).sum(dim=-1, keepdim=True).clamp(min=0.0).pow(24.0) * 0.28
        obj_rgb = obj_albedo * (0.25 + 0.85 * ndotl * light_rgb) + spec * light_rgb

        # Tabletop shading + radial grid + approximate cast shadow from object sphere
        p_table = cam_pos.view(B, 1, 1, 3) + t_plane.unsqueeze(-1) * ray_dir
        grid_x = torch.cos(p_table[..., 0] * 6.0)
        grid_z = torch.cos(p_table[..., 2] * 6.0)
        checker = ((grid_x * grid_z) > 0).float().unsqueeze(-1) * 0.08
        table_base = self.table_shade.view(B, 1, 1, 1) + checker
        # Cast shadow check against bounding sphere centered at origin (r=0.40)
        b_dot = (p_table * light_dir).sum(dim=-1, keepdim=True)
        dist_sq = (p_table - b_dot * light_dir).pow(2).sum(dim=-1, keepdim=True)
        in_shadow = (b_dot < 0.0) & (dist_sq < 0.18)
        shadow_factor = torch.where(in_shadow, 0.38, 1.0)
        table_rgb = table_base * (0.30 + 0.70 * shadow_factor * light_rgb)

        # Studio sky background gradient
        sky_t = (ray_dir[..., 1:2] * 0.5 + 0.5).clamp(0.0, 1.0)
        bg_rgb = (1.0 - sky_t) * torch.tensor([0.16, 0.18, 0.22], device=self.device) + sky_t * torch.tensor(
            [0.36, 0.42, 0.52], device=self.device
        )

        img = torch.where(hit_obj.unsqueeze(-1), obj_rgb, bg_rgb)
        img = torch.where(hit_table.unsqueeze(-1), table_rgb, img)
        img = img.clamp(0.0, 1.0).permute(0, 3, 1, 2).contiguous()  # (B, 3, H, W)
        return img

    def _eval_texture(self, p_local: torch.Tensor, texture_id: torch.Tensor) -> torch.Tensor:
        B = p_local.shape[0]
        tex = texture_id.view(B, 1, 1)
        stripes = 0.65 + 0.35 * torch.sin(p_local[..., 1] * 26.0).sign()
        checks = 0.65 + 0.35 * (
            torch.sin(p_local[..., 0] * 18.0) * torch.sin(p_local[..., 1] * 18.0) * torch.sin(p_local[..., 2] * 18.0)
        ).sign()
        mod = torch.ones_like(stripes)
        mod = torch.where(tex == 1, stripes, mod)
        mod = torch.where(tex == 2, checks, mod)
        return mod.unsqueeze(-1)

    @staticmethod
    def _intersect_sphere(ro: torch.Tensor, rd: torch.Tensor, radius: float) -> Tuple[torch.Tensor, torch.Tensor]:
        INF = 1e6
        b = (ro * rd).sum(dim=-1)
        c = (ro * ro).sum(dim=-1) - radius * radius
        disc = b * b - c
        valid = disc > 0.0
        sqrt_d = torch.sqrt(disc.clamp(min=0.0))
        t = -b - sqrt_d
        valid = valid & (t > 0.0)
        t_out = torch.where(valid, t, torch.full_like(t, INF))
        hit = ro + t_out.unsqueeze(-1) * rd
        norm = torch.nn.functional.normalize(hit, dim=-1)
        return t_out, norm

    @staticmethod
    def _intersect_box(ro: torch.Tensor, rd: torch.Tensor, half_ext: float) -> Tuple[torch.Tensor, torch.Tensor]:
        INF = 1e6
        inv_rd = 1.0 / torch.where(rd.abs() < 1e-6, torch.full_like(rd, 1e-6), rd)
        t0 = (-half_ext - ro) * inv_rd
        t1 = (half_ext - ro) * inv_rd
        tmin_v = torch.minimum(t0, t1)
        tmax_v = torch.maximum(t0, t1)
        tmin = tmin_v.max(dim=-1).values
        tmax = tmax_v.min(dim=-1).values
        valid = (tmax >= tmin.clamp(min=0.0)) & (tmin > 0.0)
        t_out = torch.where(valid, tmin, torch.full_like(tmin, INF))
        hit = ro + t_out.unsqueeze(-1) * rd
        # Face normal from dominant coordinate
        abs_hit = hit.abs()
        max_c = abs_hit.max(dim=-1, keepdim=True).values
        norm = torch.where(abs_hit >= max_c - 1e-4, hit.sign(), torch.zeros_like(hit))
        norm = torch.nn.functional.normalize(norm, dim=-1)
        return t_out, norm

    @staticmethod
    def _intersect_cylinder(
        ro: torch.Tensor, rd: torch.Tensor, radius: float, half_h: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        INF = 1e6
        ro_xz = ro[..., [0, 2]]
        rd_xz = rd[..., [0, 2]]
        a = (rd_xz * rd_xz).sum(dim=-1).clamp(min=1e-6)
        b = (ro_xz * rd_xz).sum(dim=-1)
        c = (ro_xz * ro_xz).sum(dim=-1) - radius * radius
        disc = b * b - a * c
        sqrt_d = torch.sqrt(disc.clamp(min=0.0))
        t_side = (-b - sqrt_d) / a
        y_side = ro[..., 1] + t_side * rd[..., 1]
        valid_side = (disc > 0.0) & (t_side > 0.0) & (y_side.abs() <= half_h)
        t_side = torch.where(valid_side, t_side, torch.full_like(t_side, INF))
        hit_side = ro + t_side.unsqueeze(-1) * rd
        n_side = torch.stack([hit_side[..., 0], torch.zeros_like(t_side), hit_side[..., 2]], dim=-1)
        n_side = torch.nn.functional.normalize(n_side, dim=-1)

        # Top/bottom caps
        rd_y = torch.where(rd[..., 1].abs() < 1e-6, torch.full_like(rd[..., 1], 1e-6), rd[..., 1])
        t_top = (half_h - ro[..., 1]) / rd_y
        p_top = ro + t_top.unsqueeze(-1) * rd
        v_top = (t_top > 0.0) & ((p_top[..., 0] ** 2 + p_top[..., 2] ** 2) <= radius * radius)
        t_top = torch.where(v_top, t_top, torch.full_like(t_top, INF))

        t_bot = (-half_h - ro[..., 1]) / rd_y
        p_bot = ro + t_bot.unsqueeze(-1) * rd
        v_bot = (t_bot > 0.0) & ((p_bot[..., 0] ** 2 + p_bot[..., 2] ** 2) <= radius * radius)
        t_bot = torch.where(v_bot, t_bot, torch.full_like(t_bot, INF))

        t_out = torch.minimum(t_side, torch.minimum(t_top, t_bot))
        norm = torch.where(
            ((t_top < t_side) & (t_top <= t_bot)).unsqueeze(-1),
            torch.tensor([0.0, 1.0, 0.0], device=ro.device),
            n_side,
        )
        norm = torch.where(
            ((t_bot < t_side) & (t_bot < t_top)).unsqueeze(-1),
            torch.tensor([0.0, -1.0, 0.0], device=ro.device),
            norm,
        )
        return t_out, norm

    @staticmethod
    def _intersect_cone(
        ro: torch.Tensor, rd: torch.Tensor, base_r: float, y_min: float, y_max: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        INF = 1e6
        h = y_max - y_min
        k = base_r / h
        k2 = k * k

        oy = ro[..., 1] - y_max
        dy = rd[..., 1]
        a = rd[..., 0] ** 2 + rd[..., 2] ** 2 - k2 * dy * dy
        b = ro[..., 0] * rd[..., 0] + ro[..., 2] * rd[..., 2] - k2 * oy * dy
        c = ro[..., 0] ** 2 + ro[..., 2] ** 2 - k2 * oy * oy

        disc = b * b - a * c
        sqrt_d = torch.sqrt(disc.clamp(min=0.0))
        safe_a = torch.where(a.abs() < 1e-6, torch.full_like(a, 1e-6), a)
        t1 = (-b - sqrt_d) / safe_a
        t2 = (-b + sqrt_d) / safe_a
        # Choose smallest positive t whose y lies within [y_min, y_max]
        y1 = ro[..., 1] + t1 * dy
        y2 = ro[..., 1] + t2 * dy
        v1 = (disc > 0.0) & (t1 > 0.0) & (y1 >= y_min) & (y1 <= y_max)
        v2 = (disc > 0.0) & (t2 > 0.0) & (y2 >= y_min) & (y2 <= y_max)
        inf_t = torch.full_like(t1, INF)
        t_side = torch.minimum(torch.where(v1, t1, inf_t), torch.where(v2, t2, inf_t))

        hit_side = ro + t_side.unsqueeze(-1) * rd
        n_side = torch.stack(
            [hit_side[..., 0], -k2 * (hit_side[..., 1] - y_max), hit_side[..., 2]],
            dim=-1,
        )
        n_side = torch.nn.functional.normalize(n_side, dim=-1)

        # Base cap at y_min
        rd_y = torch.where(dy.abs() < 1e-6, torch.full_like(dy, 1e-6), dy)
        t_cap = (y_min - ro[..., 1]) / rd_y
        p_cap = ro + t_cap.unsqueeze(-1) * rd
        v_cap = (t_cap > 0.0) & ((p_cap[..., 0] ** 2 + p_cap[..., 2] ** 2) <= base_r * base_r)
        t_cap = torch.where(v_cap, t_cap, torch.full_like(t_cap, INF))

        t_out = torch.minimum(t_side, t_cap)
        norm = torch.where((t_cap < t_side).unsqueeze(-1), torch.tensor([0.0, -1.0, 0.0], device=ro.device), n_side)
        return t_out, norm

    @staticmethod
    def _intersect_capsule(
        ro: torch.Tensor, rd: torch.Tensor, radius: float, half_seg: float
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        INF = 1e6
        # Cylinder body
        ro_xz = ro[..., [0, 2]]
        rd_xz = rd[..., [0, 2]]
        a = (rd_xz * rd_xz).sum(dim=-1).clamp(min=1e-6)
        b = (ro_xz * rd_xz).sum(dim=-1)
        c = (ro_xz * ro_xz).sum(dim=-1) - radius * radius
        disc = b * b - a * c
        t_cyl = (-b - torch.sqrt(disc.clamp(min=0.0))) / a
        y_cyl = ro[..., 1] + t_cyl * rd[..., 1]
        v_cyl = (disc > 0.0) & (t_cyl > 0.0) & (y_cyl.abs() <= half_seg)
        t_cyl = torch.where(v_cyl, t_cyl, torch.full_like(t_cyl, INF))
        h_cyl = ro + t_cyl.unsqueeze(-1) * rd
        n_cyl = torch.nn.functional.normalize(
            torch.stack([h_cyl[..., 0], torch.zeros_like(t_cyl), h_cyl[..., 2]], dim=-1), dim=-1
        )

        # Top/bottom hemispherical caps
        ro_top = ro.clone()
        ro_top[..., 1] = ro_top[..., 1] - half_seg
        t_top, n_top = TabletopWorld._intersect_sphere(ro_top, rd, radius)

        ro_bot = ro.clone()
        ro_bot[..., 1] = ro_bot[..., 1] + half_seg
        t_bot, n_bot = TabletopWorld._intersect_sphere(ro_bot, rd, radius)

        t_out = torch.minimum(t_cyl, torch.minimum(t_top, t_bot))
        norm = torch.where(((t_top < t_cyl) & (t_top <= t_bot)).unsqueeze(-1), n_top, n_cyl)
        norm = torch.where(((t_bot < t_cyl) & (t_bot < t_top)).unsqueeze(-1), n_bot, norm)
        return t_out, norm

    def observe(self, emit_word: bool = True) -> Dict[str, torch.Tensor]:
        """Returns visual frame, sparse grounded word, and ground-truth state metadata."""
        images = self.render()
        if emit_word:
            word_id, has_word = self._sample_teacher_words()
        else:
            word_id = torch.zeros(self.batch_size, dtype=torch.long, device=self.device)
            has_word = torch.zeros(self.batch_size, dtype=torch.bool, device=self.device)

        return {
            "images": images,
            "word_id": word_id,
            "has_word": has_word,
            "shape_id": self.shape_id.clone(),
            "color_id": self.color_id.clone(),
            "texture_id": self.texture_id.clone(),
            "light_tint_id": self.light_tint_id.clone(),
            "obj_yaw": self.obj_yaw.clone(),
            "obj_pitch": self.obj_pitch.clone(),
        }
