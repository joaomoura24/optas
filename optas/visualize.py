"""! @brief Visualizer: draws robots, trajectories and shapes in the browser with viser."""

import os
import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple

import casadi as cs
import numpy as np
from scipy.spatial.transform import Rotation as Rot
from urdf_parser_py.urdf import Box, Cylinder, Mesh, Sphere

try:
    import trimesh
    import viser
except ImportError as e:
    raise ImportError("Visualizer requires viser: pip install 'pyoptas[vis]'") from e

from .models import RobotModel
from .spatialmath import ArrayType

## Color of robot links whose URDF gives none.
DEFAULT_ROBOT_RGB = (0.7, 0.7, 0.7)


def _vec(value: ArrayType) -> np.ndarray:
    """! Flatten a list, numpy array or CasADi DM into a float numpy vector."""
    return np.asarray(value, dtype=float).flatten()


def _color(rgb: Optional[ArrayType], default=(1.0, 1.0, 1.0)) -> Tuple[int, int, int]:
    """! RGB in [0, 1] (default when None) as the uint8 tuple viser takes."""
    rgb = default if rgb is None else _vec(rgb)[:3]
    return tuple(int(round(255 * float(c))) for c in rgb)


def _opacity(alpha: float) -> Optional[float]:
    """! viser opacity: None when opaque (cheaper to render)."""
    alpha = float(alpha)
    return None if alpha >= 1.0 else alpha


def _wxyz_from_matrix(R: np.ndarray) -> np.ndarray:
    """! Quaternion (w, x, y, z), the order viser takes, of a rotation matrix."""
    x, y, z, w = Rot.from_matrix(R).as_quat()
    return np.array([w, x, y, z])


def _wxyz(
    orientation: ArrayType, euler_seq: str = "xyz", euler_degrees: bool = False
) -> np.ndarray:
    """! Quaternion (w, x, y, z) of a quaternion (x, y, z, w) or Euler angles."""
    orientation = _vec(orientation)
    if orientation.size == 4:
        R = Rot.from_quat(orientation)
    elif orientation.size == 3:
        R = Rot.from_euler(euler_seq, orientation, degrees=euler_degrees)
    else:
        raise ValueError(
            f"orientation must be a quaternion (x, y, z, w) or Euler angles, got {orientation.size} values"
        )
    x, y, z, w = R.as_quat()
    return np.array([w, x, y, z])


def _alphas(alpha_spec: Optional[Dict], n: int) -> List[float]:
    """! Transparency of each of the n elements of a trajectory.

    Styles: "none" (all 1), "const" (all alpha), "A" (from alpha_min to alpha_max),
    "B" (alpha_min, last alpha_max), "C" (alpha_start, alpha_mid..., alpha_end).
    """
    spec = {"style": "none"} if alpha_spec is None else alpha_spec
    style = spec["style"]
    lo, hi = spec.get("alpha_min", 0.1), spec.get("alpha_max", 1.0)
    if style == "none":
        return [1.0] * n
    if style == "const":
        return [spec.get("alpha", 1.0)] * n
    if style == "A":
        return np.linspace(lo, hi, n).tolist()
    if style == "B":
        return [lo] * (n - 1) + [hi]
    if style == "C":
        mid = [spec.get("alpha_mid", 0.1)] * (n - 2)
        return [spec.get("alpha_start", 1.0)] + mid + [spec.get("alpha_end", 1.0)]
    raise ValueError(f"unknown alpha_spec style '{style}'")


class _RobotVisuals:
    """! The visual elements of a robot and a CasADi function of their poses."""

    def __init__(self, robot_model: RobotModel):
        urdf = robot_model.get_urdf()
        materials = {
            m.name: m.color.rgba[:3] for m in urdf.materials if m.color is not None
        }

        q = cs.SX.sym("q", robot_model.ndof)
        ## Names of the links, in the order of the outputs of link_poses.
        self.link_names = [link.name for link in urdf.links]
        link_tfs = [
            robot_model.get_global_link_transform(name, q) for name in self.link_names
        ]
        ## Transforms of every link in the global frame, stacked as a 4 x 4n matrix.
        self.link_poses = cs.Function("link_poses", [q], [cs.horzcat(*link_tfs)])

        ## Visual elements: (link index, transform in the link frame, geometry, rgb).
        self.visuals = []
        for i, link in enumerate(urdf.links):
            for visual in link.visuals:
                if visual is None or visual.geometry is None:
                    continue
                T = np.eye(4)
                if visual.origin is not None:
                    T[:3, :3] = Rot.from_euler("xyz", visual.origin.rpy).as_matrix()
                    T[:3, 3] = visual.origin.xyz
                rgb = DEFAULT_ROBOT_RGB
                material = visual.material
                if material is not None:
                    if material.color is not None:
                        rgb = material.color.rgba[:3]
                    elif material.name in materials:
                        rgb = materials[material.name]
                self.visuals.append((i, T, visual.geometry, rgb))

        ## Directory that relative mesh filenames are resolved against.
        self.dirname = str(robot_model.get_urdf_dirname())

    def poses(self, q: ArrayType) -> List[np.ndarray]:
        """! Transforms (4 x 4) of the links in the global frame at configuration q."""
        tfs = np.asarray(self.link_poses(cs.vec(cs.DM(q))))
        return [tfs[:, 4 * i : 4 * i + 4] for i in range(len(self.link_names))]


class Visualizer:
    """! Visualizer that draws robots, trajectories and shapes in the browser (viser).

    Create it, add elements, then call start(), which prints the address to open
    (http://localhost:8080 by default) and keeps the server running. Elements are
    added as viser scene nodes; every drawing method returns the viser handles, so
    they can be moved or removed later (e.g. handle.position = ..., handle.remove()).
    """

    def __init__(
        self,
        camera_position: ArrayType = [2, 2, 2],
        camera_focal_point: ArrayType = [0, 0, 0],
        camera_view_up: ArrayType = [0, 0, 1],
        quit_after_delay: Optional[float] = None,
        host: str = "0.0.0.0",
        port: int = 8080,
    ):
        """! Initializer for the Visualizer class (starts the viser server).

        @param camera_position The position of the camera in the global frame.
        @param camera_focal_point The point the camera looks at.
        @param camera_view_up The up direction of the camera.
        @param quit_after_delay Number of seconds start() runs for; when None, it runs until Ctrl+C.
        @param host Host the server listens on.
        @param port Port the server listens on (viser takes the next free one when it is in use).
        @return An instance of the Visualizer class.
        """
        ## The viser server, for anything this class does not wrap.
        self.server = viser.ViserServer(host=host, port=port, verbose=False)
        camera = self.server.initial_camera
        camera.position = _vec(camera_position)
        camera.look_at = _vec(camera_focal_point)
        camera.up = _vec(camera_view_up)

        self._quit_after_delay = quit_after_delay
        self._count = 0
        self._robot_visuals = {}
        self._meshes = {}
        ## Animations as (duration, update(t)) where t is the time in seconds.
        self._animations: List[Tuple[float, Callable[[float], None]]] = []

    def _name(self, kind: str) -> str:
        """! Unique scene node name."""
        self._count += 1
        return f"/optas/{kind}_{self._count}"

    def _visuals(self, robot_model: RobotModel) -> _RobotVisuals:
        """! Visual elements of a robot, built once per robot model."""
        key = id(robot_model)
        if key not in self._robot_visuals:
            self._robot_visuals[key] = _RobotVisuals(robot_model)
        return self._robot_visuals[key]

    def _load_mesh(self, filename: str, scale: Optional[ArrayType] = None):
        """! Load a mesh file once (any format trimesh reads), None when it cannot be read."""
        key = (filename, None if scale is None else tuple(_vec(scale)))
        if key not in self._meshes:
            try:
                mesh = trimesh.load(filename, force="mesh")
                if scale is not None:
                    mesh.apply_scale(_vec(scale))
            except Exception as e:  # e.g. a format that needs an extra package
                warnings.warn(f"skipping mesh {filename}: {e}")
                mesh = None
            self._meshes[key] = mesh
        return self._meshes[key]

    #
    # Shapes
    #

    def line(
        self,
        start: ArrayType = [0.0, 0.0, 0.0],
        end: ArrayType = [1.0, 0.0, 0.0],
        rgb: Optional[ArrayType] = None,
        width: float = 0.005,
    ):
        """! Draw a line.

        @param start The start of the line.
        @param end The end of the line.
        @param rgb The Red-Green-Blue values in range [0, 1].
        @param width The width of the line (m).
        @return The viser handle.
        """
        points = np.stack([_vec(start), _vec(end)])[None]
        return self.server.scene.add_line_segments(
            self._name("line"), points, _color(rgb), thickness=width
        )

    def sphere(
        self,
        radius: float = 1.0,
        position: ArrayType = [0.0, 0.0, 0.0],
        rgb: Optional[ArrayType] = None,
        alpha: float = 1.0,
    ):
        """! Draw a sphere.

        @param radius The radius of the sphere.
        @param position The position of the sphere.
        @param rgb The Red-Green-Blue values in range [0, 1].
        @param alpha Transparency in range [0, 1].
        @return The viser handle.
        """
        return self.server.scene.add_icosphere(
            self._name("sphere"),
            radius=float(radius),
            color=_color(rgb),
            opacity=_opacity(alpha),
            position=_vec(position),
        )

    def sphere_traj(
        self,
        position_traj: ArrayType,
        radius: float = 1.0,
        rgb: Optional[ArrayType] = None,
        alpha_spec: Optional[Dict] = None,
        animate: bool = False,
        duration: float = 5.0,
    ) -> list:
        """! Draw a sphere at every position of a trajectory, or one moving sphere.

        @param position_traj The positions, as a 3 x n array.
        @param radius The radius of the spheres.
        @param rgb The Red-Green-Blue values in range [0, 1].
        @param alpha_spec Transparency of the spheres, see _alphas (ignored when animated).
        @param animate When True, one sphere moves through the positions in duration seconds (played by start()).
        @param duration Duration of the animation (s).
        @return The viser handles.
        """
        traj = np.asarray(position_traj, dtype=float).reshape(3, -1)
        n = traj.shape[1]
        if not animate:
            return [
                self.sphere(radius, traj[:, i], rgb, alpha)
                for i, alpha in enumerate(_alphas(alpha_spec, n))
            ]

        handle = self.sphere(radius, traj[:, 0], rgb)

        def update(t):
            handle.position = traj[:, self._frame(t, duration, n)]

        self._animations.append((float(duration), update))
        return [handle]

    def box(
        self,
        scale: ArrayType = [1, 1, 1],
        rgb: Optional[ArrayType] = None,
        alpha: float = 1.0,
        position: ArrayType = [0.0, 0.0, 0.0],
        orientation: ArrayType = [0.0, 0.0, 0.0],
        euler_seq: str = "xyz",
        euler_degrees: bool = False,
    ):
        """! Draw a box.

        @param scale The length, width, and height of the box [L, W, H].
        @param rgb The Red-Green-Blue values in range [0, 1].
        @param alpha Transparency in range [0, 1].
        @param position The position of the box center.
        @param orientation Orientation, either a quaternion (x, y, z, w) or Euler angles.
        @param euler_seq Sequence of axes for Euler angles (scipy convention, e.g. "xyz" extrinsic, "XYZ" intrinsic).
        @param euler_degrees If True, the Euler angles are in degrees.
        @return The viser handle.
        """
        return self.server.scene.add_box(
            self._name("box"),
            color=_color(rgb),
            dimensions=_vec(scale),
            opacity=_opacity(alpha),
            wxyz=_wxyz(orientation, euler_seq, euler_degrees),
            position=_vec(position),
        )

    def cylinder(
        self,
        radius: float = 1.0,
        height: float = 1.0,
        rgb: Optional[ArrayType] = None,
        alpha: float = 1.0,
        position: ArrayType = [0.0, 0.0, 0.0],
        orientation: ArrayType = [0.0, 0.0, 0.0],
        euler_seq: str = "xyz",
        euler_degrees: bool = False,
    ):
        """! Draw a cylinder with its axis along z (as in URDF).

        @param radius The cylinder radius.
        @param height The height of the cylinder.
        @param rgb The Red-Green-Blue values in range [0, 1].
        @param alpha Transparency in range [0, 1].
        @param position The position of the cylinder center.
        @param orientation Orientation, either a quaternion (x, y, z, w) or Euler angles.
        @param euler_seq Sequence of axes for Euler angles (scipy convention).
        @param euler_degrees If True, the Euler angles are in degrees.
        @return The viser handle.
        """
        return self.server.scene.add_cylinder(
            self._name("cylinder"),
            radius=float(radius),
            height=float(height),
            color=_color(rgb),
            opacity=_opacity(alpha),
            wxyz=_wxyz(orientation, euler_seq, euler_degrees),
            position=_vec(position),
        )

    def mesh(
        self,
        filename: str,
        scale: Optional[ArrayType] = None,
        rgb: Optional[ArrayType] = None,
        alpha: float = 1.0,
        texture_filename: Optional[str] = None,
        position: ArrayType = [0.0, 0.0, 0.0],
        orientation: ArrayType = [0.0, 0.0, 0.0],
        euler_seq: str = "xyz",
        euler_degrees: bool = False,
    ):
        """! Draw a mesh file (STL, OBJ, PLY, GLB, ...; anything trimesh reads).

        @param filename The mesh file.
        @param scale Scale applied to the mesh in the xyz directions.
        @param rgb The Red-Green-Blue values in range [0, 1]; when None, the colors or texture of the file are kept.
        @param alpha Transparency in range [0, 1] (used with rgb).
        @param texture_filename An image to use as texture, for meshes with texture coordinates (e.g. an OBJ without its .mtl).
        @param position The position of the mesh.
        @param orientation Orientation, either a quaternion (x, y, z, w) or Euler angles.
        @param euler_seq Sequence of axes for Euler angles (scipy convention).
        @param euler_degrees If True, the Euler angles are in degrees.
        @return The viser handle, or None when the file cannot be read.
        """
        mesh = self._load_mesh(filename, scale)
        if mesh is None:
            return None
        pose = dict(
            wxyz=_wxyz(orientation, euler_seq, euler_degrees), position=_vec(position)
        )
        if rgb is not None:
            return self.server.scene.add_mesh_simple(
                self._name("mesh"),
                mesh.vertices,
                mesh.faces,
                color=_color(rgb),
                opacity=_opacity(alpha),
                **pose,
            )
        if texture_filename is not None:
            from PIL import Image  # viser depends on pillow

            mesh = mesh.copy()
            mesh.visual = trimesh.visual.TextureVisuals(
                uv=mesh.visual.uv, image=Image.open(texture_filename)
            )
        return self.server.scene.add_mesh_trimesh(self._name("mesh"), mesh, **pose)

    def text(
        self,
        msg: str = "Hello, world!",
        position: ArrayType = [0.0, 0.0, 0.0],
    ):
        """! Draw a text label (always facing the camera).

        @param msg The text.
        @param position The position of the text in the global frame.
        @return The viser handle.
        """
        return self.server.scene.add_label(
            self._name("text"), str(msg), position=_vec(position)
        )

    def link(
        self,
        T: Optional[ArrayType] = None,
        axis_scale: float = 0.1,
        center_radius: float = 0.01,
    ):
        """! Draw a coordinate frame (x red, y green, z blue).

        @param T Homogeneous transformation matrix of the frame (default: identity).
        @param axis_scale The axis length.
        @param center_radius The radius of the sphere at the origin.
        @return The viser handle.
        """
        T = np.eye(4) if T is None else np.asarray(T, dtype=float)
        return self.server.scene.add_frame(
            self._name("frame"),
            axes_length=float(axis_scale),
            axes_radius=0.05 * float(axis_scale),
            origin_radius=float(center_radius),
            wxyz=_wxyz_from_matrix(T[:3, :3]),
            position=T[:3, 3],
        )

    def grid_floor(
        self,
        num_cells: int = 10,
        stride: float = 1.0,
        rgb: Optional[ArrayType] = None,
        euler: ArrayType = [0, 0, 0],
        euler_seq: str = "xyz",
        euler_degrees: bool = True,
    ):
        """! Draw a square grid, centered at the origin, in the xy plane (rotated by euler).

        @param num_cells The number of cells along each side.
        @param stride The length of each cell (thin lines are drawn at half of it).
        @param rgb The Red-Green-Blue values in range [0, 1] of the cell lines.
        @param euler Euler angles of the orientation of the plane.
        @param euler_seq Sequence of axes for the Euler angles (scipy convention).
        @param euler_degrees If True, the Euler angles are in degrees (default).
        @return The viser handle.
        """
        size = int(num_cells) * float(stride)
        color = _color(rgb, default=(0.55, 0.55, 0.55))
        return self.server.scene.add_grid(
            self._name("grid"),
            width=size,
            height=size,
            cell_size=0.5 * float(stride),
            cell_color=color,
            section_size=float(stride),
            section_color=color,
            section_thickness=2.0,
            wxyz=_wxyz(euler, euler_seq, euler_degrees),
        )

    #
    # Robots
    #

    def robot(
        self,
        robot_model: RobotModel,
        q: Optional[ArrayType] = None,
        alpha: float = 1.0,
        show_links: bool = False,
        link_axis_scale: float = 0.2,
        display_link_names: bool = False,
    ) -> list:
        """! Draw a robot from its URDF visuals (meshes, boxes, cylinders and spheres).

        @param robot_model The robot model defining the kinematics and visuals.
        @param q Joint configuration (default: zero).
        @param alpha Transparency in range [0, 1].
        @param show_links When True, the link frames are drawn.
        @param link_axis_scale The axis length of the link frames.
        @param display_link_names When True, the link names are drawn.
        @return The viser handles; handles[i] is drawn by visual i, so they can be moved by _set_robot_pose.
        """
        visuals = self._visuals(robot_model)
        q = np.zeros(robot_model.ndof) if q is None else q
        link_tfs = visuals.poses(q)
        scene = self.server.scene

        handles = []
        for i, T_link, geometry, rgb in visuals.visuals:
            T = link_tfs[i] @ T_link
            pose = dict(wxyz=_wxyz_from_matrix(T[:3, :3]), position=T[:3, 3])
            style = dict(color=_color(rgb), opacity=_opacity(alpha))
            if isinstance(geometry, Mesh):
                filename = geometry.filename
                if not os.path.exists(filename):
                    filename = os.path.join(visuals.dirname, filename)
                mesh = self._load_mesh(filename, geometry.scale)
                if mesh is None:
                    continue
                handle = scene.add_mesh_simple(
                    self._name("robot"), mesh.vertices, mesh.faces, **style, **pose
                )
            elif isinstance(geometry, Box):
                handle = scene.add_box(
                    self._name("robot"), dimensions=geometry.size, **style, **pose
                )
            elif isinstance(geometry, Cylinder):
                handle = scene.add_cylinder(
                    self._name("robot"),
                    radius=geometry.radius,
                    height=geometry.length,
                    **style,
                    **pose,
                )
            elif isinstance(geometry, Sphere):
                handle = scene.add_icosphere(
                    self._name("robot"), radius=geometry.radius, **style, **pose
                )
            else:
                continue
            handle._optas_link = (i, T_link)  # to move it with the robot
            handles.append(handle)

        for i, name in enumerate(visuals.link_names):
            if show_links:
                handle = self.link(link_tfs[i], axis_scale=link_axis_scale)
                handle._optas_link = (i, np.eye(4))
                handles.append(handle)
            if display_link_names:
                handle = self.text(name, link_tfs[i][:3, 3])
                handle._optas_link = (i, np.eye(4))
                handles.append(handle)

        return handles

    def _set_robot_pose(
        self, robot_model: RobotModel, handles: list, q: ArrayType
    ) -> None:
        """! Move the handles returned by robot() to configuration q."""
        link_tfs = self._visuals(robot_model).poses(q)
        with self.server.atomic():
            for handle in handles:
                i, T_link = handle._optas_link
                T = link_tfs[i] @ T_link
                handle.wxyz = _wxyz_from_matrix(T[:3, :3])
                handle.position = T[:3, 3]

    def robot_traj(
        self,
        robot_model: RobotModel,
        Q: ArrayType,
        alpha_spec: Optional[Dict] = None,
        show_links: bool = False,
        link_axis_scale: float = 0.2,
        display_link_names: bool = False,
        animate: bool = False,
        duration: float = 5.0,
    ) -> list:
        """! Draw a robot at every configuration of a trajectory, or one moving robot.

        @param robot_model The robot model defining the kinematics and visuals.
        @param Q Joint configuration trajectory, as an ndof x n array.
        @param alpha_spec Transparency of the robots, see _alphas (ignored when animated).
        @param show_links When True, the link frames are drawn.
        @param link_axis_scale The axis length of the link frames.
        @param display_link_names When True, the link names are drawn.
        @param animate When True, one robot moves through Q in duration seconds (played by start()).
        @param duration Duration of the animation (s).
        @return The viser handles.
        """
        Q = np.asarray(Q, dtype=float).reshape(robot_model.ndof, -1)
        n = Q.shape[1]
        options = dict(
            show_links=show_links,
            link_axis_scale=link_axis_scale,
            display_link_names=display_link_names,
        )
        if not animate:
            handles = []
            for i, alpha in enumerate(_alphas(alpha_spec, n)):
                handles += self.robot(robot_model, Q[:, i], alpha=alpha, **options)
            return handles

        handles = self.robot(robot_model, Q[:, 0], **options)

        def update(t):
            q = Q[:, self._frame(t, duration, n)]
            self._set_robot_pose(robot_model, handles, q)

        self._animations.append((float(duration), update))
        return handles

    #
    # Running
    #

    @staticmethod
    def _frame(t: float, duration: float, n: int) -> int:
        """! Index of the trajectory element shown at time t (the last one after duration)."""
        if n == 1 or duration <= 0.0:
            return n - 1
        return int(round(min(max(t / duration, 0.0), 1.0) * (n - 1)))

    def show_time(self, t: float) -> None:
        """! Show every animation at time t (s)."""
        for _, update in self._animations:
            update(t)

    def start(self) -> None:
        """! Keep the server running (and play the animations) until Ctrl+C, or for quit_after_delay seconds.

        Animations loop; the viser panel has a play/pause checkbox and a time slider.
        """
        print(f"Visualizer: open http://localhost:{self.server.get_port()}")
        end = None
        if self._quit_after_delay is not None:
            end = time.time() + float(self._quit_after_delay)

        period = max([d for d, _ in self._animations], default=0.0)
        if period > 0.0:
            gui = self.server.gui
            play = gui.add_checkbox("Play", initial_value=True)
            slider = gui.add_slider(
                "Time [s]", min=0.0, max=period, step=0.01, initial_value=0.0
            )

            @slider.on_update
            def _(_):
                if not play.value:
                    self.show_time(slider.value)

        try:
            t0 = time.time()
            while end is None or time.time() < end:
                if period > 0.0 and play.value:
                    t = (time.time() - t0) % period
                    slider.value = t
                    self.show_time(t)
                elif period > 0.0:
                    t0 = time.time() - slider.value  # resume from the slider
                time.sleep(0.02)
        except KeyboardInterrupt:
            pass
        finally:
            self.server.stop()
