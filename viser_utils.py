import torch
import os
import threading
import json
import ast
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib as mpl
import cv2
import numpy as np
import matplotlib.cm as cm
import viser
import viser.transforms as tf
import time
import trimesh
import dataclasses
from datetime import datetime
from scipy.spatial.transform import Rotation
from src.dust3r.viz import (
    add_scene_cam,
    CAM_COLORS,
    OPENGL,
    pts3d_to_trimesh,
    cat_meshes,
)

_VISER_WEBSOCKET_PING_DISABLED = False
_CAMERA_PRESET_FORMAT = "ttt3r_viewer_camera"
_CAMERA_PRESET_VERSION = 1


def disable_viser_websocket_ping():
    global _VISER_WEBSOCKET_PING_DISABLED
    if _VISER_WEBSOCKET_PING_DISABLED:
        return

    try:
        import websockets.asyncio.server as ws_async_server
    except Exception as exc:
        print(f"Warning: failed to import websockets for ping disable shim: {exc}")
        return

    original_serve = ws_async_server.serve

    def serve_without_ping(*args, **kwargs):
        kwargs.setdefault("ping_interval", None)
        kwargs.setdefault("ping_timeout", None)
        return original_serve(*args, **kwargs)

    ws_async_server.serve = serve_without_ping
    _VISER_WEBSOCKET_PING_DISABLED = True


def todevice(batch, device, callback=None, non_blocking=False):
    """Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    """
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == "numpy":
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice  # alias


def to_numpy(x):
    return todevice(x, "numpy")


def segment_sky(image):
    import cv2
    from scipy import ndimage

    # Convert to HSV
    image = to_numpy(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.uint8(255 * image.clip(min=0, max=1))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for blue color and create mask
    lower_blue = np.array([0, 0, 100])
    upper_blue = np.array([30, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue).view(bool)

    # add luminous gray
    mask |= (hsv[:, :, 1] < 10) & (hsv[:, :, 2] > 150)
    mask |= (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 180)
    mask |= (hsv[:, :, 1] < 50) & (hsv[:, :, 2] > 220)

    # Morphological operations
    kernel = np.ones((5, 5), np.uint8)
    mask2 = ndimage.binary_opening(mask, structure=kernel)

    # keep only largest CC
    _, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask2.view(np.uint8), connectivity=8
    )
    cc_sizes = stats[1:, cv2.CC_STAT_AREA]
    order = cc_sizes.argsort()[::-1]  # bigger first
    i = 0
    selection = []
    while i < len(order) and cc_sizes[order[i]] > cc_sizes[order[0]] / 2:
        selection.append(1 + order[i])
        i += 1
    mask3 = np.in1d(labels, selection).reshape(labels.shape)

    # Apply mask
    return torch.from_numpy(mask3)


def convert_scene_output_to_glb(
    outdir,
    imgs,
    pts3d,
    mask,
    focals,
    cams2world,
    cam_size=0.05,
    show_cam=True,
    cam_color=None,
    as_pointcloud=False,
    transparent_cams=False,
    silent=False,
    save_name=None,
):
    assert len(pts3d) == len(mask) <= len(imgs) <= len(cams2world) == len(focals)
    pts3d = to_numpy(pts3d)
    imgs = to_numpy(imgs)
    focals = to_numpy(focals)
    cams2world = to_numpy(cams2world)

    scene = trimesh.Scene()

    # full pointcloud
    if as_pointcloud:
        pts = np.concatenate([p[m] for p, m in zip(pts3d, mask)])
        col = np.concatenate([p[m] for p, m in zip(imgs, mask)])
        pct = trimesh.PointCloud(pts.reshape(-1, 3), colors=col.reshape(-1, 3))
        scene.add_geometry(pct)
    else:
        meshes = []
        for i in range(len(imgs)):
            meshes.append(pts3d_to_trimesh(imgs[i], pts3d[i], mask[i]))
        mesh = trimesh.Trimesh(**cat_meshes(meshes))
        scene.add_geometry(mesh)

    # add each camera
    if show_cam:
        for i, pose_c2w in enumerate(cams2world):
            if isinstance(cam_color, list):
                camera_edge_color = cam_color[i]
            else:
                camera_edge_color = cam_color or CAM_COLORS[i % len(CAM_COLORS)]
            add_scene_cam(
                scene,
                pose_c2w,
                camera_edge_color,
                None if transparent_cams else imgs[i],
                focals[i],
                imsize=imgs[i].shape[1::-1],
                screen_width=cam_size,
            )

    rot = np.eye(4)
    rot[:3, :3] = Rotation.from_euler("y", np.deg2rad(180)).as_matrix()
    scene.apply_transform(np.linalg.inv(cams2world[0] @ OPENGL @ rot))
    if save_name is None:
        save_name = "scene"
    outfile = os.path.join(outdir, save_name + ".glb")
    if not silent:
        print("(exporting 3D scene to", outfile, ")")
    scene.export(file_obj=outfile)
    return outfile


@dataclasses.dataclass
class CameraState(object):
    fov: float
    aspect: float
    c2w: np.ndarray

    def get_K(self, img_wh):
        W, H = img_wh
        focal_length = H / 2.0 / np.tan(self.fov / 2.0)
        K = np.array(
            [
                [focal_length, 0.0, W / 2.0],
                [0.0, focal_length, H / 2.0],
                [0.0, 0.0, 1.0],
            ]
        )
        return K


def get_vertical_colorbar(h, vmin, vmax, cmap_name="jet", label=None, cbar_precision=2):
    """
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    """
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = mpl.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation="vertical"
    )

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)
    if label is not None:
        cb1.set_label(label)

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.0
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(
    x,
    cmap_name="jet",
    mask=None,
    range=None,
    append_cbar=False,
    cbar_in_image=False,
    cbar_precision=2,
):
    """
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    """
    if range is not None:
        vmin, vmax = range
    elif mask is not None:

        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])

        x[np.logical_not(mask)] = vmin

    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += 1e-6

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1.0 - mask)

    cbar = get_vertical_colorbar(
        h=x.shape[0],
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
        cbar_precision=cbar_precision,
    )

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1] :, :] = cbar
        else:
            x_new = np.concatenate(
                (x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1
            )
        return x_new
    else:
        return x_new


def colorize(
    x, cmap_name="jet", mask=None, range=None, append_cbar=False, cbar_in_image=False
):
    """
    turn a grayscale image into a color image
    :param x: torch.Tensor, grayscale image, [H, W] or [B, H, W]
    :param mask: torch.Tensor or None, mask image, [H, W] or [B, H, W] or None
    """

    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)

    if x.ndim == 2:
        x = x[None]
        if mask is not None:
            mask = mask[None]

    out = []
    for x_ in x:
        if mask is not None:
            mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

        x_ = colorize_np(x_, cmap_name, mask, range, append_cbar, cbar_in_image)
        out.append(torch.from_numpy(x_).to(device).float())
    out = torch.stack(out).squeeze(0)
    return out


class PointCloudViewer:
    def __init__(
        self,
        model,
        state_args,
        pc_list,
        color_list,
        conf_list,
        cam_dict,
        image_mask=None,
        edge_color_list=None,
        camera_color_list=None,
        device="cpu",
        port=8080,
        show_camera=True,
        vis_threshold=1,
        size=512,
        downsample_factor=10,
        max_points=0,
        remove_sky=False,
    ):
        self.model = model
        self.size=size
        self.state_args = state_args
        disable_viser_websocket_ping()
        self.server = viser.ViserServer(port=port)
        self.server.set_up_direction("-y")
        self.device = device
        self.conf_list = conf_list
        self.vis_threshold = vis_threshold
        self.max_points = max(0, int(max_points))
        self.remove_sky = bool(remove_sky)
        self.tt = lambda x: torch.from_numpy(x).float().to(device)
        self.pcs, self.all_steps = self.read_data(
            pc_list, color_list, conf_list, edge_color_list, camera_color_list
        )
        self.cam_dict = cam_dict
        self.num_frames = len(self.all_steps)
        self.image_mask = image_mask
        self.show_camera = show_camera
        self.on_replay = False
        self.vis_pts_list = []
        self.traj_list = []
        self.orig_img_list = [x[0] for x in color_list]
        self.via_points = []
        self._export_lock = threading.Lock()
        self._export_in_progress = False
        self._viewer_updates_paused = False

        gui_reset_up = self.server.gui.add_button(
            "Reset up direction",
            hint="Set the camera control 'up' direction to the current camera's 'up'.",
        )

        @gui_reset_up.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            assert client is not None
            client.camera.up_direction = tf.SO3(client.camera.wxyz) @ np.array(
                [0.0, -1.0, 0.0]
            )

        button3 = self.server.gui.add_button("4D (Only Show Current Frame)")
        button4 = self.server.gui.add_button("3D (Show All Frames)")
        self.is_render = False
        self.fourd = False

        @button3.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = True

        @button4.on_click
        def _(event: viser.GuiEvent) -> None:
            self.fourd = False

        self.focal_slider = self.server.add_gui_slider(
            "Focal Length",
            min=0.1,
            max=99999,
            step=1,
            initial_value=533,
        )

        self.psize_slider = self.server.add_gui_slider(
            "Point Size",
            min=0.0001,
            max=0.1,
            step=0.0001,
            initial_value=0.005,
        )
        self.camsize_slider = self.server.add_gui_slider(
            "Camera Size",
            min=0.01,
            max=0.5,
            step=0.01,
            initial_value=0.1,
        )

        # point cloud downsample control
        self.downsample_slider = self.server.add_gui_slider(
            "Downsample Factor",
            min=1,
            max=1000,
            step=1,
            initial_value=downsample_factor,
        )

        # camera visualization control
        self.show_camera_checkbox = self.server.add_gui_checkbox(
            "Show Camera", 
            initial_value=self.show_camera
        )
        self.remove_sky_checkbox = self.server.add_gui_checkbox(
            "Remove Sky",
            initial_value=self.remove_sky,
        )

        # visualization threshold control slider
        self.vis_threshold_slider = self.server.add_gui_slider(
            "Visibility Threshold",
            min=0.1,
            max=30.0,
            step=0.1,
            initial_value=self.vis_threshold,
        )

        # camera downsample control slider
        self.camera_downsample_slider = self.server.add_gui_slider(
            "Camera Downsample Factor",
            min=1,
            max=50,
            step=1,
            initial_value=1,
        )

        with self.server.add_gui_folder("Export"):
            self.export_filename = self.server.gui.add_text(
                "PNG Filename",
                initial_value="viewer_snapshot_transparent.png",
            )
            self.export_width = self.server.gui.add_number(
                "PNG Width",
                initial_value=1280,
                min=64,
                max=8192,
                step=1,
            )
            self.export_height = self.server.gui.add_number(
                "PNG Height",
                initial_value=720,
                min=64,
                max=8192,
                step=1,
            )
            self.export_button = self.server.gui.add_button(
                "Export Transparent PNG",
                hint="Capture the current client viewport as an RGBA PNG and download it.",
            )

        self.pc_handles = []
        self.cam_handles = []

        @self.psize_slider.on_update
        def _(_) -> None:
            for handle in self.pc_handles:
                handle.point_size = self.psize_slider.value

        @self.camsize_slider.on_update
        def _(_) -> None:
            for handle in self.cam_handles:
                handle.scale = self.camsize_slider.value
                handle.line_thickness = 0.03 * handle.scale

        @self.downsample_slider.on_update
        def _(_) -> None:
            # when the downsample factor changes, regenerate all point clouds
            self.refresh_point_clouds()

        @self.show_camera_checkbox.on_update
        def _(_) -> None:
            # update the internal state
            self.show_camera = self.show_camera_checkbox.value
            
            if self.show_camera:
                # if the camera display is enabled, recreate the camera according to the downsample factor
                # first clear the existing camera
                for handle in self.cam_handles:
                    try:
                        handle.remove()
                    except (KeyError, AttributeError):
                        pass
                self.cam_handles.clear()
                
                # add camera according to the downsample factor
                if hasattr(self, 'frame_nodes'):
                    downsample_factor = int(self.camera_downsample_slider.value)
                    for i, step in enumerate(self.all_steps):
                        if i % downsample_factor == 0:
                            self.add_camera(step)
            else:
                # if the camera display is disabled, hide all cameras
                for handle in self.cam_handles:
                    handle.visible = False

        @self.vis_threshold_slider.on_update
        def _(_) -> None:
            # when the visualization threshold changes, update the threshold and regenerate the point cloud
            self.vis_threshold = self.vis_threshold_slider.value
            self.refresh_point_clouds()

        @self.remove_sky_checkbox.on_update
        def _(_) -> None:
            self.remove_sky = self.remove_sky_checkbox.value
            self.refresh_point_clouds()

        @self.camera_downsample_slider.on_update
        def _(_) -> None:
            # when the camera downsample factor changes, update the camera display
            if hasattr(self, 'frame_nodes'):
                # clear the existing camera display
                for handle in self.cam_handles:
                    try:
                        handle.remove()
                    except (KeyError, AttributeError):
                        # ignore the handle that has been deleted or does not exist
                        pass
                self.cam_handles.clear()
                
                # add camera according to the downsample factor
                if self.show_camera:
                    downsample_factor = int(self.camera_downsample_slider.value)
                    for i, step in enumerate(self.all_steps):
                        # only show every N cameras, where N is the downsample factor
                        if i % downsample_factor == 0:
                            self.add_camera(step)

        @self.export_button.on_click
        def _(event: viser.GuiEvent) -> None:
            client = event.client
            if client is None:
                print("Export skipped: no active client is associated with the button click.")
                return

            width = max(64, int(self.export_width.value))
            height = max(64, int(self.export_height.value))
            filename = self._normalize_export_filename(self.export_filename.value)
            with self._export_lock:
                if self._export_in_progress:
                    print("Export skipped: another PNG export is already in progress.")
                    return
                self._export_in_progress = True
                self.export_button.disabled = True

            try:
                from viser import _messages as viser_messages

                export_script = f"""
(() => {{
  const findCanvas = () => {{
    const candidates = Array.from(document.querySelectorAll("canvas"));
    const visibleWebglCanvases = candidates.filter((canvas) => {{
      try {{
        const rect = canvas.getBoundingClientRect();
        const style = window.getComputedStyle(canvas);
        const hasWebgl = !!(
          canvas.getContext("webgl2") ||
          canvas.getContext("webgl") ||
          canvas.getContext("experimental-webgl")
        );
        return hasWebgl && rect.width > 0 && rect.height > 0 && style.visibility !== "hidden";
      }} catch (err) {{
        return false;
      }}
    }});

    if (visibleWebglCanvases.length === 0) {{
      return null;
    }}

    visibleWebglCanvases.sort((a, b) => (b.width * b.height) - (a.width * a.height));
    return visibleWebglCanvases[0];
  }};

  const sourceCanvas = findCanvas();
  if (!sourceCanvas) {{
    console.error("Viser export failed: no visible WebGL canvas found.");
    return;
  }}

  const doExport = () => {{
    const exportCanvas = document.createElement("canvas");
    exportCanvas.width = sourceCanvas.width;
    exportCanvas.height = sourceCanvas.height;
    const exportCtx = exportCanvas.getContext("2d");
    if (!exportCtx) {{
      console.error("Viser export failed: could not create export canvas.");
      return;
    }}

    exportCtx.clearRect(0, 0, exportCanvas.width, exportCanvas.height);
    exportCtx.drawImage(sourceCanvas, 0, 0);

    let finalCanvas = exportCanvas;
    if ({width} > 0 && {height} > 0 && (exportCanvas.width !== {width} || exportCanvas.height !== {height})) {{
      const resizedCanvas = document.createElement("canvas");
      resizedCanvas.width = {width};
      resizedCanvas.height = {height};
      const resizedCtx = resizedCanvas.getContext("2d");
      if (!resizedCtx) {{
        console.error("Viser export failed: could not create resized canvas.");
        return;
      }}
      resizedCtx.clearRect(0, 0, resizedCanvas.width, resizedCanvas.height);
      resizedCtx.drawImage(exportCanvas, 0, 0, resizedCanvas.width, resizedCanvas.height);
      finalCanvas = resizedCanvas;
    }}

    finalCanvas.toBlob((blob) => {{
      if (!blob) {{
        console.error("Viser export failed: canvas.toBlob() returned null.");
        return;
      }}
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = {json.dumps(filename)};
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.setTimeout(() => URL.revokeObjectURL(url), 1000);
    }}, "image/png");
  }};

  requestAnimationFrame(() => requestAnimationFrame(doExport));
}})();
"""
                client._websock_connection.queue_message(
                    viser_messages.RunJavascriptMessage(source=export_script)
                )
                print(f"Triggered browser-local PNG export: {filename}")
            except Exception as exc:
                print(f"Failed to trigger browser-local PNG export: {exc}")
            finally:
                with self._export_lock:
                    self._export_in_progress = False
                    self.export_button.disabled = False

        self.server.on_client_connect(self._connect_client)

    @staticmethod
    def _normalize_export_filename(filename):
        filename = str(filename).strip()
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"viewer_snapshot_transparent_{timestamp}.png"
        if not filename.lower().endswith(".png"):
            filename = f"{filename}.png"
        return os.path.basename(filename)

    @staticmethod
    def _encode_png_bytes(image):
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dimensions, got shape {image.shape}.")

        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating):
                image = np.clip(image, 0.0, 255.0)
                if image.max() <= 1.0:
                    image = image * 255.0
            image = image.astype(np.uint8)

        if image.shape[-1] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGRA)
        elif image.shape[-1] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise ValueError(
                f"Expected RGB or RGBA image for PNG export, got shape {image.shape}."
            )

        success, encoded = cv2.imencode(".png", image)
        if not success:
            raise RuntimeError("OpenCV failed to encode PNG bytes.")
        return encoded.tobytes()

    @staticmethod
    def _capture_client_render(client, width, height):
        if hasattr(client, "get_render"):
            return client.get_render(height=height, width=width, transport_format="png")
        if hasattr(client, "camera") and hasattr(client.camera, "get_render"):
            return client.camera.get_render(
                height=height,
                width=width,
                transport_format="png",
            )
        raise AttributeError("viser client does not expose a render capture API.")

    def _run_export_capture(self, client, width, height, filename):
        try:
            image = self._capture_client_render(client, width, height)
            client.send_file_download(filename, self._encode_png_bytes(image))
            print(f"Exported transparent PNG to client download: {filename}")
        except Exception as exc:
            print(f"Failed to export transparent PNG: {exc}")
        finally:
            with self._export_lock:
                self._viewer_updates_paused = False
                self._export_in_progress = False
                self.export_button.disabled = False

    @staticmethod
    def _format_camera_vector(values):
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        return json.dumps([float(v) for v in values], ensure_ascii=True)

    @staticmethod
    def _parse_camera_vector(raw_value, expected_length, field_name):
        text = str(raw_value).strip()
        if not text:
            raise ValueError(f"{field_name} is empty.")

        parsed = None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(text)
            except (ValueError, SyntaxError):
                normalized = text
                if normalized.startswith("[") and normalized.endswith("]"):
                    normalized = normalized[1:-1]
                normalized = normalized.replace(",", " ")
                parsed = np.fromstring(normalized, sep=" ", dtype=np.float64)

        values = np.asarray(parsed, dtype=np.float64).reshape(-1)
        if values.size != expected_length:
            raise ValueError(
                f"{field_name} must contain {expected_length} numbers, got {values.size}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{field_name} must contain only finite numbers.")
        return values.astype(np.float32)

    @staticmethod
    def _format_fov_degrees(fov_radians):
        return f"{np.rad2deg(float(fov_radians)):.8f}"

    def _sync_camera_panels(self, client, wxyz_panel, position_panel, fov_panel, aspect_panel):
        with self.server.atomic():
            wxyz_panel.value = self._format_camera_vector(client.camera.wxyz)
            position_panel.value = self._format_camera_vector(client.camera.position)
            fov_panel.value = self._format_fov_degrees(client.camera.fov)
            aspect_panel.value = f"{float(client.camera.aspect):.8f}"

    def _get_camera_payload(self, client):
        return {
            "format": _CAMERA_PRESET_FORMAT,
            "version": _CAMERA_PRESET_VERSION,
            "camera": {
                "wxyz": [float(x) for x in np.asarray(client.camera.wxyz, dtype=np.float64)],
                "position": [
                    float(x) for x in np.asarray(client.camera.position, dtype=np.float64)
                ],
                "fov_degrees": float(np.rad2deg(float(client.camera.fov))),
                "aspect": float(client.camera.aspect),
            },
        }

    def _parse_camera_payload(self, payload):
        if not isinstance(payload, dict):
            raise ValueError("Camera preset must be a JSON object.")
        if payload.get("format") != _CAMERA_PRESET_FORMAT:
            raise ValueError(
                f"Unsupported camera preset format: expected '{_CAMERA_PRESET_FORMAT}'."
            )
        version = payload.get("version")
        if version != _CAMERA_PRESET_VERSION:
            raise ValueError(
                f"Unsupported camera preset version: expected {_CAMERA_PRESET_VERSION}, got {version}."
            )
        camera_payload = payload.get("camera")
        if not isinstance(camera_payload, dict):
            raise ValueError("Camera preset is missing the 'camera' object.")

        wxyz = self._parse_camera_vector(camera_payload.get("wxyz"), 4, "camera.wxyz")
        position = self._parse_camera_vector(
            camera_payload.get("position"), 3, "camera.position"
        )
        fov_degrees = float(camera_payload.get("fov_degrees"))
        aspect = float(camera_payload.get("aspect"))

        if not np.isfinite(fov_degrees) or fov_degrees <= 0.0 or fov_degrees >= 179.0:
            raise ValueError("camera.fov_degrees must be in (0, 179).")
        if not np.isfinite(aspect) or aspect <= 0.0:
            raise ValueError("camera.aspect must be positive.")

        quat_norm = np.linalg.norm(wxyz)
        if not np.isfinite(quat_norm) or quat_norm <= 0.0:
            raise ValueError("camera.wxyz must have non-zero length.")
        wxyz = (wxyz / quat_norm).astype(np.float32)

        return {
            "wxyz": wxyz,
            "position": position.astype(np.float32),
            "fov_degrees": float(fov_degrees),
            "aspect": float(aspect),
        }

    @staticmethod
    def _normalize_camera_preset_filename(filename):
        filename = str(filename).strip()
        if not filename:
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            filename = f"viewer_camera_{timestamp}.json"
        if not filename.lower().endswith(".json"):
            filename = f"{filename}.json"
        return os.path.basename(filename)

    def _apply_camera_values(self, client, camera_values):
        with self.server.atomic():
            client.camera.wxyz = camera_values["wxyz"]
            client.camera.position = camera_values["position"]
            client.camera.fov = np.deg2rad(camera_values["fov_degrees"])

    def get_camera_state(self, client: viser.ClientHandle) -> CameraState:
        camera = client.camera
        c2w = np.concatenate(
            [
                np.concatenate(
                    [tf.SO3(camera.wxyz).as_matrix(), camera.position[:, None]], 1
                ),
                [[0, 0, 0, 1]],
            ],
            0,
        )
        return CameraState(
            fov=camera.fov,
            aspect=camera.aspect,
            c2w=c2w,
        )

    @staticmethod
    def generate_pseudo_intrinsics(h, w):
        focal = (h**2 + w**2) ** 0.5
        return np.array([[focal, 0, w // 2], [0, focal, h // 2], [0, 0, 1]]).astype(
            np.float32
        )

    def get_ray_map(self, c2w, h, w, intrinsics=None):
        if intrinsics is None:
            intrinsics = self.generate_pseudo_intrinsics(h, w)
        i, j = np.meshgrid(np.arange(w), np.arange(h), indexing="xy")
        grid = np.stack([i, j, np.ones_like(i)], axis=-1)
        ro = c2w[:3, 3]
        rd = np.linalg.inv(intrinsics) @ grid.reshape(-1, 3).T
        rd = (c2w @ np.vstack([rd, np.ones_like(rd[0])])).T[:, :3].reshape(h, w, 3)
        rd = rd / np.linalg.norm(rd, axis=-1, keepdims=True)
        ro = np.broadcast_to(ro, (h, w, 3))
        ray_map = np.concatenate([ro, rd], axis=-1)
        return ray_map

    def set_camera_loc(camera, pose, K):
        """
        pose: 4x4 matrix
        K: 3x3 matrix
        """
        fx, fy = K[0, 0], K[1, 1]
        cx, cy = K[0, 2], K[1, 2]
        fov = 2 * np.arctan(2 * cx / fx)
        wxyz_xyz = tf.SE3.from_matrix(pose).wxyz_xyz
        wxyz = wxyz_xyz[:4]
        xyz = wxyz_xyz[4:]
        camera.wxyz = wxyz
        camera.position = xyz
        camera.fov = fov

    def _connect_client(self, client: viser.ClientHandle):
        from src.dust3r.inference import inference_step
        from src.dust3r.utils.geometry import geotrf

        wxyz_panel = client.gui.add_text(
            "wxyz:", self._format_camera_vector(client.camera.wxyz)
        )
        position_panel = client.gui.add_text(
            "position:", self._format_camera_vector(client.camera.position)
        )
        fov_panel = client.gui.add_text(
            "fov:", self._format_fov_degrees(client.camera.fov)
        )
        aspect_panel = client.gui.add_text("aspect:", f"{float(client.camera.aspect):.8f}")
        save_camera_panel = client.gui.add_button(
            "Save View",
            hint="Write the current camera pose and intrinsics into the text fields below.",
        )
        apply_camera_panel = client.gui.add_button(
            "Apply View",
            hint="Parse the text fields below and apply wxyz / position / fov to the current camera. Aspect is recorded but follows the viewer canvas shape in viser.",
        )
        camera_file_panel = client.gui.add_text(
            "view file:",
            "viewer_camera.json",
            hint="Filename used for direct download of the camera preset JSON file.",
        )
        save_camera_file_button = client.gui.add_button(
            "Save View File",
            hint="Download the current camera into a reusable JSON preset file.",
        )
        upload_camera_button = client.gui.add_upload_button(
            "Upload View File",
            hint="Upload a camera preset JSON file and apply it immediately.",
            mime_type="application/json",
        )

        @client.camera.on_update
        def _(_: viser.CameraHandle):
            self._sync_camera_panels(
                client,
                wxyz_panel,
                position_panel,
                fov_panel,
                aspect_panel,
            )

        @save_camera_panel.on_click
        def _(_: viser.GuiEvent) -> None:
            self._sync_camera_panels(
                client,
                wxyz_panel,
                position_panel,
                fov_panel,
                aspect_panel,
            )
            print("Saved current viewer camera parameters into the control panel.")

        @apply_camera_panel.on_click
        def _(_: viser.GuiEvent) -> None:
            try:
                wxyz = self._parse_camera_vector(wxyz_panel.value, 4, "wxyz")
                position = self._parse_camera_vector(position_panel.value, 3, "position")
                fov_degrees = float(str(fov_panel.value).strip())
                aspect = float(str(aspect_panel.value).strip())
            except ValueError as exc:
                print(f"Failed to apply viewer camera parameters: {exc}")
                return

            if not np.isfinite(fov_degrees) or fov_degrees <= 0.0 or fov_degrees >= 179.0:
                print("Failed to apply viewer camera parameters: fov must be in (0, 179) degrees.")
                return
            if not np.isfinite(aspect) or aspect <= 0.0:
                print("Failed to apply viewer camera parameters: aspect must be positive.")
                return

            quat_norm = np.linalg.norm(wxyz)
            if not np.isfinite(quat_norm) or quat_norm <= 0.0:
                print("Failed to apply viewer camera parameters: wxyz must have non-zero length.")
                return
            wxyz = (wxyz / quat_norm).astype(np.float32)

            self._apply_camera_values(
                client,
                {
                    "wxyz": wxyz,
                    "position": position,
                    "fov_degrees": float(fov_degrees),
                    "aspect": float(aspect),
                },
            )

            self._sync_camera_panels(
                client,
                wxyz_panel,
                position_panel,
                fov_panel,
                aspect_panel,
            )
            if abs(float(aspect) - float(client.camera.aspect)) > 1e-6:
                print(
                    "Applied viewer camera parameters from the control panel. "
                    f"Requested aspect={float(aspect):.8f}, current viewer aspect={float(client.camera.aspect):.8f}; "
                    "viser derives aspect from the browser canvas and does not allow assigning it."
                )
            else:
                print("Applied viewer camera parameters from the control panel.")

        @save_camera_file_button.on_click
        def _(_: viser.GuiEvent) -> None:
            try:
                self._sync_camera_panels(
                    client,
                    wxyz_panel,
                    position_panel,
                    fov_panel,
                    aspect_panel,
                )
                filename = self._normalize_camera_preset_filename(camera_file_panel.value)
                payload = self._get_camera_payload(client)
                payload_bytes = (json.dumps(payload, indent=2, ensure_ascii=True) + "\n").encode(
                    "utf-8"
                )
                client.send_file_download(filename, payload_bytes)
            except Exception as exc:
                print(f"Failed to save viewer camera preset: {exc}")
                return

            camera_file_panel.value = filename
            print(f"Triggered viewer camera preset download: {filename}")

        @upload_camera_button.on_upload
        def _(_: viser.GuiEvent) -> None:
            try:
                uploaded = upload_camera_button.value
                payload = json.loads(uploaded.content.decode("utf-8"))
                camera_values = self._parse_camera_payload(payload)
                self._apply_camera_values(client, camera_values)
            except Exception as exc:
                print(f"Failed to upload viewer camera preset: {exc}")
                return

            if uploaded.name:
                camera_file_panel.value = self._normalize_camera_preset_filename(uploaded.name)
            self._sync_camera_panels(
                client,
                wxyz_panel,
                position_panel,
                fov_panel,
                aspect_panel,
            )
            requested_aspect = float(camera_values["aspect"])
            current_aspect = float(client.camera.aspect)
            if abs(requested_aspect - current_aspect) > 1e-6:
                print(
                    f"Uploaded and applied viewer camera preset: {uploaded.name or '<unnamed>'}. "
                    f"Requested aspect={requested_aspect:.8f}, current viewer aspect={current_aspect:.8f}; "
                    "viser derives aspect from the browser canvas and does not allow assigning it."
                )
            else:
                print(f"Uploaded and applied viewer camera preset: {uploaded.name or '<unnamed>'}")

        # gui_set_current_camera = client.gui.add_button(
        #     "Set Current Camera to Infer Raymap"
        # )

        # @gui_set_current_camera.on_click
        # def _(_) -> None:
        #     try:
        #         cam = self.get_camera_state(client)
        #         cam.fov = 2 * np.arctan(self.size / self.focal_slider.value)
        #         cam.aspect = (512 / 384) if self.size==512 else 1.0
        #         pose = cam.c2w
        #         if self.size == 512:
        #             intrins = self.generate_pseudo_intrinsics(384, 512)
        #             raymap = torch.from_numpy(self.get_ray_map(pose, 384, 512, intrins))[
        #                 None
        #             ].float()
        #         else:
        #             intrins = self.generate_pseudo_intrinsics(224, 224)
        #             raymap = torch.from_numpy(self.get_ray_map(pose, 224, 224, intrins))[
        #                 None
        #             ].float()
                
                
        #         view = {
        #             "img": torch.full((1, 3, 384, 512), torch.nan) if self.size==512 else torch.full((1, 3, 224, 224), torch.nan),
        #             "ray_map": raymap,
        #             "true_shape": torch.from_numpy(np.int32([raymap.shape[1:-1]])),
        #             "idx": self.num_frames + 1,
        #             "instance": str(self.num_frames + 1),
        #             "camera_pose": torch.from_numpy(np.eye(4).astype(np.float32)).unsqueeze(
        #                 0
        #             ),
        #             "img_mask": torch.tensor(False).unsqueeze(0),
        #             "ray_mask": torch.tensor(True).unsqueeze(0),
        #             "update": torch.tensor(False).unsqueeze(0),
        #             "reset": torch.tensor(False).unsqueeze(0),
        #         }
        #         print("Start Inference Raymap")
        #         output = inference_step(
        #             view, self.state_args[-1], self.model, device=self.device
        #         )
        #         print("Finish Inference Raymap")
        #         pts3ds = output["pred"]["pts3d_in_self_view"].cpu().numpy()
        #         pts3ds = geotrf(pose[None], pts3ds)
        #         colors = 0.5 * (output["pred"]["rgb"].cpu().numpy() + 1.0)
        #         depthmap = output["pred"]["pts3d_in_self_view"].cpu().numpy()[0][..., -1]
        #         conf = output["pred"]["conf"].cpu().numpy()
        #         disp = 1.0 / depthmap
        #         pts3ds, colors = self.parse_pc_data(
        #             pts3ds, colors, set_border_color=True, 
        #             downsample_factor=self.downsample_slider.value
        #         )
        #         mask = (conf > 1.0).reshape(-1)
        #         self.num_frames += 1
        #         self.pc_handles.append(
        #             self.server.add_point_cloud(
        #                 name=f"/frames/{self.num_frames-1}/pred_pts",
        #                 points=pts3ds[mask],
        #                 colors=colors[mask],
        #                 point_size=0.005,
        #             )
        #         )

        #         self.server.add_camera_frustum(
        #             name=f"/frames/{self.num_frames-1}/camera",
        #             fov=cam.fov,
        #             aspect=cam.aspect,
        #             wxyz=client.camera.wxyz,
        #             position=client.camera.position,
        #             scale=0.1,
        #             color=[64, 179, 230],
        #         )
        #         print("Adding new pointcloud: ", pts3ds.shape)
        #     except Exception as e:
        #         print(e)

    @staticmethod
    def set_color_border(image, border_width=5, color=[1, 0, 0]):

        image[:border_width, :, 0] = color[0]  # Red channel
        image[:border_width, :, 1] = color[1]  # Green channel
        image[:border_width, :, 2] = color[2]  # Blue channel
        image[-border_width:, :, 0] = color[0]
        image[-border_width:, :, 1] = color[1]
        image[-border_width:, :, 2] = color[2]

        image[:, :border_width, 0] = color[0]
        image[:, :border_width, 1] = color[1]
        image[:, :border_width, 2] = color[2]
        image[:, -border_width:, 0] = color[0]
        image[:, -border_width:, 1] = color[1]
        image[:, -border_width:, 2] = color[2]

        return image

    def read_data(
        self,
        pc_list,
        color_list,
        conf_list,
        edge_color_list=None,
        camera_color_list=None,
    ):
        pcs = {}
        step_list = []
        for i, pc in enumerate(pc_list):
            step = i
            pcs.update(
                {
                    step: {
                        "pc": pc,
                        "color": color_list[i],
                        "conf": conf_list[i],
                        "edge_color": (
                            None if edge_color_list[i] is None else edge_color_list[i]
                        ),
                    }
                }
            )
            step_list.append(step)

        num_cameras = len(pc_list)
        if camera_color_list is not None:
            # use caller-provided per-camera colors; values may be 0-255 or 0-1
            camera_colors = np.asarray(camera_color_list, dtype=np.float64)
            if camera_colors.ndim == 1:
                camera_colors = camera_colors[None, :]
            if camera_colors.shape[0] < num_cameras:
                pad = np.tile(
                    camera_colors[-1] if camera_colors.shape[0] > 0 else np.array([0.5, 0.5, 0.5, 1.0]),
                    (num_cameras - camera_colors.shape[0], 1),
                )
                camera_colors = np.concatenate([camera_colors, pad], axis=0)
            if camera_colors.shape[1] == 3:
                alpha = np.ones((camera_colors.shape[0], 1), dtype=np.float64)
                camera_colors = np.concatenate([camera_colors, alpha], axis=1)
            if camera_colors[:, :3].max() > 1.0:
                camera_colors[:, :3] = camera_colors[:, :3] / 255.0
            self.camera_colors = camera_colors
        else:
            # generate camera gradient colors
            if num_cameras > 1:
                normalized_indices = np.array(list(range(num_cameras))) / (num_cameras - 1)
            else:
                normalized_indices = np.array([0.0])
            cmap = cm.viridis
            self.camera_colors = cmap(normalized_indices)
        return pcs, step_list

    def _stable_random_sample(self, points, colors, step):
        if self.max_points <= 0 or len(points) <= self.max_points:
            return points, colors

        rng = np.random.default_rng(0 if step is None else int(step))
        indices = rng.choice(len(points), size=self.max_points, replace=False)
        return points[indices], colors[indices]

    def parse_pc_data(
        self,
        pc,
        color,
        conf=None,
        edge_color=[0.251, 0.702, 0.902],
        set_border_color=False,
        downsample_factor=1,
        step=None,
    ):

        pred_pts = pc.reshape(-1, 3)  # [N, 3]
        color_img = color[0] if np.asarray(color).ndim == 4 else color

        if set_border_color and edge_color is not None:
            color = self.set_color_border(color[0], color=edge_color)
        if np.isnan(color).any():

            color = np.zeros((pred_pts.shape[0], 3))
            color[:, 2] = 1
        else:
            color = color.reshape(-1, 3)
        count = min(len(pred_pts), len(color))
        pred_pts = pred_pts[:count]
        color = color[:count]

        keep = np.ones(count, dtype=bool)
        if self.remove_sky and count > 0:
            try:
                sky_mask = to_numpy(segment_sky(color_img)).reshape(-1).astype(bool)
                if len(sky_mask) < count:
                    padded_sky_mask = np.zeros(count, dtype=bool)
                    padded_sky_mask[: len(sky_mask)] = sky_mask
                    sky_mask = padded_sky_mask
                keep &= ~sky_mask[:count]
            except Exception as exc:
                print(f"Warning: failed to remove sky in viewer: {exc}")
        if conf is not None:
            conf = conf[0].reshape(-1)
            conf_keep = np.zeros(count, dtype=bool)
            valid_count = min(count, len(conf))
            conf_keep[:valid_count] = conf[:valid_count] > self.vis_threshold
            keep &= conf_keep

        pred_pts = pred_pts[keep]
        color = color[keep]
        
        # apply downsample
        if downsample_factor > 1 and len(pred_pts) > 0:
            indices = np.arange(0, len(pred_pts), downsample_factor)
            pred_pts = pred_pts[indices]
            color = color[indices]

        pred_pts, color = self._stable_random_sample(pred_pts, color, step)

        return pred_pts, color

    def refresh_point_clouds(self):
        if not hasattr(self, "frame_nodes"):
            return

        for handle in self.pc_handles:
            try:
                handle.remove()
            except (KeyError, AttributeError):
                pass
        self.pc_handles.clear()
        self.vis_pts_list.clear()

        for step in self.all_steps:
            self.add_pc(step)

    def add_pc(self, step):
        pc = self.pcs[step]["pc"]
        color = self.pcs[step]["color"]
        conf = self.pcs[step]["conf"]
        edge_color = self.pcs[step].get("edge_color", None)

        pred_pts, color = self.parse_pc_data(
            pc,
            color,
            conf,
            edge_color,
            set_border_color=True,
            downsample_factor=self.downsample_slider.value,
            step=step,
        )

        self.vis_pts_list.append(pred_pts)
        self.pc_handles.append(
            self.server.add_point_cloud(
                name=f"/frames/{step}/pred_pts",
                points=pred_pts,
                colors=color,
                point_size=self.psize_slider.value,
            )
        )

    def add_camera(self, step):
        cam = self.cam_dict
        focal = cam["focal"][step]
        pp = cam["pp"][step]
        R = cam["R"][step]
        t = cam["t"][step]

        q = tf.SO3.from_matrix(R).wxyz
        fov = 2 * np.arctan(pp[0] / focal)
        aspect = pp[0] / pp[1]
        self.traj_list.append((q, t))
        
        # use gradient color instead of hardcoded green
        # find the index of step in all_steps
        step_index = self.all_steps.index(step) if step in self.all_steps else 0
        camera_color = self.camera_colors[step_index]
        # convert to 0-255 range RGB values
        camera_color_rgb = tuple((camera_color[:3] * 255).astype(int))
        
        self.cam_handles.append(
            self.server.add_camera_frustum(
                name=f"/frames/{step}/camera",
                fov=fov,
                aspect=aspect,
                wxyz=q,
                position=t,
                scale=0.1,
                color=camera_color_rgb,
                # color=(50, 205, 50),
            )
        )

    def animate(self):
        with self.server.add_gui_folder("Playback"):
            gui_timestep = self.server.add_gui_slider(
                "Train Step",
                min=0,
                max=self.num_frames - 1,
                step=1,
                initial_value=0,
                disabled=False,
            )
            gui_next_frame = self.server.add_gui_button("Next Step", disabled=False)
            gui_prev_frame = self.server.add_gui_button("Prev Step", disabled=False)
            gui_playing = self.server.add_gui_checkbox("Playing", False)
            gui_framerate = self.server.add_gui_slider(
                "FPS", min=1, max=60, step=0.1, initial_value=1
            )
            gui_framerate_options = self.server.add_gui_button_group(
                "FPS options", ("10", "20", "30", "60")
            )

        @gui_next_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

        @gui_prev_frame.on_click
        def _(_) -> None:
            gui_timestep.value = (gui_timestep.value - 1) % self.num_frames

        @gui_playing.on_update
        def _(_) -> None:
            gui_timestep.disabled = gui_playing.value
            gui_next_frame.disabled = gui_playing.value
            gui_prev_frame.disabled = gui_playing.value

        @gui_framerate_options.on_click
        def _(_) -> None:
            gui_framerate.value = int(gui_framerate_options.value)

        prev_timestep = gui_timestep.value

        @gui_timestep.on_update
        def _(_) -> None:
            nonlocal prev_timestep
            if self._viewer_updates_paused:
                return
            current_timestep = gui_timestep.value
            with self.server.atomic():
                self.frame_nodes[current_timestep].visible = True
                self.frame_nodes[prev_timestep].visible = False
            prev_timestep = current_timestep
            self.server.flush()  # Optional!

        self.server.add_frame(
            "/frames",
            show_axes=False,
        )
        self.frame_nodes = []
        for i in range(self.num_frames):
            step = self.all_steps[i]
            self.frame_nodes.append(
                self.server.add_frame(
                    f"/frames/{step}",
                    show_axes=False,
                )
            )
            self.add_pc(step)
            if self.show_camera:
                # decide whether to add camera according to the downsample factor
                downsample_factor = int(self.camera_downsample_slider.value)
                if i % downsample_factor == 0:
                    self.add_camera(step)

        prev_timestep = gui_timestep.value
        while True:
            if self._viewer_updates_paused:
                time.sleep(0.01)
                continue
            if self.on_replay:
                pass
            else:
                if gui_playing.value:
                    gui_timestep.value = (gui_timestep.value + 1) % self.num_frames

                for i, frame_node in enumerate(self.frame_nodes):
                    frame_node.visible = (
                        i <= gui_timestep.value
                        if not self.fourd
                        else i == gui_timestep.value
                    )

            time.sleep(1.0 / gui_framerate.value)

    def run(self):
        self.animate()
        while True:
            time.sleep(10.0)
