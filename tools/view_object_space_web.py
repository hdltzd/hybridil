#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ D455 OBJECT_SPACE 点云裁剪范围实时Web预览工具【带3D点云渲染】
适配遨博AUBO C5按摩机器人
✅ 2D RGB实时画面 + 3D彩色点云 + OBJECT_SPACE边界框可视化
✅ 无X11、无外网/CDN依赖，浏览器即可访问，支持远程SSH访问
✅ 相机坐标系下的预处理：裁剪 → 体素下采样 → 固定点数采样

【使用方法】
1. 安装依赖：pip install numpy opencv-python pyrealsense2 open3d
2. 运行：python tools/view_object_space_web.py
3. 浏览器打开终端显示的地址（默认http://<IP>:8766）
4. 交互：
   - 2D画面：实时显示RGB图像和点云统计
   - 3D点云：左键旋转/滚轮缩放/右键平移
   - 右侧面板调整X/Y/Z裁剪范围，点击应用实时生效
   - 一键保存点云为PLY文件
"""
import os
import sys
import time
import json
import threading
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import numpy as np
import cv2

# 中文绘制支持
try:
    from PIL import Image, ImageDraw, ImageFont
    _FONT_PATHS = [
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
        "C:/Windows/Fonts/msyh.ttc",
        "/System/Library/Fonts/PingFang.ttc",
    ]
    _CN_FONT = None
    for fp in _FONT_PATHS:
        if os.path.exists(fp):
            try:
                _CN_FONT = ImageFont.truetype(fp, 20)
                break
            except Exception:
                continue
    _USE_CN_TEXT = _CN_FONT is not None
except ImportError:
    _USE_CN_TEXT = False
    _CN_FONT = None


def put_cn_text(img, text, pos, color=(0, 255, 0), font_scale=0.6, thickness=2, fallback_text=None):
    """中文文本绘制，无中文字体时自动降级为英文"""
    if fallback_text:
        cv2.putText(img, fallback_text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return img
    text = str(text).translate(str.maketrans({
        '✅': '', '❌': '', '⚠': '', '💡': '', '📌': '', '📏': '', '🔍': '',
        '📂': '', '📸': '', '🌐': '', '🚀': '', '🛑': '', '️': '',
    }))
    if not _USE_CN_TEXT:
        return img
    if all(ord(c) < 128 for c in text):
        cv2.putText(img, text, pos, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
        return img
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    pil_color = (color[2], color[1], color[0])
    draw.text(pos, text, font=_CN_FONT, fill=pil_color)
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    return img


def validate_bounds(bounds):
    """Return finite (3, 2) bounds whose lower limits are below upper limits."""
    result = np.asarray(bounds, dtype=np.float32)
    if result.shape != (3, 2) or not np.isfinite(result).all():
        raise ValueError("bounds must be a finite 3x2 array")
    if np.any(result[:, 0] >= result[:, 1]):
        raise ValueError("each lower bound must be below its upper bound")
    return result


def voxel_downsample(xyz, colors, voxel_size):
    """Keep the first point in each voxel, matching deployment semantics."""
    if voxel_size <= 0 or len(xyz) == 0:
        return xyz, colors
    voxel_indices = np.floor(xyz / voxel_size).astype(np.int64)
    _, selected = np.unique(voxel_indices, axis=0, return_index=True)
    return xyz[selected], colors[selected]


class D455:
    """RealSense D455相机驱动"""
    def __init__(self, serial=None, width=1280, height=720, fps=30):
        try:
            import pyrealsense2 as rs
        except ImportError:
            raise RuntimeError("未安装pyrealsense2，请执行 pip install pyrealsense2")
        self.rs = rs
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        if serial is not None and serial:
            self.config.enable_device(str(serial))
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.profile = self.pipeline.start(self.config)
        self.align = rs.align(rs.stream.color)
        color_profile = self.profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self.depth_scale = self.profile.get_device().first_depth_sensor().get_depth_scale()
        self._pixel_grid = None
        # 预热相机
        for _ in range(20):
            self.pipeline.wait_for_frames()
        print(f"✅ D455初始化完成，分辨率：{width}x{height} @ {fps}fps")

    def get(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        return np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data())

    def get_pointcloud(self, color_image, depth_image, object_bounds, voxel_size):
        """Build and crop a colored point cloud in the D455 camera frame."""
        h, w = depth_image.shape
        fx, fy = self.intrinsics.fx, self.intrinsics.fy
        cx, cy = self.intrinsics.ppx, self.intrinsics.ppy
        depth_scale = self.depth_scale

        if self._pixel_grid is None or self._pixel_grid[0].shape != (h * w,):
            v, u = np.indices((h, w), dtype=np.float32)
            self._pixel_grid = (u.reshape(-1), v.reshape(-1))
        u, v = self._pixel_grid
        depth = depth_image.reshape(-1).astype(np.float32) * depth_scale

        valid_mask = depth > 0
        u, v, depth = u[valid_mask], v[valid_mask], depth[valid_mask]
        # RealSense color stream is BGR for OpenCV; browser/Open3D require RGB.
        colors = color_image.reshape(-1, 3)[valid_mask, ::-1].astype(np.float32) / 255.0

        if len(depth) == 0:
            return np.zeros((0,3), dtype=np.float32), np.zeros((0,3), dtype=np.float32), 0

        # 像素坐标转相机坐标系
        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth
        xyz = np.stack([x, y, z], axis=1)

        # OBJECT_SPACE裁剪
        bounds = validate_bounds(object_bounds)
        crop_mask = np.all((xyz > bounds[:, 0]) & (xyz < bounds[:, 1]), axis=1)
        xyz = xyz[crop_mask]
        colors = colors[crop_mask]
        raw_count = len(xyz)

        if raw_count == 0:
            return np.zeros((0,3), dtype=np.float32), np.zeros((0,3), dtype=np.float32), 0

        xyz, colors = voxel_downsample(xyz, colors, voxel_size)

        return xyz.astype(np.float32), colors.astype(np.float32), raw_count

    def stop(self):
        if hasattr(self, 'pipeline'):
            self.pipeline.stop()
        print("[D455] 相机已停止")


def get_lan_ip():
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(('8.8.8.8', 80))
        return sock.getsockname()[0]
    except OSError:
        return '127.0.0.1'
    finally:
        sock.close()


class BrowserPreview:
    """Thread-safe latest-frame store and small HTTP API for the browser UI."""

    def __init__(self, host='0.0.0.0', port=8766, jpeg_quality=80,
                 default_bounds=None, output_dir=None):
        self.host = host
        self.port = port
        self.jpeg_quality = jpeg_quality
        self.default_bounds = default_bounds if default_bounds is not None else (
            (-0.113, 0.031), (-0.075, 0.116), (0.7, 0.95)
        )
        self.output_dir = Path(output_dir or Path.cwd() / "object_space_exports")
        self._condition = threading.Condition()
        self._jpeg = None
        self._jpeg_version = 0
        self._running = False
        self._server = None
        self._thread = None
        self._control_lock = threading.Lock()

        self._bounds = validate_bounds(self.default_bounds)
        self._frame_version = 0
        self._stats = {
            "raw_count": 0,
            "sampled_count": 0,
            "pointcloud": None,
            "pointcloud_rgb": None,
            "point_x": [0, 0],
            "point_y": [0, 0],
            "point_z": [0, 0],
            "fps": 0,
            "pointcloud_version": 0,
        }

    def start(self):
        preview = self

        class PreviewHandler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path in ('/', '/index.html'):
                    html = generate_index_html(preview.default_bounds)
                    content = html.encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/html; charset=utf-8')
                    self.send_header('Content-Length', str(len(content)))
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(content)
                    return

                if self.path == '/stream.mjpg':
                    self.send_response(200)
                    self.send_header('Age', '0')
                    self.send_header('Cache-Control', 'no-cache, private')
                    self.send_header('Pragma', 'no-cache')
                    self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
                    self.end_headers()
                    try:
                        last_jpeg_version = -1
                        while preview._running:
                            with preview._condition:
                                preview._condition.wait_for(
                                    lambda: not preview._running or preview._jpeg_version != last_jpeg_version,
                                    timeout=2.0,
                                )
                                jpeg = preview._jpeg
                                last_jpeg_version = preview._jpeg_version
                            if jpeg is None:
                                continue
                            self.wfile.write(b'--FRAME\r\n')
                            self.send_header('Content-Type', 'image/jpeg')
                            self.send_header('Content-Length', str(len(jpeg)))
                            self.end_headers()
                            self.wfile.write(jpeg)
                            self.wfile.write(b'\r\n')
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return

                if self.path == '/api/status':
                    with preview._control_lock:
                        stats = {
                            "fps": round(preview._stats["fps"], 1),
                            "raw_count": preview._stats["raw_count"],
                            "sampled_count": preview._stats["sampled_count"],
                            "point_x": preview._stats["point_x"],
                            "point_y": preview._stats["point_y"],
                            "point_z": preview._stats["point_z"],
                            "bounds": preview._bounds.tolist(),
                            "pointcloud_version": preview._stats["pointcloud_version"],
                        }
                    body = json.dumps(stats, ensure_ascii=False).encode('utf-8')
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json; charset=utf-8')
                    self.send_header('Content-Length', str(len(body)))
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(body)
                    return

                if self.path == '/api/pointcloud.bin':
                    with preview._control_lock:
                        xyz = preview._stats["pointcloud"]
                        rgb = preview._stats.get("pointcloud_rgb", None)
                        if xyz is None or len(xyz) == 0:
                            self.send_response(204)
                            self.end_headers()
                            return
                        xyz = xyz.astype(np.float32)
                        version = preview._stats["pointcloud_version"]
                        if rgb is None or len(rgb) != len(xyz):
                            rgb = np.ones_like(xyz) * np.array([0.2, 0.8, 0.2], dtype=np.float32)
                        else:
                            rgb = rgb.astype(np.float32)
                        data = np.concatenate([xyz, rgb], axis=1).tobytes()
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/octet-stream')
                    self.send_header('Content-Length', str(len(data)))
                    self.send_header('X-Pointcloud-Version', str(version))
                    self.send_header('Cache-Control', 'no-store')
                    self.end_headers()
                    self.wfile.write(data)
                    return

                self.send_error(404)

            def do_POST(self):
                if self.path == '/api/bounds':
                    try:
                        length = min(int(self.headers.get('Content-Length', '0')), 2048)
                        payload = json.loads(self.rfile.read(length).decode('utf-8'))
                        new_bounds = validate_bounds(payload["bounds"])
                    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                        self.send_error(400, f'Invalid bounds: {error}')
                        return
                    with preview._control_lock:
                        preview._bounds = new_bounds
                    self.send_response(200)
                    self.send_header('Content-Type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True}).encode('utf-8'))
                    return

                if self.path == '/api/save':
                    try:
                        import open3d as o3d
                        with preview._control_lock:
                            xyz = preview._stats["pointcloud"]
                            if xyz is None or len(xyz) == 0:
                                raise RuntimeError("当前无有效点云")
                            xyz = xyz.copy()
                            rgb = preview._stats["pointcloud_rgb"]
                            rgb = None if rgb is None else rgb.copy()
                        pcd = o3d.geometry.PointCloud()
                        pcd.points = o3d.utility.Vector3dVector(xyz)
                        if rgb is not None and rgb.shape == xyz.shape:
                            pcd.colors = o3d.utility.Vector3dVector(rgb)
                        preview.output_dir.mkdir(parents=True, exist_ok=True)
                        save_path = preview.output_dir / f"object_space_{time.time_ns()}.ply"
                        if not o3d.io.write_point_cloud(str(save_path), pcd):
                            raise RuntimeError("Open3D 写入 PLY 失败")
                        body = json.dumps({"success": True, "path": os.path.abspath(save_path)}).encode('utf-8')
                        self.send_response(200)
                        self.send_header('Content-Type', 'application/json; charset=utf-8')
                        self.send_header('Content-Length', str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return
                    except Exception as e:
                        body = json.dumps({"success": False, "error": str(e)}).encode('utf-8')
                        self.send_response(500)
                        self.send_header('Content-Type', 'application/json; charset=utf-8')
                        self.send_header('Content-Length', str(len(body)))
                        self.end_headers()
                        self.wfile.write(body)
                        return

                self.send_error(404)

            def log_message(self, format, *args):
                return

        class ReusableThreadingHTTPServer(ThreadingHTTPServer):
            allow_reuse_address = True
            daemon_threads = True

        self._server = ReusableThreadingHTTPServer((self.host, self.port), PreviewHandler)
        self._running = True
        self._thread = threading.Thread(target=self._server.serve_forever, name='PointcloudPreview', daemon=True)
        self._thread.start()

    def update(self, image, stats):
        if image is None or not self._running:
            return
        ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        if not ok:
            return
        with self._control_lock:
            self._frame_version += 1
            stats["pointcloud_version"] = self._frame_version
            self._stats.update(stats)
        with self._condition:
            self._jpeg = encoded.tobytes()
            self._jpeg_version += 1
            self._condition.notify_all()

    def get_bounds(self):
        with self._control_lock:
            return self._bounds.copy()

    def stop(self):
        if not self._running:
            return
        self._running = False
        with self._condition:
            self._condition.notify_all()
        self._server.shutdown()
        self._server.server_close()
        if self._thread is not None:
            self._thread.join(timeout=2.0)


def generate_index_html(default_bounds):
    """Return a self-contained UI with an offline Canvas point-cloud viewer."""
    default_bounds_json = json.dumps(np.asarray(default_bounds).tolist())
    html = """<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>OBJECT_SPACE 点云裁剪预览</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #0d1117; color: #e6edf3; font-family: system-ui, sans-serif; padding: 16px; }
    h2,h3 { text-align: center; margin: 8px 0 12px; }
    .grid { display: grid; grid-template-columns: 1fr 360px; gap: 14px; margin-bottom: 14px; }
    @media (max-width: 1000px) { .grid { grid-template-columns: 1fr; } }
    .card { border: 1px solid #30363d; border-radius: 9px; background: #161b22; padding: 12px; }
    #camera { width: 100%; max-height: 500px; object-fit: contain; background: #000; border-radius: 6px; }
    #pointcloud-canvas { width: 100%; height: 600px; touch-action: none; background: #000; border-radius: 6px; display: block; cursor: grab; }
    #pointcloud-canvas:active { cursor: grabbing; }
    .stat-row { display: flex; justify-content: space-between; margin: 8px 0; font-size: 14px; }
    .stat-label { color: #9da7b3; }
    .stat-value { font-family: monospace; color: #58a6ff; font-weight: bold; }
    .stat-value.ok { color: #3fb950; }
    .stat-value.warn { color: #d29922; }
    .control-group { margin: 16px 0; }
    .axis-row { display: grid; grid-template-columns: 60px 1fr 1fr; gap: 8px; align-items: center; margin: 6px 0; }
    .axis-row input { width: 100%; background: #21262d; border: 1px solid #30363d; border-radius: 4px; color: #fff; padding: 6px 8px; font-family: monospace; }
    .axis-name { font-weight: bold; font-size: 16px; text-align: center; }
    .axis-x { color: #f85149; } .axis-y { color: #3fb950; } .axis-z { color: #58a6ff; }
    .range-info { font-size: 12px; color: #8b949e; margin: 4px 0 12px; text-align: center; }
    button { width: 100%; min-height: 40px; padding: 8px 16px; border: 1px solid #6e7681; border-radius: 6px; background: #238636; color: #fff; font-weight: bold; cursor: pointer; margin-top: 8px; }
    button:hover { background: #2ea043; }
    button.secondary { background: #30363d; }
    button.secondary:hover { background: #484f58; }
    #message { padding: 10px; margin-top: 12px; border-radius: 6px; background: #252b33; text-align: center; font-size: 13px; min-height: 40px; display: flex; align-items: center; justify-content: center; }
    #message.success { background: #1a4722; color: #3fb950; }
    #message.error { background: #5a1d1d; color: #f85149; }
    .hint { font-size: 12px; color: #8b949e; margin-top: 12px; line-height: 1.5; }
    .legend { text-align: center; color: #9da7b3; margin-top: 8px; font-size: 13px; }
  </style>
</head>
<body>
<h2>📏 OBJECT_SPACE 点云裁剪范围实时预览</h2>
<p class="legend">坐标系：D455相机坐标系 | X右/Y下/Z前 | 单位：米</p>

<div class="grid">
  <section class="card">
    <h3>2D RGB实时画面</h3>
    <img id="camera" src="/stream.mjpg" alt="D455画面">
    <div class="legend">绿色文字=统计信息 | 红色文字=警告</div>
  </section>

  <section class="card">
    <h3>裁剪范围控制</h3>
    <div class="control-group">
      <div class="axis-row"><span class="axis-name axis-x">X轴</span><input type="number" step="0.001" id="xmin"><input type="number" step="0.001" id="xmax"></div>
      <div class="range-info" id="xrange">范围宽度：-- m</div>
    </div>
    <div class="control-group">
      <div class="axis-row"><span class="axis-name axis-y">Y轴</span><input type="number" step="0.001" id="ymin"><input type="number" step="0.001" id="ymax"></div>
      <div class="range-info" id="yrange">范围高度：-- m</div>
    </div>
    <div class="control-group">
      <div class="axis-row"><span class="axis-name axis-z">Z轴</span><input type="number" step="0.001" id="zmin"><input type="number" step="0.001" id="zmax"></div>
      <div class="range-info" id="zrange">范围深度：-- m</div>
    </div>

    <button id="apply">应用裁剪范围</button>
    <button id="reset" class="secondary">恢复默认</button>
    <button id="save" class="secondary">💾 保存点云为PLY</button>

    <h3 style="margin-top: 20px;">实时统计</h3>
    <div class="stat-row"><span class="stat-label">帧率</span><span class="stat-value" id="fps">-- FPS</span></div>
    <div class="stat-row"><span class="stat-label">裁剪后原始点数</span><span class="stat-value" id="rawcount">--</span></div>
    <div class="stat-row"><span class="stat-label">输出采样点数</span><span class="stat-value" id="sampledcount">--</span></div>
    <div class="stat-row"><span class="stat-label">点云X范围</span><span class="stat-value" id="px">--</span></div>
    <div class="stat-row"><span class="stat-label">点云Y范围</span><span class="stat-value" id="py">--</span></div>
    <div class="stat-row"><span class="stat-label">点云Z范围</span><span class="stat-value" id="pz">--</span></div>

    <div id="message">等待数据...</div>
    <div class="hint">
      💡 提示：<br>
      • 调整范围后点击应用实时生效<br>
      • 裁剪后点数建议≥3000保证模型识别精度<br>
      • 3D点云：左键旋转，滚轮缩放，右键或 Shift+左键平移
    </div>
  </section>
</div>

<section class="card">
  <h3>3D彩色点云实时预览</h3>
  <canvas id="pointcloud-canvas"></canvas>
  <div class="legend">彩色点=有效点云 | 🟥 红色线框=OBJECT_SPACE裁剪边界 | 离线 Canvas 渲染</div>
</section>

<script>
const DEFAULT_BOUNDS = __DEFAULT_BOUNDS__;
let lastPointCloudVersion = -1;
let pointData = new Float32Array(0), currentBounds = DEFAULT_BOUNDS;
const view = {yaw: -0.45, pitch: 0.25, zoom: 1, panX: 0, panY: 0, drag: null};

document.addEventListener('DOMContentLoaded', function() {
  initUI();
  initCanvasViewer();
  setInterval(updateStatus, 200);
  setInterval(fetchPointCloud, 500);
});

function initUI() {
  updateBoundsInputs(DEFAULT_BOUNDS);
  document.getElementById('apply').onclick = applyBounds;
  document.getElementById('reset').onclick = function() { updateBoundsInputs(DEFAULT_BOUNDS); applyBounds(); };
  document.getElementById('save').onclick = savePointCloud;
}

function initCanvasViewer() {
  const canvas = document.getElementById('pointcloud-canvas');
  const redraw = () => { resizeCanvas(canvas); renderPointCloud(canvas); };
  new ResizeObserver(redraw).observe(canvas);
  canvas.addEventListener('contextmenu', e => e.preventDefault());
  canvas.addEventListener('pointerdown', e => { canvas.setPointerCapture(e.pointerId); view.drag = {x:e.clientX, y:e.clientY, pan:e.button === 2 || e.shiftKey}; });
  canvas.addEventListener('pointerup', () => { view.drag = null; });
  canvas.addEventListener('pointermove', e => {
    if (!view.drag) return;
    const dx = e.clientX - view.drag.x, dy = e.clientY - view.drag.y;
    view.drag.x = e.clientX; view.drag.y = e.clientY;
    if (view.drag.pan) { view.panX += dx; view.panY += dy; }
    else { view.yaw += dx * 0.01; view.pitch = Math.max(-1.45, Math.min(1.45, view.pitch + dy * 0.01)); }
    renderPointCloud(canvas);
  });
  canvas.addEventListener('wheel', e => { e.preventDefault(); view.zoom = Math.max(0.25, Math.min(8, view.zoom * (e.deltaY < 0 ? 1.12 : 0.89))); renderPointCloud(canvas); }, {passive:false});
  redraw();
}

function resizeCanvas(canvas) {
  const ratio = Math.min(window.devicePixelRatio || 1, 2), width = Math.max(1, Math.floor(canvas.clientWidth * ratio)), height = Math.max(1, Math.floor(canvas.clientHeight * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
}

function rotateProject(x, y, z, center, width, height) {
  x -= center[0]; y -= center[1]; z -= center[2];
  const cy = Math.cos(view.yaw), sy = Math.sin(view.yaw), cp = Math.cos(view.pitch), sp = Math.sin(view.pitch);
  const rx = cy*x + sy*z, rz = -sy*x + cy*z, ry = cp*y - sp*rz, depth = sp*y + cp*rz + 0.9;
  if (depth <= 0.02) return null;
  const focal = Math.min(width, height) * 1.15 * view.zoom;
  return [width / 2 + view.panX + rx / depth * focal, height / 2 + view.panY + ry / depth * focal, depth];
}

function renderPointCloud(canvas) {
  const ctx = canvas.getContext('2d'), width = canvas.width, height = canvas.height;
  ctx.fillStyle = '#000'; ctx.fillRect(0, 0, width, height);
  const b = currentBounds, center = [(b[0][0]+b[0][1])/2, (b[1][0]+b[1][1])/2, (b[2][0]+b[2][1])/2];
  const projected = [];
  for (let i=0; i<pointData.length; i+=6) { const p = rotateProject(pointData[i], pointData[i+1], pointData[i+2], center, width, height); if (p) projected.push([p, i]); }
  projected.sort((a,b) => b[0][2] - a[0][2]);
  for (const item of projected) { const p=item[0], i=item[1]; ctx.fillStyle='rgb('+Math.round(pointData[i+3]*255)+','+Math.round(pointData[i+4]*255)+','+Math.round(pointData[i+5]*255)+')'; ctx.fillRect(p[0], p[1], 2, 2); }
  const corners=[]; for (const x of b[0]) for (const y of b[1]) for (const z of b[2]) corners.push(rotateProject(x,y,z,center,width,height));
  const edges=[[0,1],[0,2],[0,4],[1,3],[1,5],[2,3],[2,6],[3,7],[4,5],[4,6],[5,7],[6,7]];
  ctx.strokeStyle='#ff4d4d'; ctx.lineWidth=2; ctx.beginPath();
  for (const edge of edges) { const a=corners[edge[0]], c=corners[edge[1]]; if (a && c) { ctx.moveTo(a[0],a[1]); ctx.lineTo(c[0],c[1]); } }
  ctx.stroke(); ctx.fillStyle='#9da7b3'; ctx.font='14px system-ui'; ctx.fillText(pointData.length/6+' points | drag rotate | wheel zoom', 12, 22);
}

async function fetchPointCloud() {
  try {
    const res = await fetch('/api/status', {cache: 'no-store'});
    const s = await res.json();
    currentBounds = s.bounds;
    renderPointCloud(document.getElementById('pointcloud-canvas'));

    if (s.pointcloud_version === lastPointCloudVersion) return;

    if (s.raw_count === 0) {
      pointData = new Float32Array(0);
      lastPointCloudVersion = s.pointcloud_version;
      renderPointCloud(document.getElementById('pointcloud-canvas'));
      return;
    }

    const pcRes = await fetch('/api/pointcloud.bin', {cache: 'no-store'});
    if (!pcRes.ok || pcRes.status === 204) return;
    if (Number(pcRes.headers.get('X-Pointcloud-Version')) !== s.pointcloud_version) return;
    const buf = await pcRes.arrayBuffer();
    const data = new Float32Array(buf);
    const numPoints = data.length / 6;

    if (!Number.isInteger(numPoints)) throw new Error('invalid point-cloud payload');
    pointData = data;
    lastPointCloudVersion = s.pointcloud_version;
    renderPointCloud(document.getElementById('pointcloud-canvas'));
  } catch(e) { console.error("点云加载失败:", e); }
}

async function updateStatus() {
  try {
    const res = await fetch('/api/status', {cache: 'no-store'});
    const s = await res.json();
    document.getElementById('fps').textContent = s.fps + ' FPS';
    const rawEl = document.getElementById('rawcount');
    rawEl.textContent = s.raw_count;
    rawEl.className = 'stat-value ' + (s.raw_count >= 3000 ? 'ok' : s.raw_count > 0 ? 'warn' : '');
    document.getElementById('sampledcount').textContent = s.sampled_count;
    document.getElementById('px').textContent = s.point_x[0].toFixed(3) + ' ~ ' + s.point_x[1].toFixed(3) + ' m';
    document.getElementById('py').textContent = s.point_y[0].toFixed(3) + ' ~ ' + s.point_y[1].toFixed(3) + ' m';
    document.getElementById('pz').textContent = s.point_z[0].toFixed(3) + ' ~ ' + s.point_z[1].toFixed(3) + ' m';
    if (!document.activeElement || document.activeElement.tagName !== 'INPUT') updateBoundsInputs(s.bounds);
  } catch(e) { console.error("状态更新失败:", e); }
}

async function applyBounds() {
  const bounds = [
    [parseFloat(document.getElementById('xmin').value), parseFloat(document.getElementById('xmax').value)],
    [parseFloat(document.getElementById('ymin').value), parseFloat(document.getElementById('ymax').value)],
    [parseFloat(document.getElementById('zmin').value), parseFloat(document.getElementById('zmax').value)],
  ];
  if (bounds.some(function(a) { return isNaN(a[0]) || isNaN(a[1]) || a[0] >= a[1]; })) {
    showMessage('❌ 范围无效：最小值必须小于最大值', 'error'); return;
  }
  try {
    const res = await fetch('/api/bounds', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({bounds: bounds})});
    if (res.ok) { updateRangeInfo(bounds); showMessage('✅ 裁剪范围已更新'); }
    else showMessage('❌ 更新失败', 'error');
  } catch(e) { showMessage('❌ 网络错误', 'error'); }
}

async function savePointCloud() {
  showMessage('⏳ 正在保存...');
  try {
    const res = await fetch('/api/save', {method:'POST'});
    const d = await res.json();
    d.success ? showMessage('✅ 已保存到: ' + d.path, 'success') : showMessage('❌ 失败: ' + d.error, 'error');
  } catch(e) { showMessage('❌ 保存失败', 'error'); }
}

function updateBoundsInputs(b) {
  document.getElementById('xmin').value = b[0][0].toFixed(3); document.getElementById('xmax').value = b[0][1].toFixed(3);
  document.getElementById('ymin').value = b[1][0].toFixed(3); document.getElementById('ymax').value = b[1][1].toFixed(3);
  document.getElementById('zmin').value = b[2][0].toFixed(3); document.getElementById('zmax').value = b[2][1].toFixed(3);
  updateRangeInfo(b);
}

function updateRangeInfo(b) {
  document.getElementById('xrange').textContent = '宽度：' + (b[0][1]-b[0][0]).toFixed(3) + ' m';
  document.getElementById('yrange').textContent = '高度：' + (b[1][1]-b[1][0]).toFixed(3) + ' m';
  document.getElementById('zrange').textContent = '深度：' + (b[2][1]-b[2][0]).toFixed(3) + ' m';
}

function showMessage(text, type) {
  if (!type) type = 'info';
  const el = document.getElementById('message');
  el.textContent = text; el.className = type;
  if (type !== 'error') setTimeout(function() { el.className = ''; el.textContent = ''; }, 5000);
}
</script>
</body>
</html>
"""
    # 替换占位符
    html = html.replace("__DEFAULT_BOUNDS__", default_bounds_json)
    return html


def draw_pointcloud_overlay(img, raw_count, sampled_count, point_range, bounds):
    h, w = img.shape[:2]
    if raw_count == 0:
        put_cn_text(img, "❌ OBJECT_SPACE内无有效点云", (15, 30), color=(0,0,255), font_scale=0.7, thickness=2, fallback_text="❌ NO VALID POINTS IN OBJECT_SPACE")
    else:
        color = (0,255,0) if raw_count >= 3000 else (0,165,255)
        put_cn_text(img, f"✅ 裁剪后点数: {raw_count}  输出点数: {sampled_count}", (15,30), color=color, font_scale=0.7, thickness=2, fallback_text=f"✅ Raw points: {raw_count}  Sampled: {sampled_count}")
        put_cn_text(img, f"X: {point_range[0][0]:.3f}~{point_range[0][1]:.3f}  Y: {point_range[1][0]:.3f}~{point_range[1][1]:.3f}  Z: {point_range[2][0]:.3f}~{point_range[2][1]:.3f}", (15,60), color=(255,255,255), font_scale=0.6, thickness=1, fallback_text=f"X:{point_range[0][0]:.3f}~{point_range[0][1]:.3f} Y:{point_range[1][0]:.3f}~{point_range[1][1]:.3f} Z:{point_range[2][0]:.3f}~{point_range[2][1]:.3f}")
        if raw_count < 3000:
            put_cn_text(img, "⚠️ 点数偏少，建议扩大裁剪范围", (15,85), color=(0,165,255), font_scale=0.6, thickness=1, fallback_text="⚠️ Low point count")
    cv2.putText(img, f"OBJECT_SPACE: X[{bounds[0,0]:.3f},{bounds[0,1]:.3f}] Y[{bounds[1,0]:.3f},{bounds[1,1]:.3f}] Z[{bounds[2,0]:.3f},{bounds[2,1]:.3f}]", (15, h-15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,255), 1)
    return img


def main():
    parser = argparse.ArgumentParser(description="Web版OBJECT_SPACE点云预览工具")
    parser.add_argument("--host", default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8766, help="端口")
    parser.add_argument("--serial", default=None, help="D455序列号")
    parser.add_argument("--width", type=int, default=1280, help="分辨率宽")
    parser.add_argument("--height", type=int, default=720, help="分辨率高")
    parser.add_argument("--fps", type=int, default=30, help="采集帧率")
    parser.add_argument("--point-count", type=int, default=10000, help="输出点数")
    parser.add_argument("--voxel-size", type=float, default=0.001, help="体素下采样边长(米)")
    parser.add_argument("--output-dir", default="object_space_exports", help="PLY 输出目录")
    args = parser.parse_args()
    if args.point_count <= 0:
        parser.error("--point-count 必须为正数")
    if args.voxel_size < 0:
        parser.error("--voxel-size 不能小于 0")

    default_bounds = ((-0.113, 0.031), (-0.075, 0.116), (0.7, 0.95))
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from configs.pose import OBJECT_SPACE
        default_bounds = tuple(tuple(x) for x in OBJECT_SPACE)
        print(f"✅ 从configs/pose.py加载默认OBJECT_SPACE: {default_bounds}")
    except Exception:
        print("⚠️ 使用内置默认OBJECT_SPACE")

    print("="*70)
    print("🚀 OBJECT_SPACE点云预览工具启动")
    print("="*70)

    print("\n[1/2] 初始化D455相机...")
    camera = D455(serial=args.serial, width=args.width, height=args.height, fps=args.fps)

    print("\n[2/2] 启动Web服务...")
    preview = BrowserPreview(
        host=args.host,
        port=args.port,
        default_bounds=default_bounds,
        output_dir=args.output_dir,
    )
    preview.start()

    print(f"\n🌐 访问地址：")
    print(f"   本机：http://127.0.0.1:{args.port}")
    print(f"   局域网：http://{get_lan_ip()}:{args.port}")
    print("\n按 Ctrl+C 退出\n")

    prev_time = time.time()
    frame_count = 0
    fps = 0
    rng = np.random.default_rng(1)

    try:
        while True:
            color_img, depth_img = camera.get()
            if color_img is None or depth_img is None:
                continue

            bounds = preview.get_bounds()
            xyz, rgb, raw_count = camera.get_pointcloud(
                color_img, depth_img, bounds, args.voxel_size
            )

            pointcloud_for_save = None
            rgb_for_3d = None
            point_range = [[0,0],[0,0],[0,0]]
            sampled_count = 0

            if len(xyz) > 0:
                replace = len(xyz) < args.point_count
                selected = rng.choice(len(xyz), args.point_count, replace=replace)
                xyz_sampled = xyz[selected]
                rgb_sampled = rgb[selected]
                pointcloud_for_save = xyz_sampled
                rgb_for_3d = rgb_sampled
                sampled_count = len(xyz_sampled)
                point_range = [
                    [float(xyz[:,0].min()), float(xyz[:,0].max())],
                    [float(xyz[:,1].min()), float(xyz[:,1].max())],
                    [float(xyz[:,2].min()), float(xyz[:,2].max())],
                ]

            display_img = color_img.copy()
            draw_pointcloud_overlay(display_img, raw_count, sampled_count, point_range, bounds)

            frame_count += 1
            current_time = time.time()
            if current_time - prev_time >= 1.0:
                fps = frame_count / (current_time - prev_time)
                frame_count = 0
                prev_time = current_time

            preview.update(display_img, {
                "raw_count": raw_count,
                "sampled_count": sampled_count,
                "pointcloud": pointcloud_for_save,
                "pointcloud_rgb": rgb_for_3d,
                "point_x": point_range[0],
                "point_y": point_range[1],
                "point_z": point_range[2],
                "fps": fps,
            })

            print(f"\rFPS: {fps:.1f} | 点数: {raw_count:5d} | Z: {point_range[2][0]:.3f}~{point_range[2][1]:.3f}", end="", flush=True)

    except KeyboardInterrupt:
        print("\n\n[WARN] 正在退出...")
    finally:
        preview.stop()
        camera.stop()
        print("✅ 已退出")


if __name__ == "__main__":
    main()
