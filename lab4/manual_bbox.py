import cv2
import os
import numpy as np
import json
from datetime import datetime
# ═══════════════════════════════════════════════════════════
#  Enhanced BBox Selector — OpenCV
#  Controls:
#    Left Click      — Place point (2-click mode) or drag
#    Right Click     — Undo last point / cancel current drag
#    Z              — Undo last point
#    S              — Save current bbox to history
#    C              — Copy last bbox info to console
#    M              — Toggle mode (2-click / drag)
#    R              — Reset current selection
#    G              — Toggle crosshair guides
#    H              — Toggle help overlay
#    F              — Toggle output format
#    N              — Next image (if folder mode)
#    P              — Previous image (if folder mode)
#    Q / Esc        — Quit
# ═══════════════════════════════════════════════════════════
# ── Configuration ──────────────────────────────────────────
IMG_PATH = os.path.join("test_images", "test.jpg")
WINDOW_NAME = "BBox Selector"
COLORS = {
    "crosshair": (100, 100, 100),
    "point": (0, 200, 255),
    "point_outline": (0, 0, 0),
    "rect_active": (0, 255, 0),
    "rect_preview": (0, 255, 0),
    "rect_saved": (255, 180, 0),
    "text_bg": (30, 30, 30),
    "text_fg": (255, 255, 255),
    "help_bg": (20, 20, 20),
    "help_fg": (200, 200, 200),
    "help_key": (0, 200, 255),
    "banner_bg": (0, 160, 80),
    "banner_fg": (255, 255, 255),
    "status_bg": (40, 40, 40),
    "info_label": (160, 160, 160),
    "info_value": (0, 220, 255),
}
FORMATS = ["x y w h", "x1 y1 x2 y2", "YOLO", "JSON", "CSV"]
# ───────────────────────────────────────────────────────────
class BBoxSelector:
    def __init__(self, img_path):
        self.images = []
        self.img_index = 0
        self.load_images(img_path)
        # State
        self.points = []
        self.history = []
        self.mouse_pos = (0, 0)
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        # Toggles
        self.mode = "click"
        self.show_crosshair = True
        self.show_help = False
        self.format_index = 0
        self.needs_redraw = True
    def load_images(self, path):
        """Load single image or all images from a folder."""
        supported = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp")
        if os.path.isdir(path):
            files = sorted(
                [
                    os.path.join(path, f)
                    for f in os.listdir(path)
                    if f.lower().endswith(supported)
                ]
            )
            if not files:
                print(f"No images found in '{path}'")
                exit(1)
            self.images = files
            print(f"Loaded {len(files)} images from '{path}'")
        elif os.path.isfile(path):
            self.images = [path]
        else:
            print(f"Path not found: '{path}'")
            exit(1)
    @property
    def current_path(self):
        return self.images[self.img_index]
    def load_current_image(self):
        img = cv2.imread(self.current_path)
        if img is None:
            print(f"Cannot read: {self.current_path}")
            return False
        self.original = img.copy()
        self.img_h, self.img_w = img.shape[:2]
        self.points = []
        self.dragging = False
        self.drag_start = None
        self.drag_end = None
        self.needs_redraw = True
        return True
    # ── Drawing helpers ────────────────────────────────────
    def draw_text(self, img, text, pos, font_scale=0.5, color=(255, 255, 255),
                  bg_color=None, thickness=1, padding=5):
        """Draw text with optional background."""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        x, y = pos
        if bg_color is not None:
            cv2.rectangle(img, (x - padding, y - th - padding),
                          (x + tw + padding, y + baseline + padding), bg_color, -1)
        cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
        return th + baseline + padding * 2
    def draw_point(self, img, pt, label="", radius=6):
        """Draw a labeled point marker."""
        cv2.circle(img, pt, radius + 2, COLORS["point_outline"], -1, cv2.LINE_AA)
        cv2.circle(img, pt, radius, COLORS["point"], -1, cv2.LINE_AA)
        cv2.circle(img, pt, radius - 2, (255, 255, 255), 1, cv2.LINE_AA)
        if label:
            self.draw_text(img, label, (pt[0] + 12, pt[1] + 5), 0.45,
                           COLORS["text_fg"], COLORS["text_bg"], padding=3)
    def draw_dashed_line(self, img, pt1, pt2, color, thickness=1, dash_len=8):
        """Draw a dashed line."""
        x1, y1 = pt1
        x2, y2 = pt2
        dist = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        if dist == 0:
            return
        dashes = int(dist / dash_len)
        for i in range(0, dashes, 2):
            s = i / dashes
            e = min((i + 1) / dashes, 1.0)
            sx = int(x1 + (x2 - x1) * s)
            sy = int(y1 + (y2 - y1) * s)
            ex = int(x1 + (x2 - x1) * e)
            ey = int(y1 + (y2 - y1) * e)
            cv2.line(img, (sx, sy), (ex, ey), color, thickness, cv2.LINE_AA)
    def draw_rect_with_dims(self, img, x1, y1, x2, y2, color, thickness=2, dashed=False):
        """Draw rectangle with dimension labels."""
        if dashed:
            self.draw_dashed_line(img, (x1, y1), (x2, y1), color, thickness)
            self.draw_dashed_line(img, (x2, y1), (x2, y2), color, thickness)
            self.draw_dashed_line(img, (x2, y2), (x1, y2), color, thickness)
            self.draw_dashed_line(img, (x1, y2), (x1, y1), color, thickness)
        else:
            cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
        # Dimension labels
        w = abs(x2 - x1)
        h = abs(y2 - y1)
        if w > 40 and h > 20:
            # Width label on top
            mid_x = (x1 + x2) // 2
            self.draw_text(img, f"{w}px", (mid_x - 15, min(y1, y2) - 8),
                           0.4, color, COLORS["text_bg"], padding=2)
            # Height label on right
            mid_y = (y1 + y2) // 2
            self.draw_text(img, f"{h}px", (max(x1, x2) + 6, mid_y + 4),
                           0.4, color, COLORS["text_bg"], padding=2)
    def draw_crosshair(self, img, pos):
        """Draw crosshair guides at mouse position."""
        mx, my = pos
        self.draw_dashed_line(img, (0, my), (self.img_w, my),
                              COLORS["crosshair"], 1, 6)
        self.draw_dashed_line(img, (mx, 0), (mx, self.img_h),
                              COLORS["crosshair"], 1, 6)
    def draw_cursor_info(self, img, pos):
        """Draw coordinate info near cursor."""
        mx, my = pos
        text = f"({mx}, {my})"
        offset_x = 15 if mx < self.img_w - 120 else -100
        offset_y = -15 if my > 30 else 25
        self.draw_text(img, text, (mx + offset_x, my + offset_y),
                       0.4, COLORS["info_value"], COLORS["text_bg"], padding=3)
    def draw_status_bar(self, img):
        """Draw bottom status bar."""
        bar_h = 32
        h = img.shape[0]
        cv2.rectangle(img, (0, h - bar_h), (self.img_w, h), COLORS["status_bg"], -1)
        # Mode
        mode_text = f"Mode: {self.mode.upper()}"
        self.draw_text(img, mode_text, (8, h - 10), 0.42,
                       COLORS["info_value"], padding=0)
        # Format
        fmt_text = f"Format: {FORMATS[self.format_index]}"
        self.draw_text(img, fmt_text, (160, h - 10), 0.42,
                       COLORS["info_label"], padding=0)
        # Image info
        fname = os.path.basename(self.current_path)
        if len(self.images) > 1:
            nav = f"[{self.img_index + 1}/{len(self.images)}] "
        else:
            nav = ""
        info = f"{nav}{fname}  ({self.img_w}x{self.img_h})"
        self.draw_text(img, info, (360, h - 10), 0.42,
                       COLORS["info_label"], padding=0)
        # History count
        hist = f"Saved: {len(self.history)}"
        self.draw_text(img, hist, (self.img_w - 100, h - 10), 0.42,
                       COLORS["rect_saved"], padding=0)
    def draw_top_banner(self, img, text, color=None):
        """Draw a top notification banner."""
        bg = color or COLORS["banner_bg"]
        cv2.rectangle(img, (0, 0), (self.img_w, 28), bg, -1)
        self.draw_text(img, text, (10, 18), 0.48,
                       COLORS["banner_fg"], padding=0)
    def draw_help_overlay(self, img):
        """Draw help overlay."""
        overlay = img.copy()
        oh, ow = img.shape[:2]
        # Semi-transparent background
        cv2.rectangle(overlay, (0, 0), (ow, oh), COLORS["help_bg"], -1)
        cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
        title = "KEYBOARD SHORTCUTS"
        self.draw_text(img, title, (ow // 2 - 100, 40), 0.7,
                       COLORS["help_key"], thickness=2, padding=0)
        shortcuts = [
            ("Left Click", "Place point / Start drag"),
            ("Right Click", "Undo / Cancel"),
            ("Z", "Undo last point"),
            ("S", "Save bbox to history"),
            ("C", "Print bbox info to console"),
            ("M", "Toggle mode (click/drag)"),
            ("R", "Reset current selection"),
            ("G", "Toggle crosshair guides"),
            ("F", "Cycle output format"),
            ("H", "Toggle this help"),
            ("N / P", "Next / Previous image"),
            ("Q / Esc", "Quit"),
        ]
        y = 80
        for key, desc in shortcuts:
            self.draw_text(img, f"  {key:>12s}", (ow // 2 - 180, y), 0.5,
                           COLORS["help_key"], padding=0)
            self.draw_text(img, f"  —  {desc}", (ow // 2 - 50, y), 0.5,
                           COLORS["help_fg"], padding=0)
            y += 28
        self.draw_text(img, "Press H to close", (ow // 2 - 70, y + 20), 0.45,
                       (120, 120, 120), padding=0)
    # ── BBox formatting ────────────────────────────────────
    def get_bbox(self, p1, p2):
        """Return normalized bbox dict from two points."""
        x1, y1 = p1
        x2, y2 = p2
        return {
            "x": min(x1, x2),
            "y": min(y1, y2),
            "w": abs(x2 - x1),
            "h": abs(y2 - y1),
            "x1": min(x1, x2),
            "y1": min(y1, y2),
            "x2": max(x1, x2),
            "y2": max(y1, y2),
            "image": os.path.basename(self.current_path),
            "img_w": self.img_w,
            "img_h": self.img_h,
        }
    def format_bbox(self, bbox):
        """Format bbox according to current format."""
        fmt = FORMATS[self.format_index]
        if fmt == "x y w h":
            return f"{bbox['x']} {bbox['y']} {bbox['w']} {bbox['h']}"
        elif fmt == "x1 y1 x2 y2":
            return f"{bbox['x1']} {bbox['y1']} {bbox['x2']} {bbox['y2']}"
        elif fmt == "YOLO":
            cx = (bbox['x1'] + bbox['x2']) / 2.0 / bbox['img_w']
            cy = (bbox['y1'] + bbox['y2']) / 2.0 / bbox['img_h']
            nw = bbox['w'] / bbox['img_w']
            nh = bbox['h'] / bbox['img_h']
            return f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}"
        elif fmt == "JSON":
            return json.dumps({"x": bbox['x'], "y": bbox['y'],
                               "w": bbox['w'], "h": bbox['h']})
        elif fmt == "CSV":
            return f"{bbox['image']},{bbox['x']},{bbox['y']},{bbox['w']},{bbox['h']}"
        return ""
    def print_bbox(self, bbox, prefix="BBox"):
        """Print bbox to console in all formats."""
        print(f"\n{'─' * 50}")
        print(f"  {prefix}  [{bbox['image']}]")
        print(f"{'─' * 50}")
        print(f"  x y w h      :  {bbox['x']} {bbox['y']} {bbox['w']} {bbox['h']}")
        print(f"  x1 y1 x2 y2  :  {bbox['x1']} {bbox['y1']} {bbox['x2']} {bbox['y2']}")
        cx = (bbox['x1'] + bbox['x2']) / 2.0 / bbox['img_w']
        cy = (bbox['y1'] + bbox['y2']) / 2.0 / bbox['img_h']
        nw = bbox['w'] / bbox['img_w']
        nh = bbox['h'] / bbox['img_h']
        print(f"  YOLO         :  0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")
        print(f"  Area         :  {bbox['w'] * bbox['h']} px²")
        print(f"{'─' * 50}\n")
    # ── Mouse callback ─────────────────────────────────────
    def mouse_callback(self, event, x, y, flags, params):
        x = max(0, min(x, self.img_w - 1))
        y = max(0, min(y, self.img_h - 1))
        self.mouse_pos = (x, y)
        self.needs_redraw = True
        if self.mode == "click":
            if event == cv2.EVENT_LBUTTONDOWN:
                if len(self.points) < 2:
                    self.points.append((x, y))
                    print(f"  Point {len(self.points)}: ({x}, {y})")
                    if len(self.points) == 2:
                        bbox = self.get_bbox(self.points[0], self.points[1])
                        self.print_bbox(bbox)
            elif event == cv2.EVENT_RBUTTONDOWN:
                if self.points:
                    removed = self.points.pop()
                    print(f"  Undone point: {removed}")
        elif self.mode == "drag":
            if event == cv2.EVENT_LBUTTONDOWN:
                self.dragging = True
                self.drag_start = (x, y)
                self.drag_end = (x, y)
                self.points = []
            elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
                self.drag_end = (x, y)
            elif event == cv2.EVENT_LBUTTONUP and self.dragging:
                self.dragging = False
                self.drag_end = (x, y)
                if self.drag_start and self.drag_end:
                    dx = abs(self.drag_end[0] - self.drag_start[0])
                    dy = abs(self.drag_end[1] - self.drag_start[1])
                    if dx > 3 and dy > 3:  # minimum size
                        self.points = [self.drag_start, self.drag_end]
                        bbox = self.get_bbox(self.points[0], self.points[1])
                        self.print_bbox(bbox)
                    else:
                        self.drag_start = None
                        self.drag_end = None
            elif event == cv2.EVENT_RBUTTONDOWN:
                self.dragging = False
                self.drag_start = None
                self.drag_end = None
                self.points = []
                print("  Selection cancelled")
    # ── Main render ────────────────────────────────────────
    def render(self):
        """Render the current frame."""
        img = self.original.copy()
        # Draw saved bboxes (from history for this image)
        fname = os.path.basename(self.current_path)
        for entry in self.history:
            if entry["image"] == fname:
                cv2.rectangle(img, (entry["x1"], entry["y1"]),
                              (entry["x2"], entry["y2"]),
                              COLORS["rect_saved"], 2, cv2.LINE_AA)
                self.draw_text(img, f"#{entry['id']}", (entry["x1"] + 4, entry["y1"] + 16),
                               0.4, COLORS["rect_saved"], COLORS["text_bg"], padding=2)
        # Crosshair
        if self.show_crosshair and not self.show_help:
            self.draw_crosshair(img, self.mouse_pos)
            self.draw_cursor_info(img, self.mouse_pos)
        # Draw current selection
        if self.mode == "click":
            if len(self.points) >= 1:
                self.draw_point(img, self.points[0], "P1")
                # Preview rectangle from P1 to mouse
                if len(self.points) == 1:
                    mx, my = self.mouse_pos
                    self.draw_rect_with_dims(img, self.points[0][0], self.points[0][1],
                                             mx, my, COLORS["rect_preview"], 1, dashed=True)
            if len(self.points) == 2:
                self.draw_point(img, self.points[0], "P1")
                self.draw_point(img, self.points[1], "P2")
                bbox = self.get_bbox(self.points[0], self.points[1])
                self.draw_rect_with_dims(img, bbox["x1"], bbox["y1"],
                                         bbox["x2"], bbox["y2"],
                                         COLORS["rect_active"], 2)
                # Show formatted bbox on image
                fmt_text = self.format_bbox(bbox)
                self.draw_text(img, fmt_text, (bbox["x1"], bbox["y2"] + 20),
                               0.45, COLORS["rect_active"], COLORS["text_bg"], padding=4)
        elif self.mode == "drag":
            if self.dragging and self.drag_start and self.drag_end:
                self.draw_rect_with_dims(img, self.drag_start[0], self.drag_start[1],
                                         self.drag_end[0], self.drag_end[1],
                                         COLORS["rect_preview"], 2, dashed=True)
                self.draw_point(img, self.drag_start, "")
            elif len(self.points) == 2:
                bbox = self.get_bbox(self.points[0], self.points[1])
                self.draw_point(img, self.points[0], "P1")
                self.draw_point(img, self.points[1], "P2")
                self.draw_rect_with_dims(img, bbox["x1"], bbox["y1"],
                                         bbox["x2"], bbox["y2"],
                                         COLORS["rect_active"], 2)
                fmt_text = self.format_bbox(bbox)
                self.draw_text(img, fmt_text, (bbox["x1"], bbox["y2"] + 20),
                               0.45, COLORS["rect_active"], COLORS["text_bg"], padding=4)
        # Top banner — context-aware hint
        if len(self.points) == 0 and not self.dragging:
            hint = "Click to place P1" if self.mode == "click" else "Click & drag to select"
            hint += "  |  H = Help"
            self.draw_top_banner(img, hint, (60, 60, 60))
        elif len(self.points) == 1 and self.mode == "click":
            self.draw_top_banner(img, "Click to place P2", (60, 60, 60))
        elif len(self.points) == 2:
            self.draw_top_banner(img, "S = Save  |  C = Print  |  R = Reset  |  H = Help",
                                 COLORS["banner_bg"])
        # Status bar
        self.draw_status_bar(img)
        # Help overlay (on top of everything)
        if self.show_help:
            self.draw_help_overlay(img)
        return img
    # ── Keyboard handling ──────────────────────────────────
    def handle_key(self, key):
        """Handle keyboard input. Returns False to quit."""
        if key == -1:
            return True
        char = chr(key & 0xFF) if key & 0xFF < 128 else ""
        # Quit
        if key == 27 or char in ("q", "Q"):
            return False
        # Help
        elif char in ("h", "H"):
            self.show_help = not self.show_help
            self.needs_redraw = True
        # Mode toggle
        elif char in ("m", "M"):
            self.mode = "drag" if self.mode == "click" else "click"
            self.points = []
            self.dragging = False
            self.drag_start = None
            self.drag_end = None
            print(f"  ⚙ Mode: {self.mode.upper()}")
            self.needs_redraw = True
        # Reset
        elif char in ("r", "R"):
            self.points = []
            self.dragging = False
            self.drag_start = None
            self.drag_end = None
            print("  ↺ Selection reset")
            self.needs_redraw = True
        # Undo
        elif char in ("z", "Z"):
            if self.points:
                removed = self.points.pop()
                print(f"  ↩ Undone: {removed}")
                self.needs_redraw = True
        # Save to history
        elif char in ("s", "S"):
            if len(self.points) == 2:
                bbox = self.get_bbox(self.points[0], self.points[1])
                bbox["id"] = len(self.history) + 1
                bbox["timestamp"] = datetime.now().strftime("%H:%M:%S")
                self.history.append(bbox)
                print(f"  💾 Saved bbox #{bbox['id']}  ({self.format_bbox(bbox)})")
                self.points = []
                self.drag_start = None
                self.drag_end = None
                self.needs_redraw = True
            else:
                print(" No complete bbox to save")
        # Copy / print
        elif char in ("c", "C"):
            if len(self.points) == 2:
                bbox = self.get_bbox(self.points[0], self.points[1])
                self.print_bbox(bbox, "📋 Copied")
            elif self.history:
                self.print_bbox(self.history[-1], "📋 Last saved")
            else:
                print(" No bbox to print")
        # Format cycle
        elif char in ("f", "F"):
            self.format_index = (self.format_index + 1) % len(FORMATS)
            print(f"  Format: {FORMATS[self.format_index]}")
            self.needs_redraw = True
        # Next image
        elif char in ("n", "N"):
            if len(self.images) > 1:
                self.img_index = (self.img_index + 1) % len(self.images)
                self.load_current_image()
                print(f"  ➡ Image: {os.path.basename(self.current_path)}")
        # Previous image
        elif char in ("p", "P"):
            if len(self.images) > 1:
                self.img_index = (self.img_index - 1) % len(self.images)
                self.load_current_image()
                print(f"  ⬅ Image: {os.path.basename(self.current_path)}")
        return True
    # ── Main loop ──────────────────────────────────────────
    def run(self):
        if not self.load_current_image():
            return
        print("\n╔══════════════════════════════════════════╗")
        print("║        Enhanced BBox Selector            ║")
        print("║   Press H for help  |  Q/Esc to quit     ║")
        print("╚══════════════════════════════════════════╝\n")
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        while True:
            if self.needs_redraw:
                frame = self.render()
                cv2.imshow(WINDOW_NAME, frame)
                self.needs_redraw = False
            key = cv2.waitKey(30)
            if key != -1:
                self.needs_redraw = True
            if not self.handle_key(key):
                break
        cv2.destroyAllWindows()
        # Print summary
        if self.history:
            print(f"\n{'═' * 50}")
            print(f"  SESSION SUMMARY — {len(self.history)} bbox(es) saved")
            print(f"{'═' * 50}")
            for entry in self.history:
                print(f"  #{entry['id']:>2d}  [{entry['timestamp']}]  "
                      f"{entry['image']}  →  "
                      f"{entry['x']} {entry['y']} {entry['w']} {entry['h']}")
            print(f"{'═' * 50}\n")
# ── Entry point ────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else IMG_PATH
    selector = BBoxSelector(path)
    selector.run()