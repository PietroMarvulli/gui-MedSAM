# -*- coding: utf-8 -*-
import sys
import os
import time
from PyQt5.QtGui import (
    QBrush, QPainter, QPen, QPixmap, QKeySequence, QColor, QImage
)
from PyQt5.QtWidgets import (
    QFileDialog, QApplication, QGraphicsEllipseItem, QGraphicsScene, QGraphicsView,
    QGraphicsPixmapItem, QHBoxLayout, QPushButton, QVBoxLayout, QWidget, QShortcut, QLabel, QSlider,
    QListWidget, QListWidgetItem, QComboBox
)
from PyQt5.QtCore import Qt

import numpy as np
from skimage import transform, io
import torch
import torch.nn.functional as F
from PIL import Image
from segment_anything import sam_model_registry

# Set seeds
torch.manual_seed(2023)
torch.cuda.empty_cache()
torch.cuda.manual_seed(2023)
np.random.seed(2023)

SAM_MODEL_TYPE = "vit_b"
MedSAM_CKPT_PATH = "work_dir/MedSAM/medsam_vit_b.pth"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def np2pixmap(np_img):
    height, width, channel = np_img.shape
    bytesPerLine = 3 * width
    qImg = QImage(np_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
    return QPixmap.fromImage(qImg)


@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, height, width):
    box_torch = torch.as_tensor(box_1024, dtype=torch.float, device=img_embed.device)
    if len(box_torch.shape) == 2:
        box_torch = box_torch[:, None, :]
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points=None, boxes=box_torch, masks=None
    )
    low_res_logits, _ = medsam_model.mask_decoder(
        image_embeddings=img_embed,
        image_pe=medsam_model.prompt_encoder.get_dense_pe(),
        sparse_prompt_embeddings=sparse_embeddings,
        dense_prompt_embeddings=dense_embeddings,
        multimask_output=False,
    )
    low_res_pred = torch.sigmoid(low_res_logits)
    low_res_pred = F.interpolate(
        low_res_pred, size=(height, width), mode="bilinear", align_corners=False
    )
    low_res_pred = low_res_pred.squeeze().cpu().numpy()
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    return medsam_seg


colors = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (255, 0, 255), (0, 255, 255),
    (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
]


class ZoomableGraphicsView(QGraphicsView):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorUnderMouse)
        self.zoom_factor = 1.15

    def wheelEvent(self, event):
        if event.angleDelta().y() > 0:
            zoom = self.zoom_factor
        else:
            zoom = 1 / self.zoom_factor
        self.scale(zoom, zoom)


class Window(QWidget):
    def __init__(self):
        super().__init__()

        self.eraser_mode = False

        self.is_painting = False
        self.brush_radius = 3

        self.show_only_mask = False

        self.mask_history = []
        self.max_history = 10

        self.half_point_size = 5
        self.color_idx = 0
        self.prev_mask = None
        self.is_mouse_down = False
        self.start_point = None
        self.end_point = None
        self.rect = None
        self.image_path = None
        self.embedding = None

        self.label_display = QLabel(f"Label selected: {self.color_idx}")
        self.label_display.setStyleSheet("font-size: 16px; font-weight: bold; padding: 5px;")

        self.view = ZoomableGraphicsView()

        self.image_list = QListWidget()
        self.image_list.setMaximumWidth(350)
        self.image_list.itemClicked.connect(self.image_selected)

        self.view.setRenderHint(QPainter.Antialiasing)

        load_button = QPushButton("Load Image")
        save_button = QPushButton("Save Mask")

        self.brush_toggle_button = QPushButton("🖌️ Paintbrush")
        self.brush_toggle_button.setCheckable(True)
        self.brush_toggle_button.setMinimumHeight(40)
        self.brush_toggle_button.setFixedWidth(180)
        self.brush_toggle_button.setStyleSheet("font-size: 16px; font-weight: bold; padding: 8px")
        self.brush_toggle_button.toggled.connect(self.toggle_brush_mode)

        self.slider_label = QLabel(f"Brush Size: {self.brush_radius}")
        self.slider_label.setStyleSheet("font-size: 14px; font-weight: bold; padding: 5px")

        self.brush_slider = QSlider(Qt.Horizontal)
        self.brush_slider.setMinimum(1)
        self.brush_slider.setMaximum(50)
        self.brush_slider.setValue(self.brush_radius)
        self.brush_slider.setTickPosition(QSlider.TicksBelow)
        self.brush_slider.setTickInterval(5)
        self.brush_slider.valueChanged.connect(self.update_brush_size)

        self.folder_path_combo = QComboBox()
        self.folder_path_combo.setEditable(True)
        self.folder_path_combo.setMinimumWidth(450)
        self.folder_path_combo.setInsertPolicy(QComboBox.InsertAtTop)
        self.folder_path_combo.setStyleSheet("font-size: 16px;")

        self.folder_browse_button = QPushButton("📁")
        self.folder_browse_button.setFixedWidth(60)
        self.folder_browse_button.setStyleSheet("font-size: 20px; font-weight: bold;")
        self.folder_browse_button.clicked.connect(self.select_image_folder)
        self.folder_path_combo.activated.connect(self.load_from_combo)
        self.folder_path_combo.lineEdit().editingFinished.connect(self.load_from_combo)

        self.load_image_list_from_folder(r"C:\Users\pietr\Downloads\data\video01\video01_seq4")

        main_layout = QHBoxLayout(self)
        layout = QVBoxLayout()

        top_bar = QHBoxLayout()
        top_bar.addWidget(self.label_display)
        top_bar.addSpacing(20)
        top_bar.addWidget(self.brush_toggle_button)
        top_bar.addSpacing(20)
        top_bar.addWidget(self.slider_label)
        top_bar.addWidget(self.brush_slider)
        top_bar.addStretch()
        path_bar = QHBoxLayout()
        path_bar.addWidget(QLabel("Cartella immagini:"))
        path_bar.addWidget(self.folder_path_combo)
        path_bar.addWidget(self.folder_browse_button)

        layout.addLayout(path_bar)
        layout.addLayout(top_bar)

        layout.addWidget(self.view)

        buttons = QHBoxLayout()
        buttons.addWidget(load_button)
        buttons.addWidget(save_button)
        layout.addLayout(buttons)

        main_layout.addWidget(self.image_list)  # colonna sinistra
        main_layout.addLayout(layout)  # colonna destra (già esistente)
        self.setLayout(main_layout)
        self.setFocusPolicy(Qt.StrongFocus)

        load_button.clicked.connect(self.load_image)
        save_button.clicked.connect(self.save_mask)

        QShortcut(QKeySequence("Ctrl+Q"), self).activated.connect(lambda: quit())
        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self.undo)

    def keyPressEvent(self, event):
        key = event.key()
        if Qt.Key_0 <= key <= Qt.Key_9:
            self.color_idx = key - Qt.Key_0
            print(f"[INFO] Etichetta selezionata: {self.color_idx}")
            self.label_display.setText(f"Label selected : {self.color_idx}")
        elif key == Qt.Key_M:
            self.show_only_mask = not self.show_only_mask
            self.update_display()
        elif key == Qt.Key_R:
            self.mask_c = np.zeros_like(self.mask_c)
            self.mask_history = []
            self.update_display()
        elif key == Qt.Key_N:
            self.eraser_mode = not self.eraser_mode

    def select_image_folder(self):
        folder_path = QFileDialog.getExistingDirectory(
            self, "Seleziona una cartella con immagini", r"C:\Users\pietr\Downloads\data"
        )
        if folder_path:
            self.folder_path_combo.insertItem(0, folder_path)
            self.folder_path_combo.setCurrentText(folder_path)
            self.load_image_list_from_folder(folder_path)

    def load_from_combo(self):
        path = self.folder_path_combo.currentText()
        if os.path.isdir(path):
            self.load_image_list_from_folder(path)

    def load_image_list_from_folder(self, folder_path):
        if not os.path.exists(folder_path):
            print(f"[WARN] Cartella '{folder_path}' non trovata.")
            return

        image_extensions = [".png", ".jpg", ".jpeg", ".bmp"]
        self.image_list.clear()

        for fname in sorted(os.listdir(folder_path)):
            if fname.lower().endswith("_endo.png"):
                full_path = os.path.join(folder_path, fname)
                mask_path = full_path.replace("_endo.png", "_endo_mask.png")

                item = QListWidgetItem(fname)
                item.setData(Qt.UserRole, full_path)

                # Aggiungi la spunta se esiste la maschera corrispondente
                if os.path.exists(mask_path):
                    item.setCheckState(Qt.Checked)
                else:
                    item.setCheckState(Qt.Unchecked)

                self.image_list.addItem(item)

    def image_selected(self, item):
        image_path = item.data(Qt.UserRole)
        self.load_image_from_path(image_path)

    def load_image_from_path(self, file_path):
        if not file_path or file_path.strip() == "":
            print("[INFO] Caricamento annullato.")
            return

        self.is_mouse_down = False
        self.is_painting = False
        self.start_point = None
        self.end_point = None
        self.rect = None
        self.mask_history.clear()

        img_np = io.imread(file_path)
        img_3c = np.repeat(img_np[:, :, None], 3, axis=-1) if len(img_np.shape) == 2 else img_np
        self.img_3c = img_3c
        self.mask_c = np.zeros((*img_3c.shape[:2], 3), dtype="uint8")

        self.image_path = file_path
        self.get_embeddings()

        pixmap = np2pixmap(img_3c)
        H, W, _ = self.img_3c.shape
        self.scene = QGraphicsScene(0, 0, W, H)

        # Visualizza immagine + maschera in trasparenza
        img_overlay = Image.blend(
            Image.fromarray(self.img_3c.astype("uint8")),
            Image.fromarray(self.mask_c.astype("uint8")),
            0.2
        )
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img_overlay)))
        self.view.setScene(self.scene)

        mask_path = file_path.replace("_endo.png", "_endo_mask.png")
        if os.path.exists(mask_path):
            print(f"[INFO] Maschera trovata: {mask_path}")
            mask_loaded = io.imread(mask_path)
            if len(mask_loaded.shape) == 2:  # (H, W)
                mask_rgb = np.zeros((*mask_loaded.shape, 3), dtype=np.uint8)
                mask_rgb[mask_loaded != 0] = colors[self.color_idx % len(colors)]
                self.mask_c = mask_rgb
            elif len(mask_loaded.shape) == 3 and mask_loaded.shape[2] == 3:
                self.mask_c = mask_loaded
            else:
                print("[WARN] Formato maschera non valido.")
                self.mask_c = np.zeros((*img_3c.shape[:2], 3), dtype="uint8")
        else:
            self.mask_c = np.zeros((*img_3c.shape[:2], 3), dtype="uint8")

        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

    def toggle_brush_mode(self, checked):
        self.eraser_mode = checked
        if self.eraser_mode:
            self.brush_toggle_button.setText("🆑 Eraser")
        else:
            self.brush_toggle_button.setText("🖌️ Paintbrush")

    def update_brush_size(self, value):
        self.brush_radius = value
        self.slider_label.setText(f"Brush Size: {value}")

    def draw_brush(self, x, y):
        x = int(x)
        y = int(y)
        H, W, _ = self.mask_c.shape
        rr, cc = np.ogrid[:H, :W]
        mask_area = (rr - y) ** 2 + (cc - x) ** 2 <= self.brush_radius ** 2

        # Salva stato precedente per Undo
        self.mask_history.append(self.mask_c.copy())
        if len(self.mask_history) > self.max_history:
            self.mask_history.pop(0)

        if self.eraser_mode:
            self.mask_c[mask_area] = 0  # cancella
        else:
            self.mask_c[mask_area] = colors[self.color_idx % len(colors)]

        self.update_display()

    def load_image(self):
        self.is_mouse_down = False
        self.is_painting = False
        self.start_point = None
        self.end_point = None
        self.rect = None
        file_path, file_type = QFileDialog.getOpenFileName(
            self, "Choose Image to Segment", ".", "Image Files (*.png *.jpg *.bmp)"
        )

        if file_path is None or len(file_path) == 0:
            print("No image path specified, plz select an image")
            exit()

        img_np = io.imread(file_path)
        if len(img_np.shape) == 2:
            img_3c = np.repeat(img_np[:, :, None], 3, axis=-1)
        else:
            img_3c = img_np

        self.img_3c = img_3c
        self.image_path = file_path
        self.get_embeddings()
        pixmap = np2pixmap(self.img_3c)

        H, W, _ = self.img_3c.shape

        self.scene = QGraphicsScene(0, 0, W, H)
        self.end_point = None
        self.rect = None
        self.bg_img = self.scene.addPixmap(pixmap)
        self.bg_img.setPos(0, 0)
        self.mask_c = np.zeros((*self.img_3c.shape[:2], 3), dtype="uint8")
        self.view.setScene(self.scene)

        # events
        self.scene.mousePressEvent = self.mouse_press
        self.scene.mouseMoveEvent = self.mouse_move
        self.scene.mouseReleaseEvent = self.mouse_release

    def mouse_press(self, ev):
        button = ev.button()
        x, y = ev.scenePos().x(), ev.scenePos().y()

        # Reset modalità attive
        self.is_mouse_down = False
        self.is_painting = False

        if button == Qt.LeftButton:
            self.is_mouse_down = True
            self.start_pos = x, y
            self.start_point = self.scene.addEllipse(
                x - self.half_point_size, y - self.half_point_size,
                self.half_point_size * 2, self.half_point_size * 2,
                pen=QPen(QColor("red")), brush=QBrush(QColor("red"))
            )
        elif button == Qt.RightButton:
            self.is_painting = True
            self.draw_brush(x, y)

    def mouse_move(self, ev):
        x, y = ev.scenePos().x(), ev.scenePos().y()
        if self.is_mouse_down:
            if self.end_point:
                self.scene.removeItem(self.end_point)
            self.end_point = self.scene.addEllipse(
                x - self.half_point_size, y - self.half_point_size,
                self.half_point_size * 2, self.half_point_size * 2,
                pen=QPen(QColor("red")), brush=QBrush(QColor("red"))
            )
            if self.rect:
                self.scene.removeItem(self.rect)
            sx, sy = self.start_pos
            self.rect = self.scene.addRect(
                min(x, sx), min(y, sy), abs(x - sx), abs(y - sy),
                pen=QPen(QColor("red"))
            )
        elif self.is_painting:
            self.draw_brush(x, y)

    def mouse_release(self, ev):
        button = ev.button()
        if button == Qt.LeftButton:
            self.is_mouse_down = False
            x, y = ev.scenePos().x(), ev.scenePos().y()
            sx, sy = self.start_pos
            H, W, _ = self.img_3c.shape
            box = np.array([[min(x, sx), min(y, sy), max(x, sx), max(y, sy)]])
            box_1024 = box / np.array([W, H, W, H]) * 1024

            if len(self.mask_history) >= self.max_history:
                self.mask_history.pop(0)
            self.mask_history.append(self.mask_c.copy())

            sam_mask = medsam_inference(medsam_model, self.embedding, box_1024, H, W)
            self.mask_c[sam_mask != 0] = colors[self.color_idx % len(colors)]
            self.color_idx += 1
            self.label_display.setText(f"Label selected: {self.color_idx % 10}")
            self.update_display()
        elif button == Qt.RightButton:
            self.is_painting = False

    def undo(self):
        if not self.mask_history:
            print("No undo available.")
            return

        self.mask_c = self.mask_history.pop()

        self.update_display()


    def save_mask(self):
        out_path = f"{self.image_path.split('.')[0]}_mask.png"
        io.imsave(out_path, self.mask_c)

    def update_display(self):
        if self.show_only_mask:
            img = Image.fromarray(self.mask_c.astype("uint8"), "RGB")
        else:
            img = Image.blend(
                Image.fromarray(self.img_3c.astype("uint8"), "RGB"),
                Image.fromarray(self.mask_c.astype("uint8"), "RGB"),
                0.2
            )
        self.scene.removeItem(self.bg_img)
        self.bg_img = self.scene.addPixmap(np2pixmap(np.array(img)))

    @torch.no_grad()
    def get_embeddings(self):
        img_1024 = transform.resize(self.img_3c, (1024, 1024), preserve_range=True, anti_aliasing=True).astype(np.uint8)
        img_1024 = (img_1024 - img_1024.min()) / max(img_1024.max() - img_1024.min(), 1e-8)
        img_tensor = torch.tensor(img_1024).float().permute(2, 0, 1).unsqueeze(0).to(device)
        self.embedding = medsam_model.image_encoder(img_tensor)


# Carica MedSAM
print("Loading MedSAM...")
tic = time.perf_counter()
medsam_model = sam_model_registry["vit_b"](checkpoint=MedSAM_CKPT_PATH).to(device)
medsam_model.eval()

# Avvia GUI
app = QApplication(sys.argv)
window = Window()
window.setWindowTitle("MedSAM GUI")
window.show()
app.exec()
