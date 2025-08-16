# src/app/pyqt_gan.py
import sys, io
from PyQt5 import QtWidgets, QtGui
from src.service.gan_infer import GANService, GenerateParams


class Main(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.svc = GANService()
        self.setWindowTitle("GAN Sampler")
        self.resize(900, 700)

        # UI
        self.ckpt_edit = QtWidgets.QLineEdit("logs/stage3_wgangp/ckpt_epoch1.pt")
        self.device_combo = QtWidgets.QComboBox()
        self.device_combo.addItems(["cuda", "cpu"])
        self.load_btn = QtWidgets.QPushButton("Load")

        self.n_spin = QtWidgets.QSpinBox()
        self.n_spin.setRange(4, 256)
        self.n_spin.setValue(64)
        self.n_spin.setSingleStep(4)
        self.nrow_spin = QtWidgets.QSpinBox()
        self.nrow_spin.setRange(1, 32)
        self.nrow_spin.setValue(8)
        self.seed_spin = QtWidgets.QSpinBox()
        self.seed_spin.setRange(0, 10_000_000)
        self.seed_spin.setValue(42)
        self.ema_chk = QtWidgets.QCheckBox("Use EMA shadow (if available)")
        self.gen_btn = QtWidgets.QPushButton("Generate")

        self.lbl = QtWidgets.QLabel()
        self.lbl.setAlignment(QtCore.Qt.AlignCenter)  # type: ignore

        form = QtWidgets.QFormLayout()
        form.addRow("Checkpoint", self.ckpt_edit)
        form2 = QtWidgets.QHBoxLayout()
        form2.addWidget(QtWidgets.QLabel("Device"))
        form2.addWidget(self.device_combo)
        form2.addWidget(self.load_btn)
        form.addRow(form2)

        grid = QtWidgets.QGridLayout()
        grid.addWidget(QtWidgets.QLabel("n"), 0, 0)
        grid.addWidget(self.n_spin, 0, 1)
        grid.addWidget(QtWidgets.QLabel("nrow"), 0, 2)
        grid.addWidget(self.nrow_spin, 0, 3)
        grid.addWidget(QtWidgets.QLabel("seed"), 1, 0)
        grid.addWidget(self.seed_spin, 1, 1)
        grid.addWidget(self.ema_chk, 1, 2)
        grid.addWidget(self.gen_btn, 1, 3)

        v = QtWidgets.QVBoxLayout(self)
        v.addLayout(form)
        v.addLayout(grid)
        v.addWidget(self.lbl, 1)

        self.load_btn.clicked.connect(self.on_load)
        self.gen_btn.clicked.connect(self.on_gen)

    def on_load(self):
        device = self.device_combo.currentText()
        self.svc.__init__(device=device)
        self.svc.load_checkpoint(self.ckpt_edit.text())
        QtWidgets.QMessageBox.information(
            self, "Loaded", f"Device={self.svc.device}, size={self.svc.cfg['img_size']}"  # type: ignore
        )

    def on_gen(self):
        img = self.svc.generate_grid(
            GenerateParams(
                n=self.n_spin.value(),
                seed=self.seed_spin.value(),
                nrow=self.nrow_spin.value(),
                use_ema_shadow=self.ema_chk.isChecked(),
            )
        )
        data = io.BytesIO()
        img.save(data, format="PNG")
        qimg = QtGui.QImage.fromData(data.getvalue(), "PNG")
        self.lbl.setPixmap(
            QtGui.QPixmap.fromImage(qimg).scaled(
                self.lbl.size(), QtCore.Qt.KeepAspectRatio  # type: ignore
            )
        )


if __name__ == "__main__":
    from PyQt5 import QtCore  # placed here to avoid unused warning

    app = QtWidgets.QApplication(sys.argv)
    w = Main()
    w.show()
    sys.exit(app.exec_())
