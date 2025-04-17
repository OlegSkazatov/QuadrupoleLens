from PyQt5 import QtWidgets, QtCore, QtGui


class ElementEditorWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Magnetic Optics Editor")
        self.setup_ui()

    def setup_ui(self):
        # Главный контейнер
        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)

        # Основной layout
        layout = QtWidgets.QHBoxLayout(main_widget)

        # 1. Схематический вид (левая панель)
        self.canvas = QtWidgets.QGraphicsView()
        self.scene = QtWidgets.QGraphicsScene()
        self.canvas.setScene(self.scene)
        self.canvas.setFixedSize(600, 400)
        layout.addWidget(self.canvas, 70)  # 70% ширины

        # 2. Панель управления (правая панель)
        right_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(right_panel, 30)  # 30% ширины

        # Таблица элементов
        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "Position"])
        right_panel.addWidget(self.table)

        # Кнопки управления
        btn_layout = QtWidgets.QHBoxLayout()
        right_panel.addLayout(btn_layout)

        self.add_btn = QtWidgets.QPushButton("Add")
        self.remove_btn = QtWidgets.QPushButton("Remove")
        self.save_btn = QtWidgets.QPushButton("Save")
        self.load_btn = QtWidgets.QPushButton("Load")
        self.set_btn = QtWidgets.QPushButton("Set Configuration")

        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.remove_btn)
        right_panel.addWidget(self.save_btn)
        right_panel.addWidget(self.load_btn)
        right_panel.addWidget(self.set_btn)

        # Контекстное меню для элементов
        self.canvas.setContextMenuPolicy(QtCore.Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self.show_context_menu)

        # Стилизация
        self.setStyleSheet("""
            QTableWidget { font-size: 10pt; }
            QPushButton { padding: 5px; min-width: 80px; }
        """)

    def show_context_menu(self, pos):
        menu = QtWidgets.QMenu()
        color_action = menu.addAction("Change Color")
        edit_action = menu.addAction("Edit Parameters")
        action = menu.exec_(self.canvas.mapToGlobal(pos))

        if action == color_action:
            self.change_element_color()
        elif action == edit_action:
            self.open_parameters_dialog()


class ElementParametersDialog(QtWidgets.QDialog):
    def __init__(self, element_type):
        super().__init__()
        self.setup_ui(element_type)

    def setup_ui(self, element_type):
        self.setWindowTitle(f"{element_type} Parameters")
        layout = QtWidgets.QFormLayout(self)

        # Общие параметры
        self.name_edit = QtWidgets.QLineEdit()
        self.x_pos = QtWidgets.QDoubleSpinBox()
        self.y_pos = QtWidgets.QDoubleSpinBox()

        layout.addRow("Name:", self.name_edit)
        layout.addRow("X Position [m]:", self.x_pos)
        layout.addRow("Y Position [m]:", self.y_pos)

        # Параметры специфичные для типа
        if element_type == "Quadrupole":
            self.strength = QtWidgets.QDoubleSpinBox()
            self.length = QtWidgets.QDoubleSpinBox()
            layout.addRow("Strength [T/m]:", self.strength)
            layout.addRow("Length [m]:", self.length)

        elif element_type == "Dipole":
            # ... аналогично для других типов
            pass

        # Кнопки
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addRow(btn_box)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)