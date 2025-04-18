from PyQt5 import QtWidgets, QtCore, QtGui



class ElementEditorWindow(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super().__init__()
        self.setWindowTitle("Magnetic Optics Editor")
        self.parent = parent
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

        self.add_btn.clicked.connect(self.add_element)
        self.save_btn.clicked.connect(self.save_config)
        self.table.itemDoubleClicked.connect(self.on_table_double_click)

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

    def update_table(self):
        """Обновляет таблицу элементов на основе объектов сцены"""
        self.table.setRowCount(0)  # Очищаем таблицу

        # Собираем только элементы типа CanvasElement
        elements = [
            item for item in self.scene.items()
            if isinstance(item, CanvasElement)
        ]

        for idx, elem in enumerate(elements):
            # Добавляем новую строку
            self.table.insertRow(idx)

            # Название элемента
            name_item = QtWidgets.QTableWidgetItem(elem.name)
            name_item.setData(QtCore.Qt.UserRole, elem)  # Сохраняем ссылку на элемент

            # Тип элемента
            type_item = QtWidgets.QTableWidgetItem(elem.element_type)

            # Позиция в формате (X, Y)
            pos = elem.scenePos()
            pos_item = QtWidgets.QTableWidgetItem(f"({pos.x():.2f}, {pos.y():.2f})")

            # Заполняем колонки
            self.table.setItem(idx, 0, name_item)
            self.table.setItem(idx, 1, type_item)
            self.table.setItem(idx, 2, pos_item)

            # Раскрашиваем строку в цвет элемента
            color = elem.brush().color()
            for col in range(3):
                self.table.item(idx, col).setBackground(color)

    def on_table_double_click(self, item):
        """Открывает диалог редактирования при двойном клике по строке"""
        elem = item.data(QtCore.Qt.UserRole)
        self.open_parameters_dialog(elem)

    def show_context_menu(self, pos):
        menu = QtWidgets.QMenu()
        color_action = menu.addAction("Change Color")
        edit_action = menu.addAction("Edit Parameters")
        action = menu.exec_(self.canvas.mapToGlobal(pos))

        if action == color_action:
            self.change_element_color()
        elif action == edit_action:
            self.open_parameters_dialog()

    def add_element(self):
        element_types = ["Quadrupole", "Dipole"]
        type_dialog = QtWidgets.QInputDialog()
        type_, ok = type_dialog.getItem(self, "Select Type", "Element type:", element_types)
        if ok:
            new_element = CanvasElement(0, 0, type_)
            self.scene.addItem(new_element)
            self.update_table()

    def save_config(self):
        config = {
            "elements": [
                {
                    "type": elem.element_type,
                    "x": elem.x(),
                    "y": elem.y(),
                    "color": elem.brush().color().name(),
                    "params": elem.parameters
                }
                for elem in self.scene.items()
                if isinstance(elem, CanvasElement)
            ]
        }
        # Сохранение в JSON через QFileDialog


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


class CanvasElement(QtWidgets.QGraphicsRectItem):
    def __init__(self, x, y, element_type):
        super().__init__(0, 0, 40, 20)
        self.element_type = element_type
        self.name = f"{element_type}_{id(self)}"  # Уникальное имя по умолчанию
        self.setPos(x, y)
        self.setBrush(QtGui.QBrush(QtCore.Qt.blue))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            # При перемещении элемента обновляем таблицу
            if self.scene():
                self.scene().parent().update_table()
        return super().itemChange(change, value)

    def update_parameters(self, params):
        """Обновляет параметры элемента"""
        self.name = params.get('name', self.name)
        self.setPos(params.get('x', self.x()), params.get('y', self.y()))
        self.setBrush(QtGui.QBrush(params.get('color', self.brush().color())))