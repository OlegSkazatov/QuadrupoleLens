from PyQt5 import QtWidgets, QtCore, QtGui



class ElementEditorWindow(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super().__init__()
        self.setWindowTitle("Magnetic Optics Editor")
        self.parent = parent
        self.setup_ui()
        self.setup_coordinate_label()
        self.setup_grid_and_axes()

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
        self.scene.parent_window = self  # Явно связываем сцену с окном
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

    def setup_coordinate_label(self):
        # Создаем статус бар для отображения координат
        self.coord_label = QtWidgets.QLabel("X: 0.00, Y: 0.00")
        self.statusBar().addPermanentWidget(self.coord_label)

        # Подключаем отслеживание движения мыши
        self.canvas.mouseMoveEvent = self.update_coord_label
        self.canvas.setMouseTracking(True)

    def update_coord_label(self, event):
        """Обновляет координаты в статус баре"""
        # Преобразуем координаты курсора в координаты сцены
        scene_pos = self.canvas.mapToScene(event.pos())
        self.coord_label.setText(f"X: {scene_pos.x():.2f}, Y: {scene_pos.y():.2f}")
        event.accept()

    def setup_grid_and_axes(self):
        """Добавляет оси координат и сетку на схему"""
        # Настройка сетки
        grid_pen = QtGui.QPen(QtGui.QColor(200, 200, 200), 1, QtCore.Qt.DotLine)
        for x in range(-500, 501, 50):
            self.scene.addLine(x, -500, x, 500, grid_pen)
        for y in range(-500, 501, 50):
            self.scene.addLine(-500, y, 500, y, grid_pen)

        # Оси координат
        axis_pen = QtGui.QPen(QtCore.Qt.black, 2)
        self.scene.addLine(-500, 0, 500, 0, axis_pen)  # X-axis
        self.scene.addLine(0, -500, 0, 500, axis_pen)  # Y-axis

        # Подписи осей
        font = QtGui.QFont("Arial", 10)
        self.scene.addText("X [m]", font).setPos(480, -20)
        self.scene.addText("Y [m]", font).setPos(10, 480)

    def open_parameters_dialog(self, element=None):
        """Открывает диалог редактирования параметров элемента"""
        if not element:
            # Получаем выделенный элемент из таблицы
            selected = self.table.selectedItems()
            if not selected:
                return
            element = selected[0].data(QtCore.Qt.UserRole)

        dialog = ElementParametersDialog(element)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            new_params = dialog.get_values()
            element.update_parameters(new_params)
            self.update_table()

            # Обновляем отображение элемента
            element.setPos(new_params['x'], new_params['y'])
            element.setBrush(QtGui.QBrush(new_params['color']))

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

    def wheelEvent(self, event):
        factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        self.canvas.scale(factor, factor)


class ElementParametersDialog(QtWidgets.QDialog):
    def __init__(self, element):  # Изменяем аргумент на элемент
        super().__init__()
        self.element = element
        self.setup_ui()

    def setup_ui(self):
        self.setWindowTitle(f"{self.element.element_type} Parameters")
        layout = QtWidgets.QFormLayout(self)

        # Общие параметры
        self.name_edit = QtWidgets.QLineEdit(self.element.name)
        self.x_spin = QtWidgets.QDoubleSpinBox()
        self.x_spin.setValue(self.element.x())
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setValue(self.element.y())

        layout.addRow("Element name:", self.name_edit)
        layout.addRow("X Position [m]:", self.x_spin)
        layout.addRow("Y Position [m]:", self.y_spin)

        # Параметры для квадруполя
        if self.element.element_type == "Quadrupole":
            self.gradient_spin = QtWidgets.QDoubleSpinBox()
            self.gradient_spin.setRange(0, 1000)
            self.gradient_spin.setValue(self.element.parameters.get('gradient', 0))

            self.radius_spin = QtWidgets.QDoubleSpinBox()
            self.radius_spin.setRange(0.01, 1.0)
            self.radius_spin.setValue(self.element.parameters.get('radius', 0.1))

            layout.addRow("Gradient [T/m]:", self.gradient_spin)
            layout.addRow("Radius [m]:", self.radius_spin)

        # Выбор цвета
        self.color_btn = QtWidgets.QPushButton()
        self.color_btn.setStyleSheet(f"background-color: {self.element.brush().color().name()}")
        self.color_btn.clicked.connect(self.choose_color)
        layout.addRow("Color:", self.color_btn)

        # Кнопки
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        layout.addRow(btn_box)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)

    def choose_color(self):
        color = QtWidgets.QColorDialog.getColor()
        if color.isValid():
            self.color_btn.setStyleSheet(f"background-color: {color.name()}")

    def get_values(self):
        values = {
            'name': self.name_edit.text(),
            'x': self.x_spin.value(),
            'y': self.y_spin.value(),
            'color': self.color_btn.palette().button().color()
        }

        if self.element.element_type == "Quadrupole":
            values.update({
                'gradient': self.gradient_spin.value(),
                'radius': self.radius_spin.value()
            })

        return values


class CanvasElement(QtWidgets.QGraphicsRectItem):
    def __init__(self, x, y, element_type):
        super().__init__(0, 0, 40, 20)
        self.element_type = element_type
        self.name = f"{element_type}_{id(self)}"
        self.parameters = {}  # Инициализируем хранилище параметров

        # Инициализация параметров по умолчанию
        if self.element_type == "Quadrupole":
            self.parameters = {
                'gradient': 10.0,  # T/m
                'radius': 0.05  # m
            }

        self.setPos(x, y)
        self.setBrush(QtGui.QBrush(QtCore.Qt.blue))
        self.setFlag(QtWidgets.QGraphicsItem.ItemIsMovable, True)
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setToolTip(self._update_tooltip())

    def _update_tooltip(self):
        params = "\n".join([f"{k}: {v}" for k, v in self.parameters.items()])
        return f"{self.name}\n({self.x():.2f}, {self.y():.2f})\n{params}"

    def itemChange(self, change, value):
        if change == QtWidgets.QGraphicsItem.ItemPositionHasChanged:
            scene = self.scene()
            if scene and hasattr(scene, 'parent_window'):
                scene.parent_window.update_table()
            else:
                main_window = self.get_main_window()
                if main_window:
                    main_window.update_table()
        return super().itemChange(change, value)

    def get_main_window(self):
        """Рекурсивно ищет главное окно в иерархии родителей"""
        parent = self.parent()
        while parent is not None:
            if isinstance(parent, ElementEditorWindow):
                return parent
            parent = parent.parent()
        return None

    def update_parameters(self, params):
        self.name = params.get('name', self.name)
        self.setPos(params.get('x', self.x()), params.get('y', self.y()))
        self.setBrush(QtGui.QBrush(params.get('color', self.brush().color())))
        self.parameters.update(params)  # Обновляем специфичные параметры
        self.setToolTip(self._update_tooltip())