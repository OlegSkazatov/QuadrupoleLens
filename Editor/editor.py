# editor.py
import json

import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui

# ВАЖНО! Разрешение холста - 1 метр равен 10000 пикселей!

class ElementEditorWindow(QtWidgets.QMainWindow):
    def __init__(self, parent):
        super().__init__()
        self.setWindowTitle("Magnetic Optics Editor")
        self.parent = parent
        self.setup_ui()
        self.setup_navigation()
        self.setup_coordinate_label()
        self.setup_grid()

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
        self.zoom_level = 1.0
        self.base_grid_step = 0.1  # Базовый шаг сетки в метрах (10 см)
        self.canvas.setRenderHint(QtGui.QPainter.Antialiasing)
        self.canvas.setTransformationAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        self.canvas.setResizeAnchor(QtWidgets.QGraphicsView.AnchorUnderMouse)
        # Отключаем скролл-бары
        self.canvas.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        self.canvas.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        # Настройки сцены
        self.scene.setSceneRect(-1e5, -1e5, 2e5, 2e5)  # Большая сцена
        self.scene.parent_window = self  # Явно связываем сцену с окном
        layout.addWidget(self.canvas, 70)  # 70% ширины

        # 2. Панель управления (правая панель)
        right_panel = QtWidgets.QVBoxLayout()
        layout.addLayout(right_panel, 30)  # 30% ширины

        # Таблица элементов
        self.table = QtWidgets.QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Name", "Type", "Position", "Color"])
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
        self.remove_btn.clicked.connect(self.remove_element)
        self.save_btn.clicked.connect(self.save_config)
        self.load_btn.clicked.connect(self.load_config)
        self.set_btn.clicked.connect(self.set_configuration)
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
        self.canvas.setMouseTracking(True)

    def setup_navigation(self):
        self.drag_mode = False
        self.last_mouse_pos = QtCore.QPoint()

        # Обработчики событий
        self.canvas.mousePressEvent = self.canvas_mouse_press
        self.canvas.mouseMoveEvent = self.canvas_mouse_move
        self.canvas.mouseReleaseEvent = self.canvas_mouse_release

    def canvas_mouse_press(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drag_mode = True
            self.last_mouse_pos = event.pos()
            self.canvas.setCursor(QtCore.Qt.ClosedHandCursor)
        super(QtWidgets.QGraphicsView, self.canvas).mousePressEvent(event)

    def canvas_mouse_move(self, event):
        # Преобразуем координаты курсора в координаты сцены
        scene_pos = self.canvas.mapToScene(event.pos())
        x, y = scene_pos.x(), scene_pos.y()
        x_unit, y_unit = 'm', 'm'

        # Подбираем, в чём отображать координату (метры, сантиметры, миллиметры)
        if 100 <= abs(x) < 10000:
            x_unit = 'cm'
            x = x / 100
        elif abs(x) < 100:
            x_unit = 'mm'
            x = x / 10
        elif abs(x) >= 10000:
            x = x / 10000
        if 100 <= abs(y) < 10000:
            y_unit = 'cm'
            y = y / 100
        elif abs(y) < 100:
            y_unit = 'mm'
            y = y / 10
        elif abs(y) >= 10000:
            y = y / 10000
        self.coord_label.setText(f"X: {x:.3f}{x_unit}, Y: {y:.3f}{y_unit}")
        if self.drag_mode:
            delta = event.pos() - self.last_mouse_pos
            self.last_mouse_pos = event.pos()
            self.canvas.translate(delta.x(), delta.y())
        super(QtWidgets.QGraphicsView, self.canvas).mouseMoveEvent(event)
        self.update_grid()

    def canvas_mouse_release(self, event):
        if event.button() == QtCore.Qt.LeftButton:
            self.drag_mode = False
            self.canvas.setCursor(QtCore.Qt.ArrowCursor)
        super(QtWidgets.QGraphicsView, self.canvas).mouseReleaseEvent(event)

    def setup_grid(self):
        self.grid_group = QtWidgets.QGraphicsItemGroup()
        self.scene.addItem(self.grid_group)
        self.update_grid()

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


    def update_table(self):
        """Обновляет таблицу элементов на основе объектов сцены"""
        self.table.setRowCount(0)
        elements = [
            item for item in self.scene.items()
            if isinstance(item, CanvasElement)
        ]

        for idx, elem in enumerate(elements):
            self.table.insertRow(idx)

            # Name
            name_item = QtWidgets.QTableWidgetItem(elem.name)
            name_item.setData(QtCore.Qt.UserRole, elem)
            self.table.setItem(idx, 0, name_item)

            # Type
            type_item = QtWidgets.QTableWidgetItem(elem.element_type)
            self.table.setItem(idx, 1, type_item)

            # Position
            pos = elem.scenePos()
            pos_item = QtWidgets.QTableWidgetItem(f"({pos.x() / 10000:.2f}, {pos.y() / 10000:.2f})")
            self.table.setItem(idx, 2, pos_item)

            # Color
            color_item = QtWidgets.QTableWidgetItem()
            color_item.setBackground(elem.brush().color())
            self.table.setItem(idx, 3, color_item)

    def on_table_double_click(self, item):
        """Открывает диалог редактирования при двойном клике по строке"""
        # Получаем элемент из первой колонки строки
        row = item.row()
        elem_item = self.table.item(row, 0)
        elem = elem_item.data(QtCore.Qt.UserRole)
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
            new_element = CanvasElement(0, 0, type_, self)
            self.scene.addItem(new_element)
            self.update_table()

    def remove_element(self):
        """Удаляет выбранные элементы из сцены и таблицы"""
        selected_items = self.table.selectedItems()

        # Собираем уникальные строки
        rows_to_delete = set()
        for item in selected_items:
            rows_to_delete.add(item.row())

        # Удаляем элементы сцены
        for row in sorted(rows_to_delete, reverse=True):
            elem_item = self.table.item(row, 0)
            element = elem_item.data(QtCore.Qt.UserRole)
            self.scene.removeItem(element)

        self.update_table()

    def generate_json(self):
        config = {
            "elements": [
                {
                    "type": elem.element_type,
                    "x": elem.x() / 10000,  # Конвертация пикселей в метры
                    "y": elem.y() / 10000,
                    "color": elem.brush().color().name(),
                    "rotation": elem.rotation(),
                    "parameters": {
                        k: v if not isinstance(v, QtGui.QColor) else v.name()
                        for k, v in elem.parameters.items()
                    }
                }
                for elem in self.scene.items()
                if isinstance(elem, CanvasElement)
            ]
        }
        return config

    def save_config(self):
        """Сохраняет конфигурацию в JSON файл"""
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save Configuration",
            "",
            "JSON Files (*.json)",
            options=options
        )

        if not file_name:
            return

        config = self.generate_json()

        try:
            with open(file_name, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Ошибка сохранения",
                f"Не удалось сохранить файл:\n{str(e)}"
            )

    def load_config(self):
        """Загружает конфигурацию из JSON файла"""
        options = QtWidgets.QFileDialog.Options()
        file_name, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load Configuration",
            "",
            "JSON Files (*.json)",
            options=options
        )

        if not file_name:
            return

        try:
            with open(file_name, 'r', encoding='utf-8') as f:
                config = json.load(f)

            # Очищаем сцену
            self.scene.clear()
            self.grid_group = QtWidgets.QGraphicsItemGroup()
            self.scene.addItem(self.grid_group)

            # Создаем элементы
            for elem_data in config.get('elements', []):
                elem_type = elem_data['type']
                x = elem_data['x'] * 10000  # Конвертация метров в пиксели
                y = elem_data['y'] * 10000
                color = QtGui.QColor(elem_data['color'])

                # Создаем элемент
                element = CanvasElement(x, y, elem_type, self, name=elem_data['parameters']['name'])
                element.parameters = elem_data.get('parameters', {})

                # Восстанавливаем параметры
                element.setPos(x, y)
                element.setBrush(QtGui.QBrush(color))

                # Обновляем размеры
                if elem_type == "Quadrupole":
                    element.set_position(
                        x, y,
                        element.parameters.get('length', 0.2) * 10000,
                        element.parameters.get('radius', 0.1) * 10000, rotation=elem_data.get('rotation', 0)
                    )
                elif elem_type == "Dipole":
                    element.set_position(
                        x, y,
                        element.parameters.get('width', 0.3) * 10000,
                        element.parameters.get('length', 0.3) * 10000, rotation=elem_data.get('rotation', 0)
                    )

                self.scene.addItem(element)

            self.update_table()
            self.update_grid()

        except Exception as e:
            QtWidgets.QMessageBox.critical(
                self,
                "Ошибка загрузки",
                f"Не удалось загрузить файл:\n{str(e)}"
            )

    def set_configuration(self):
        configuration = self.generate_json()
        self.parent.load_magnetic_system(configuration)

    def wheelEvent(self, event):
        zoom_factor = 1.1  # Более плавное масштабирование
        if event.angleDelta().y() > 0:
            self.canvas.scale(zoom_factor, zoom_factor)
        else:
            self.canvas.scale(1 / zoom_factor, 1 / zoom_factor)

        # Фиксируем позицию курсора относительно сцены
        old_pos = self.canvas.mapToScene(event.pos())
        self.update_grid()
        new_pos = self.canvas.mapToScene(event.pos())
        delta = new_pos - old_pos
        self.canvas.translate(delta.x(), delta.y())

        event.accept()

    def update_grid(self):
        # Очищаем старую сетку
        for item in self.grid_group.childItems():
            self.grid_group.removeFromGroup(item)
            self.scene.removeItem(item)

        # Определяем шаг сетки в зависимости от масштаба
        view_rect = self.canvas.mapToScene(self.canvas.viewport().geometry()).boundingRect()

        visible_width = view_rect.width()

        # Автоматический выбор шага сетки
        steps = [2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
        for s in steps:
            if 20 <= visible_width / s <= 50:
                step = s
                break
        else:
            return  # Убираем сетку, если вышли за пределы
        x0, y0 = view_rect.x() + view_rect.width() / 2, view_rect.y() + view_rect.height() / 2
        x0, y0 = x0 // step * step, y0 // step * step
        # Параметры отрисовки
        pen = QtGui.QPen(QtGui.QColor(200, 200, 200), 1.0 * step / 20)
        bounds = visible_width * 1.1
        border = bounds // step * step
        num_x, num_y = 0, 0
        # Рисуем вертикальные линии
        while num_x <= border // step:
            line1 = QtWidgets.QGraphicsLineItem(x0 + step * num_x, y0 - border, x0 + step * num_x, y0 + border)
            line2 = QtWidgets.QGraphicsLineItem(x0 - step * num_x, y0 - border, x0 - step * num_x, y0 + border)

            line1.setPen(pen)
            line2.setPen(pen)
            self.grid_group.addToGroup(line1)
            self.grid_group.addToGroup(line2)
            num_x += 1

        # Рисуем горизонтальные линии
        while num_y <= border // step:
            line1 = QtWidgets.QGraphicsLineItem(x0 - border, y0 + step*num_y, x0 + border, y0 + step*num_y)
            line2 = QtWidgets.QGraphicsLineItem(x0 - border, y0 - step*num_y, x0 + border, y0 - step*num_y)
            line1.setPen(pen)
            line2.setPen(pen)
            self.grid_group.addToGroup(line1)
            self.grid_group.addToGroup(line2)
            num_y += 1

        # Рисуем оси
        ax_pen = QtGui.QPen(QtGui.QColor(130, 130, 130), 2.0 * step / 20)
        x_ax = QtWidgets.QGraphicsLineItem(x0 - border, 0, x0 + border, 0)
        y_ax = QtWidgets.QGraphicsLineItem(0, y0 - border, 0, y0 + border)
        x_ax.setPen(ax_pen)
        y_ax.setPen(ax_pen)
        self.grid_group.addToGroup(x_ax)
        self.grid_group.addToGroup(y_ax)


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
        self.x_spin.setRange(-20, 20)
        self.x_spin.setValue(self.element.parameters.get('x'))
        self.y_spin = QtWidgets.QDoubleSpinBox()
        self.y_spin.setRange(-20, 20)
        self.y_spin.setValue(self.element.parameters.get('y'))

        layout.addRow("Element name:", self.name_edit)
        layout.addRow("X Position [m]:", self.x_spin)
        layout.addRow("Y Position [m]:", self.y_spin)
        self.rotation_spin = QtWidgets.QDoubleSpinBox()
        self.rotation_spin.setRange(-360, 360)
        self.rotation_spin.setValue(self.element.rotation())
        layout.addRow("Rotation [deg]:", self.rotation_spin)

        # Параметры для квадруполя
        if self.element.element_type == "Quadrupole":
            self.gradient_spin = QtWidgets.QDoubleSpinBox()
            self.gradient_spin.setRange(-1000, 1000)
            self.gradient_spin.setValue(self.element.parameters.get('gradient'))

            self.length_spin = QtWidgets.QDoubleSpinBox()
            self.length_spin.setRange(1, 10000)  # мм
            self.length_spin.setValue(self.element.parameters.get('length') * 1000)
            layout.addRow("Length [mm]:", self.length_spin)

            self.radius_spin = QtWidgets.QDoubleSpinBox()
            self.radius_spin.setRange(1, 10000)  # мм
            self.radius_spin.setValue(self.element.parameters.get('radius') * 1000)
            layout.addRow("Radius [mm]:", self.radius_spin)

            layout.addRow("Gradient [T/m]:", self.gradient_spin)
        # Параметры для диполя
        if self.element.element_type == "Dipole":
            self.field_spin = QtWidgets.QDoubleSpinBox()
            self.field_spin.setRange(-1000, 1000)
            self.field_spin.setValue(self.element.parameters.get('field'))
            layout.addRow("Field [T]:", self.field_spin)

            self.width_spin = QtWidgets.QDoubleSpinBox()
            self.width_spin.setRange(1, 10000)  # мм
            self.width_spin.setValue(self.element.parameters.get('width') * 1000)
            layout.addRow("Width [mm]:", self.width_spin)

            self.length_spin = QtWidgets.QDoubleSpinBox()
            self.length_spin.setRange(1, 10000)  # мм
            self.length_spin.setValue(self.element.parameters.get('length') * 1000)
            layout.addRow("Length [mm]:", self.length_spin)

            self.height_spin = QtWidgets.QDoubleSpinBox()
            self.height_spin.setRange(1, 10000)  # мм
            self.height_spin.setValue(self.element.parameters.get('height') * 1000)
            layout.addRow("Height [mm]:", self.height_spin)

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
            'color': self.color_btn.palette().button().color(),
            'rotation': self.rotation_spin.value()
        }

        if self.element.element_type == "Quadrupole":
            values.update({
                'length': self.length_spin.value() / 1000,  # Конвертация мм → м
                'radius': self.radius_spin.value() / 1000,
                'gradient': self.gradient_spin.value()
            })

        if self.element.element_type == "Dipole":
            values.update({
                'width': self.width_spin.value() / 1000,  # Конвертация мм → м
                'length': self.length_spin.value() / 1000,
                'height': self.height_spin.value() / 1000,
                'field': self.field_spin.value()
            })

        return values


class CanvasElement(QtWidgets.QGraphicsRectItem):
    def __init__(self, x, y, element_type, parent, name=''):
        if element_type == "Quadrupole":
            super().__init__(x, y, 2000, 1000)
            self.set_position(x, y, 2000, 1000)
        else:
            super().__init__(x, y, 3000, 3000)
            self.set_position(x, y, 3000, 3000)
        self.element_type = element_type
        if name == '':
            self.name = f"{element_type}_{id(self)}"
        else:
            self.name = name
        self.parent = parent
        self.parameters = {}  # Инициализируем хранилище параметров
        self.setTransformOriginPoint(self.rect().center())  # Центр для вращения


        # Инициализация параметров по умолчанию
        if self.element_type == "Quadrupole":
            self.parameters = {
                'name': self.name,
                'x': 0,
                'y': 0,
                'gradient': 1.0,  # T/m
                'radius': 0.1,  # m
                'length': 0.2  # m
            }

        if self.element_type == 'Dipole':
            self.parameters = {
                'name': self.name,
                'x': 0,
                'y': 0,
                'field': 1.0,  # T
                'width': 0.3,  # m
                'length': 0.3,  # m
                'height': 0.1 # m
            }

        self.label = QtWidgets.QGraphicsTextItem(self)  # Текстовая метка

        # Настройки текста
        self.label.setDefaultTextColor(QtGui.QColor(0, 0, 0))
        font = QtGui.QFont("Arial", 140)
        font.setBold(True)
        self.label.setFont(font)
        self.label.setZValue(1)  # Текст поверх элемента
        self.setBrush(QtGui.QBrush(QtCore.Qt.blue))
        self.setFlag(QtWidgets.QGraphicsItem.ItemSendsGeometryChanges, True)
        self.setToolTip(self._update_tooltip())
        self.update_label()

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
        self.update_label()
        if self.element_type == 'Quadrupole':
            self.set_position(params['x'] * 10000, params['y'] * 10000, params['length'] * 10000,
                                  params['radius'] * 10000, rotation=params.get('rotation', 0))  # Обновляем размеры
        if self.element_type == 'Dipole':
            self.set_position(params['x'] * 10000, params['y'] * 10000, params['width'] * 10000,
                              params['length'] * 10000, rotation=params.get('rotation', 0))  # Обновляем размеры
        self.setBrush(QtGui.QBrush(params['color']))
        self.parameters.update(params)
        self.setToolTip(self._update_tooltip())

    def set_position(self, x, y, w, h, rotation=0):
        self.setPos(x, y)
        self.setRect(-w / 2, -h / 2, w, h)
        self.setRotation(rotation)

    def update_label(self):
        """Обновляет текст и позиционирует его по центру"""
        self.label.setPlainText(self.name)

        # Центрирование текста
        text_rect = self.label.boundingRect()
        elem_rect = self.rect()
        self.label.setPos(
            elem_rect.center().x() - text_rect.width() / 2,
            elem_rect.center().y() - text_rect.height() / 2
        )
