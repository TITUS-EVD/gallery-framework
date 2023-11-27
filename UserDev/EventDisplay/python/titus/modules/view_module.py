#!/usr/bin/env python3

"""
Adds a menu which controls view switching. Sub-modules added to this are added
to the menu. Menu selections show/hide main view widget & relevant controls if
the sub-modules implement central widget and dock widget members
"""

from PyQt5 import QtWidgets, QtGui, QtCore

from .module import Module


class ViewSelectModule(Module):
    viewChanged = QtCore.pyqtSignal()

    def __init__(self):
        super().__init__()
        self._modules = {}
        self._view_menu = None
        self._view_action_map = {}
        self._current_view = None

        # flag to automatically trigger the first view added. Additional views
        # are added in hidden state
        self._first_module = None

    def _initialize(self):
        self.populate_menu()

    def populate_menu(self):
        if self._view_menu is not None:
            self._view_menu.clear()
        else:
            self._view_menu = QtWidgets.QMenu("&View", self._gui)
            self._gui.menuBar().addMenu(self._view_menu)

        view_action_group = QtWidgets.QActionGroup(self._gui)
        if not self._modules:
            return 

        for name, mod in self._modules.items():
            action = QtWidgets.QAction(name, self._view_menu)
            action.setCheckable(True)
            action.triggered.connect(lambda checked, sender=action:\
                                     self._on_action_checked(checked, sender))
            action.setActionGroup(view_action_group)
            self._view_menu.addAction(action)
            self._view_action_map[action] = name

            if mod == self._first_module:
                action.setChecked(True)
                action.triggered.emit(True)

        restore_action = QtWidgets.QAction('Restore controls...', self._view_menu)        
        restore_action.triggered.connect(self._on_restore_action)
        self._view_menu.addSeparator()
        self._view_menu.addAction(restore_action)

    def _on_action_checked(self, checked, sender):
        """ call back for QAction in geometry menu bar """
        view = self._view_action_map[sender]
        if view != self._current_view:
            for _, mod in self._modules.items():
                mod.deactivate()
            self._modules[view].activate()
            self._current_view = view
            self.viewChanged.emit()

    def _on_restore_action(self):
        ''' resets current view (for now basically just restores closed dock widgets) '''
        if self._current_view is None:
            for _, mod in self._modules.items():
                mod.deactivate()
                return

        self._modules[self._current_view].activate()

    def add_module(self, module, view_name):
        ''' allow sub-modules. Give a reference to this module & add to gui '''
        if view_name in self._modules:
            raise RuntimeError('Error: Cannot add multiple views with the same name to the view module')

        super().add_module(module)
        self._modules[view_name] = module

        if self._first_module is None:
            self._first_module = module
            module.activate()
        else:
            module.deactivate()

        self.populate_menu()
