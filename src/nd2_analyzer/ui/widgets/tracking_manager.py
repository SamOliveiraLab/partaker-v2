# tracking_manager.py
from PySide6.QtWidgets import QWidget, QVBoxLayout
from pubsub import pub

from .lineage_dialog import LineageDialog
from .motility_widget import MotilityDialog
from .tracking_widget import TrackingWidget


class TrackingManager(QWidget):
    """
    Manager widget that coordinates the tracking, lineage, and motility widgets.
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.tracking_widget = TrackingWidget()

        # Initialize UI
        layout = QVBoxLayout(self)
        layout.addWidget(self.tracking_widget)
        self.setLayout(layout)

        # Subscribe to relevant messages
        pub.subscribe(self.show_lineage_dialog, "show_lineage_dialog_request")
        pub.subscribe(self.show_motility_dialog, "show_motility_dialog_request")
        pub.subscribe(self.show_time_comparison, "show_time_comparison_request")

    def show_lineage_dialog(self, lineage_tracks):
        """Show the lineage dialog"""
        dialog = LineageDialog(lineage_tracks, self)
        dialog.exec()

    def show_motility_dialog(self, tracked_cells, lineage_tracks, image_data=None):
        """Show the motility analysis dialog"""
        dialog = MotilityDialog(tracked_cells, lineage_tracks, image_data, self)
        dialog.exec()

    def show_time_comparison(self, lineage_tracks):
        """Show time comparison dialog - this would be a separate dialog class"""
        # This would instantiate and show a TimeComparisonDialog
        pass
