"""The tab bar in biofilm mode (Bukola, 18 Nov 2026): Segmentation, then the four stages as tabs of their own, then
Live-Dead Analysis. The mode is already the biofilm layer, so there is no "Biofilm Stages" tab around the stages, and
the older EPS Analysis, Colony Separation and Colony Analysis tabs are kept in the program but not shown."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QPushButton, QTabWidget, QWidget

WANTED = ["Segmentation", "Seeding", "Microcolonies", "Maturation", "Dispersion", "Live-Dead Analysis"]


def names(tabs):
    return [tabs.tabText(i) for i in range(tabs.count())]


def test_the_stage_pages_become_tabs_of_the_main_bar():
    from nd2_analyzer.ui.biofilms.stages.tab import BiofilmStagesTab
    main = QTabWidget(); main.addTab(QWidget(), "Segmentation")
    stages = BiofilmStagesTab(); stages.move_pages_to(main)
    main.addTab(QWidget(), "Live-Dead Analysis")
    assert names(main) == WANTED
    assert main.widget(1) is stages.seeding_page and main.widget(2) is stages.microcolony_page


def test_the_stage_tabs_still_share_one_cursor_and_other_tabs_are_left_alone():
    from nd2_analyzer.ui.biofilms.stages.tab import BiofilmStagesTab
    main = QTabWidget(); main.addTab(QWidget(), "Segmentation")
    stages = BiofilmStagesTab(); stages.move_pages_to(main); main.addTab(QWidget(), "Live-Dead Analysis")
    main.setCurrentIndex(2)
    stages.microcolony_page.hour_changed.emit(30.0)          # the user moves the cursor on the Microcolonies tab
    main.setCurrentIndex(3)
    assert main.widget(3).hour == 30.0                       # the Maturation tab opens at the same hour
    main.setCurrentIndex(0); main.setCurrentIndex(5)         # Segmentation and Live-Dead are not stage pages: nothing to do
    assert stages.shared_hour == 30.0


def test_the_pages_that_still_read_a_study_file_carry_their_own_open_button():
    from nd2_analyzer.ui.biofilms.stages.tab import BiofilmStagesTab
    main = QTabWidget()
    stages = BiofilmStagesTab(); stages.move_pages_to(main)
    for k, wanted in ((0, False), (1, False), (2, False), (3, True)):       # only Dispersion still reads a study file
        assert (main.widget(k).findChild(QPushButton, "open_study") is not None) is wanted


def test_biofilm_mode_shows_segmentation_the_four_stages_and_live_dead(monkeypatch):
    from nd2_analyzer.ui import app as A

    def biofilm(config, parent=None):
        config.set_mode(A.AnalysisMode.BIOFILM_CLOUD, save=False)
        return A.AnalysisMode.BIOFILM_CLOUD

    monkeypatch.setattr(A.ModeSelectionDialog, "show_mode_selection", staticmethod(biofilm))
    win = A.App()
    try:
        assert names(win.tab_widget) == WANTED
        for older in (win.eps_analysis_tab, win.colony_separation_tab, win.cube_analysis_tab):
            assert older.isHidden() and win.tab_widget.indexOf(older) == -1
    finally:
        win.close()


def test_every_stage_controller_listens_with_the_shape_the_tools_messages_have():
    """The message bus takes the shape of a message from its first listener, so whichever stage controller is built
    first must ask for the same names as the Seeding controller does: experiment_loaded(experiment),
    image_data_loaded(image_data), view_index_changed(index), batch_segmentation_completed(time_range, positions, model)."""
    import inspect
    from nd2_analyzer.ui.biofilms.stages.growth_controller import GrowthController
    from nd2_analyzer.ui.biofilms.stages.microcolony_controller import MicrocolonyController
    from nd2_analyzer.ui.biofilms.stages.seeding_controller import SeedingController

    def shape(method):
        return [(name, p.default is inspect.Parameter.empty, p.kind) for name, p in inspect.signature(method).parameters.items() if name != "self"]

    seeding = {"experiment_loaded": SeedingController.on_experiment_loaded, "image_data_loaded": SeedingController.on_image_data_loaded,
               "view_index_changed": SeedingController.on_view_index_changed, "batch_segmentation_completed": SeedingController.on_segmentation_completed}
    for controller in (MicrocolonyController, GrowthController):
        assert {topic for topic, _method in controller.LISTENS} >= {"experiment_loaded", "image_data_loaded", "batch_segmentation_completed"}
        for topic, method in controller.LISTENS:
            assert shape(getattr(controller, method)) == shape(seeding[topic]), (controller.__name__, topic)
