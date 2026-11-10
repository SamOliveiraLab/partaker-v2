"""The Seeding page end to end, offscreen: the App in biofilm mode, the Lee 2018 FC1 film loaded as the recording, the
tool's own cell rows of all 35,400 frames fed to MetricsService, then Track and Measure through the controller, the top
candidate followed, and the page grabbed as PNGs under ~/Workspace/results/partaker2_fixtures/seeding_e2e/.

The cell rows come from tests/fixtures/build_seeding_fixture.py's full run (full_rows/rows_*.parquet), so no frame is
segmented here. Run with the partaker-v2 venv:
    QT_QPA_PLATFORM=offscreen ~/Workspace/partaker-v2/.venv/bin/python tests/fixtures/run_seeding_end_to_end.py
"""
import glob
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
SRC = Path(__file__).resolve().parents[2] / "src"
sys.path.insert(0, str(SRC))

import polars as pl
from pubsub import pub
from PySide6.QtWidgets import QApplication

FILM = Path("/Users/smdoliveira/Workspace/Datasets/external/lee2018/tiff/lee2018_WT_FC1_29.5h.ome.tif")
FIXTURES = Path.home() / "Workspace/results/partaker2_fixtures/seeding_lee_fc1"
OUT = Path.home() / "Workspace/results/partaker2_fixtures/seeding_e2e"
WINDOW = (1600, 1000)
WAIT_S = 600


def biofilm_mode_app():
    """The App with the mode dialog answered 'biofilm' before it opens."""
    from nd2_analyzer.ui.biofilms.analysis_mode import AnalysisMode
    from nd2_analyzer.ui.dialogs import mode_selection_dialog
    mode_selection_dialog.ModeSelectionDialog.show_mode_selection = staticmethod(lambda config, parent=None: AnalysisMode.BIOFILM_CLOUD)
    from nd2_analyzer.ui.app import App
    return App()


def load_recording():
    """The film as the tool's ImageData and an Experiment with the fixture's biofilm block."""
    from nd2_analyzer.data.experiment import Experiment
    from nd2_analyzer.data.image_data import ImageData
    meta = json.loads((FIXTURES / "late" / "meta.json").read_text())
    image_data = ImageData.load_nd2(str(FILM))
    n_frames = int(image_data.data.shape[0])
    biofilm = {"start_hour": meta["start_hour"], "channel_roles": {"0": "cell_marker", "1": "other", "2": "other"},
               "channel_labels": {"0": "masks", "1": "YFP", "2": "CFP"}, "flow": meta["flow"]}
    experiment = Experiment(name="lee2018_WT_FC1", image_files=[str(FILM)], interval=meta["dt_s"], selected_positions=[0],
                            time_range=(0, n_frames - 1), biofilm=biofilm)
    pub.sendMessage("experiment_loaded", experiment=experiment)
    return image_data, experiment


def feed_cell_rows():
    """All the tool's cell rows of the recording into MetricsService, as if every frame had been segmented."""
    from nd2_analyzer.analysis.metrics_service import MetricsService
    files = sorted(glob.glob(str(FIXTURES / "full_rows" / "rows_*.parquet")))
    df = pl.concat([pl.read_parquet(f) for f in files], how="vertical")
    MetricsService().df = df
    return df


def wait_until(app, condition, what):
    t0 = time.time()
    while not condition():
        app.processEvents()
        time.sleep(0.05)
        if time.time() - t0 > WAIT_S:
            raise TimeoutError(what)
    app.processEvents()


def settle(app, n=6):
    for _ in range(n):
        app.processEvents()
        time.sleep(0.02)


def grab(app, page, name):
    settle(app)
    img = page.grab()
    path = OUT / f"{name}.png"
    img.save(str(path))
    print("saved", path, img.width(), "x", img.height())


def step_words(page):
    return {step: page.step_labels[step].text().replace("<b>", "").replace("</b>", "") for step in page.step_labels}


def card_words(results):
    return [{"key": m.key, "group": m.group, "available": m.available, "text": m.text if m.available else "", "reason": m.reason,
             "n": m.n} for m in results.metrics]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    app = QApplication.instance() or QApplication([])
    win = biofilm_mode_app()
    win.resize(*WINDOW)
    win.main_splitter.setSizes([340, WINDOW[0] - 340])
    tab = win.biofilm_stages_tab
    page = tab.seeding_page
    controller = tab.controller
    win.tab_widget.setCurrentWidget(tab); tab.tabs.setCurrentWidget(page)
    win.show(); settle(app)

    t0 = time.time()
    image_data, experiment = load_recording()
    print(f"recording: shape {tuple(image_data.data.shape)}, pixel {image_data.pixel_size_um()}, loaded in {time.time() - t0:.1f} s")
    df = feed_cell_rows()
    print(f"cell rows: {df.height:,} over frames {df['time'].min()} to {df['time'].max()}")
    controller.refresh(); settle(app)
    print("steps before:", step_words(page))
    assert page.track_button.isEnabled() and not page.measure_button.isEnabled()
    grab(app, page, "01_before_track")

    t0 = time.time()
    page.track_button.click()
    wait_until(app, lambda: not controller.is_tracking(), "tracking did not finish")
    tracks = controller.tracks(controller.position())
    print(f"tracked: {len(tracks):,} tracks in {time.time() - t0:.1f} s; steps:", step_words(page))
    assert page.measure_button.isEnabled()
    grab(app, page, "02_after_track")

    t0 = time.time()
    page.measure_button.click()
    wait_until(app, lambda: not controller.is_measuring() and controller.results is not None, "measure did not finish")
    results = controller.results
    print(f"measured in {time.time() - t0:.1f} s; steps:", step_words(page))
    for m in results.metrics:
        print(f"  {m.group:9s} {m.key:26s} {m.text if m.available else 'not available: ' + m.reason}  (n={m.n})")
    for note in results.notes:
        print("  note:", note)
    grab(app, page, "03_after_measure")

    page.candidates.setCurrentRow(0)
    top = page.candidates.item(0).text()
    page.follow_button.click(); settle(app, 10)
    print("followed:", top, "| film hours:", page.followed_span, "| slider frame:", win.viewArea.slider_t.value())
    grab(app, page, "04_after_follow")

    saved = controller.save_results(str(OUT)); print("tables saved under", saved)
    image = controller.save_image(str(OUT)); print("report image", image)
    (OUT / "cards.json").write_text(json.dumps({"steps": step_words(page), "cards": card_words(results), "top_candidate": top,
                                                "notes": list(results.notes)}, indent=2))
    print("done")


if __name__ == "__main__":
    main()
