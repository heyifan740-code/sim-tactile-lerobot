from __future__ import annotations

import numpy as np


def create_tactile_heatmap_panel(num_rows: int, num_cols: int, scale: int = 12, title: str = "SO100 Tactile Heatmap (fn)"):
    """Create a docked omni.ui window for a tactile heatmap.

    Returns a dict with an ``update(fn_grid)`` callback, or ``None`` when omni.ui
    is unavailable (for headless mode or minimal IsaacSim installs).
    """

    import asyncio

    try:
        import omni.kit.app
        import omni.ui as ui
    except ImportError:
        return None

    try:
        from matplotlib import cm

        cmap = cm.get_cmap("viridis")
    except ImportError:
        cmap = None

    height_px, width_px = num_rows * scale, num_cols * scale
    provider = ui.ByteImageProvider()
    blank = np.zeros((height_px, width_px, 4), dtype=np.uint8)
    blank[..., 3] = 255
    provider.set_bytes_data(blank.flatten().data, [width_px, height_px])

    window = ui.Window(
        title,
        width=width_px + 60,
        height=height_px + 80,
        visible=True,
        dock_preference=ui.DockPreference.RIGHT_TOP,
    )
    with window.frame:
        with ui.VStack(spacing=4):
            ui.Label("12x32 tactile heatmap   viridis   blue->green->yellow")
            ui.ImageWithProvider(provider, width=width_px, height=height_px)

    async def _dock_window(window_title: str):
        app = omni.kit.app.get_app()
        for _ in range(60):
            if ui.Workspace.get_window(window_title):
                break
            await app.next_update_async()
        win = ui.Workspace.get_window(window_title)
        prop = ui.Workspace.get_window("Property")
        if win is not None:
            if prop is not None:
                win.dock_in(prop, ui.DockPosition.SAME, 1.0)
            win.visible = True
            win.focus()

    asyncio.ensure_future(_dock_window(window.title))

    def _to_rgba(img01: np.ndarray) -> np.ndarray:
        img01 = np.clip(np.nan_to_num(img01), 0.0, 1.0)
        if cmap is None:
            gray = (img01 * 255).astype(np.uint8)
            return np.dstack((gray, gray, gray, np.full_like(gray, 255)))
        rgb = (cmap(img01)[..., :3] * 255).astype(np.uint8)
        alpha = np.full((*rgb.shape[:2], 1), 255, dtype=np.uint8)
        return np.concatenate((rgb, alpha), axis=2)

    def update(fn_grid: np.ndarray):
        upscaled = np.kron(fn_grid.astype(np.float32), np.ones((scale, scale), dtype=np.float32))
        provider.set_bytes_data(_to_rgba(upscaled).flatten().data, [width_px, height_px])

    return {"window": window, "provider": provider, "update": update}