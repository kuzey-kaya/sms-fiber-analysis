"""The data behind one graph as a table (2026-11-17, the advisor: every graph exportable on its
own, as a picture and as data). The picture is made in the browser; this is the data half."""

import io

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from sms_analysis.figure_export import ascii_text, figure_csv, figure_table, safe_name


def test_curves_on_the_same_x_share_one_column():
    t = np.arange(5.0)
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=t, y=1550 + t, name="dip@1552.8"), secondary_y=False)
    fig.add_trace(go.Scatter(x=t, y=20 + t, name="Temperature (°C)"), secondary_y=True)
    fig.update_xaxes(title_text="Time (s)")
    fig.update_yaxes(title_text="<b>Peak Wavelength</b> (nm)", secondary_y=False)
    fig.update_yaxes(title_text="Temperature (°C)", secondary_y=True)
    table = figure_table(fig)
    # the axis title is plain text, and a curve named like its axis does not repeat it
    assert list(table.columns) == ["Time (s)", "dip@1552.8 [Peak Wavelength (nm)]", "Temperature (°C)"]
    np.testing.assert_allclose(table["dip@1552.8 [Peak Wavelength (nm)]"], 1550 + t)


def test_curves_on_other_x_values_come_as_groups_side_by_side():
    x = np.linspace(1500, 1510, 11)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=np.cos(x), name="spectrum"))
    fig.add_trace(go.Scatter(x=[1502.3, 1508.6], y=[0.9, 0.8], mode="markers", name="peaks",
                             text=["λ+1", "λ+2"], error_y=dict(type="data", array=[0.01, 0.02])))
    fig.update_layout(xaxis_title="Wavelength (nm)", yaxis_title="Transmittance")
    table = figure_table(fig)
    assert list(table.columns) == ["x1: Wavelength (nm)", "y1: spectrum [Transmittance]",
                                   "x2: Wavelength (nm)", "y2: peaks [Transmittance]", "y2: peaks label",
                                   "y2: peaks ±"]
    assert len(table) == 11 and table["x2: Wavelength (nm)"].notna().sum() == 2    # padded with empty cells
    text = figure_csv(fig)
    assert "nan" not in text.lower() and text.isascii()
    back = pd.read_csv(io.StringIO(text))
    assert list(back["y2: peaks label"].dropna()) == ["lambda+1", "lambda+2"]
    assert back.columns[-1] == "y2: peaks +/-"


def test_a_fringe_that_lost_lock_joins_the_others():
    t = np.arange(6.0)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t[[0, 1, 4, 5]], y=[1.0, 2.0, 5.0, 6.0], name="short"))    # two frames missing
    fig.add_trace(go.Scatter(x=t, y=t * 10, name="full"))
    fig.update_layout(xaxis_title="Time (s)")
    table = figure_table(fig)
    assert list(table.columns) == ["Time (s)", "short", "full"]
    np.testing.assert_allclose(table["Time (s)"], t)
    np.testing.assert_allclose(table["short"], [1, 2, np.nan, np.nan, 5, 6], equal_nan=True)
    # repeated x values (shift against mass: many frames per mass) are never merged by value
    fig2 = go.Figure([go.Scatter(x=[0, 0, 50, 50], y=[1, 2, 3, 4], name="frames"),
                      go.Scatter(x=[0, 50], y=[1.5, 3.5], name="plateau mean")])
    assert list(figure_table(fig2).columns) == ["x1: x", "y1: frames", "x2: x", "y2: plateau mean"]


def test_stacked_panels_take_the_title_of_the_axis_they_share():
    x = np.linspace(20, 40, 8)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True)
    fig.add_trace(go.Scatter(x=x, y=x * 2, name="tracked"), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=x * 0, name="residual"), row=2, col=1)
    fig.update_xaxes(title_text="Temperature (°C)", row=2, col=1)
    fig.update_yaxes(title_text="Wavelength (nm)", row=1, col=1)
    fig.update_yaxes(title_text="Residual (pm)", row=2, col=1)
    assert list(figure_table(fig).columns) == ["Temperature (°C)", "tracked [Wavelength (nm)]",
                                               "residual [Residual (pm)]"]


def test_a_map_is_the_matrix_as_drawn():
    wl = np.array([1500.0, 1500.2, 1500.4], dtype=np.float32)          # float32, as in the strain file
    time = np.array([0.0, 1.0])
    z = np.arange(6.0).reshape(2, 3)
    fig = go.Figure(go.Heatmap(x=wl, y=time, z=z))
    fig.add_trace(go.Scatter(x=[1500.1, 1500.1], y=time, name="dip"))          # an overlaid track
    fig.update_layout(xaxis_title="Wavelength (nm)", yaxis_title="Time (min)")
    table = figure_table(fig)
    assert list(table.columns) == ["Time (min) \\ Wavelength (nm)", "1500", "1500.2", "1500.4"]
    np.testing.assert_allclose(table.iloc[:, 1:].to_numpy(), z)
    surface = go.Figure(go.Surface(x=wl, y=time, z=z))
    surface.update_layout(scene=dict(xaxis_title="Wavelength (nm)", yaxis_title="Time (min)"))
    assert figure_table(surface).shape == (2, 4)
    assert figure_table(surface).columns[0] == "Time (min) \\ Wavelength (nm)"


def test_empty_figures_and_names():
    assert figure_table(go.Figure()).empty
    assert figure_csv(go.Figure(go.Scatter(x=[], y=[]))).strip() == ""
    unnamed = figure_table(go.Figure([go.Scatter(y=[1, 2]), go.Scatter(y=[3, 4])]))
    assert list(unnamed.columns) == ["x", "trace 1", "trace 2"]
    twins = figure_table(go.Figure([go.Scatter(y=[1, 2], name="a"), go.Scatter(y=[3, 4], name="a")]))
    assert list(twins.columns) == ["x", "a", "a (2)"]
    assert safe_name("sms strain 2026/10/15 (norm)") == "sms_strain_2026_10_15_norm" and safe_name("  ") == "figure"
    assert ascii_text("|φ(1570) − φ(1545)| (rad)") == "|phi(1570) - phi(1545)| (rad)"
    assert ascii_text("λc +31.0 pm/°C · R² 0.998") == "lambdac +31.0 pm/degC - R^2 0.998"
