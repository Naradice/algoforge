"""Geometric table border and grid line extraction from PDF layout objects.

Extends the graph-based line segment adjacency logic originally developed in stocknet_study.
Supports both fully-bordered grids and open tables with horizontal rules only.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from pdfminer.layout import LTLine, LTRect

BBOX_COLUMNS = ["x_min", "y_min", "x_max", "y_max"]


class Point:
    """Represents a cell intersection corner with 4 directional line pointers."""

    __edges_dict = {
        8: "left",
        4: "right",
        2: "top",
        1: "bottom",
        10: "left_top",
        9: "left_bottom",
        6: "right_top",
        5: "right_bottom",
    }

    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y
        self.__lines: list[Any] = [None, None, None, None]

    @property
    def left(self):
        return self.__lines[0]

    @left.setter
    def left(self, value):
        self.__lines[0] = value

    @property
    def right(self):
        return self.__lines[1]

    @right.setter
    def right(self, value):
        self.__lines[1] = value

    @property
    def top(self):
        return self.__lines[2]

    @top.setter
    def top(self, value):
        self.__lines[2] = value

    @property
    def bottom(self):
        return self.__lines[3]

    @bottom.setter
    def bottom(self, value):
        self.__lines[3] = value

    def has_left(self) -> bool:
        return self.__lines[0] is not None

    def has_right(self) -> bool:
        return self.__lines[1] is not None

    def has_top(self) -> bool:
        return self.__lines[2] is not None

    def has_bottom(self) -> bool:
        return self.__lines[3] is not None

    def get_edge_type(self) -> int:
        b = 1 if self.__lines[0] is None else 0
        for t in self.__lines[1:]:
            b = b << 1
            if t is None:
                b += 1
        return b

    def edge_type_to_str(self, value: int) -> str | None:
        return self.__edges_dict.get(value)

    def __str__(self) -> str:
        return f"Point({self.x:.1f}, {self.y:.1f})"


def is_near_point(p1: float, p2: float, margin: float = 1.0) -> bool:
    """Check if two coordinate points are within distance margin."""
    return abs(p1 - p2) <= margin


def is_adj(base_node: tuple | pd.Series, target_node: tuple | pd.Series, margin: float = 1.0) -> tuple[int, Any]:
    """Check if target_node line is adjacent/connected to base_node line."""
    x_l, y_l, x_r, y_t = base_node[:4]
    xt_l, yt_l, xt_r, yt_t = target_node[:4]
    height = abs(yt_t - yt_l)
    width = abs(xt_r - xt_l)
    default_response = (-1, None)

    is_horizontal_line = abs(x_r - x_l) > margin
    is_vertical_line = abs(y_t - y_l) > margin

    if height > margin and width > margin:
        return default_response
    if is_vertical_line and is_horizontal_line:
        return default_response

    if height > margin:
        # target is vertical
        if is_horizontal_line:
            if is_near_point(y_t, yt_t, margin):
                if is_near_point(x_l, xt_l, margin):
                    return (0, target_node)
                elif is_near_point(x_r, xt_r, margin):
                    return (1, target_node)
                elif (x_l < xt_l) and (x_r > xt_l):
                    return (-1, target_node)
            return default_response
        else:
            if is_near_point(x_l, xt_l, margin):
                if is_near_point(y_t, yt_l, margin):
                    return (2, target_node)
                elif is_near_point(y_l, yt_t, margin):
                    return (3, target_node)
                elif (y_l < yt_l) and (y_t > yt_t):
                    return (-1, target_node)
            return default_response
    elif width > margin:
        # target is horizontal
        if is_vertical_line:
            if is_near_point(x_l, xt_l, margin):
                if is_near_point(y_t, yt_t, margin):
                    return (2, target_node)
                elif is_near_point(y_l, yt_l, margin):
                    return (3, target_node)
                elif (y_l < yt_l) and (y_t > yt_t):
                    return (-1, target_node)
            return default_response
        else:
            if is_near_point(y_t, yt_t, margin):
                if is_near_point(x_l, xt_r, margin):
                    return (0, target_node)
                elif is_near_point(x_r, xt_l, margin):
                    return (1, target_node)
                elif (x_l < xt_l) and (x_r > xt_l):
                    return (-1, target_node)
            return default_response
    return default_response


def separate_adjs(
    target_node: pd.Series, remaining_nodes: pd.DataFrame, margin: float = 1.0
) -> tuple[list[pd.Series], pd.DataFrame]:
    """Partition remaining line segments into adjacent and non-adjacent."""
    adjs = []
    others = []
    for index in remaining_nodes.index:
        item = remaining_nodes.loc[index]
        _, coord = is_adj(target_node, item, margin=margin)
        if coord is None:
            others.append(item)
        else:
            adjs.append(item)
    return adjs, pd.DataFrame(others, columns=BBOX_COLUMNS) if others else pd.DataFrame(columns=BBOX_COLUMNS)


def retrieve_table_elements(page: Any) -> pd.DataFrame:
    """Extract line segments and split rectangle strokes into lines from a PDF page."""
    lines: list[tuple[float, float, float, float]] = []
    rects: list[LTRect] = []

    for item in page:
        if isinstance(item, LTLine):
            lines.append(item.bbox)
        elif isinstance(item, LTRect):
            rects.append(item)

    if len(rects) >= 2:
        for rect in rects:
            x_min, y_min, x_max, y_max = rect.bbox
            w = abs(x_max - x_min)
            h = abs(y_max - y_min)
            if w <= 2.0 or h <= 2.0:
                lines.append(rect.bbox)
            else:
                left_line = (x_min, y_min, x_min, y_max)
                right_line = (x_max, y_min, x_max, y_max)
                top_line = (x_min, y_max, x_max, y_max)
                bottom_line = (x_min, y_min, x_max, y_min)
                lines.extend([left_line, right_line, top_line, bottom_line])

    if not lines:
        return pd.DataFrame(columns=BBOX_COLUMNS)

    lines_df = pd.DataFrame(lines, columns=BBOX_COLUMNS)
    lines_df = lines_df.sort_values(by="y_max", ascending=False).reset_index(drop=True)
    return lines_df


def retrieve_table_lines(target_lines: pd.DataFrame, margin: float = 1.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Find a connected component of table line segments starting from the first line."""
    if target_lines.empty:
        return pd.DataFrame(columns=BBOX_COLUMNS), pd.DataFrame(columns=BBOX_COLUMNS)

    initial_node = target_lines.iloc[0]
    table_nodes: list[pd.Series] = []
    adjs: list[pd.Series] = [initial_node]
    remaining_nodes = target_lines.iloc[1:].copy()

    while len(remaining_nodes) > 0:
        pre_length = len(remaining_nodes)
        if len(adjs) > 0:
            new_adjs = []
            for adj in adjs:
                children, remaining_nodes = separate_adjs(adj, remaining_nodes, margin=margin)
                new_adjs.extend(children)
                table_nodes.append(adj)
            adjs = new_adjs
        else:
            break
        if pre_length == len(remaining_nodes):
            break

    table_nodes_df = pd.DataFrame(table_nodes, columns=BBOX_COLUMNS)
    remaining_adjs_df = pd.DataFrame(adjs, columns=BBOX_COLUMNS)
    if not table_nodes_df.empty:
        table_nodes_df = pd.concat([table_nodes_df, remaining_adjs_df], ignore_index=True)
    else:
        remaining_nodes = pd.concat([remaining_adjs_df, remaining_nodes], ignore_index=True)

    table_nodes_df.reset_index(inplace=True, drop=True)
    return table_nodes_df, remaining_nodes


def to_bbox(table_lines: pd.DataFrame) -> tuple[float, float, float, float]:
    """Compute outer bounding box (x_min, y_min, x_max, y_max) from a set of lines."""
    return (
        float(table_lines.x_min.min()),
        float(table_lines.y_min.min()),
        float(table_lines.x_max.max()),
        float(table_lines.y_max.max()),
    )


def is_in_bb(
    target_out_bb: tuple[float, float, float, float],
    target_in_bb: tuple[float, float, float, float],
    margin: float = 1.0,
) -> bool:
    """Check if target_in_bb is fully contained inside target_out_bb."""
    x_min, y_min, x_max, y_max = target_out_bb
    xt_min, yt_min, xt_max, yt_max = target_in_bb

    return (
        (x_min - margin <= xt_min)
        and (y_min - margin <= yt_min)
        and (x_max + margin >= xt_max)
        and (y_max + margin >= yt_max)
    )


def find_table_regions(
    lines_df: pd.DataFrame,
    page_texts_df: pd.DataFrame | None = None,
    margin: float = 2.0,
    min_lines: int = 2,
) -> list[tuple[float, float, float, float]]:
    """Discover table regions supporting both fully-enclosed grids and horizontal-rule open tables.

    Returns a list of bounding boxes (x_min, y_min, x_max, y_max) sorted from top to bottom.
    """
    if lines_df.empty:
        return []

    regions: list[tuple[float, float, float, float]] = []

    # 1. First, find fully connected grid boxes (with vertical & horizontal lines)
    remaining_lines = lines_df.copy()
    orphan_h_lines: list[pd.Series] = []

    while len(remaining_lines) >= min_lines:
        pre_len = len(remaining_lines)
        grp, remaining_lines = retrieve_table_lines(remaining_lines, margin=margin)
        if grp.empty:
            break

        bb = to_bbox(grp)
        w = bb[2] - bb[0]
        h = bb[3] - bb[1]

        if w > 20.0 and h > 15.0:
            # Has both width and height -> standard enclosed table
            regions.append(bb)
        else:
            # Flat line group (only horizontal or only vertical)
            if w > 20.0 and h <= margin:
                # Merge horizontal segments on approximately the same Y
                orphan_h_lines.append(pd.Series({"x_min": bb[0], "y_min": bb[1], "x_max": bb[2], "y_max": bb[3]}))

        if len(remaining_lines) == pre_len:
            break

    # 2. Form open tables from orphan horizontal lines (very common in Japanese IR/financial notes)
    if len(orphan_h_lines) >= 2:
        h_df = pd.DataFrame(orphan_h_lines).sort_values(by="y_max", ascending=False).reset_index(drop=True)
        # Cluster horizontal lines whose X spans significantly overlap
        i = 0
        while i < len(h_df) - 1:
            line_top = h_df.iloc[i]
            cluster = [line_top]
            j = i + 1
            while j < len(h_df):
                cand = h_df.iloc[j]
                x_overlap = min(line_top["x_max"], cand["x_max"]) - max(line_top["x_min"], cand["x_min"])
                span = max(line_top["x_max"] - line_top["x_min"], cand["x_max"] - cand["x_min"])
                y_dist = line_top["y_max"] - cand["y_min"]

                # If overlap > 60% of span and vertical distance within reasonable table height (e.g. 200 pt)
                if x_overlap > 0.6 * span and 5.0 <= y_dist <= 250.0:
                    cluster.append(cand)
                    j += 1
                else:
                    break

            if len(cluster) >= 2:
                c_df = pd.DataFrame(cluster)
                # Outer bbox formed by this cluster
                c_bb = (
                    float(c_df["x_min"].min()),
                    float(c_df["y_min"].min()) - 15.0,  # add breathing room for lowest row text
                    float(c_df["x_max"].max()),
                    float(c_df["y_max"].max()) + 2.0,
                )
                regions.append(c_bb)
                i = j
            else:
                i += 1

    # Sort all discovered regions from top to bottom (descending y_max)
    regions = sorted(regions, key=lambda b: b[3], reverse=True)
    return regions
