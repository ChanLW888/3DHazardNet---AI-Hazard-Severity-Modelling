#!/usr/bin/env python3
"""3DHazardNet dual-monitor demo: two HTTP servers, one synchronized clock.

By default, serves existing dashboard/data.json, frames/ and cluster_media/.
Run --rebuild to regenerate those dashboard assets from cached V10.1 inference
and Code 1 clustering/severity files, without neural-network inference.
Run --build-only to regenerate the cache and exit without starting servers.
The original inference NPZ files and clustering metadata are never modified.

Hazard review is non-destructive. One-click removal/restoration targets the
hazard associated with the current playback frame. Exclusions persist in hazard_review.json. Demo POIs persist in hazard_pins.json. Street name plus suburb/state are resolved using road-network matching and fallback geocoders; results cached in address_cache.json. A per-location operator correction can be saved when external suburb lookups fail.
A curated data_curated.json and hazards_curated.csv are exported after each edit.
Excluded clusters disappear from the map, severity summaries and incident navigation,
while the original data.json, frames and inference files remain untouched.

Starting-point video: place starting_point.mp4 in dashboard root, or use placeholder.

Display 1: http://localhost:8890/  (satellite GPS map)
Display 2: http://localhost:8891/  (video, voxels, severity)

On Monash HPC, open both /proxy/<port>/ URLs, including trailing slashes.
All browser API and imagery paths are relative to the proxy base path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import os
import tempfile
from datetime import datetime, timezone
import bisect
import functools
import http.server
import json
import math
import re
import threading
import time
import urllib.request
import urllib.parse
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_ROOT = Path(
    "/fs04/scratch2/bt60/lcha_data/map-anything/personal_data/"
    "hazard_test_101/hazard_clusters_cached_v10/dashboard"
)


# User-specified named starting point, separate from predicted hazard clusters.
START_POINT = {'name': 'Outside Monash Innovation Labs',
               'lat': -(37 + 54/60 + 33.5/3600),
               'lon': 145 + 8/60 + 4.9/3600}
START_VIDEO_NAME = 'starting_point.mp4'

STYLE = r"""
:root{--bg:#09111b;--panel:#111e2c;--panel2:#172738;--edge:#2b4055;
--txt:#e9f2fa;--muted:#95acc0;--accent:#49b9e9;--green:#58d5af;--amber:#f7bb5c}
*{box-sizing:border-box}html,body{margin:0;width:100%;height:100%;overflow:hidden}
body{background:var(--bg);color:var(--txt);font:14px/1.4 Inter,Segoe UI,Arial,sans-serif}
button,select,input{font:inherit}button,select{border:1px solid #365269;background:#192f42;color:var(--txt);
 border-radius:7px;padding:8px 11px;cursor:pointer}button:hover{background:#27506b}
button.primary{background:#087caa;border-color:#2497c9;font-weight:750}
button:focus-visible,select:focus-visible,input:focus-visible{outline:2px solid #69cef8}
.h{height:71px;padding:11px 23px;display:flex;align-items:center;justify-content:space-between;
 border-bottom:1px solid var(--edge);background:#0d1926;gap:12px}
.brand{font-size:24px;font-weight:760;letter-spacing:-.6px}.sub{font-size:13px;color:var(--muted);margin-top:3px;font-weight:650}
.eyebrow{color:var(--accent);font-size:10px;letter-spacing:1.75px;font-weight:800}
.topright{display:flex;gap:10px;align-items:center}.tag{border:1px solid #26564b;background:#13382e;
 color:#8ee6c4;border-radius:30px;padding:7px 11px;font-size:11px;font-weight:700}
.tag.secondary{color:#b9d7e8;background:#172a3b;border-color:#365269}
.panel{background:var(--panel);border:1px solid var(--edge);border-radius:11px;min-width:0;min-height:0;overflow:hidden}
.panelhead{min-height:39px;padding:9px 13px;border-bottom:1px solid var(--edge);font-size:12px;
 display:flex;align-items:center;justify-content:space-between;gap:8px;font-weight:750}
.meta{color:var(--muted);font-size:11px;font-weight:500}.muted{color:var(--muted)}
.dot{height:8px;width:8px;display:inline-block;border-radius:50%;background:var(--green);margin-right:8px}
.value{font-size:22px;font-weight:800;letter-spacing:-.6px;font-variant-numeric:tabular-nums}
.kicker{font-size:10px;letter-spacing:1px;color:var(--muted);text-transform:uppercase;font-weight:750}
.sc{padding:12px 15px}.small{font-size:11px}.row{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
.badge{display:inline-block;padding:4px 7px;border:1px solid #365269;background:#1c3347;
 border-radius:6px;font-size:10px;color:#d4e5ef}
.sq{display:inline-block;width:9px;height:9px;border-radius:2px;margin-right:5px}
.legend{display:flex;gap:17px;align-items:center;flex-wrap:wrap;color:#b3c5d4;font-size:11px}
.hint{color:var(--muted);font-size:11px}.err{color:#ffc6a8}
a{color:#7dd5fb}strong{font-weight:800}
/* MAP MONITOR */
.mapstage{display:grid;grid-template-columns:minmax(0,1fr) 340px;gap:13px;
 padding:13px 17px 15px;height:calc(100vh - 71px);min-height:0}
.mapframe{display:flex;flex-direction:column}.mapactions{display:flex;align-items:center;gap:9px;flex-wrap:wrap}
#map{height:100%;min-height:0;width:100%;background:#1a2c3c}
.mapfooter{min-height:46px;padding:10px 13px;display:flex;justify-content:space-between;
 gap:10px;align-items:center;border-top:1px solid var(--edge)}
.maprail{display:grid;grid-template-rows:auto minmax(0,1fr) auto;gap:12px;min-height:0}
.railscroll{overflow:auto;padding:13px;min-height:0;scrollbar-color:#34556c transparent}
#inspectImage{width:100%;max-height:250px;object-fit:contain;background:#0b1621;
 border:1px solid var(--edge);border-radius:7px;margin-top:10px}
#inspectImageLink{display:none}#inspectLinks{display:flex;gap:7px;flex-wrap:wrap;margin-top:12px}
#inspectLinks a{border:1px solid #365269;background:#17394c;border-radius:6px;
 padding:7px 10px;text-decoration:none;font-size:11px}
.routeStat{border-bottom:1px solid var(--edge);padding:10px 0}
.routeStat:last-child{border-bottom:0}.topincident{width:100%;text-align:left;
 display:flex;justify-content:space-between;gap:8px;margin-bottom:6px;padding:10px}
#tileWarning{display:none;position:absolute;bottom:60px;left:14px;z-index:1000;
 border-radius:7px;background:#192a3bef;color:#ffd29b;border:1px solid #705934;
 padding:9px 11px;max-width:510px;font-size:11px;pointer-events:none}
.vanIcon{display:grid;place-items:center;width:33px;height:33px;border-radius:50%;
 background:#00a8e8;color:white;border:3px solid white;box-shadow:0 2px 12px #000a;font-size:16px}
.pinIcon{display:grid;place-items:center;border:2px solid #f0f7ff;border-radius:50%;
 width:29px;height:29px;color:white;font-weight:800;box-shadow:0 1px 9px #000a}
/* ANALYSIS MONITOR */
.console{display:grid;grid-template-rows:93px minmax(0,1fr) 119px;
 gap:11px;padding:12px 16px 13px;height:calc(100vh - 71px);min-height:0}
.kpis{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px;min-height:0}
.workarea{display:grid;grid-template-columns:minmax(0,1fr) 320px;gap:11px;min-height:0}
.quad{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));grid-template-rows:repeat(2,minmax(0,1fr));
 gap:10px;min-height:0}
.visual{display:flex;flex-direction:column}.visual canvas{display:block;width:100%;height:calc(100% - 39px);
 object-fit:contain;background:#eff2f4}
.side{display:grid;grid-template-rows:40% minmax(0,1fr);gap:10px;min-height:0}
#severityPanel{padding:13px 15px;overflow:auto}.stars{color:var(--amber);letter-spacing:2px}
#eventList{height:calc(100% - 39px);overflow:auto;padding:8px;scrollbar-color:#34556c transparent}
.event{display:flex;justify-content:space-between;gap:9px;width:100%;margin:4px 0;text-align:left}
.event.active{border-color:#2badd7;background:#174059}.event .muted{white-space:nowrap}
.controls{padding:11px 15px}.controltop{display:flex;align-items:center;justify-content:space-between;gap:12px}
.controls input[type=range]{width:100%;accent-color:#4dbbea;margin:11px 0 3px}
.controlbottom{display:flex;justify-content:space-between;gap:8px}.footline{display:flex;
 justify-content:space-between;align-items:center;gap:8px}
@media(max-width:1050px){.mapstage{grid-template-columns:minmax(0,1fr) 285px}
 .workarea{grid-template-columns:minmax(0,1fr) 275px}.console{grid-template-rows:80px minmax(0,1fr) 125px}}
@media(max-width:800px){html,body{overflow:auto;height:auto}.mapstage,.console{height:auto}
 .mapstage{display:block}.mapframe{height:68vh}.maprail{display:block;margin-top:12px}
 .maprail>.panel{margin-top:10px}.quad{min-height:700px}.workarea{display:block}
 .side{display:block}.side>.panel{height:240px;margin-top:10px}.kpis{grid-template-columns:repeat(2,1fr)}
 .console{display:block}.console>*{margin-bottom:12px}.controls{min-height:140px}}

/* V3 MAP: severity controls and visible count */
.filterLabel{display:flex;align-items:center;gap:7px;font-weight:650;color:#bad4e7;font-size:11px}
#severityFilter{background:#123a4f;border-color:#43708a;color:#e4f5fd;min-width:153px;font-weight:750}
#visibleCount{font-weight:800;color:#90e3ce}
/* V3 ANALYSIS: three larger screens, no permanent side rail */
.console{grid-template-rows:74px 62px minmax(0,1fr) 116px;gap:9px;padding:10px 15px 12px;position:relative}
.kpis .sc{padding:10px 13px}.kpis .value{font-size:20px}.kpis .hint{font-size:10px}
.assessment{display:flex;align-items:center;gap:22px;padding:9px 15px;min-width:0}
.assessment .sevTitle{min-width:174px}.assessment .sevMain{display:flex;align-items:center;gap:14px;flex:1;min-width:0}
.assessment .value{font-size:25px;white-space:nowrap}.assessment .stars{font-size:17px;white-space:nowrap}
.sevMeta{min-width:0;flex:1}.sevMeta #sevLevel{font-size:13px;font-weight:750}
.sevMeta #sevClasses{font-size:11px;color:var(--muted);white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
#sevEvents{font-size:10px;color:var(--muted)}
.visualGrid{display:grid;grid-template-columns:minmax(0,1.31fr) minmax(0,1fr);grid-template-rows:repeat(2,minmax(0,1fr));gap:11px;min-width:0;min-height:0}
.visualGrid #rgbPanel{grid-column:1;grid-row:1/span 2}
.visualGrid #segPanel{grid-column:2;grid-row:1}
.visualGrid #predPanel{grid-column:2;grid-row:2}
.visual{display:flex;flex-direction:column;min-height:0}
.visual .panelhead{flex:0 0 39px}
.visual .canvasWrap{flex:1;min-width:0;min-height:0;display:flex;align-items:center;justify-content:center;background:#111c29;overflow:hidden}
.visual canvas{width:100%;height:100%;display:block;object-fit:contain;background:transparent}
.visual .focusBtn{font-size:11px;padding:4px 8px;margin-left:auto}
.visualGrid.focus-mode{display:flex;flex-direction:column}
.visualGrid.focus-mode .visual:not(.focused){display:none}
.visualGrid.focus-mode .visual.focused{display:flex;flex:1}
.eventDrawer{display:none;position:absolute;z-index:2500;right:15px;top:155px;bottom:137px;width:min(390px,calc(100% - 30px));box-shadow:0 20px 60px #000b}
.eventDrawer.open{display:flex;flex-direction:column}
.eventDrawer #eventList{height:auto;flex:1;min-height:0;overflow:auto;padding:9px}
.eventDrawer .panelhead{flex-shrink:0}
#closeEvents{padding:3px 8px}
.controls{padding:10px 14px}.controltop{gap:10px}
@media(max-width:1200px){.assessment{gap:10px}.assessment .sevTitle{min-width:130px}
 .visualGrid{grid-template-columns:minmax(0,1.2fr) minmax(0,1fr)}}
@media(max-width:800px){.console{display:block;height:auto}.console>*{margin-bottom:10px}
 .assessment{flex-wrap:wrap}.visualGrid{display:flex;flex-direction:column;min-height:0}
 .visual{height:48vh;min-height:300px}.visualGrid.focus-mode .visual.focused{height:70vh}
 .eventDrawer{position:fixed;top:10%;bottom:10%;right:10px}.controls{min-height:140px}}

/* V5 PRESENTATION LAYOUT: remove KPI row and give the visual panels the viewport. */
.console{
  display:grid;grid-template-rows:66px minmax(0,1fr) 70px;
  gap:8px;padding:9px 13px 10px;height:calc(100vh - 71px);
  min-height:0;position:relative;
}
.assessment{
  display:flex;align-items:center;gap:18px;padding:7px 14px;
  min-width:0;min-height:0;overflow:hidden;
}
.severityBrief{display:flex;align-items:center;gap:17px;flex:1 1 45%;min-width:0}
.severityBrief .sevTitle{flex:0 0 144px;min-width:0}
.severityBrief .sevTitle .eyebrow{white-space:nowrap}
.severityBrief #currentHazard{font-size:11px;margin-top:3px;white-space:nowrap;
  overflow:hidden;text-overflow:ellipsis}
.severityValues{display:flex;align-items:center;gap:10px;min-width:0;flex:1}
.severityValues #sevScore{font-size:24px;line-height:1;font-weight:800;white-space:nowrap;
  font-variant-numeric:tabular-nums;letter-spacing:-.5px}
.severityValues #sevStars{font-size:14px;letter-spacing:0;white-space:nowrap}
.sevMeta{min-width:0;flex:1}
.sevMeta #sevLevel{font-size:12px;font-weight:700;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sevMeta #sevClasses{font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.sevMeta #sevEvents{font-size:10px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.legendBox{display:flex;flex-direction:column;justify-content:center;gap:6px;
  flex:1 1 450px;min-width:0;border-left:1px solid var(--edge);padding-left:18px}
.legendBox .kicker{font-size:10px}
.semanticLegend{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;
  font-size:12px;line-height:1.1;color:#d5e5f0}
.semanticLegend>span{white-space:nowrap}
.semanticLegend .sq{width:11px;height:11px;margin-right:5px}
#showEvents{flex:none;white-space:nowrap;font-size:12px;padding:7px 11px}
.visualGrid{grid-template-columns:minmax(0,1.28fr) minmax(0,1fr);
  grid-template-rows:repeat(2,minmax(0,1fr));gap:9px;min-height:0}
.visual .panelhead{flex:0 0 34px;min-height:34px;padding:6px 11px;font-size:12px}
.visual .focusBtn{font-size:11px;padding:3px 7px}
.visual .canvasWrap{min-height:0;flex:1;background:#0b1724}
.visual canvas{display:block;width:100%;height:100%;object-fit:contain;image-rendering:auto}
.controls{display:flex;flex-direction:column;justify-content:center;
  padding:6px 12px;min-height:0;gap:5px}
.controls .controltop{display:flex;align-items:center;gap:8px;min-width:0}
.controls .controltop .row{display:flex;gap:7px;align-items:center;flex-wrap:nowrap;min-width:0}
.controls button,.controls select{padding:4px 9px;border-radius:6px;font-size:12px;line-height:1.45}
.controls label{display:inline-flex;align-items:center;gap:5px;white-space:nowrap;font-size:11px}
.controls #frameInfo{font-size:11px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
.controls .shortcut{margin-left:auto;white-space:nowrap;font-size:10px;padding:3px 7px}
.scrubRow{display:flex;align-items:center;gap:11px;width:100%}
.scrubRow input[type=range]{min-width:0;flex:1;width:auto;margin:0;accent-color:#49b9e9;cursor:pointer}
.scrubRow .meta{font-size:11px;white-space:nowrap;font-variant-numeric:tabular-nums}
.eventDrawer{top:85px;bottom:90px}
#threshold{position:absolute;right:13px;bottom:88px;max-width:640px;
  padding:5px 9px;background:#342a1d;border:1px solid #9a7438;
  border-radius:6px;font-size:10px;line-height:1.25;z-index:2000}
@media(max-width:1440px){
 .assessment{gap:9px;padding:7px 11px}
 .severityBrief{gap:8px}
 .severityBrief .sevTitle{flex-basis:123px}
 .legendBox{padding-left:9px;gap:5px}
 .semanticLegend{font-size:10px;gap:5px 9px}
 .severityValues #sevScore{font-size:21px}
 .controls .shortcut{display:none}
}
@media(max-width:1050px){
 .console{grid-template-rows:96px minmax(0,1fr) 76px}
 .assessment{flex-wrap:wrap;align-content:center;overflow:visible}
 .severityBrief{flex-basis:48%}
 .legendBox{flex-basis:45%}
 .assessment #showEvents{font-size:10px;padding:4px 8px}
}
@media(max-width:800px){
 html,body{overflow:auto;height:auto}
 .console{display:flex;flex-direction:column;height:auto;min-height:calc(100vh - 71px)}
 .assessment{min-height:118px;height:auto;gap:10px}
 .severityBrief{flex:1 1 100%}
 .legendBox{flex:1 1 100%;border-left:0;border-top:1px solid var(--edge);padding:7px 0 0}
 .visualGrid{display:flex;flex-direction:column;min-height:0}
 .visual{height:44vh;min-height:245px}
 .visualGrid.focus-mode .visual.focused{height:75vh}
 .controls{min-height:92px}
 .controls .controltop .row{flex-wrap:wrap}
 .controls #frameInfo{display:none}
 .eventDrawer{position:fixed;top:10%;bottom:10%;right:10px}
 .controls .shortcut{display:none}
}

/* V7 SINGLE-CLICK REVIEW: actions apply to the currently displayed frame. */
.assessmentActions{display:flex;align-items:center;gap:6px;flex:none}
.assessmentActions button{flex:none;white-space:nowrap;font-size:11px;padding:6px 9px}
#reviewHazards{background:#713237;border:1px solid #ba6166;color:#fff1ef;font-weight:750}
#reviewHazards:hover:not(:disabled){background:#904047}
#reviewHazards.restoring{background:#153f38;border-color:#4d977d;color:#bdf7df}
#reviewHazards.restoring:hover:not(:disabled){background:#215447}
#restoreAll{background:#173b38;border-color:#427869;color:#bdf7df;font-weight:700}
#restoreAll:hover:not(:disabled){background:#28554c}
#reviewHazards:disabled,#restoreAll:disabled{opacity:.42;cursor:not-allowed}
#reviewStatus{position:absolute;right:13px;top:84px;z-index:2400;max-width:420px;
 font-size:11px;color:#c5ebd9;line-height:1.4;pointer-events:none;
 background:#122e33ef;border:1px solid #3e7c78;border-radius:6px;padding:6px 10px}
#reviewStatus:empty{display:none}
.eventDrawer{width:min(430px,calc(100% - 26px));max-width:none;top:85px;bottom:90px}
.eventDrawer .panelhead{gap:9px;min-height:42px}
.drawerIntro{padding:10px 12px 8px;border-bottom:1px solid var(--edge)}
.eventDrawer #eventList{height:auto;flex:1;min-height:0;overflow:auto;padding:9px}
.eventDrawer .eventRow{display:flex;gap:6px;align-items:stretch;margin:4px 0}
.eventDrawer .event{flex:1;min-width:0;margin:0;padding:8px 10px;
 justify-content:space-between;align-items:center}
.eventDrawer .event .eventText{flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis}
.eventDrawer .event .eventTime{white-space:nowrap;font-size:10px}
.eventDrawer .event.removed{opacity:.7;border-style:dashed}
.eventDrawer .event .removedMark{color:#ffb3a7;font-size:10px;font-weight:800}
#currentHazard.excluded{color:#ffb3a7}
#sevLevel.excluded{color:#ffb3a7}
/* V8: presentation POI markers have a separate purple ring; star and severity fill remain unchanged. */
.pinIcon.isPoi{outline:3px solid #c58cff;outline-offset:3px;
 box-shadow:0 0 0 7px #9d54e83b,0 2px 12px #000a}
.poiKey{display:inline-block;width:12px;height:12px;border-radius:50%;
 border:3px solid #c58cff;vertical-align:-2px;margin-right:6px}
#pinPoi{background:#3b2858;border-color:#9365c7;color:#eedcff;font-weight:750}
#pinPoi:hover:not(:disabled){background:#573774}
#pinPoi.pinned{background:#36234f;border-color:#c58cff;color:#f8eaff}
#pinPoi:disabled{opacity:.42;cursor:not-allowed}
.poiEntry{border-color:#8057a9;background:#241a38}
.poiEntry:hover{background:#35214d}
.poiHeader{display:flex;align-items:center;justify-content:space-between;gap:5px}
#poiCount{font-size:11px;font-weight:750;color:#d0a7ff}

@media(max-width:1440px){.assessmentActions{gap:4px}
 .assessmentActions button{font-size:10px;padding:5px 6px}
 .legendBox{flex-basis:310px}
 .semanticLegend{column-gap:8px}}
@media(max-width:1050px){.assessmentActions{margin-left:auto}
 #reviewStatus{top:117px}}
@media(max-width:800px){.assessmentActions{flex:1 1 100%;justify-content:flex-end}
 .eventDrawer{position:fixed;top:7%;bottom:7%;right:10px;width:min(430px,calc(100% - 20px))}}

/* V15: distinct reference point (not a severity-coded cluster). */
.startMarker{width:38px;height:38px;display:grid;place-items:center;
  background:#126f98;border:3px solid #e1f7ff;outline:3px solid #ffd788;outline-offset:2px;
  border-radius:11px;color:white;box-shadow:0 3px 16px #000c;font-size:19px;font-weight:900}
.startLink{display:block;width:100%;text-align:left;border:1px solid #4b87a4;
  background:#12344a;color:#e2f6ff;padding:9px 10px;margin-top:7px;font-weight:750}
.startLink:hover{background:#19506e}
.startKey{display:inline-block;width:11px;height:11px;background:#ffd788;
  border:2px solid #126f98;border-radius:3px;margin-right:5px;vertical-align:-1px}
#showStart{border-color:#558da8;background:#153a4e;color:#e2f6ff}
.console.intro-mode{display:flex;flex-direction:column;gap:8px}
.console.intro-mode>.assessment,.console.intro-mode>.visualGrid,
.console.intro-mode>.controls,.console.intro-mode>.eventDrawer{display:none!important}
.introStage{display:none;flex:1;min-height:0;min-width:0;flex-direction:column}
.console.intro-mode .introStage{display:flex}
.introHead{display:flex;align-items:center;gap:12px;justify-content:space-between;
  flex-wrap:wrap;padding:12px 17px;border-bottom:1px solid var(--edge)}
.introTitle{font-size:21px;letter-spacing:-.4px;font-weight:850}
.introSubtitle{color:var(--muted);font-size:12px}
.introBack{border-color:#3c7898!important;background:#164058!important;
  color:#e3f6ff;font-weight:800;padding:8px 13px}
.introMedia{flex:1;min-height:0;min-width:0;display:flex;align-items:center;
  justify-content:center;background:#07131e;overflow:hidden;position:relative}
#startVideo{width:100%;height:100%;object-fit:contain;background:black;display:none}
.introPlaceholder{display:flex;flex-direction:column;align-items:center;text-align:center;
  justify-content:center;gap:12px;padding:24px;max-width:650px}
.introPlaceholderIcon{width:88px;height:88px;display:grid;place-items:center;
  border-radius:22px;border:2px dashed #4685a6;color:#78cbee;font-size:45px}
.introPlaceholder .introEmptyTitle{font-weight:800;font-size:23px}
.introPlaceholder .introEmptyInfo{color:#a8c1d3;font-size:13px;line-height:1.7}
.introPlaceholder code{color:#8addff;background:#18384e;border:1px solid #3a6c87;
  padding:3px 6px;border-radius:5px}
.introFooter{padding:11px 17px;color:var(--muted);font-size:11px;
  border-top:1px solid var(--edge)}
@media(max-width:1050px){#showStart{font-size:10px;padding:5px 6px}}
@media(max-width:800px){.introStage{min-height:65vh}.introMedia{min-height:40vh}}

/* V14: road name + suburb with explicit operator correction option. */
#addressText{display:block;margin-top:6px;color:#e6f3fb;font-size:14px;font-weight:750;
 line-height:1.3;overflow-wrap:anywhere}
#coords{display:block;margin-top:5px;font-size:11px;color:#93aec4;font-variant-numeric:tabular-nums}
#addressAttribution{color:#819ab0;font-size:10px;margin-top:3px}
#addressRefresh,#addressSetLocality{font-size:10px;padding:3px 6px;margin-top:7px;border-color:#365269}
#addressSetLocality{border-color:#4e6e8c;color:#cce9ff;margin-left:5px}
"""

COMMON_JS = r"""
'use strict';
const $ = id => document.getElementById(id);
const esc = v => String(v ?? '').replace(/[&<>"']/g, c =>
  ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const fmt = t => {let n=Math.max(0,Math.floor(Number(t)||0));
  return [Math.floor(n/3600),Math.floor(n/60)%60,n%60].map(x=>String(x).padStart(2,'0')).join(':');};
// Presentation-only terminology mapping; never modify source metadata or classes.
const className = name => String(name||'').replaceAll('_',' ')
  .replace(/\bhazards\b/gi,'events').replace(/\bhazard\b/gi,'event')
  .replace(/\bseverity\b/gi,'priority');
const starColour = n => ({1:'#24a777',2:'#379e9c',3:'#d2a63c',4:'#ef8d36',5:'#d54848'}[n]||'#66788c');
let DB=null; let S=null;let lastSelection=null;
let excludedIds=new Set();
let reviewRevision=-1;
let reviewRecords=[];
let pinnedIds=new Set();
let pinRevision=-1;
function pinned(id){return pinnedIds.has(Number(id));}
async function fetchPins(onChange){
  try{
    const info=await getJSON('api/pins');
    if(pinRevision!==info.revision){
      pinRevision=info.revision;
      pinnedIds=new Set((Array.isArray(info.pinned)?info.pinned:[]).map(x=>Number(x.id)));
      onChange(info);
    }
  }catch(e){const status=$('status');if(status)status.textContent='POI SYNC ERROR · '+e.message;}
}
async function editPins(action,id){
  const response=await fetch('api/pins',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action,cluster_id:id}),cache:'no-store'});
  const info=await response.json().catch(()=>({error:'Invalid POI response'}));
  if(!response.ok)throw new Error(info.error||('POI update HTTP '+response.status));
  return info;
}
async function fetchReview(onChange){
  try{
    const info=await getJSON('api/review');
    if(reviewRevision!==info.revision){
      reviewRevision=info.revision;
      reviewRecords=Array.isArray(info.excluded)?info.excluded:[];
      excludedIds=new Set(reviewRecords.map(x=>Number(x.id)));
      onChange(info);
    }
  }catch(e){const status=$('status');if(status)status.textContent='REVIEW SYNC ERROR · '+e.message;}
}
async function editReview(action,id,reason=''){
  const response=await fetch('api/review',{method:'POST',
    headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action,cluster_id:id,reason}),cache:'no-store'});
  const info=await response.json().catch(()=>({error:'Server returned an invalid response'}));
  if(!response.ok)throw new Error(info.error||('Review update HTTP '+response.status));
  return info;
}
function excluded(id){return excludedIds.has(Number(id));}

async function getJSON(url){const r=await fetch(url,{cache:'no-store'});
  if(!r.ok)throw new Error(url+' returned HTTP '+r.status);return await r.json();}
async function command(action,extra={}){
  try {const r=await fetch('api/control',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({action,...extra}),cache:'no-store'});
    if(!r.ok)throw new Error(await r.text()); S=await r.json();return S;
  }catch(e){$('status').textContent='Connection error: '+e.message;throw e;}}
async function pollState(onState){try{const state=await getJSON('api/state'); S=state;
  $('status').textContent='LINKED · SHARED CLOCK';onState(state);
}catch(e){$('status').textContent='DISCONNECTED · '+e.message;}}
function bannerError(error){const box=document.createElement('div');box.className='err';
  box.style.cssText='padding:16px;background:#3d222b;position:fixed;top:70px;left:20px;z-index:5000';
  box.textContent='Dashboard error: '+error.message;document.body.prepend(box);console.error(error);}
function lowerBound(arr,t){let lo=0,hi=arr.length;while(lo<hi){const mid=(lo+hi)>>1;
  if(arr[mid][0]<t)lo=mid+1;else hi=mid;}return lo;}
function gpsAt(t){const route=DB.route,i=lowerBound(route,t),maxGap=Number(DB.gps_max_gap_s)||4;
  if(!route.length)return null;
  if(i===0)return Math.abs(route[0][0]-t)<=maxGap?[route[0][1],route[0][2]]:null;
  if(i===route.length)return Math.abs(route[i-1][0]-t)<=maxGap?[route[i-1][1],route[i-1][2]]:null;
  const a=route[i-1],b=route[i],dt=b[0]-a[0];
  if(dt<=0||dt>maxGap)return null;const f=Math.max(0,Math.min(1,(t-a[0])/dt));
  return [a[1]+f*(b[1]-a[1]),a[2]+f*(b[2]-a[2])];}
function warningThreshold(){if(DB.cluster_threshold!=null && DB.segmentation_threshold!=null &&
  Math.abs(DB.cluster_threshold-DB.segmentation_threshold)>1e-6){
  const w=$('threshold');if(w){w.textContent='Threshold mismatch: frames '+DB.segmentation_threshold+
    ' versus clusters '+DB.cluster_threshold+'. Rebuild frames to make them comparable.';w.style.display='block';}}}
"""

MAP_JS = r"""
(async function(){
  DB=await getJSON('data.json');if(!window.L)throw new Error('Leaflet could not load: check access to unpkg.com');
  if(!DB.route?.length||!DB.frames?.length)throw new Error('Missing route/frames in data.json');
  $('count').textContent=DB.clusters.length.toLocaleString();
  $('routeTime').textContent=fmt(DB.duration);warningThreshold();
  const map=L.map('map',{preferCanvas:true,zoomControl:false,fadeAnimation:false,zoomAnimation:false,
    maxZoom:19,worldCopyJump:false});
  L.control.zoom({position:'topleft'}).addTo(map);
  const b=L.latLngBounds(DB.route.map(p=>[p[1],p[2]]));
  map.fitBounds(b,{padding:[55,55],maxZoom:16,animate:false});
  const urls={satellite:'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    proxy:'_satellite/{z}/{x}/{y}.jpg',
    streets:'https://tile.openstreetmap.org/{z}/{x}/{y}.png'};
  let basemap=null,layerGeneration=0;
  function base(which){
    const gen=++layerGeneration; if(basemap)map.removeLayer(basemap);
    let loads=0,errors=0;
    basemap=L.tileLayer(urls[which],{maxZoom:19,maxNativeZoom:which==='streets'?19:18,
      attribution:which==='streets'?'© OpenStreetMap contributors':'Imagery © Esri',
      tileSize:256,updateWhenIdle:false,keepBuffer:3,crossOrigin:false});
    basemap.on('tileload',()=>{if(gen!==layerGeneration)return;loads++;
      $('tileWarning').style.display='none';});
    basemap.on('tileerror',()=>{if(gen!==layerGeneration)return;errors++;
      if(errors>=6&&loads===0){
        if(which==='satellite'){ $('basemap').value='proxy';base('proxy'); }
        else if(which==='proxy'){$('basemap').value='streets';base('streets');}
        else { $('tileWarning').textContent='Map images unavailable. Check internet access from browser/HPC.';
          $('tileWarning').style.display='block'; }
      }});
    basemap.addTo(map);
  }
  base('satellite');
  $('basemap').onchange=e=>base(e.target.value);
  $('fit').onclick=()=>{map.fitBounds(b,{padding:[55,55],maxZoom:16,animate:false});$('follow').checked=false;};
  const routeGeo=DB.route.map(p=>[p[1],p[2]]);
  L.polyline(routeGeo,{color:'#d4e1ed',opacity:.53,weight:3}).addTo(map);
  const completed=L.polyline([],{color:'#25b5ee',weight:5,opacity:.95}).addTo(map);
  const van=L.marker(routeGeo[0],{zIndexOffset:1500,
    icon:L.divIcon({html:'<div class="vanIcon">◆</div>',iconSize:[33,33],iconAnchor:[16,16],className:''})
  }).addTo(map);

  // The start pin is a user-named reference point, independent of DBSCAN clusters.
  const startPoint={lat:-37.9093055556,lon:145.1346944444,name:'Outside Monash Innovation Labs'};
  const startingMarker=L.marker([startPoint.lat,startPoint.lon],{
    zIndexOffset:2300,icon:L.divIcon({html:'<div class="startMarker">▶</div>',
      iconSize:[44,44],iconAnchor:[22,22],className:''})
  }).addTo(map);
  startingMarker.bindTooltip(startPoint.name,{direction:'top',offset:[0,-24]});
  function showStartingInspector(){
    $('selected').textContent='STARTING LOCATION · REFERENCE POINT';
    $('score').textContent=startPoint.name;
    $('stars').textContent='';
    $('level').textContent='Dedicated location video';
    $('classes').textContent='Named reference point · not a detected event';
    $('observations').textContent='No 2D segmentation or 3D reconstruction for this view';
    $('inspectImageLink').style.display='none';
    $('inspectImage').removeAttribute('src');
    $('inspectLinks').replaceChildren();
  }
  async function openStartingLocation(){
    $('follow').checked=false;
    map.setView([startPoint.lat,startPoint.lon],17,{animate:false});
    showStartingInspector();
    await command('start');
  }
  startingMarker.on('click',()=>openStartingLocation().catch(e=>console.error(e)));
  $('showStart').onclick=()=>openStartingLocation().catch(e=>console.error(e));
  $('showStartRail').onclick=()=>openStartingLocation().catch(e=>console.error(e));
  let priorView='analysis';

  const markers=new Map(),byId=new Map();let shown=new Set();
  function pinIcon(c){
    // Only the outer outline changes; the original severity colour and numeric star label stay visible.
    const col=starColour(Number(c.stars));
    const mark=pinned(c.id)?' isPoi':'';
    return L.divIcon({html:'<div class="pinIcon'+mark+'" style="background:'+col+'">'+
      (c.stars||'·')+'</div>',iconSize:[29,29],iconAnchor:[14,14],className:''});
  }
  for(const c of DB.clusters){
    byId.set(Number(c.id),c);
    const pin=L.marker([c.lat,c.lon],{icon:pinIcon(c),zIndexOffset:0});
    pin.on('click',()=>selectCluster(c,true));markers.set(Number(c.id),pin);
  }
  const list=$('topHazards');
  const poiList=$('pinnedHazards');
  function passesSeverity(c){
    const cutoff=Number($('severityFilter').value);
    const stars=Number(c.stars);
    // Pinned demo points remain visible even below the filter cutoff, but never if excluded.
    return !excluded(c.id)&&(pinned(c.id)||cutoff===0||(Number.isInteger(stars)&&stars>=cutoff&&stars<=5));
  }
  function updatePriorityList(){
    list.replaceChildren();
    const top=DB.clusters.filter(passesSeverity)
      .sort((a,b)=>(Number(b.score) || -1)-(Number(a.score) || -1)).slice(0,9);
    if(!top.length){const note=document.createElement('div');note.className='hint';
      note.textContent='No events match this priority filter.';list.append(note);return;}
    for(const c of top){const btn=document.createElement('button');btn.className='topincident';
      btn.innerHTML='<span>Event <strong>#'+esc(String(c.id).padStart(4,'0'))+'</strong>'+
        '<br><span class="muted small">'+esc((c.classes||[]).map(className).join(', ')||'No label')+'</span></span>'+
        '<span style="white-space:nowrap;color:'+starColour(Number(c.stars))+'">'+
        esc(c.score==null?'Unrated':Number(c.score).toFixed(1)+'/100')+'</span>';
      btn.onclick=()=>selectCluster(c,true);list.append(btn);
    }
  }
  function updatePoiList(){
    poiList.replaceChildren();
    const chosen=DB.clusters.filter(c=>pinned(c.id)&&!excluded(c.id))
      .sort((a,b)=>Number(a.first_s)-Number(b.first_s));
    $('poiCount').textContent=String(chosen.length);
    if(!chosen.length){const note=document.createElement('div');note.className='hint';
      note.textContent='Pin an event from the playback dashboard to mark it for your demo.';
      poiList.append(note);return;}
    for(const c of chosen){
      const btn=document.createElement('button');btn.className='topincident poiEntry';
      const severity=c.stars==null?'Unrated':c.stars+'★';
      btn.innerHTML='<span><strong>◉ #'+esc(String(c.id).padStart(4,'0'))+'</strong>'+
        '<br><span class="muted small">'+esc((c.classes||[]).map(className).join(', ')||'Unlabelled')+
        '</span></span><span style="white-space:nowrap;color:#d9aeff">'+esc(severity)+'</span>';
      btn.onclick=()=>selectCluster(c,true);poiList.append(btn);
    }
  }
  updatePriorityList();
  updatePoiList();
  function selectCluster(c,send){
    if(!c)return;
    $('selected').textContent='EVENT #'+String(c.id).padStart(4,'0')+(pinned(c.id)?' · PINNED POI':'')+(excluded(c.id)?' · REMOVED':'');
    $('score').textContent=c.score==null?'Unrated':Number(c.score).toFixed(1)+' / 100';
    $('stars').textContent=c.stars?'★'.repeat(Math.min(5,Number(c.stars))):'Unrated';
    $('level').textContent=className(c.level||'unrated');
    $('classes').textContent=(c.classes||[]).map(className).join(' · ')||'Not specified';
    $('observations').textContent=(c.events||0)+' associated recorded frames';
    const im=$('inspectImage'),link=$('inspectImageLink');
    if(c.image){link.href=c.image;im.src=c.image;link.style.display='block';}
    else {link.style.display='none';im.removeAttribute('src');}
    const links=$('inspectLinks');links.replaceChildren();
    for(const [name,url] of [['Full RGB',c.rgb],['3D voxel inspection',c.voxel]]){
      if(!url)continue;const a=document.createElement('a');a.href=url;a.target='_blank';
      a.rel='noopener';a.textContent=name;links.appendChild(a);}
    if(send){lastSelection=Number(c.id);$('follow').checked=false;
      command('jump',{time:Number(c.first_s),selected:Number(c.id)}).catch(console.error);
      map.panTo([c.lat,c.lon],{animate:false});}
  }
  $('pins').onchange=()=>{lastPinTick=-1;renderPins(S?.time??0);};
  $('severityFilter').onchange=()=>{lastPinTick=-1;renderPins(S?.time??0);updatePriorityList();};
  let lastPinTick=-1,lastTrailTick=-1,lastCameraPan=0;
  function renderPins(t){
    const whole=$('pins').checked;
    for(const c of DB.clusters){const id=Number(c.id),pin=markers.get(id);
      const allow=passesSeverity(c)&&(pinned(c.id)||whole||t>=Number(c.first_s));
      if(allow&&!shown.has(id)){pin.addTo(map);shown.add(id);}
      if(!allow&&shown.has(id)){map.removeLayer(pin);shown.delete(id);}
    }
    $('visibleCount').textContent=shown.size.toLocaleString()+' / '+(DB.clusters.length-excludedIds.size).toLocaleString();
    lastPinTick=t;
  }
  // Strictly use named-road results; never treat suburb/postcode as a street.
  // API paths are relative for Monash's /proxy/<port>/ URLs.
  let addressBusy=false,lastAddressPoll=0,lastAddressAt=null,lastAddressValue='';
  let lastAddressRequestAt=null,lastAddressInfo='',lastAddressStatus='pending';
  let lastAddressLocality=null,lastAddressState=null;
  function distanceM(a,b){
    if(!a||!b)return Infinity;
    const r=Math.PI/180,p1=a[0]*r,p2=b[0]*r,dp=(b[0]-a[0])*r,dl=(b[1]-a[1])*r;
    const q=Math.sin(dp/2)**2+Math.cos(p1)*Math.cos(p2)*Math.sin(dl/2)**2;
    return 12742000*Math.asin(Math.min(1,Math.sqrt(q)));
  }
  function showStreet(value,source){
    $('addressText').textContent=value;
    $('addressAttribution').textContent=source||'Nearest named street and suburb · © OpenStreetMap contributors';
  }
  async function updateStreet(t,pos,force=false){
    const now=performance.now();
    const far=distanceM(pos,lastAddressRequestAt)>30;
    if(addressBusy||(!force&&(now-lastAddressPoll)<(far?1100:3200)))return;
    lastAddressPoll=now;lastAddressRequestAt=[...pos];addressBusy=true;
    try{
      const res=await getJSON('api/address?time='+encodeURIComponent(t.toFixed(3)));
      // Never replace the named start location with an in-flight reverse-geocode result.
      if(S?.view==='start')return;
      lastAddressStatus=res.status;
      if(res.status==='ready'&&typeof res.address==='string'){
        lastAddressAt=[res.lat,res.lon];lastAddressValue=res.address;
        lastAddressLocality=res.locality||null;lastAddressState=res.state_abbr||null;
        lastAddressInfo=(res.source==='OSRM'?
          ('Road-network match · '+Math.round(res.road_distance_m||0)+' m from mapped road'):
          'Approximate named street')+(res.locality?
          (res.locality_source==='manual'?' · suburb manually confirmed':''):
          ' · suburb unresolved — use Set suburb')+' · © OSM contributors';
      }
      const current=S?gpsAt(S.time):null;
      if(current&&Number.isFinite(res.lat)&&Number.isFinite(res.lon)&&
          distanceM(current,[res.lat,res.lon])<=30){
        if(res.status==='ready')showStreet(res.address,lastAddressInfo);
        else if(res.status==='pending'||res.status==='rate_limited')
          showStreet('Identifying nearby road…','Road-network lookup in progress');
        else showStreet('Street name unavailable at this point','GPS coordinate shown below');
      }
    }catch(error){
      const current=S?gpsAt(S.time):null;
      if(current&&distanceM(current,pos)<30){
        lastAddressStatus='unavailable';
        showStreet('Street lookup unavailable','GPS coordinate shown below');
      }
      console.warn('Road name lookup:',error.message);
    }finally{addressBusy=false;}
  }
  $('addressRefresh').onclick=()=>{
    const t=S?.time??0,pos=gpsAt(t);
    if(pos){lastAddressAt=null;lastAddressPoll=0;lastAddressRequestAt=null;lastAddressStatus='pending';
      showStreet('Identifying nearby road…','Road-network lookup in progress');
      void updateStreet(t,pos,true);}
  };
  // A human-verified suburb is a per-point override, never a citywide assumption.
  // The actual road name and distance continue to come from the road matcher.
  $('addressSetLocality').onclick=async()=>{
    const t=S?.time??0,pos=gpsAt(t);
    if(!pos){window.alert('No GPS coordinate for the current playback frame.');return;}
    const suburb=window.prompt('Suburb for this displayed GPS location (e.g. Clayton):',lastAddressLocality||'');
    if(suburb===null)return;
    const locality=suburb.trim();
    if(!locality){window.alert('Please enter a suburb.');return;}
    const state=window.prompt('Australian state abbreviation:',lastAddressState||'VIC');
    if(state===null)return;
    const state_abbr=state.trim().toUpperCase();
    const btn=$('addressSetLocality');btn.disabled=true;
    try{
      const r=await fetch('api/locality',{method:'POST',cache:'no-store',
        headers:{'Content-Type':'application/json'},
        body:JSON.stringify({time:t,locality,state_abbr})});
      const result=await r.json();
      if(!r.ok)throw new Error(result.error||('HTTP '+r.status));
      lastAddressAt=[result.lat,result.lon];lastAddressValue=result.address;
      lastAddressLocality=result.locality;lastAddressState=result.state_abbr;
      lastAddressInfo='Road-network match · suburb manually confirmed at this GPS point · © OSM contributors';
      showStreet(lastAddressValue,lastAddressInfo);
    }catch(e){window.alert('Could not save suburb: '+e.message);}
    finally{btn.disabled=false;}
  };
  let lastClock=-1;
  function onState(state){const t=state.time;
    $('elapsed').textContent=fmt(t)+' / '+fmt(DB.duration);
    if(state.view==='start'){
      $('mode').textContent='STARTING LOCATION · VIDEO';
      if(priorView!=='start'){
        $('follow').checked=false;
        map.setView([startPoint.lat,startPoint.lon],17,{animate:false});
        showStartingInspector();
      }
      $('coords').textContent=startPoint.lat.toFixed(6)+', '+startPoint.lon.toFixed(6);
      showStreet('Outside Monash Innovation Labs','User-specified reference point · not reverse-geocoded');
      $('addressRefresh').disabled=true;
      $('addressSetLocality').disabled=true;
      priorView='start';
      return;
    }
    if(priorView==='start'){
      $('selected').textContent='EVENT INSPECTOR';
      $('score').textContent='—';$('stars').textContent='—';
      $('level').textContent='Select an event';
      $('classes').textContent='—';$('observations').textContent='—';
      lastSelection=null;
    }
    priorView='analysis';
    $('addressRefresh').disabled=false;
    $('addressSetLocality').disabled=false;
    $('mode').textContent=state.playing?'PLAYBACK RUNNING':'PLAYBACK PAUSED';
    const pos=gpsAt(t);
    if(pos){van.setLatLng(pos);$('coords').textContent=pos.map(v=>v.toFixed(6)).join(', ');
      if(distanceM(pos,lastAddressAt)<=28){showStreet(lastAddressValue,lastAddressInfo);}
      else if(distanceM(pos,lastAddressRequestAt)<=28 && lastAddressStatus==='unavailable'){
        showStreet('Street name unavailable at this point','GPS coordinate shown below');
      }else {showStreet('Identifying nearby road…','Road-network lookup in progress');}
      void updateStreet(t,pos);
      if($('follow').checked&&performance.now()-lastCameraPan>1400){
        const safe=map.getBounds().pad(-.28);
        if(safe.isValid()&&!safe.contains(pos)){
          map.panTo(pos,{animate:false});lastCameraPan=performance.now();}}}
    else {$('coords').textContent='GPS unavailable at this timestamp';
      showStreet('GPS signal unavailable','Cannot identify a road without valid GPS');}
    if(lastPinTick<0||Math.abs(t-lastPinTick)>.55)renderPins(t);
    if(lastTrailTick<0||Math.abs(t-lastTrailTick)>.55||t<lastTrailTick){
      let end=lowerBound(DB.route,t);const pts=routeGeo.slice(0,end);
      if(pos)pts.push(pos);completed.setLatLngs(pts);lastTrailTick=t;
    }
    if(state.selected!=null&&Number(state.selected)!==lastSelection){
      lastSelection=Number(state.selected);selectCluster(byId.get(lastSelection),false);
    }
    lastClock=t;
  }
  map.whenReady(()=>{requestAnimationFrame(()=>map.invalidateSize({animate:false}));});
  window.addEventListener('resize',()=>map.invalidateSize({animate:false}));
  setTimeout(()=>map.invalidateSize({animate:false}),280);
  function onReviewChange(info){
    $('count').textContent=(DB.clusters.length-info.excluded_count).toLocaleString();
    lastPinTick=-1;renderPins(S?.time??0);updatePriorityList();updatePoiList();
    if(lastSelection!=null)selectCluster(byId.get(lastSelection),false);
  }
  function onPinsChange(info){
    // Recreate only map icons, never GPS coords or underlying severity values.
    for(const c of DB.clusters){
      const marker=markers.get(Number(c.id));
      marker.setIcon(pinIcon(c));
      marker.setZIndexOffset(pinned(c.id)?900:0);
    }
    lastPinTick=-1;renderPins(S?.time??0);updatePriorityList();updatePoiList();
    if(lastSelection!=null)selectCluster(byId.get(lastSelection),false);
  }
  await fetchReview(onReviewChange);
  await fetchPins(onPinsChange);
  await pollState(onState);
  setInterval(()=>pollState(onState),200);
  setInterval(()=>fetchReview(onReviewChange),900);
  setInterval(()=>fetchPins(onPinsChange),900);
})().catch(bannerError);
"""

CONSOLE_JS = r"""
(async function(){
  DB=await getJSON('data.json');
  if(!DB.frames?.length)throw new Error('No cached frames found in data.json');
  $('end').textContent=fmt(DB.duration);warningThreshold();
  // Composite is laid out [RGB, SEG] over [GT, PRED]. Skip GT without repacking.
  const panelCrops=[['rgb',0,0],['seg',1,0],['pred',1,1]];
  const panels=panelCrops.map(([id,x,y])=>({ctx:$(id).getContext('2d'),x,y}));
  const byId=new Map(DB.clusters.map(c=>[Number(c.id),c]));
  const cache=new Map();let lastRendered=-1,requestToken=0,lastEventId=null;
  function imageFor(index){const path=DB.frames[index].file;
    if(cache.has(path))return cache.get(path);
    const image=new Image();image.decoding='async';image.src=path;
    cache.set(path,image);if(cache.size>16)cache.delete(cache.keys().next().value);
    return image;}
  function renderImage(index){
    if(index<0||index>=DB.frames.length||index===lastRendered)return;
    const token=++requestToken,img=imageFor(index);
    function paint(){if(token!==requestToken)return;
      const w=Number(DB.panel_width)||640,h=Number(DB.panel_height)||360;
      for(const panel of panels){panel.ctx.clearRect(0,0,640,360);
        panel.ctx.drawImage(img,panel.x*w,panel.y*h,w,h,0,0,640,360);}
      lastRendered=index;}
    if(img.complete&&img.naturalWidth)paint();
    else img.addEventListener('load',paint,{once:true});
    img.addEventListener('error',()=>{$('status').textContent='MISSING FRAME · '+DB.frames[index].file;},{once:true});
    if(index+1<DB.frames.length)imageFor(index+1);
  }
  // Double-size any of the three panels without obscuring the playback controls.
  const visualGrid=$('visualGrid');
  const visualIds=['rgbPanel','segPanel','predPanel'];
  function setFocus(id){
    const already=visualGrid.classList.contains('focus-mode') &&
      document.getElementById(id)?.classList.contains('focused');
    visualGrid.classList.toggle('focus-mode',!already);
    for(const panelId of visualIds){
      const panel=$(panelId);panel.classList.toggle('focused',!already&&panelId===id);
      panel.querySelector('.focusBtn').textContent=(!already&&panelId===id)?'↙ Restore':'⛶ Expand';
    }
  }
  for(const id of visualIds){$(id).querySelector('.focusBtn').onclick=()=>setFocus(id);}
  const introStage=$('introStage'),introVideo=$('startVideo');
  let introShown=false, introLookupId=0;
  async function showIntroVideo(){
    const id=++introLookupId;
    try{
      const info=await getJSON('api/start-video');
      if(id!==introLookupId||!introShown)return;
      if(info.available){
        $('introPlaceholder').style.display='none';
        introVideo.style.display='block';
        if(introVideo.getAttribute('src')!==info.url){
          introVideo.setAttribute('src',info.url);introVideo.load();
        }
      }else{
        introVideo.pause();
        introVideo.style.display='none';
        $('introPlaceholder').style.display='flex';
      }
    }catch(error){console.warn('Starting video availability:',error.message);}
  }
  function setView(view){
    const isIntro=view==='start';
    if(introShown===isIntro)return;
    introShown=isIntro;
    $('eventDrawer').classList.remove('open');
    $('mainConsole').classList.toggle('intro-mode',isIntro);
    if(isIntro){
      visualGrid.classList.remove('focus-mode');
      visualIds.forEach(id=>{$(id).classList.remove('focused');
        $(id).querySelector('.focusBtn').textContent='⛶ Expand';});
      void showIntroVideo();
    }else{
      introLookupId++;
      introVideo.pause();introVideo.currentTime=0;
    }
  }
  $('showStartConsole').onclick=()=>command('start').then(onState);
  $('returnFeed').onclick=()=>command('analysis').then(onState);

  // The only review target is the hazard associated with the current frame.
  // This uses the ORIGINAL data.json even after exclusion, so restoration is
  // still possible while displaying a previously excluded cluster.
  const events=$('eventList');
  let editBusy=false;
  let poiBusy=false;
  let lastReviewedId=null;
  let reviewNoticeTimer=null;
  function currentCluster(){
    if(!S||S.view==='start'||!DB.frames.length)return null;
    const index=Math.max(0,Math.min(DB.frames.length-1,Number(S.frame)||0));
    const id=DB.frames[index]?.cluster;
    return id==null?null:(byId.get(Number(id))||null);
  }
  function announceReview(message){
    $('reviewStatus').textContent=message;
    if(reviewNoticeTimer)clearTimeout(reviewNoticeTimer);
    reviewNoticeTimer=setTimeout(()=>{$('reviewStatus').textContent='';},4500);
  }
  function updateReviewButtons(){
    const c=currentCluster();
    const gone=c?excluded(c.id):false;
    const action=$('reviewHazards');
    action.textContent=gone?'↶ Restore event':'✕ Remove event';
    action.classList.toggle('restoring',gone);
    action.disabled=editBusy||!c;
    action.title=c?('Event #'+String(c.id).padStart(4,'0')+
      (gone?' — restore to curated results':' — exclude from curated results')):
      'No event assigned to the displayed frame';
    const restore=$('restoreAll');
    restore.disabled=editBusy||excludedIds.size===0;
    restore.textContent=excludedIds.size?('↶ Restore all ('+excludedIds.size+')'):'↶ Restore all';
  }
  function updatePoiButton(){
    const c=currentCluster();
    const isPinned=c?pinned(c.id):false;
    const btn=$('pinPoi');
    btn.classList.toggle('pinned',isPinned);
    btn.textContent=isPinned?'◆ Unpin POI':'◇ Pin POI';
    // Removed clusters are not shown on the curated map; restore them before pinning.
    btn.disabled=poiBusy||!c||excluded(c.id);
    btn.title=!c?'No event assigned to this frame':excluded(c.id)?
      'Restore this excluded event before pinning it':
      ('Event #'+String(c.id).padStart(4,'0')+
       (isPinned?' — remove demo marker':' — highlight on the map for your demo'));
  }
  function onPinsChange(info){
    pinRevision=info.revision;
    pinnedIds=new Set((Array.isArray(info.pinned)?info.pinned:[]).map(x=>Number(x.id)));
    updatePoiButton();
  }
  $('pinPoi').onclick=async()=>{
    if(poiBusy||editBusy)return;
    const c=currentCluster();
    if(!c){announceReview('No event assigned to this frame. Browse events to choose one.');return;}
    if(excluded(c.id)){announceReview('Restore the event before pinning it.');return;}
    const wasPinned=pinned(c.id);
    poiBusy=true;updatePoiButton();
    try{
      // Target the currently displayed hazard, not an arbitrary lat/lon point.
      if(S?.playing){
        const i=Math.max(0,Math.min(DB.frames.length-1,Number(S.frame)||0));
        S=await command('seek',{time:Number(DB.frames[i].s)});onState(S);
      }
      const info=await editPins(wasPinned?'unpin':'pin',Number(c.id));
      onPinsChange(info);
      announceReview('Event #'+String(c.id).padStart(4,'0')+
        (wasPinned?' unpinned.':' pinned on the GPS map with a violet outline.'));
    }catch(e){announceReview('POI update FAILED: '+e.message);}
    finally{poiBusy=false;updatePoiButton();}
  };
  function renderEvents(){
    events.replaceChildren();
    // Keep removed incidents browseable, so the user can jump back and restore
    // the precise cluster without losing the original frames or associations.
    for(const c of DB.clusters){
      const row=document.createElement('div');row.className='eventRow';
      const btn=document.createElement('button');btn.className='event';
      btn.dataset.id=String(c.id);
      if(excluded(c.id))btn.classList.add('removed');
      const name=(c.classes||[]).map(className).join(', ')||'No label';
      btn.innerHTML='<span class="eventText"><span style="color:'+starColour(Number(c.stars))+'">●</span> '+
        '<strong>#'+esc(String(c.id).padStart(4,'0'))+'</strong> '+esc(name)+
        (excluded(c.id)?'<br><span class="removedMark">EXCLUDED</span>':'')+'</span>'+
        '<span class="muted eventTime">'+fmt(c.first_s)+'</span>';
      btn.onclick=()=>command('jump',{time:Number(c.first_s),selected:Number(c.id)})
        .then(onState).catch(e=>announceReview('Cannot jump to cluster: '+e.message));
      row.append(btn);events.append(row);
    }
    updateEventHighlight();
  }
  function updateEventHighlight(){
    const c=currentCluster();
    for(const button of events.querySelectorAll('button.event')){
      button.classList.toggle('active',!!c&&Number(button.dataset.id)===Number(c.id));
    }
  }
  $('showEvents').onclick=()=>{$('eventDrawer').classList.toggle('open');};
  $('closeEvents').onclick=()=>{$('eventDrawer').classList.remove('open');};
  $('reviewHazards').onclick=async()=>{
    if(editBusy)return;
    const c=currentCluster();
    if(!c){announceReview('No event assigned to this frame. Browse events to jump to one.');return;}
    const wasRemoved=excluded(c.id);
    const description='Event #'+String(c.id).padStart(4,'0');
    editBusy=true;
    updateReviewButtons();
    try{
      // Freeze the displayed event before editing; capture ID first so the
      // action cannot accidentally switch to another frame while playing.
      if(S?.playing){
        const originalFrameIndex=Math.max(0,Math.min(DB.frames.length-1,Number(S.frame)||0));
        S=await command('seek',{time:Number(DB.frames[originalFrameIndex].s)});
        onState(S);
      }
      const result=await editReview(wasRemoved?'restore':'exclude',Number(c.id),
        wasRemoved?'':'Manual review during playback');
      onReviewChange(result);
      announceReview(wasRemoved?(description+' restored.'):
        (description+' excluded from curated results. Click Restore event to undo.'));
    }catch(e){announceReview('Review update FAILED: '+e.message);}
    finally{editBusy=false;updateReviewButtons();}
  };
  $('restoreAll').onclick=async()=>{
    if(!excludedIds.size||editBusy)return;
    const count=excludedIds.size;
    if(!window.confirm('Restore all '+count+' excluded events to the curated results?'))return;
    editBusy=true;
    updateReviewButtons();
    try{
      const result=await editReview('restore_all',null,'');
      onReviewChange(result);
      announceReview('Restored all '+count+' excluded events.');
    }catch(e){announceReview('Restore all FAILED: '+e.message);}
    finally{editBusy=false;updateReviewButtons();}
  };
  function onReviewChange(info){
    reviewRevision=info.revision;
    reviewRecords=Array.isArray(info.excluded)?info.excluded:[];
    excludedIds=new Set(reviewRecords.map(x=>Number(x.id)));
    renderEvents();
    lastEventId=NaN; // ensure severity strip updates on exclusion/restoration
    if(S)onState(S);
    updateReviewButtons();
    updatePoiButton();
  }
  renderEvents();
  updateReviewButtons();
  updatePoiButton();
  function severity(c){
    const gone=c?excluded(c.id):false;
    $('currentHazard').textContent=c?('EVENT #'+String(c.id).padStart(4,'0')+
      (gone?' · EXCLUDED':'')):'NO ACTIVE EVENT';
    $('currentHazard').classList.toggle('excluded',gone);
    $('sevScore').textContent=c?(c.score==null?'Unrated':Number(c.score).toFixed(1)+' / 100'):'—';
    $('sevStars').textContent=c?.stars?'★'.repeat(Math.min(5,Number(c.stars))):'—';
    $('sevLevel').textContent=c?(gone?'EXCLUDED FROM CURATED RESULTS':
      className(c.level||'unrated')):'No event detected in current frame';
    $('sevLevel').classList.toggle('excluded',gone);
    $('sevClasses').textContent=c?(c.classes||[]).map(className).join(' · '):'—';
    $('sevEvents').textContent=c?(c.events||0)+' recorded observations':'—';
  }
  function onState(state){
    setView(state.view||'analysis');
    if(state.view==='start')return;
    const i=Math.max(0,Math.min(DB.frames.length-1,state.frame));
    const frame=DB.frames[i];
    renderImage(i);
    $('play').textContent=state.playing?'❚❚ Pause':'▶ Play';
    $('speed').value=String(state.speed);
    if(!dragging)$('seek').value=String(DB.duration?Math.round(1000*state.time/DB.duration):0);
    $('elapsed').textContent=fmt(state.time);
    $('frameInfo').textContent='Frame '+(i+1).toLocaleString()+' / '+DB.frames.length.toLocaleString()+' · '+frame.cam+' · Sample '+frame.id;
    const c=frame.cluster==null?null:byId.get(Number(frame.cluster));
    const severityKey=c?(Number(c.id)+':'+(excluded(c.id)?'excluded':'active')):'none';
    if(severityKey!==lastEventId){
      severity(c);lastEventId=severityKey;
      updateEventHighlight();
    }
    updateReviewButtons();
    updatePoiButton();
  }
  let dragging=false;
  $('play').onclick=()=>command('toggle').then(onState);
  $('speed').onchange=()=>command('speed',{speed:Number($('speed').value)}).then(onState);
  $('seek').addEventListener('pointerdown',()=>dragging=true);
  $('seek').addEventListener('pointerup',()=>dragging=false);
  $('seek').oninput=()=>{$('elapsed').textContent=fmt(DB.duration*Number($('seek').value)/1000);};
  $('seek').onchange=()=>{dragging=false;command('seek',{
    time:DB.duration*Number($('seek').value)/1000}).then(onState);};
  function seekHazard(direction){
    const t=S?.time??0;
    const active=DB.clusters.filter(c=>!excluded(c.id));
    const next=direction>0?active.find(c=>c.first_s>t+.5):
      [...active].reverse().find(c=>c.first_s<t-.5);
    if(next)command('jump',{time:Number(next.first_s),selected:Number(next.id)}).then(onState);
  }
  $('prev').onclick=()=>seekHazard(-1);$('next').onclick=()=>seekHazard(1);
  document.addEventListener('keydown',e=>{
    if(['INPUT','SELECT','BUTTON'].includes(document.activeElement?.tagName))return;
    if(e.code==='Space'){e.preventDefault();$('play').click();}
    if(e.code==='ArrowRight')seekHazard(1);
    if(e.code==='ArrowLeft')seekHazard(-1);
    if(['Digit1','Digit2','Digit3'].includes(e.code))setFocus(visualIds[Number(e.code.slice(-1))-1]);
    if(e.code==='Escape'){$('eventDrawer').classList.remove('open');visualGrid.classList.remove('focus-mode');
      visualIds.forEach(id=>{$(id).classList.remove('focused');$(id).querySelector('.focusBtn').textContent='⛶ Expand';});}
  });
  await fetchReview(onReviewChange);
  await fetchPins(onPinsChange);
  await pollState(onState);setInterval(()=>pollState(onState),170);
  setInterval(()=>fetchReview(onReviewChange),900);
  setInterval(()=>fetchPins(onPinsChange),900);
  setInterval(()=>{if(introShown&&!introVideo.getAttribute('src'))void showIntroVideo();},5000);
})().catch(bannerError);
"""

MAP_BODY = r"""
<header class="h"><div>
<div class="brand">AI-Based Asset Management and Assessment</div>
<div class="sub">Georeferenced Inspection Map <span style="color:var(--accent)">/ 01</span></div></div>
<div class="topright"><span class="tag secondary" id="mode">PLAYBACK PAUSED</span>
<span class="tag"><span class="dot"></span><span id="status">CONNECTING</span></span></div></header>
<main class="mapstage">
<section class="panel mapframe"><div class="panelhead">
<div>VEHICLE TRACKING <span class="meta">· route and recorded inspection events</span></div>
<div class="mapactions"><select id="basemap" aria-label="Base map">
<option value="satellite">Satellite · Esri</option><option value="proxy">Satellite · server relay</option>
<option value="streets">Streets · OSM</option></select>
<label class="small muted"><input type="checkbox" id="follow"> Follow vehicle</label>
<label class="filterLabel" for="severityFilter">Event priority <select id="severityFilter" aria-label="Minimum event priority"><option value="0">All priorities</option><option value="2">2★ and above</option><option value="3">3★ and above</option><option value="4" selected>4★ and above</option><option value="5">5★ only</option></select></label><label class="small muted"><input type="checkbox" id="pins" checked> All times</label>
<button id="showStart" type="button">▶ Starting point</button><button id="fit">Fit route</button></div></div>
<div style="position:relative;flex:1;min-height:0"><div id="map"></div>
<div id="tileWarning"></div></div>
<div class="mapfooter"><div class="legend"><span><i class="sq" style="background:#25b5ee"></i>Traversed route</span>
<span><i class="sq" style="background:#d4e1ed"></i>Full recorded route</span>
<span><i class="sq" style="background:#d54848"></i>High priority (5★)</span>
<span><i class="sq" style="background:#24a777"></i>Low priority (1★)</span></div>
  <span class="hint"><i class="startKey"></i>Starting point &nbsp; <i class="poiKey"></i>POI violet outline · original priority</span></div></section>
<aside class="maprail">
<section class="panel sc"><div class="eyebrow">MISSION SUMMARY</div>
<div class="row" style="margin-top:12px;justify-content:space-between">
<div><div class="kicker">Identified events</div><div class="value" id="count">—</div><div class="hint"><span id="visibleCount">—</span> matching filter</div></div>
<div><div class="kicker">Route duration</div><div class="value" id="routeTime">—</div></div></div>
<div class="routeStat"><div class="kicker">Synced time</div><strong id="elapsed">—</strong></div>
<div class="routeStat"><div class="kicker">Vehicle street location (approx.)</div>
<strong id="addressText">Identifying nearby road…</strong>
<span id="coords">—</span>
<div id="addressAttribution">Road-network match · © OpenStreetMap contributors</div>
<button type="button" id="addressRefresh" title="Retry lookup for the current GPS point">↻ Check street</button><button type="button" id="addressSetLocality" title="Save a verified suburb for this specific GPS location">✎ Set suburb</button></div>
<div id="threshold" class="err small" style="display:none;margin-top:9px"></div></section>
<section class="panel railscroll"><div class="eyebrow">STARTING LOCATION</div>
<div style="margin-top:7px;font-weight:800">Outside Monash Innovation Labs</div>
<div class="hint">37°54′33.5″S · 145°08′04.9″E</div>
<button class="startLink" id="showStartRail" type="button">▶ Open starting-point video</button>
<hr style="border:0;border-top:1px solid var(--edge);margin:17px 0 11px">
<div class="eyebrow" id="selected">EVENT INSPECTOR</div>
<div style="font-size:10px;color:var(--muted);margin-top:6px">Select a map marker or listed event to inspect.</div>
<div class="value" style="margin-top:12px" id="score">—</div>
<div class="stars" id="stars">—</div>
<div style="margin-top:7px" id="level">Select an event</div>
<div class="muted small" style="margin-top:5px" id="classes">—</div>
<div class="muted small" style="margin-top:7px" id="observations">—</div>
<a id="inspectImageLink" target="_blank" rel="noopener" title="Open complete representative image">
<img id="inspectImage" alt="Representative face-blurred segmentation image"></a>
<div id="inspectLinks"></div>
  <hr style="border:0;border-top:1px solid var(--edge);margin:17px 0 11px">
  <div class="poiHeader"><div class="eyebrow">PINNED POINTS OF INTEREST</div>
    <span id="poiCount">0</span></div>
  <div class="hint" style="margin:6px 0 9px">Pinned demo events · click to jump both monitors</div>
  <div id="pinnedHazards"></div>
  <hr style="border:0;border-top:1px solid var(--edge);margin:17px 0 11px">
  <div class="eyebrow">PRIORITY LOCATIONS</div><div class="small muted" style="margin:6px 0 10px">
Highest scored clusters · click to jump both screens</div>
<div id="topHazards"></div></section>
<section class="panel sc"><div class="kicker">DATA INTERPRETATION</div>
<div class="small muted" style="margin-top:6px">Recorded, time-synchronized visualisation.
Map markers show vehicle observation locations, not surveyed event positions.
Event priority uses the cached cluster score and star rating.</div></section>
</aside></main>
"""

CONSOLE_BODY = r"""
<header class="h"><div>
<div class="brand">AI-Based Asset Management and Assessment</div>
<div class="sub">Interactable Dashboard <span style="color:var(--accent)">/ 02</span></div></div>
<div class="topright"><span class="tag secondary" id="camLabel">RECORDED REPLAY</span>
<span class="tag"><span class="dot"></span><span id="status">CONNECTING</span></span></div></header>
<main class="console" id="mainConsole">
<section class="panel assessment" aria-label="Event priority and semantic legend">
  <div class="severityBrief">
    <div class="sevTitle"><div class="eyebrow">EVENT PRIORITY</div>
      <div class="hint" id="currentHazard">NO ACTIVE EVENT</div></div>
    <div class="severityValues"><strong id="sevScore">—</strong>
      <span class="stars" id="sevStars">—</span>
      <div class="sevMeta"><div id="sevLevel">No event at current frame</div>
        <div id="sevClasses">—</div><div id="sevEvents">—</div></div>
    </div>
  </div>
  <div class="legendBox" aria-label="Semantic classes">
    <div class="kicker">SEMANTIC CLASSES</div>
    <div class="semanticLegend">
      <span><i class="sq" style="background:#ed4e41"></i>Loose trash</span>
      <span><i class="sq" style="background:#ffa52b"></i>Mud spill</span>
      <span><i class="sq" style="background:#4aa63a"></i>Public greenway</span>
      <span><i class="sq" style="background:#50b2e4"></i>Sidewalk</span>
    </div>
  </div>
  <div class="assessmentActions">
    <button id="showEvents" type="button">☷ Browse events</button>
    <button id="pinPoi" type="button" disabled title="No event at current frame">◇ Pin POI</button>
    <button id="reviewHazards" type="button" disabled title="No event at current frame">✕ Remove event</button>
    <button id="restoreAll" type="button" disabled>↶ Restore all</button>
  </div>
  <div id="reviewStatus" role="status" aria-live="polite"></div>
</section>

<section class="panel introStage" id="introStage" aria-label="Starting location video">
  <div class="introHead">
    <div><div class="eyebrow">STARTING LOCATION · VIDEO VIEW</div>
      <div class="introTitle">Outside Monash Innovation Labs</div>
      <div class="introSubtitle">Reference location · 37°54′33.5″S, 145°08′04.9″E</div></div>
    <button id="returnFeed" class="introBack" type="button">↶ Return to Dashboard</button>
  </div>
  <div class="introMedia">
    <div id="introPlaceholder" class="introPlaceholder">
      <div class="introPlaceholderIcon">▶</div>
      <div class="introEmptyTitle">Starting-point video placeholder</div>
      <div class="introEmptyInfo">The dedicated location footage will appear here when provided.<br>
        Place your video in the existing dashboard folder as <code>starting_point.mp4</code>.<br>
        This view intentionally contains no segmentation or 3D reconstruction.</div>
    </div>
    <video id="startVideo" preload="metadata" playsinline controls aria-label="Outside Monash Innovation Labs video"></video>
  </div>
  <div class="introFooter">Named presentation reference point · independent of the recorded inspection timeline and event priority analysis</div>
</section>
<section class="visualGrid" id="visualGrid">
  <section class="panel visual" id="rgbPanel"><div class="panelhead">
    01 · RGB CAMERA <span class="meta">Face blurred</span>
    <button class="focusBtn" type="button">⛶ Expand</button></div>
    <div class="canvasWrap"><canvas id="rgb" width="640" height="360"></canvas></div></section>
  <section class="panel visual" id="segPanel"><div class="panelhead">
    02 · RF-DETR SEGMENTATION <span class="meta">2D semantic prior</span>
    <button class="focusBtn" type="button">⛶ Expand</button></div>
    <div class="canvasWrap"><canvas id="seg" width="640" height="360"></canvas></div></section>
  <section class="panel visual" id="predPanel"><div class="panelhead">
    03 · 3D CONDITION MODEL <span class="meta">Predicted semantic voxels</span>
    <button class="focusBtn" type="button">⛶ Expand</button></div>
    <div class="canvasWrap"><canvas id="pred" width="640" height="360"></canvas></div></section>
</section>
<aside class="panel eventDrawer" id="eventDrawer" aria-label="Event index">
  <div class="panelhead">EVENT INDEX
    <button id="closeEvents" type="button" aria-label="Close event panel">✕</button></div>
  <div class="drawerIntro hint">Choose a cluster to jump both monitors to its first observation.
    Excluded clusters remain browseable and can be restored at the displayed frame.</div>
  <div id="eventList"></div>
</aside>
<section class="panel controls" aria-label="Playback controls">
  <div class="controltop">
    <div class="row">
      <button class="primary" id="play">▶ Play</button>
      <button id="showStartConsole" type="button">▶ Starting point</button>
      <button id="prev">◀ Previous event</button>
      <button id="next">Next event ▶</button>
      <label class="muted">Speed <select id="speed">
        <option value="0.5">0.5×</option><option value="1">1×</option>
        <option value="2">2×</option><option value="5">5×</option>
        <option value="10">10×</option></select></label>
      <span class="meta" id="frameInfo">—</span>
    </div>
    <span class="badge shortcut">SPACE · PLAY / PAUSE</span>
  </div>
  <div class="scrubRow">
    <span class="meta" id="elapsed">00:00:00</span>
    <input id="seek" aria-label="Timeline" type="range" min="0" max="1000" step="1" value="0">
    <span class="meta" id="end">00:00:00</span>
  </div>
  <div id="threshold" class="err small" style="display:none"></div>
</section></main>
"""


def make_page(title: str, body: str, script: str, leaflet: bool = False) -> bytes:
    head = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        f'<title>{title} | AI-Based Asset Management and Assessment</title>'
    )
    if leaflet:
        head += '<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">'
    head += '<style>' + STYLE + '</style></head><body>'
    tail = '<script>' + COMMON_JS + '</script>'
    if leaflet:
        tail += '<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>'
    tail += '<script>' + script + '</script></body></html>'
    return (head + body + tail).encode('utf-8')


class PlaybackClock:
    def __init__(self, duration: float, frame_times: list[float]):
        self.lock = threading.Lock()
        self.duration = duration
        self.frame_times = frame_times
        self.seconds = 0.0
        self.origin = time.monotonic()
        self.playing = False
        self.speed = 1.0
        self.selected = None
        self.view = 'analysis'

    def _time_locked(self):
        sec = self.seconds
        if self.playing:
            sec += max(0.0, time.monotonic() - self.origin) * self.speed
            if sec >= self.duration:
                sec = self.duration
                self.seconds = sec
                self.playing = False
        return max(0.0, min(self.duration, sec))

    def _snapshot_locked(self):
        seconds = self._time_locked()
        idx = max(0, bisect.bisect_right(self.frame_times, seconds) - 1)
        return {'time': round(seconds, 4), 'playing': self.playing,
                'speed': self.speed, 'frame': idx, 'selected': self.selected,
                'duration': self.duration, 'view': self.view}

    def snapshot(self):
        with self.lock:
            return self._snapshot_locked()

    def control(self, request: dict):
        action = request.get('action')
        if action not in {'toggle', 'play', 'pause', 'seek', 'jump', 'speed', 'reset', 'start', 'analysis'}:
            raise ValueError('Unsupported playback action')
        with self.lock:
            now = self._time_locked()
            self.seconds = now
            self.origin = time.monotonic()
            if action == 'start':
                self.seconds = 0.0
                self.playing = False
                self.selected = None
                self.view = 'start'
            elif action == 'analysis':
                self.playing = False
                self.view = 'analysis'
                self.selected = None
            elif action == 'toggle':
                self.view = 'analysis'
                self.playing = not self.playing
            elif action == 'play':
                self.view = 'analysis'
                if self.seconds >= self.duration:
                    self.seconds = 0
                self.playing = True
            elif action == 'pause':
                self.playing = False
            elif action in {'seek', 'jump'}:
                self.view = 'analysis'
                t = float(request['time'])
                if not math.isfinite(t):
                    raise ValueError('Non-finite seek time')
                self.seconds = max(0.0, min(self.duration, t))
                self.playing = False
                self.selected = None
                if action == 'jump' and request.get('selected') is not None:
                    self.selected = int(request['selected'])
            elif action == 'speed':
                value = float(request['speed'])
                if value not in {0.5, 1.0, 2.0, 5.0, 10.0}:
                    raise ValueError('Unsupported speed')
                self.speed = value
            elif action == 'reset':
                self.view = 'analysis'
                self.seconds = 0.0
                self.playing = False
                self.selected = None
            if self.playing and self.seconds >= self.duration:
                self.seconds = 0
                self.origin = time.monotonic()
            return self._snapshot_locked()


def atomic_write(path: Path, content: bytes):
    """Atomic replace in the same directory; never modify source metadata."""
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode='wb', dir=path.parent, prefix='.' + path.name + '.',
            suffix='.tmp', delete=False
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class HazardReview:
    """Persistent, auditable, non-destructive exclusion list for this dataset."""

    def __init__(self, root: Path, data: dict):
        self.root = root
        self.raw = data
        self.lock = threading.RLock()
        self.ids = {int(c['id']) for c in data.get('clusters', [])}
        self.path = root / 'hazard_review.json'
        self.export_path = root / 'data_curated.json'
        self.csv_path = root / 'hazards_curated.csv'
        # Bind review records to the SAME clustering run. IDs can be reused.
        identity = sorted((int(c['id']), round(float(c['lat']), 7),
                           round(float(c['lon']), 7), round(float(c['first_s']), 3))
                          for c in data.get('clusters', []))
        self.fingerprint = hashlib.sha256(json.dumps(identity, separators=(',', ':')).encode()).hexdigest()
        self.records = {}
        self.history = []
        self.revision = 0
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            if saved.get('dataset_fingerprint') != self.fingerprint:
                raise RuntimeError(
                    f'Review file {self.path} belongs to another clustering run. '
                    'Back it up or rename it, then restart; do not silently reuse old cluster IDs.')
            self.records = {str(int(k)): v for k, v in saved.get('excluded', {}).items()}
            if not all(int(k) in self.ids for k in self.records):
                raise RuntimeError(f'Review file {self.path} contains invalid cluster IDs')
            self.history = saved.get('history', [])
            self.revision = int(saved.get('revision', 0))
        # Always regenerate lightweight derived metadata on startup. Cached
        # frames remain untouched, while newer source metadata is included.
        self._write_exports(self.records)

    def _review_document(self, records, history, revision):
        return {'schema_version': 1, 'dataset_fingerprint': self.fingerprint,
                'source': 'data.json', 'revision': revision,
                'excluded': records, 'history': history}

    def _curated_document(self, records):
        excluded = {int(key) for key in records}
        result = dict(self.raw)
        result['clusters'] = [dict(c) for c in self.raw['clusters'] if int(c['id']) not in excluded]
        result['frames'] = [{**fr, 'cluster': None if fr.get('cluster') is not None and
                             int(fr['cluster']) in excluded else fr.get('cluster')}
                            for fr in self.raw['frames']]
        result['review'] = {'curated': True, 'source': 'data.json',
                            'excluded_clusters': sorted(excluded),
                            'included_clusters': len(result['clusters']),
                            'original_clusters': len(self.raw['clusters']),
                            'dataset_fingerprint': self.fingerprint,
                            'note': 'Manual exclusions only; not ground truth or model evaluation.'}
        return result

    def _csv_bytes(self, curated):
        s = io.StringIO()
        writer = csv.writer(s)
        writer.writerow(['cluster_id','latitude_observation','longitude_observation',
                         'severity_score','severity_stars','severity_level',
                         'classes','event_frames','first_seconds','last_seconds'])
        for c in curated['clusters']:
            writer.writerow([c.get('id'), c.get('lat'), c.get('lon'), c.get('score'),
                             c.get('stars'), c.get('level'),
                             '; '.join(map(str,c.get('classes') or [])),
                             c.get('events'), c.get('first_s'), c.get('last_s')])
        return s.getvalue().encode('utf-8')

    def _write_exports(self, records):
        curated = self._curated_document(records)
        atomic_write(self.export_path, json.dumps(curated, indent=2, allow_nan=False).encode('utf-8'))
        atomic_write(self.csv_path, self._csv_bytes(curated))

    def snapshot(self):
        with self.lock:
            return {'revision': self.revision,
                    'excluded_count': len(self.records),
                    'active_count': len(self.ids) - len(self.records),
                    'excluded': [{'id': int(k), **v} for k, v in sorted(
                        self.records.items(), key=lambda x: int(x[0]))]}

    def curated_bytes(self):
        with self.lock:
            return json.dumps(self._curated_document(self.records), allow_nan=False).encode('utf-8')

    def edit(self, request: dict):
        action = request.get('action')
        if action not in {'exclude','restore','restore_all'}:
            raise ValueError('Unsupported review action')
        with self.lock:
            records = dict(self.records)
            cluster_id = request.get('cluster_id')
            if action != 'restore_all':
                if type(cluster_id) is not int or cluster_id not in self.ids:
                    raise ValueError('Unknown cluster ID')
                key = str(cluster_id)
            reason = request.get('reason', '')
            if action == 'exclude':
                if type(reason) is not str or not 1 <= len(reason.strip()) <= 140:
                    raise ValueError('Provide a reason (1–140 characters)')
                if key in records:
                    return self.snapshot()
                records[key] = {'reason': reason.strip(),
                                'removed_at': datetime.now(timezone.utc).isoformat()}
            elif action == 'restore':
                if key not in records:
                    return self.snapshot()
                del records[key]
            else:
                if not records:
                    return self.snapshot()
                records.clear()
            timestamp = datetime.now(timezone.utc).isoformat()
            history = self.history + [{'action': action, 'id': cluster_id if action!='restore_all' else None,
                                       'reason': reason if action=='exclude' else '', 'at': timestamp}]
            revision = self.revision + 1
            # Derived outputs first; persistent review state last.
            # A failed write leaves the original inputs and current review unchanged.
            self._write_exports(records)
            document = self._review_document(records, history, revision)
            try:
                atomic_write(self.path, json.dumps(document, indent=2, allow_nan=False).encode('utf-8'))
            except OSError:
                # Roll back derived output if saving the audit record failed.
                try:
                    self._write_exports(self.records)
                except OSError:
                    pass
                raise
            self.records, self.history, self.revision = records, history, revision
            return self.snapshot()


class HazardPins:
    """Saved demonstration waypoints refer to existing cluster IDs, never arbitrary GPS positions.

    Keep these separate from manual exclusion so presentation markers don't affect
    the research results, severity ratings or curated CSV.
    """
    def __init__(self, root: Path, data: dict, dataset_fingerprint: str):
        self.path = root / 'hazard_pins.json'
        self.ids = {int(c['id']) for c in data.get('clusters', [])}
        self.fingerprint = dataset_fingerprint
        self.lock = threading.RLock()
        self.records = {}
        self.revision = 0
        if self.path.exists():
            saved = json.loads(self.path.read_text(encoding='utf-8'))
            if saved.get('dataset_fingerprint') != self.fingerprint:
                raise RuntimeError('POI file belongs to another clustering run: '+str(self.path)+
                                   '. Back it up before renaming/removing it.')
            raw = saved.get('pinned', {})
            if not isinstance(raw, dict):
                raise RuntimeError('Invalid POI file: pinned must be an object')
            self.records = {str(int(k)): dict(v) for k,v in raw.items()}
            if not all(int(k) in self.ids for k in self.records):
                raise RuntimeError('POI file contains unknown cluster IDs')
            self.revision = int(saved.get('revision', 0))

    def snapshot(self):
        with self.lock:
            return {'revision': self.revision, 'count': len(self.records),
                    'pinned': [{'id': int(k), **v} for k,v in
                               sorted(self.records.items(), key=lambda x:int(x[0]))]}

    def edit(self, request: dict):
        action = request.get('action')
        cluster_id = request.get('cluster_id')
        if action not in {'pin','unpin'}:
            raise ValueError('Unsupported POI action')
        if type(cluster_id) is not int or cluster_id not in self.ids:
            raise ValueError('Unknown cluster ID')
        key = str(cluster_id)
        with self.lock:
            records = dict(self.records)
            if action == 'pin':
                if key in records:
                    return self.snapshot()
                records[key] = {'pinned_at': datetime.now(timezone.utc).isoformat()}
            else:
                if key not in records:
                    return self.snapshot()
                del records[key]
            revision = self.revision + 1
            document = {'schema_version': 1, 'dataset_fingerprint': self.fingerprint,
                        'source': 'data.json', 'revision': revision, 'pinned': records}
            atomic_write(self.path, json.dumps(document, indent=2, allow_nan=False).encode('utf-8'))
            self.records, self.revision = records, revision
            return self.snapshot()


class AddressResolver:
    """Nearest named road plus suburb/state from OpenStreetMap.

    OSRM supplies the nearest named drivable street. Nominatim supplies the
    actual locality and Australian state abbreviation. A street-only match
    stays honest when the locality service is unavailable; it never silently
    labels an entire drive as belonging to one suburb.

    Old V11 road-only cache entries are enriched in the background and saved.
    All work is rate-limited, nonblocking, and compatible with Monash's proxy.
    """

    MAX_ROAD_OFFSET_M = 70.0
    CACHE_RADIUS_M = 28.0
    MIN_REQUEST_INTERVAL_S = 12.0
    STATE_ABBR = {
        'victoria': 'VIC', 'new south wales': 'NSW', 'queensland': 'QLD',
        'western australia': 'WA', 'south australia': 'SA', 'tasmania': 'TAS',
        'australian capital territory': 'ACT', 'northern territory': 'NT'
    }

    def __init__(self, root: Path, route: list, duration: float,
                 max_gap_s: float = 4.0,
                 endpoint: str = 'https://nominatim.openstreetmap.org/reverse',
                 user_agent: str = 'PublicSpaceConditionAssessment/14 (Monash University research demo)',
                 road_endpoint: str = 'https://router.project-osrm.org/nearest/v1/driving'):
        self.route = [tuple(map(float, fix[:3])) for fix in route]
        self.times = [fix[0] for fix in self.route]
        self.duration = float(duration)
        self.max_gap_s = float(max_gap_s)
        self.endpoint = endpoint
        self.user_agent = user_agent
        self.road_endpoint = road_endpoint.rstrip('/')
        self.path = root / 'address_cache.json'
        self.lock = threading.Lock()
        self.entries = []
        self.inflight = False
        self.last_lookup_at = -1e12
        self.retry_after = 0.0
        if self.path.is_file():
            try:
                saved = json.loads(self.path.read_text(encoding='utf-8'))
                for entry in saved.get('entries', []):
                    lat, lon = float(entry['lat']), float(entry['lon'])
                    street = str(entry.get('address', ''))
                    # Reject the suburb-only result that older versions cached.
                    if not (street.startswith('Near ') and len(street) < 260):
                        continue
                    if -90 <= lat <= 90 and -180 <= lon <= 180:
                        self.entries.append({
                            'lat': lat, 'lon': lon, 'address': street,
                            'source': entry.get('source', 'Nominatim'),
                            'road_distance_m': entry.get('road_distance_m'),
                            'locality': self._clean_name(entry.get('locality')),
                            'state_abbr': self._clean_name(entry.get('state_abbr')),
                            'locality_source': entry.get('locality_source')
                        })
                print(f'Named-road cache: {len(self.entries)} location(s) loaded')
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
                print('WARNING: Street-name cache unavailable:', exc)

    @staticmethod
    def distance_m(a, b):
        p1, p2 = math.radians(a[0]), math.radians(b[0])
        dlat = p2 - p1
        dlon = math.radians(b[1] - a[1])
        h = math.sin(dlat / 2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dlon / 2)**2
        return 12_742_000 * math.asin(min(1.0, math.sqrt(h)))

    def position(self, seconds):
        if not math.isfinite(seconds) or seconds < 0 or seconds > self.duration:
            raise ValueError('Playback time out of range')
        i = bisect.bisect_left(self.times, seconds)
        if i == 0:
            p = self.route[0]
            return (p[1], p[2]) if abs(p[0] - seconds) <= self.max_gap_s else None
        if i == len(self.route):
            p = self.route[-1]
            return (p[1], p[2]) if abs(p[0] - seconds) <= self.max_gap_s else None
        a, b = self.route[i-1], self.route[i]
        dt = b[0] - a[0]
        if dt <= 0 or dt > self.max_gap_s:
            return None
        f = min(1.0, max(0.0, (seconds - a[0]) / dt))
        return (a[1] + f*(b[1]-a[1]), a[2] + f*(b[2]-a[2]))

    @staticmethod
    def _clean_name(value):
        if not isinstance(value, str):
            return None
        value = value.strip()
        return value if 2 <= len(value) <= 120 else None

    @classmethod
    def _road_name(cls, value):
        name = cls._clean_name(value)
        if name and name.casefold() not in {'unnamed road', 'unknown', 'none', 'unnamed', 'road'}:
            return name
        return None

    @classmethod
    def format_address(cls, result):
        """Fallback named road only; NEVER substitute a suburb as a road."""
        address = result.get('address') or {}
        road = next((cls._road_name(address.get(k)) for k in (
            'road', 'residential', 'living_street', 'pedestrian', 'service',
            'unclassified', 'tertiary', 'secondary', 'primary', 'highway',
            'footway', 'path', 'cycleway') if cls._road_name(address.get(k))), None)
        return 'Near ' + road if road else None

    @classmethod
    def _locality_from_result(cls, result):
        """Prefer the Australian suburb over the metro city or council area."""
        addr = result.get('address') or {}
        if not isinstance(addr, dict):
            return {'locality': None, 'state_abbr': None}
        locality = next((cls._clean_name(addr.get(k)) for k in (
            'suburb', 'quarter', 'neighbourhood', 'village', 'town',
            'city_district', 'hamlet', 'city') if cls._clean_name(addr.get(k))), None)
        state = cls._clean_name(addr.get('state'))
        code = cls._clean_name(addr.get('ISO3166-2-lvl4'))
        if code and code.upper().startswith('AU-'):
            state_abbr = code.upper()[3:]
        elif state:
            state_abbr = cls.STATE_ABBR.get(state.casefold(), state)
        else:
            state_abbr = None
        if state_abbr and not (2 <= len(state_abbr) <= 45):
            state_abbr = None
        return {'locality': locality, 'state_abbr': state_abbr}

    @staticmethod
    def _display_address(entry):
        """Display a road-matched location clearly, with its actual suburb/state.

        Keep the conservative road snap limit: if OSRM reports an offset of
        more than 25 m, 'On' is not justified, so show 'Near' instead.
        This is a street-level approximation, never a surveyed address.
        Older 'Near ...' cache entries are converted at display time.
        """
        name = re.sub(r'^(?:Near|On)\s+', '', str(entry['address']), flags=re.I).strip()
        if not name:
            return 'Street name unavailable'
        offset = entry.get('road_distance_m')
        source = entry.get('source', '')
        if source == 'OSRM':
            try:
                on_road = math.isfinite(float(offset)) and 0 <= float(offset) <= 25.0
            except (ValueError, TypeError):
                on_road = False
        else:
            # Nominatim's road field is only a reverse-geocoded estimate.
            # Maintain 'Near' when the road-network offset is unknown.
            on_road = False
        street = ('On ' if on_road else 'Near ') + name
        locality, state = entry.get('locality'), entry.get('state_abbr')
        if locality and state:
            return f'{street}, {locality} {state}'
        if locality:
            return f'{street}, {locality}'
        if state:
            return f'{street}, {state}'
        return street

    def _request_json(self, url):
        request = urllib.request.Request(url, headers={
            'User-Agent': self.user_agent, 'Accept': 'application/json'})
        with urllib.request.urlopen(request, timeout=6) as response:
            if 'json' not in response.headers.get('Content-Type', '').lower():
                raise ValueError('Map provider returned non-JSON response')
            raw = response.read(150_001)
            if len(raw) > 150_000:
                raise ValueError('Map response too large')
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError('Unexpected road service response type')
            return result

    def _nearest_driveable_street(self, latitude, longitude):
        url = f'{self.road_endpoint}/{longitude:.7f},{latitude:.7f}?number=5'
        data = self._request_json(url)
        if data.get('code') != 'Ok' or not isinstance(data.get('waypoints'), list):
            raise ValueError('Road snap returned no routeable roads')
        candidates = []
        for waypoint in data['waypoints']:
            if not isinstance(waypoint, dict):
                continue
            road = self._road_name(waypoint.get('name'))
            try:
                dist = float(waypoint.get('distance', float('inf')))
            except (ValueError, TypeError):
                continue
            if road and math.isfinite(dist) and 0 <= dist <= self.MAX_ROAD_OFFSET_M:
                candidates.append((dist, road))
        if not candidates:
            raise ValueError('No named drivable road within 70 metres')
        dist, road = min(candidates)
        return {'address': 'Near ' + road, 'source': 'OSRM',
                'road_distance_m': round(dist, 1)}

    def _reverse_result(self, latitude, longitude):
        query = urllib.parse.urlencode({
            'format': 'jsonv2', 'lat': f'{latitude:.7f}',
            'lon': f'{longitude:.7f}', 'zoom': 17,
            'addressdetails': 1, 'accept-language': 'en-AU'})
        url = self.endpoint + ('&' if '?' in self.endpoint else '?') + query
        return self._request_json(url)

    def _reverse_named_street(self, latitude, longitude):
        data = self._reverse_result(latitude, longitude)
        label = self.format_address(data)
        if not label:
            raise ValueError('Reverse geocoder returned suburb only')
        return {'address': label, 'source': 'Nominatim',
                'road_distance_m': None, **self._locality_from_result(data)}

    def _photon_locality(self, latitude, longitude):
        """Second independent OSM-based provider when Nominatim is blocked.

        Photon returns GeoJSON; its district is normally the suburb-level field.
        Do not replace a missing suburb with the entire metropolitan area.
        """
        url = ('https://photon.komoot.io/reverse?' + urllib.parse.urlencode({
            'lat': f'{latitude:.7f}', 'lon': f'{longitude:.7f}', 'limit': 5, 'lang': 'en'
        }))
        request = urllib.request.Request(url, headers={
            'User-Agent': self.user_agent, 'Accept': 'application/json'})
        with urllib.request.urlopen(request, timeout=7) as response:
            if 'json' not in response.headers.get('Content-Type', '').lower():
                raise ValueError('Photon returned non-JSON response')
            raw = response.read(180_001)
            if len(raw) > 180_000:
                raise ValueError('Photon response too large')
            data = json.loads(raw)
        if not isinstance(data, dict) or not isinstance(data.get('features'), list):
            raise ValueError('Photon response has no features')
        for feature in data['features']:
            if not isinstance(feature, dict):
                continue
            p = feature.get('properties') or {}
            if not isinstance(p, dict):
                continue
            country = str(p.get('countrycode') or '').upper()
            if country and country != 'AU':
                continue
            # Australian state names are typically expanded in Photon.
            state = self._clean_name(p.get('state'))
            state_abbr = self.STATE_ABBR.get((state or '').casefold())
            if not state_abbr:
                code = str(p.get('statecode') or '').upper()
                if code in self.STATE_ABBR.values():
                    state_abbr = code
            suburb = next((self._clean_name(p.get(key)) for key in (
                'district', 'suburb', 'neighbourhood', 'locality', 'quarter'
            ) if self._clean_name(p.get(key))), None)
            # A city's name is usable only for regional-town addresses, not
            # as a supposed Melbourne-suburb lookup.
            if suburb and (suburb.casefold().startswith(('city of ', 'shire of ', 'municipality of '))
                           or suburb.casefold() in {'city of monash', 'greater melbourne'}):
                suburb = None
            if not suburb:
                city = self._clean_name(p.get('city'))
                if city and city.casefold() not in {'melbourne', 'greater melbourne'}:
                    suburb = city
            if suburb and state_abbr:
                return {'locality': suburb, 'state_abbr': state_abbr,
                        'locality_source': 'Photon'}
        raise ValueError('Photon returned no suburb/state pair')

    def _lookup_locality(self, latitude, longitude):
        """Retrieve a meaningful locality using two independently hosted APIs."""
        try:
            result = self._locality_from_result(self._reverse_result(latitude, longitude))
            if (result.get('locality') and result.get('state_abbr') and
                    result['locality'].casefold() not in {'melbourne', 'greater melbourne'}):
                return {**result, 'locality_source': 'Nominatim'}
            raise ValueError('Nominatim returned no complete suburb/state pair')
        except (OSError, ValueError, TypeError, KeyError, TimeoutError, json.JSONDecodeError) as exc:
            print('Primary suburb lookup failed; trying Photon:', exc, flush=True)
        return self._photon_locality(latitude, longitude)

    def set_locality(self, request):
        """Operator correction for the *current GPS point*, not the entire route.

        It requires an existing verified road match within 28 m and persists
        to address_cache.json. The road name and on/near status are preserved.
        """
        seconds = float(request['time'])
        pos = self.position(seconds)
        if pos is None:
            raise ValueError('GPS unavailable at this playback time')
        locality = self._clean_name(request.get('locality'))
        state = str(request.get('state_abbr') or '').upper().strip()
        if not locality or len(locality) > 80 or not re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,79}", locality):
            raise ValueError('Invalid suburb name')
        if state not in self.STATE_ABBR.values():
            raise ValueError('State must be VIC, NSW, QLD, WA, SA, TAS, ACT or NT')
        with self.lock:
            nearest = min(self.entries, key=lambda e: self.distance_m(
                pos, (e['lat'], e['lon'])), default=None)
            if not nearest or self.distance_m(pos, (nearest['lat'], nearest['lon'])) > self.CACHE_RADIUS_M:
                raise ValueError('Street match not yet cached at this GPS position. Click Check street first.')
            previous = dict(nearest)
            nearest.update({'locality': locality, 'state_abbr': state,
                            'locality_source': 'manual'})
            try:
                self._save_cache_locked()
            except OSError:
                nearest.clear()
                nearest.update(previous)
                raise
            return {'status': 'ready', 'lat': pos[0], 'lon': pos[1],
                    'address': self._display_address(nearest),
                    'locality': locality, 'state_abbr': state,
                    'locality_source': 'manual',
                    'source': nearest.get('source', 'OSRM'),
                    'road_distance_m': nearest.get('road_distance_m')}

    def _save_cache_locked(self):
        atomic_write(self.path, json.dumps({
            'schema_version': 3,
            'source': 'OpenStreetMap road matching and locality enrichment',
            'entries': self.entries}, indent=2).encode('utf-8'))

    def _do_lookup(self, latitude, longitude):
        try:
            try:
                named = self._nearest_driveable_street(latitude, longitude)
            except (OSError, ValueError, TypeError, KeyError, TimeoutError, json.JSONDecodeError) as err:
                print('Road matching unavailable; trying named street reverse lookup:', err, flush=True)
                named = self._reverse_named_street(latitude, longitude)
            else:
                # OSRM has a road name but NO suburb. Enrich with the same GPS
                # point's administrative locality, not a citywide hardcoded value.
                try:
                    named.update(self._lookup_locality(latitude, longitude))
                except (OSError, ValueError, TypeError, KeyError, TimeoutError, json.JSONDecodeError) as err:
                    print('Locality lookup unavailable; keeping verified road:', err, flush=True)
            entry = {'lat': latitude, 'lon': longitude, **named}
            with self.lock:
                self.entries = [v for v in self.entries if self.distance_m(
                    (latitude, longitude), (v['lat'], v['lon'])) > 12]
                self.entries.append(entry)
                try:
                    self._save_cache_locked()
                except OSError as err:
                    print('WARNING: Could not persist street/locality cache:', err, flush=True)
        except (OSError, ValueError, TypeError, KeyError, TimeoutError, json.JSONDecodeError) as err:
            print('Named street unavailable; GPS fallback retained:', err, flush=True)
            with self.lock:
                self.retry_after = time.monotonic() + 16.0
        finally:
            with self.lock:
                self.inflight = False

    def _enrich_cached_locality(self, entry):
        """Upgrade cached V11 street labels without discarding/re-snapping them."""
        try:
            parts = self._lookup_locality(entry['lat'], entry['lon'])
            if not parts['locality']:
                raise ValueError('Reverse geocoder returned no named locality')
            with self.lock:
                for existing in self.entries:
                    if (existing['address'] == entry['address'] and
                            self.distance_m((entry['lat'], entry['lon']),
                                            (existing['lat'], existing['lon'])) <= 12 and
                            existing.get('locality_source') != 'manual'):
                        existing.update(parts)
                try:
                    self._save_cache_locked()
                except OSError as err:
                    print('WARNING: Could not persist locality cache:', err, flush=True)
        except (OSError, ValueError, TypeError, KeyError, TimeoutError, json.JSONDecodeError) as err:
            print('Locality enrichment unavailable; retaining street:', err, flush=True)
            with self.lock:
                self.retry_after = time.monotonic() + 60.0
        finally:
            with self.lock:
                self.inflight = False

    def resolve(self, seconds):
        pos = self.position(float(seconds))
        if pos is None:
            return {'status': 'gps_unavailable', 'address': None, 'lat': None, 'lon': None}
        lat, lon = pos
        with self.lock:
            nearest = min(self.entries, key=lambda p: self.distance_m(
                (lat, lon), (p['lat'], p['lon'])), default=None)
            now = time.monotonic()
            can_lookup = (not self.inflight and
                          now-self.last_lookup_at >= self.MIN_REQUEST_INTERVAL_S and
                          now >= self.retry_after)
            if nearest and self.distance_m((lat, lon), (nearest['lat'], nearest['lon'])) <= self.CACHE_RADIUS_M:
                if can_lookup and not nearest.get('locality'):
                    self.last_lookup_at = now
                    self.inflight = True
                    threading.Thread(target=self._enrich_cached_locality,
                                     args=(dict(nearest),), daemon=True).start()
                return {'status': 'ready', 'lat': lat, 'lon': lon,
                        'address': self._display_address(nearest),
                        'locality': nearest.get('locality'),
                        'state_abbr': nearest.get('state_abbr'),
                        'locality_source': nearest.get('locality_source'),
                        'source': nearest['source'],
                        'road_distance_m': nearest.get('road_distance_m')}
            if can_lookup:
                self.last_lookup_at = now
                self.inflight = True
                threading.Thread(target=self._do_lookup,
                                 args=(lat, lon), daemon=True).start()
            status = ('pending' if self.inflight else
                      'unavailable' if now < self.retry_after else 'rate_limited')
        return {'status': status, 'address': None, 'lat': lat, 'lon': lon}


# Strictly restricted imagery relay. No arbitrary URL proxying.
TILE_PATH = re.compile(r'^/_satellite/(\d{1,2})/(\d{1,8})/(\d{1,8})\.jpg$')


@functools.lru_cache(maxsize=512)
def satellite_tile(z: int, x: int, y: int):
    url = ('https://server.arcgisonline.com/ArcGIS/rest/services/'
           f'World_Imagery/MapServer/tile/{z}/{y}/{x}')
    request = urllib.request.Request(url, headers={
        'User-Agent': 'Mozilla/5.0 (3DHazardNet academic research dashboard)',
        'Accept': 'image/jpeg,image/*'})
    with urllib.request.urlopen(request, timeout=6) as response:
        if not response.headers.get('Content-Type', '').startswith('image/'):
            raise ValueError('Esri returned non-image response')
        body = response.read(800_001)
    if not body or len(body) > 800_000:
        raise ValueError('Satellite tile exceeds size limit')
    return body


def build_handler(root: Path, page: bytes, clock: PlaybackClock, review: HazardReview, pins: HazardPins, addresses: AddressResolver):
    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, directory=str(root), **kwargs)

        def log_message(self, format, *args):
            # Reduce spam from 8-12 frame image loads / second and map tiles.
            if self.path.startswith(('/api/', '/frames/', '/_satellite/')):
                return
            super().log_message(format, *args)

        def _bytes(self, code: int, body: bytes, mime='application/json', cache='no-store'):
            self.send_response(code)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', cache)
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(body)

        def _json(self, obj, code=200):
            self._bytes(code, json.dumps(obj, allow_nan=False).encode('utf-8'))


        def _send_start_video(self):
            video = root / START_VIDEO_NAME
            if not video.is_file():
                self.send_error(404, 'Starting-point video not uploaded')
                return
            size = video.stat().st_size
            if size < 1:
                self.send_error(404, 'Starting-point video is empty')
                return
            header = self.headers.get('Range', '')
            begin, end = 0, size - 1
            if header:
                match = re.fullmatch(r'bytes=(\d*)-(\d*)', header.strip())
                if not match or not any(match.groups()):
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                aa,bb=match.groups()
                if aa:
                    begin=int(aa)
                    end=int(bb) if bb else size-1
                else:
                    suffix=int(bb)
                    begin=max(0,size-suffix) if suffix else size
                if begin>=size or end<begin:
                    self.send_response(416)
                    self.send_header('Content-Range', f'bytes */{size}')
                    self.send_header('Content-Length', '0')
                    self.end_headers()
                    return
                end=min(end,size-1)
            self.send_response(206 if header else 200)
            self.send_header('Content-Type','video/mp4')
            self.send_header('Content-Length',str(end-begin+1))
            self.send_header('Cache-Control','no-store')
            self.send_header('X-Content-Type-Options','nosniff')
            self.send_header('Accept-Ranges','bytes')
            if header:
                self.send_header('Content-Range',f'bytes {begin}-{end}/{size}')
            self.end_headers()
            with video.open('rb') as f:
                f.seek(begin)
                remaining=end-begin+1
                try:
                    while remaining>0:
                        chunk=f.read(min(128*1024,remaining))
                        if not chunk:break
                        self.wfile.write(chunk)
                        remaining-=len(chunk)
                except (BrokenPipeError,ConnectionResetError):
                    pass

        def do_GET(self):
            path = urlsplit(self.path).path
            if path in {'/', '/index.html'}:
                self._bytes(200, page, 'text/html; charset=utf-8')
                return
            if path == '/api/start-video':
                video=root / START_VIDEO_NAME
                self._json({'available':video.is_file() and video.stat().st_size>0,
                            'url':START_VIDEO_NAME,'name':START_POINT['name'],
                            'lat':START_POINT['lat'],'lon':START_POINT['lon']})
                return
            if path == '/' + START_VIDEO_NAME:
                self._send_start_video()
                return
            if path == '/api/state':
                self._json(clock.snapshot())
                return
            if path == '/api/address':
                query = urllib.parse.parse_qs(urlsplit(self.path).query)
                try:
                    if set(query) != {'time'} or len(query['time']) != 1:
                        raise ValueError('Expected playback time only')
                    self._json(addresses.resolve(float(query['time'][0])))
                except (ValueError, TypeError, OverflowError) as exc:
                    self._json({'error': str(exc)}, 400)
                return
            if path == '/api/review':
                self._json(review.snapshot())
                return
            if path == '/api/pins':
                self._json(pins.snapshot())
                return
            if path == '/api/export-curated':
                body = review.curated_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.send_header('Content-Disposition', 'attachment; filename="data_curated.json"')
                self.end_headers()
                self.wfile.write(body)
                return
            if path == '/api/health':
                self._json({'ok': True, 'frames': len(clock.frame_times), 'root': str(root)})
                return
            if path == '/data.json':
                # Serve the live cached metadata; do not let Chrome cache stale JSON.
                try:
                    content = (root / 'data.json').read_bytes()
                    self._bytes(200, content)
                except OSError:
                    self.send_error(404, 'Existing dashboard data.json missing')
                return
            m = TILE_PATH.fullmatch(path)
            if m:
                z, x, y = map(int, m.groups())
                if not (0 <= z <= 19 and 0 <= x < 2**z and 0 <= y < 2**z):
                    self.send_error(400, 'Out of bounds satellite tile')
                    return
                try:
                    content = satellite_tile(z, x, y)
                    self._bytes(200, content, 'image/jpeg', 'public, max-age=86400')
                except (OSError, ValueError, TimeoutError) as exc:
                    print('Satellite relay unavailable:', exc, flush=True)
                    self.send_error(502, 'Satellite tile unavailable')
                return
            return super().do_GET()

        def do_POST(self):
            api_path = urlsplit(self.path).path
            if api_path not in {'/api/control', '/api/review', '/api/pins', '/api/locality'}:
                self.send_error(404)
                return
            try:
                if self.headers.get('Content-Type', '').split(';')[0].strip() != 'application/json':
                    raise ValueError('Content-Type must be application/json')
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length < 2048:
                    raise ValueError('Invalid request size')
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError('Expected a JSON object')
                if api_path == '/api/control':
                    self._json(clock.control(body))
                elif api_path == '/api/review':
                    self._json(review.edit(body))
                elif api_path == '/api/locality':
                    self._json(addresses.set_locality(body))
                else:
                    self._json(pins.edit(body))
            except (KeyError, ValueError, TypeError, OverflowError, json.JSONDecodeError) as exc:
                self._json({'error': str(exc)}, 400)
            except OSError as exc:
                self._json({'error': 'Could not persist review: '+str(exc)}, 500)

    return Handler


# ============================================================
# OFFLINE DASHBOARD BUILDER
# ============================================================
# Generates the data consumed by the existing two-monitor HTTP server.
# It does not run RF-DETR or 3DHazardNet inference.

DASHBOARD_PANEL_W = 640
DASHBOARD_PANEL_H = 360
DASHBOARD_CLASSES = ("loose_trash", "mud_spill", "public_greenway", "sidewalk")
DASHBOARD_BGR = ((65, 78, 237), (43, 164, 255), (58, 166, 74), (228, 178, 80))
DASHBOARD_SAMPLE_RE = re.compile(r"^sample_(\d+)_(CAM\d+)$")


def _db_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _db_rows(path):
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _db_number(value, integer=False):
    try:
        if value is None or str(value).strip() == "":
            return None
        n = float(value)
        if not math.isfinite(n):
            return None
        if integer:
            # Nanosecond timestamps must never lose precision via float.
            return int(str(value).strip())
        return n
    except (TypeError, ValueError, OverflowError):
        return None


def _db_column(columns, options):
    normalized = {str(c).strip().lower(): c for c in columns}
    for name in options:
        if name in normalized:
            return normalized[name]
    for name in options:
        for candidate, original in normalized.items():
            if name in candidate:
                return original
    return None


def _db_samples(inference_root, camera, start_ts=None, end_ts=None, limit=None):
    items = []
    diagnostics = {"folders": 0, "incomplete": 0, "invalid_metadata": 0}
    for folder in sorted(inference_root.glob("sample_*")):
        if not folder.is_dir():
            continue
        match = DASHBOARD_SAMPLE_RE.fullmatch(folder.name)
        if not match or match.group(2) != camera:
            continue
        diagnostics["folders"] += 1
        try:
            meta = _db_json(folder / "metadata.json")
            index = int(meta["sample_index"])
            stamp = int(meta["timestamp"])
            if index != int(match.group(1)) or str(meta["camera"]).upper() != camera:
                raise ValueError("Folder/metadata mismatch")
            if start_ts is not None and stamp < start_ts:
                continue
            if end_ts is not None and stamp > end_ts:
                continue
            if not (folder / "prediction.npz").is_file() or not (folder / "rfdetr_prior.npz").is_file():
                diagnostics["incomplete"] += 1
                continue
            images = sorted(p for p in folder.glob("image_original.*")
                            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"})
            if not images:
                original = meta.get("original_image")
                if original and Path(original).is_file():
                    images = [Path(original)]
            if not images:
                diagnostics["incomplete"] += 1
                continue
            items.append({"id": index, "t": stamp, "folder": folder, "image": images[0]})
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            diagnostics["invalid_metadata"] += 1
    items.sort(key=lambda item: (item["t"], item["id"]))
    unique, seen = [], set()
    for item in items:
        if item["t"] not in seen:
            unique.append(item)
            seen.add(item["t"])
    if limit is not None:
        unique = unique[:limit]
    if not unique:
        raise RuntimeError(f"No usable {camera} saved inference samples in {inference_root}")
    print(f"Cached {camera} frames: {len(unique):,} | sample scan: {diagnostics}", flush=True)
    return unique, diagnostics


def _db_route(clusters_root, gps_csv, first_ts, last_ts, max_gap_s):
    path = gps_csv or clusters_root / "route.csv"
    rows = _db_rows(path)
    if not rows:
        raise RuntimeError(f"GPS route CSV missing or empty: {path}")
    keys = rows[0].keys()
    tc = _db_column(keys, ("ros_timestamp_ns", "timestamp_ns", "timestamp", "time_ns"))
    yc = _db_column(keys, ("latitude_deg", "latitude", "lat"))
    xc = _db_column(keys, ("longitude_deg", "longitude", "lon", "lng"))
    if not all((tc, yc, xc)):
        raise RuntimeError(f"GPS CSV missing timestamp/lat/lon columns: {path}")
    margin = int(max_gap_s * 1e9)
    fixes = {}
    for row in rows:
        t = _db_number(row[tc], integer=True)
        lat = _db_number(row[yc])
        lon = _db_number(row[xc])
        if t is None or lat is None or lon is None:
            continue
        if first_ts - margin <= t <= last_ts + margin and -90 <= lat <= 90 and -180 <= lon <= 180 and (lat or lon):
            fixes[t] = (lat, lon)
    if len(fixes) < 2:
        raise RuntimeError(f"Need 2 or more valid route GPS fixes around selected camera frames: {path}")
    return [[round((stamp - first_ts) / 1e9, 6), *fixes[stamp]] for stamp in sorted(fixes)]


def _db_clusters(clusters_root, camera, frames):
    rows = _db_rows(clusters_root / "clusters.csv")
    if not (clusters_root / "clusters.csv").is_file():
        raise RuntimeError(f"Code 1 clusters.csv is missing: {clusters_root / 'clusters.csv'}")
    allowed = {value for row in rows if (value := _db_number(row.get("location_id"), integer=True)) is not None}
    keys = {(int(frame["id"]), int(frame["t"])) for frame in frames}
    lookup, clusters, used = {}, [], set()
    for meta_path in sorted(clusters_root.glob("cluster_*/metadata.json")):
        try:
            meta = _db_json(meta_path)
            cluster_id = int(meta["location_id"])
            if cluster_id not in allowed or cluster_id in used:
                continue
            lat, lon = float(meta["latitude"]), float(meta["longitude"])
            if not (math.isfinite(lat) and math.isfinite(lon) and -90 <= lat <= 90 and -180 <= lon <= 180):
                raise ValueError("Invalid cluster GPS")
            pairs = []
            for row in _db_rows(meta_path.parent / "events.csv"):
                index = _db_number(row.get("index", row.get("sample_index")), integer=True)
                stamp = _db_number(row.get("timestamp"), integer=True)
                if str(row.get("camera", "")).upper() == camera and (index, stamp) in keys:
                    pairs.append((index, stamp))
            if not pairs:
                continue
            first = int(frames[0]["t"])
            for key in pairs:
                if key in lookup and lookup[key] != cluster_id:
                    raise RuntimeError(f"Frame {key} assigned to two clusters; check Code 1 outputs")
                lookup[key] = cluster_id
            classes = meta.get("hazard_classes", [])
            if isinstance(classes, str):
                classes = [name.strip() for name in re.split(r"[;,]", classes) if name.strip()]
            score = _db_number(meta.get("severity_score"))
            stars = _db_number(meta.get("severity_stars"), integer=True)
            if stars not in (1, 2, 3, 4, 5):
                stars = None
            clusters.append({
                "id": cluster_id, "lat": lat, "lon": lon, "classes": list(classes),
                "first_s": round((min(t for _, t in pairs) - first) / 1e9, 6),
                "last_s": round((max(t for _, t in pairs) - first) / 1e9, 6),
                "score": score, "stars": stars,
                "level": str(meta.get("severity_level") or "unrated"),
                "events": len(pairs),
                "representative": _db_number(meta.get("representative_sample_index"), integer=True),
            })
            used.add(cluster_id)
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            print(f"WARNING: Skipped invalid cluster {meta_path}: {exc}", flush=True)
    if allowed and not clusters:
        raise RuntimeError("No Code 1 cluster records match these camera frames. Check --camera, --clusters-root and dataset timestamps.")
    unmatched = allowed - used
    if unmatched:
        print(f"WARNING: {len(unmatched)} clustering records have no matching {camera} frame in this replay", flush=True)
    for frame in frames:
        frame["cluster"] = lookup.get((int(frame["id"]), int(frame["t"])))
    active = {fr["cluster"] for fr in frames if fr["cluster"] is not None}
    clusters = sorted((c for c in clusters if c["id"] in active), key=lambda c: c["first_s"])
    print(f"Matched dashboard clusters: {len(clusters):,} | annotated frames: {len(lookup):,}", flush=True)
    return clusters


class DashboardFaceBlurrer:
    """Face pixelation shared by the replay and its map evidence images."""

    def __init__(self, model_path, mode="yunet"):
        import cv2
        self.cv2 = cv2
        self.mode = mode
        if mode == "yunet":
            if not model_path.is_file() or not hasattr(cv2, "FaceDetectorYN"):
                raise RuntimeError(f"YuNet face model not found or unsupported: {model_path}. "
                                   "Supply --yunet /path/to/yunet.onnx; or explicitly use --face-mode haar (weaker detection).")
            self.detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), 0.55, 0.3, 5000)
        else:
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if self.detector.empty():
                raise RuntimeError("OpenCV frontal-face cascade could not load")
            print("WARNING: Haar misses more faces than YuNet. Validate privacy before public display.", flush=True)

    def boxes(self, original):
        cv2 = self.cv2
        height, width = original.shape[:2]
        scale = min(1.0, 960.0 / max(height, width))
        reduced = cv2.resize(original, (max(1, round(width*scale)), max(1, round(height*scale))),
                             interpolation=cv2.INTER_AREA)
        if self.mode == "yunet":
            self.detector.setInputSize((reduced.shape[1], reduced.shape[0]))
            _, faces = self.detector.detect(reduced)
            faces = [] if faces is None else [face[:4] for face in faces]
        else:
            gray = cv2.cvtColor(reduced, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.1, 4, minSize=(20, 20))
        results = []
        for x, y, fw, fh in faces:
            x, y, fw, fh = (float(v)/scale for v in (x, y, fw, fh))
            pad = 0.22*max(fw, fh)
            bounds = (max(0, int(x-pad)), max(0, int(y-pad)),
                      min(width, int(x+fw+pad)), min(height, int(y+fh+pad)))
            if bounds[2] > bounds[0] and bounds[3] > bounds[1]:
                results.append(bounds)
        return results

    def apply(self, original, boxes):
        cv2 = self.cv2
        result = original.copy()
        for x0, y0, x1, y1 in boxes:
            patch = result[y0:y1, x0:x1]
            if not patch.size:
                continue
            h, w = patch.shape[:2]
            small = cv2.resize(patch, (max(1, w//18), max(1, h//18)), interpolation=cv2.INTER_AREA)
            pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
            result[y0:y1, x0:x1] = cv2.GaussianBlur(pixelated, (0, 0), max(12, max(w,h)/8))
        return result


def _db_letterbox(image):
    import cv2
    import numpy as np
    height, width = image.shape[:2]
    scale = min(DASHBOARD_PANEL_W / width, DASHBOARD_PANEL_H / height)
    w, h = max(1, round(width*scale)), max(1, round(height*scale))
    resized = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)
    canvas = np.full((DASHBOARD_PANEL_H, DASHBOARD_PANEL_W, 3), (21, 27, 37), dtype=np.uint8)
    dx, dy = (DASHBOARD_PANEL_W - w)//2, (DASHBOARD_PANEL_H - h)//2
    canvas[dy:dy+h, dx:dx+w] = resized
    return canvas


def _db_overlay(original, prior_path, threshold):
    import numpy as np
    import cv2
    with np.load(prior_path, allow_pickle=False) as data:
        if "full_rfdetr_prior" in data:
            full = np.asarray(data["full_rfdetr_prior"])
            if full.ndim != 3 or full.shape[0] != 6:
                raise ValueError(f"Invalid 6-channel prior at {prior_path}")
            planes = full[2:6]
        elif "prior" in data:
            planes = np.asarray(data["prior"])
        else:
            raise ValueError(f"No semantic prior in {prior_path}")
    if planes.ndim != 3 or planes.shape[0] != 4:
        raise ValueError(f"Invalid 4-channel semantic prior at {prior_path}: {planes.shape}")
    height, width = original.shape[:2]
    sh, sw = planes.shape[1:]
    scale = min(sw/width, sh/height)
    nw, nh = max(1, round(width*scale)), max(1, round(height*scale))
    dx, dy = (sw-nw)//2, (sh-nh)//2
    crops = planes[:, dy:dy+nh, dx:dx+nw]
    if crops.shape[1] < 1 or crops.shape[2] < 1:
        raise ValueError(f"Degenerate RF-DETR prior crop for {prior_path}")
    restored = np.stack([cv2.resize(np.asarray(c, np.float32), (width, height)) for c in crops])
    argmax = np.argmax(restored, axis=0)
    active = np.max(restored, axis=0) >= threshold
    output = original.copy()
    for cls_idx, bgr in enumerate(DASHBOARD_BGR):
        mask = active & (argmax == cls_idx)
        if np.any(mask):
            output[mask] = (0.57*output[mask].astype(np.float32) + 0.43*np.asarray(bgr, np.float32)).astype(np.uint8)
    return output


def _db_projection(points):
    import numpy as np
    pts = np.asarray(points, dtype=np.float64)
    return np.column_stack((0.89*pts[:,0] + 0.47*pts[:,2],
                            0.87*pts[:,1] - 0.36*pts[:,2] + 0.18*pts[:,0]))


def _db_empty_panel(message):
    import numpy as np
    import cv2
    img = np.full((DASHBOARD_PANEL_H, DASHBOARD_PANEL_W, 3), 244, dtype=np.uint8)
    cv2.putText(img, message[:60], (20, DASHBOARD_PANEL_H//2), cv2.FONT_HERSHEY_SIMPLEX,
                0.57, (100, 110, 120), 1, cv2.LINE_AA)
    return img


def _db_voxel_panel(semantic, bounds, max_points, occupancy=None, is_gt=False):
    import numpy as np
    import cv2
    if semantic.ndim != 3 or len(bounds) != 6:
        raise ValueError("Invalid voxel tensor or bounds")
    xmin,xmax,ymin,ymax,zmin,zmax = map(float, bounds)
    if not (xmin < xmax and ymin < ymax and zmin < zmax):
        raise ValueError("Invalid voxel bounds ordering")
    if occupancy is not None and occupancy.shape != semantic.shape:
        raise ValueError("Prediction occupancy and semantic dimensions do not match")
    img = np.full((DASHBOARD_PANEL_H, DASHBOARD_PANEL_W, 3), 250, dtype=np.uint8)
    corners = np.array([(x,y,z) for x in (xmin,xmax) for y in (ymin,ymax) for z in (zmin,zmax)])
    projected = _db_projection(corners)
    lo, hi = projected.min(axis=0), projected.max(axis=0)
    scale = min((DASHBOARD_PANEL_W-48)/max(hi[0]-lo[0], 1e-6),
                (DASHBOARD_PANEL_H-48)/max(hi[1]-lo[1], 1e-6))
    shift = np.array([DASHBOARD_PANEL_W/2, DASHBOARD_PANEL_H/2]) - (lo+hi)*scale/2

    def xy(points):
        return np.rint(_db_projection(points)*scale+shift).astype(np.int32)

    for x in np.linspace(xmin,xmax,9):
        edge=xy([(x,ymax,zmin),(x,ymax,zmax)])
        cv2.line(img,tuple(edge[0]),tuple(edge[1]),(220,226,232),1,cv2.LINE_AA)
    for z in np.linspace(zmin,zmax,8):
        edge=xy([(xmin,ymax,z),(xmax,ymax,z)])
        cv2.line(img,tuple(edge[0]),tuple(edge[1]),(220,226,232),1,cv2.LINE_AA)

    step=np.array([(xmax-xmin)/semantic.shape[0], (ymax-ymin)/semantic.shape[1],
                   (zmax-zmin)/semantic.shape[2]])
    origin=np.array([xmin,ymin,zmin])
    draw=[(4,(188,193,197), max_points*2, 1)] if is_gt else []
    draw += [(2,DASHBOARD_BGR[2],max_points,1), (3,DASHBOARD_BGR[3],max_points,1),
             (1,DASHBOARD_BGR[1],max_points,2), (0,DASHBOARD_BGR[0],max_points,2)]
    for label, bgr, cap, radius in draw:
        active = (semantic == label)
        if occupancy is not None:
            active &= occupancy
        indices = np.argwhere(active)
        if not len(indices):
            continue
        if len(indices) > cap:
            indices=indices[np.linspace(0,len(indices)-1,cap,dtype=np.int64)]
        pixels=xy(origin+(indices+0.5)*step)
        pixels=pixels[(pixels[:,0]>=0)&(pixels[:,0]<DASHBOARD_PANEL_W)&
                      (pixels[:,1]>=0)&(pixels[:,1]<DASHBOARD_PANEL_H)]
        if not len(pixels):
            continue
        mask=np.zeros((DASHBOARD_PANEL_H,DASHBOARD_PANEL_W),dtype=np.uint8)
        mask[pixels[:,1],pixels[:,0]]=255
        if radius>1:
            mask=cv2.dilate(mask,np.ones((3,3),dtype=np.uint8))
        img[mask>0]=bgr
    cv2.putText(img,"X right / Z forward / Y down",(12,DASHBOARD_PANEL_H-14),
                cv2.FONT_HERSHEY_SIMPLEX,.45,(70,76,85),1,cv2.LINE_AA)
    return img


def _db_voxel_pair(folder, max_points):
    import numpy as np
    with np.load(folder/"prediction.npz",allow_pickle=False) as data:
        if "semantic" not in data:
            raise ValueError(f"prediction.npz lacks semantic at {folder}")
        predicted=np.asarray(data["semantic"],dtype=np.uint8)
        occupied=np.asarray(data["occupancy_binary"],dtype=bool) if "occupancy_binary" in data else None
        if "bounds" in data:
            bounds=np.asarray(data["bounds"],dtype=np.float64).ravel()
        else:
            origin=np.asarray(data["origin"],dtype=np.float64).ravel()
            size=float(data["voxel_size"])
            if len(origin)!=3:
                raise ValueError("Incorrect voxel origin dimensions")
            bounds=np.array([origin[0],origin[0]+size*predicted.shape[0],
                             origin[1],origin[1]+size*predicted.shape[1],
                             origin[2],origin[2]+size*predicted.shape[2]])
    if occupied is None:
        raise ValueError(f"prediction.npz missing occupancy_binary: {folder}; refusing to plot background as voxels")
    prediction_panel=_db_voxel_panel(predicted,bounds,max_points,occupied)
    gt_path=folder/"gt_target.npz"
    if not gt_path.is_file():
        return _db_empty_panel("Pseudo-GT unavailable"),prediction_panel
    try:
        with np.load(gt_path,allow_pickle=False) as data:
            if "semantic_visual" in data:
                gt=np.asarray(data["semantic_visual"],dtype=np.uint8)
            elif "semantic_replay" in data:
                gt=np.asarray(data["semantic_replay"],dtype=np.uint8)
            elif "semantic" in data:
                raw=np.asarray(data["semantic"])
                gt=np.full(raw.shape,255,dtype=np.uint8)
                good=(raw>=0)&(raw<=3)
                gt[good]=raw[good]
            else:
                raise ValueError("No semantic GT array")
            if "semantic_visual" not in data and "background_observed" in data:
                bg=np.asarray(data["background_observed"])>0
                gt[bg & (gt==255)]=4
        if gt.shape != predicted.shape:
            raise ValueError("Pseudo-GT grid dimensions disagree with prediction")
        gt_panel=_db_voxel_panel(gt,bounds,max_points,is_gt=True)
    except (OSError,KeyError,ValueError,TypeError) as exc:
        print(f"WARNING: Missing or incompatible pseudo-GT for {folder.name}: {exc}",flush=True)
        gt_panel=_db_empty_panel("Pseudo-GT unavailable")
    return gt_panel,prediction_panel


def _db_jpg(path, img, quality=85):
    import cv2
    ok, buf=cv2.imencode(".jpg",img,[cv2.IMWRITE_JPEG_QUALITY,int(quality)])
    if not ok:
        raise OSError(f"Failed to JPEG-encode {path}")
    path.parent.mkdir(parents=True,exist_ok=True)
    atomic_write(path,buf.tobytes())


def _db_render_tile(sample, dest, blurrer, threshold, max_points):
    import cv2
    import numpy as np
    original=cv2.imread(str(sample["image"]))
    if original is None:
        raise OSError(f"Failed reading original RGB: {sample['image']}")
    boxes=blurrer.boxes(original)
    rgb=_db_letterbox(blurrer.apply(original,boxes))
    segmentation=_db_overlay(original,sample["folder"]/"rfdetr_prior.npz",threshold)
    seg=_db_letterbox(blurrer.apply(segmentation,boxes))
    gt,pred=_db_voxel_pair(sample["folder"],max_points)
    tile=np.vstack((np.hstack((rgb,seg)),np.hstack((gt,pred))))
    if tile.shape != (2*DASHBOARD_PANEL_H,2*DASHBOARD_PANEL_W,3):
        raise RuntimeError(f"Unexpected composite shape: {tile.shape}")
    _db_jpg(dest,tile,85)


def _db_build_media(data, inference_root, clusters_root, root, run_name, samples, blurrer, threshold):
    import cv2
    import shutil
    by_sample={fr["id"]:fr for fr in data["frames"]}
    by_cluster={}
    for fr in data["frames"]:
        if fr["cluster"] is not None:
            by_cluster.setdefault(fr["cluster"],fr)
    sample_lookup={item["id"]:item for item in samples}
    count=0
    for cluster in data["clusters"]:
        cid=cluster["id"]
        target_dir=root/"cluster_media"/run_name/f"cluster_{cid:04d}"
        target_dir.mkdir(parents=True,exist_ok=True)
        frame=by_sample.get(cluster.get("representative")) if cluster.get("representative") is not None else None
        if frame is None or frame["cluster"] != cid:
            frame=by_cluster.get(cid)
        if frame:
            sample=sample_lookup.get(frame["id"])
            if sample is None:
                raise RuntimeError(f"Cluster {cid} representative source was not found")
            original=cv2.imread(str(sample["image"]))
            if original is None:
                raise RuntimeError(f"Cluster {cid} representative image unreadable")
            boxes=blurrer.boxes(original)
            blurred=blurrer.apply(original,boxes)
            overlaid=_db_overlay(original,sample["folder"]/"rfdetr_prior.npz",threshold)
            blurred_overlay=blurrer.apply(overlaid,boxes)
            rgb=target_dir/"rgb_blurred.jpg"
            overlay=target_dir/"segmentation.jpg"
            _db_jpg(rgb,blurred,87)
            _db_jpg(overlay,blurred_overlay,88)
            cluster.update({"rgb":rgb.relative_to(root).as_posix(),
                            "image":overlay.relative_to(root).as_posix(),
                            "image_resolution":f"{original.shape[1]}x{original.shape[0]}"})
            count+=1
        viewer=clusters_root/f"cluster_{cid:04d}"/"voxel_3d.html"
        if viewer.is_file():
            dest=target_dir/"voxel_3d.html"
            shutil.copy2(viewer,dest)
            cluster["voxel"]=dest.relative_to(root).as_posix()
    print(f"Prepared privacy-blurred cluster images: {count}/{len(data['clusters'])}",flush=True)


def _db_archive_stale_review(root, clusters):
    """Avoid applying old exclusions to unrelated newly numbered event IDs."""
    identity=sorted((int(c["id"]),round(float(c["lat"]),7),round(float(c["lon"]),7),
                     round(float(c["first_s"]),3)) for c in clusters)
    fingerprint=hashlib.sha256(json.dumps(identity,separators=(",",":")).encode()).hexdigest()
    for filename in ("hazard_review.json","hazard_pins.json"):
        path=root/filename
        if not path.is_file():
            continue
        existing=_db_json(path)
        if existing.get("dataset_fingerprint") != fingerprint:
            suffix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
            destination=root/f"{path.stem}.previous_{suffix}_{os.urandom(3).hex()}.json"
            path.rename(destination)
            print(f"Archived stale review/POI file: {destination.name}",flush=True)


def rebuild_dashboard_data(root, inference_root, clusters_root, gps_csv, camera,
                           threshold, face_mode, yunet_path, max_points,
                           gps_max_gap_s=4.0, max_frames=None, start_ts=None, end_ts=None):
    """Build versioned image/media folders, then atomically publish data.json.

    Existing data.json keeps pointing to the previous complete cache if the
    build fails halfway through. Previous media is never deleted automatically.
    """
    import shutil
    if not inference_root.is_dir():
        raise RuntimeError(f"Saved inference folder not found: {inference_root}")
    if not clusters_root.is_dir():
        raise RuntimeError(f"Code 1 cluster folder not found: {clusters_root}")
    if not (clusters_root/"clusters.csv").is_file():
        raise RuntimeError(f"Run Code 1 first: {clusters_root/'clusters.csv'} is missing")
    if not (clusters_root/"route.csv").is_file() and gps_csv is None:
        raise RuntimeError(f"Run Code 1 first: {clusters_root/'route.csv'} is missing")
    if not 0<=threshold<=1 or max_points<100 or gps_max_gap_s<=0:
        raise ValueError("Invalid rendering confidence, max-points or GPS gap")
    samples,diagnostics=_db_samples(inference_root,camera,start_ts,end_ts,max_frames)
    # Fail before rendering thousands of files if YuNet cannot be loaded.
    blurrer=DashboardFaceBlurrer(yunet_path,face_mode)
    first=int(samples[0]["t"])
    run_name="build_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+os.urandom(4).hex()
    frame_dir=root/"frames"/run_name
    media_dir=root/"cluster_media"/run_name
    root.mkdir(parents=True,exist_ok=True)
    frames=[]
    try:
        for number,sample in enumerate(samples):
            frame_path=frame_dir/f"{number:06d}.jpg"
            _db_render_tile(sample,frame_path,blurrer,threshold,max_points)
            frames.append({"id":sample["id"],"t":str(sample["t"]),
                           "s":round((int(sample["t"])-first)/1e9,6),"cam":camera,
                           "file":frame_path.relative_to(root).as_posix(),"cluster":None})
            if (number+1)%100==0 or number+1==len(samples):
                print(f"Composite frames generated: {number+1:,}/{len(samples):,}",flush=True)
        route=_db_route(clusters_root,gps_csv,int(frames[0]["t"]),int(frames[-1]["t"]),gps_max_gap_s)
        clusters=_db_clusters(clusters_root,camera,frames)
        summary_path=clusters_root/"pipeline_summary.json"
        cluster_threshold=None
        if summary_path.is_file():
            cluster_threshold=_db_number(_db_json(summary_path).get("two_d_threshold"))
        if cluster_threshold is not None and abs(cluster_threshold-threshold)>1e-7:
            print(f"WARNING: Event clustering used threshold {cluster_threshold}, but replay segmentation used {threshold}.",flush=True)
        data={"name":"AI-Based Asset Management and Assessment","camera":camera,
              "frames":frames,"route":route,"clusters":clusters,
              "duration":frames[-1]["s"],"gps_max_gap_s":gps_max_gap_s,
              "panel_width":DASHBOARD_PANEL_W,"panel_height":DASHBOARD_PANEL_H,
              "segmentation_threshold":threshold,"cluster_threshold":cluster_threshold}
        _db_build_media(data,inference_root,clusters_root,root,run_name,samples,blurrer,threshold)
        if not frames or not route:
            raise RuntimeError("Empty dashboard data after rebuild")
        config={"camera":camera,"threshold":threshold,"max_points":max_points,
                "face_mode":face_mode,"yunet":str(yunet_path),
                "inference_root":str(inference_root),"clusters_root":str(clusters_root),
                "start":int(samples[0]["t"]),"end":int(samples[-1]["t"]),
                "frames":len(frames),"build_id":run_name,
                "rendered_dimensions":[2*DASHBOARD_PANEL_W,2*DASHBOARD_PANEL_H]}
        # Metadata and review writes precede publishing the new dashboard.
        _db_archive_stale_review(root,clusters)
        atomic_write(root/"build_config.json",json.dumps(config,indent=2).encode())
        atomic_write(root/"build_report.json",json.dumps({"frames":len(frames),
                     "clusters":len(clusters),"sample_scan":diagnostics,"build_id":run_name},indent=2).encode())
        atomic_write(root/"data.json",json.dumps(data,indent=2,allow_nan=False).encode())
    except Exception:
        shutil.rmtree(frame_dir,ignore_errors=True)
        shutil.rmtree(media_dir,ignore_errors=True)
        raise
    print(f"DASHBOARD REBUILT: {root/'data.json'} | {len(frames):,} frames | {len(clusters):,} clusters",flush=True)
    print("Old versioned frame/media folders were deliberately left intact until you verify this rebuild.",flush=True)
    return data

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dashboard-root', type=Path, default=DEFAULT_ROOT,
                        help='Dashboard folder containing data.json and frames/')
    parser.add_argument('--rebuild', action='store_true',
                        help='Regenerate data.json, privacy-blurred composite frames, and cluster media before serving')
    parser.add_argument('--build-only', action='store_true',
                        help='Regenerate dashboard cache and exit without launching web servers')
    parser.add_argument('--inference-root', type=Path,
                        help='Saved V10.1 inference sample folders (defaults to sibling of clusters-root)')
    parser.add_argument('--clusters-root', type=Path,
                        help='Output directory from Code 1 (defaults to parent of dashboard-root)')
    parser.add_argument('--gps-csv', type=Path,
                        help='GPS CSV override (default clusters-root/route.csv)')
    parser.add_argument('--camera', choices=('CAM2','CAM6'), default='CAM2')
    parser.add_argument('--threshold', type=float, default=None,
                        help='RF-DETR overlay threshold (default: Code 1 two_d_threshold)')
    parser.add_argument('--face-mode', choices=('yunet','haar'), default='yunet',
                        help='YuNet ONNX is recommended; Haar is a weaker explicit fallback')
    parser.add_argument('--yunet', type=Path,
                        help='YuNet ONNX model path (default: inference-root/yunet.onnx)')
    parser.add_argument('--max-points', type=int, default=18000)
    parser.add_argument('--gps-max-gap-s', type=float, default=4.0)
    parser.add_argument('--max-frames', type=int,
                        help='Debug only: render first N selected frames')
    parser.add_argument('--start-ts', type=int, help='Minimum nanosecond image timestamp')
    parser.add_argument('--end-ts', type=int, help='Maximum nanosecond image timestamp')
    parser.add_argument('--map-port', type=int, default=8890)
    parser.add_argument('--console-port', type=int, default=8891)
    parser.add_argument('--host', default='127.0.0.1',
                        help='Bind address; keep 127.0.0.1 for SSH tunnels')
    parser.add_argument('--geocode-endpoint',
                        default='https://nominatim.openstreetmap.org/reverse',
                        help='Nominatim-compatible reverse geocoding API endpoint')
    parser.add_argument('--geocode-user-agent',
                        default='PublicSpaceConditionAssessment/14 (Monash University research demo)',
                        help='Identifying User-Agent for road-name services')
    parser.add_argument('--road-endpoint',
                        default='https://router.project-osrm.org/nearest/v1/driving',
                        help='OSRM-compatible nearest-road service (driving profile)')
    args = parser.parse_args()
    root = args.dashboard_root.expanduser().resolve()
    if args.build_only or args.rebuild:
        if args.max_frames is not None and args.max_frames < 1:
            parser.error('--max-frames must be positive')
        if args.start_ts is not None and args.end_ts is not None and args.start_ts > args.end_ts:
            parser.error('--start-ts must be <= --end-ts')
        cluster_root = (args.clusters_root or root.parent).expanduser().resolve()
        inference_root = (args.inference_root or cluster_root.parent / 'v10_1_sequence_inference').expanduser().resolve()
        gps_csv = args.gps_csv.expanduser().resolve() if args.gps_csv else None
        yunet = (args.yunet or inference_root / 'yunet.onnx').expanduser().resolve()
        summary_path = cluster_root / 'pipeline_summary.json'
        summary = _db_json(summary_path) if summary_path.is_file() else {}
        threshold = args.threshold
        if threshold is None:
            threshold = _db_number(summary.get('two_d_threshold'))
            if threshold is None:
                threshold = 0.25
                print('WARNING: Code 1 threshold unavailable; defaulting to 0.25. Verify this matches your clustering run.')
        print('DATA REBUILD: saved inference, no neural network re-inference')
        print('Inference:', inference_root)
        print('Clusters: ', cluster_root)
        print('Dashboard:', root)
        print('Threshold:', threshold)
        rebuild_dashboard_data(root, inference_root, cluster_root, gps_csv, args.camera,
                               float(threshold), args.face_mode, yunet, args.max_points,
                               args.gps_max_gap_s, args.max_frames, args.start_ts, args.end_ts)
        if args.build_only:
            print('Build-only requested. Ready to start dashboard servers in a subsequent run.')
            return
    if not 1 <= args.map_port <= 65535 or not 1 <= args.console_port <= 65535:
        parser.error('Invalid port')
    if args.map_port == args.console_port:
        parser.error('Two distinct ports are required')
    data_file = root / 'data.json'
    if not data_file.is_file():
        parser.error(f'Dashboard data missing: {data_file}. Run this same script with --rebuild first.')
    with data_file.open(encoding='utf-8') as f:
        data = json.load(f)
    frames = data.get('frames', [])
    route = data.get('route', [])
    if not frames or len(route) < 2:
        parser.error('Dashboard data.json needs frames[] and at least two route points')
    if any(not (root / frame['file']).is_file() for frame in (frames[0], frames[-1])):
        parser.error('First or last composite frame is missing; check dashboard-root')
    times = [float(fr['s']) for fr in frames]
    if any(b < a for a, b in zip(times, times[1:])):
        parser.error('Frame times in data.json are not sorted')
    clock = PlaybackClock(float(data.get('duration', times[-1])), times)
    review = HazardReview(root, data)
    pins = HazardPins(root, data, review.fingerprint)
    addresses = AddressResolver(root, route, clock.duration,
                                float(data.get('gps_max_gap_s', 4.0)),
                                args.geocode_endpoint, args.geocode_user_agent,
                                args.road_endpoint)
    map_html = make_page('Georeferenced Inspection Map', MAP_BODY, MAP_JS, leaflet=True)
    console_html = make_page('Interactable Dashboard', CONSOLE_BODY, CONSOLE_JS)

    servers = []
    started = False
    try:
        for port, html in ((args.map_port, map_html), (args.console_port, console_html)):
            server = http.server.ThreadingHTTPServer(
                (args.host, port), build_handler(root, html, clock, review, pins, addresses))
            server.daemon_threads = True
            servers.append(server)
        print('\nAI-BASED ASSET MANAGEMENT AND ASSESSMENT · DUAL-MONITOR DASHBOARD')
        print(f'Cached frames: {len(frames):,} | Clusters: {len(data.get("clusters", []))}')
        print('No inference or rendering will be rerun.')
        print(f'Review file:       {review.path}')
        print(f'Pinned demo POIs:  {pins.path} ({len(pins.records)} saved)')
        print(f'Address cache:     {addresses.path}')
        print(f'Starting point:    {START_POINT["name"]} ({START_POINT["lat"]:.6f}, {START_POINT["lon"]:.6f})')
        print(f'Optional video:    {root / START_VIDEO_NAME}')
        print(f'Curated metadata:  {review.export_path}')
        print(f'Curated CSV:       {review.csv_path}')
        print(f'GPS MAP:           http://localhost:{args.map_port}/')
        print(f'PLAYBACK/ANALYSIS: http://localhost:{args.console_port}/')
        print('Shared server clock synchronizes both displays. Ctrl+C stops both.\n')
        threads = [threading.Thread(target=s.serve_forever, daemon=True) for s in servers]
        for thread in threads:
            thread.start()
        started = True
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print('\nClosing both dashboard servers.')
    finally:
        for server in servers:
            if started:
                server.shutdown()
            server.server_close()


if __name__ == '__main__':
    main()
