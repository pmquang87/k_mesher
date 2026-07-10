"""tkinter GUI for k_mesher: STEP -> TET4/TET10/TRI3/QUAD4 -> LS-DYNA .k"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import traceback
import tkinter as tk

from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

from k_mesher import job_runner
from k_mesher import mesher

ELEMENT_TYPES = {
    "TET4 (linear tetrahedron)": "TET4",
    "TET10 (quadratic tetrahedron)": "TET10",
    "Shell TRI3 (triangles)": "TRI3",
    "Shell QUAD4 (quad-dominant)": "QUAD4",
}
ELFORMS_BY_ETYPE = {
    "TET4": {
        "10 - 1-point tetrahedron": 10,
        "13 - 1-point nodal pressure tetrahedron": 13,
    },
    "TET10": {
        "16 - 4/5-point quadratic tetrahedron": 16,
        "17 - 10-noded composite tetrahedron": 17,
    },
    "TRI3": {
        "4 - C0 triangular shell": 4,
        "17 - fully integrated DKT triangle": 17,
    },
    "QUAD4": {
        "16 - fully integrated shell": 16,
        "2 - Belytschko-Tsay": 2,
    },
}
ELFORMS_ALL = {k: v for d in ELFORMS_BY_ETYPE.values() for k, v in d.items()}
REFINE_HINTS = {
    "Sphere": "params: cx, cy, cz, radius",
    "Box": "params: xmin, ymin, zmin, xmax, ymax, zmax",
}
FACE_ROLES = ["Set only", "Fix (SPC)", "Pressure", "Force X", "Force Y",
              "Force Z", "Mesh size", "Suppress (defeature)"]
# symmetry-plane boundary-condition kinds (label -> dyna_writer constraint id)
SYM_CONSTRAINTS = {
    "Symmetric (normal transl. + in-plane rot.)": "symmetric",
    "Anti-symmetric": "antisymmetric",
    "Fixed (all 6 DOFs)": "fixed",
    "Custom DOFs": "custom",
}
# per-body output modes (label -> job_runner split_mode id)
SPLIT_MODES = {
    "Off (single file)": None,
    "Standalone .k per body": "parts",
    "*INCLUDE fragments + master deck": "include",
}
# contact card kinds (label -> job_runner contact type id). The first entry is
# the default: it keeps the legacy single-surface behaviour driven by contact_fs
# (kopts["contacts"] stays empty); any other choice emits a *CONTACT via
# kopts["contacts"].
CONTACT_TYPES = {
    "Automatic single surface (default)": "automatic_single_surface",
    "Automatic surface to surface": "automatic_surface_to_surface",
    "Tied surface to surface": "tied_surface_to_surface",
    "Tied nodes to surface": "tied_nodes_to_surface",
}
# auto-connect modes (label -> job_runner auto_connect "mode" id). "None" leaves
# kopts["auto_connect"] as None (no interface detection); the others auto-detect
# body interfaces after meshing and write spotwelds or a tied contact.
CONNECT_MODES = {
    "None (no auto-detect)": "none",
    "Spotweld (*CONSTRAINED_SPOTWELD)": "spotweld",
    "Tied contact": "tied",
}
# coordinate node-set kinds (label -> (mesher.select_nodes kind, params hint))
COORD_KINDS = {
    "Plane": ("plane", "params: axis (x/y/z), offset  e.g. z, 0"),
    "Box": ("box", "params: xmin, ymin, zmin, xmax, ymax, zmax"),
    "Sphere": ("sphere", "params: cx, cy, cz, radius"),
}
COORD_ROLES = ["Set only", "Fix (SPC)", "Force X", "Force Y", "Force Z"]
# input file dialog filter (STEP/IGES/BREP CAD + STL/OBJ/PLY tessellation)
CAD_FILETYPES = [
    ("CAD & mesh files",
     "*.step *.stp *.iges *.igs *.brep *.brp *.stl *.obj *.ply"),
    ("STEP", "*.step *.stp"), ("IGES", "*.iges *.igs"),
    ("BREP", "*.brep *.brp"), ("Tessellation (STL/OBJ/PLY)", "*.stl *.obj *.ply"),
    ("All files", "*.*"),
]
_HERE = os.path.dirname(os.path.abspath(__file__))
SETTINGS_FILE = os.path.join(_HERE, "k_mesher_settings.json")
PRESETS_FILE = os.path.join(_HERE, "k_mesher_presets.json")


class KMesherGUI:
    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("k_mesher - STEP to LS-DYNA mesher")
        root.minsize(700, 680)
        root.geometry("880x800")

        self.log_queue: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.last_preview: str | None = None
        self.refinements: list[dict] = []
        self.part_mats: list[dict] = []    # {"body","e","pr","ro"[,"rigid"]}
        self.coord_sets: list[dict] = []   # {"kind","params","role",...}
        self.face_roles: list[dict] = []   # {"tag","role","value","dofs"}
        self.faces_by_tag: dict[int, dict] = {}
        self.face_scan_sig: str | None = None
        self.batch_jobs: list[dict] = []

        pad = {"padx": 6, "pady": 3}
        body = ttk.Frame(root, padding=8)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)

        # ------------------------------------------------------ files -----
        files = ttk.LabelFrame(body, text="Files", padding=6)
        files.grid(row=0, column=0, sticky="ew", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="CAD/mesh input:").grid(row=0, column=0, sticky="w")
        self.var_step = tk.StringVar()
        ttk.Entry(files, textvariable=self.var_step).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(files, text="Browse...", command=self._browse_step).grid(row=0, column=2)

        ttk.Label(files, text=".k output:").grid(row=1, column=0, sticky="w")
        self.var_out = tk.StringVar()
        ttk.Entry(files, textvariable=self.var_out).grid(row=1, column=1, sticky="ew", padx=4)
        ttk.Button(files, text="Browse...", command=self._browse_out).grid(row=1, column=2)

        # ---------------------------------------------------- notebook ----
        nb = ttk.Notebook(body)
        nb.grid(row=1, column=0, sticky="ew", **pad)
        tab_mesh = ttk.Frame(nb, padding=4)
        tab_dyna = ttk.Frame(nb, padding=4)
        tab_sym = ttk.Frame(nb, padding=4)
        tab_batch = ttk.Frame(nb, padding=4)
        nb.add(tab_mesh, text=" Mesh ")
        nb.add(tab_dyna, text=" LS-DYNA output ")
        nb.add(tab_sym, text=" Symmetry & face sets ")
        nb.add(tab_batch, text=" Batch & presets ")
        for tab in (tab_mesh, tab_dyna, tab_sym, tab_batch):
            tab.columnconfigure(0, weight=1)

        self._build_mesh_tab(tab_mesh, pad)
        self._build_dyna_tab(tab_dyna, pad)
        self._build_sym_tab(tab_sym, pad)
        self._build_batch_tab(tab_batch, pad)

        # ---------------------------------------------------- actions -----
        actions = ttk.Frame(body)
        actions.grid(row=2, column=0, sticky="ew", **pad)
        self.btn_run = ttk.Button(actions, text="Generate mesh", command=self._start_run)
        self.btn_run.pack(side="left")
        self.var_preview = tk.BooleanVar(value=False)
        ttk.Checkbutton(actions, text="Open Gmsh viewer when done",
                        variable=self.var_preview).pack(side="left", padx=12)
        self.btn_preview = ttk.Button(actions, text="Preview last mesh",
                                      command=self._open_preview, state="disabled")
        self.btn_preview.pack(side="left")
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=140)

        # -------------------------------------------------------- log -----
        logf = ttk.LabelFrame(body, text="Log", padding=4)
        logf.grid(row=3, column=0, sticky="nsew", **pad)
        body.rowconfigure(3, weight=1)
        self.log = ScrolledText(logf, height=12, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True)

        self._load_settings()
        self._sync_elforms()
        self._sync_sym_bc()
        self._sync_connect()
        self._refresh_presets()
        root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._poll_log()

    # ------------------------------------------------------- tab: mesh ----
    def _build_mesh_tab(self, tab, pad):
        mesh = ttk.LabelFrame(tab, text="Mesh parameters", padding=6)
        mesh.grid(row=0, column=0, sticky="ew", **pad)
        for c in (1, 3):
            mesh.columnconfigure(c, weight=1)

        ttk.Label(mesh, text="Max element size:").grid(row=0, column=0, sticky="w")
        self.var_smax = tk.StringVar(value="10.0")
        ttk.Entry(mesh, textvariable=self.var_smax, width=10).grid(row=0, column=1, sticky="w", padx=4)

        ttk.Label(mesh, text="Min element size:").grid(row=0, column=2, sticky="w")
        self.var_smin = tk.StringVar(value="0.0")
        ttk.Entry(mesh, textvariable=self.var_smin, width=10).grid(row=0, column=3, sticky="w", padx=4)

        self.var_curv = tk.BooleanVar(value=True)
        ttk.Checkbutton(mesh, text="Refine by curvature, elements per 2π:",
                        variable=self.var_curv).grid(row=1, column=0, columnspan=2, sticky="w")
        self.var_curv_n = tk.StringVar(value="16")
        ttk.Entry(mesh, textvariable=self.var_curv_n, width=10).grid(row=1, column=2, sticky="w", padx=4)

        ttk.Label(mesh, text="3D algorithm:").grid(row=2, column=0, sticky="w")
        self.var_algo = tk.StringVar(value="Delaunay")
        ttk.Combobox(mesh, textvariable=self.var_algo, state="readonly",
                     values=list(mesher.ALGO3D)).grid(row=2, column=1, sticky="w", padx=4)

        ttk.Label(mesh, text="Geometry units:").grid(row=2, column=2, sticky="w")
        self.var_unit = tk.StringVar(value=next(iter(mesher.OCC_UNITS)))
        ttk.Combobox(mesh, textvariable=self.var_unit, state="readonly", width=24,
                     values=list(mesher.OCC_UNITS)).grid(row=2, column=3, sticky="w", padx=4)

        self.var_opt = tk.BooleanVar(value=True)
        ttk.Checkbutton(mesh, text="Optimize element quality",
                        variable=self.var_opt).grid(row=3, column=0, columnspan=2, sticky="w")
        self.var_heal = tk.BooleanVar(value=False)
        ttk.Checkbutton(mesh, text="Heal geometry on import",
                        variable=self.var_heal).grid(row=3, column=2, columnspan=2, sticky="w")
        self.var_glue = tk.BooleanVar(value=False)
        ttk.Checkbutton(mesh, text="Glue touching bodies (conformal shared nodes)",
                        variable=self.var_glue).grid(row=4, column=0, columnspan=3, sticky="w")
        self.var_autoref = tk.BooleanVar(value=False)
        ttk.Checkbutton(mesh, text="Auto-refine bad spots and remesh (max 2 extra rounds)",
                        variable=self.var_autoref).grid(row=5, column=0, columnspan=3, sticky="w")

        ref = ttk.LabelFrame(tab, text="Local refinement regions", padding=6)
        ref.grid(row=1, column=0, sticky="ew", **pad)
        ref.columnconfigure(1, weight=1)

        self.var_ref_kind = tk.StringVar(value="Sphere")
        cmb = ttk.Combobox(ref, textvariable=self.var_ref_kind, state="readonly",
                           width=8, values=list(REFINE_HINTS))
        cmb.grid(row=0, column=0, sticky="w")
        cmb.bind("<<ComboboxSelected>>", lambda e: self._sync_ref_hint())
        self.var_ref_params = tk.StringVar()
        ttk.Entry(ref, textvariable=self.var_ref_params).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Label(ref, text="size:").grid(row=0, column=2, sticky="w")
        self.var_ref_size = tk.StringVar()
        ttk.Entry(ref, textvariable=self.var_ref_size, width=8).grid(row=0, column=3, padx=4)
        ttk.Button(ref, text="Add", command=self._add_refinement).grid(row=0, column=4)

        self.lbl_ref_hint = ttk.Label(ref, foreground="gray")
        self.lbl_ref_hint.grid(row=1, column=0, columnspan=3, sticky="w")
        self._sync_ref_hint()

        self.lst_ref = tk.Listbox(ref, height=3)
        self.lst_ref.grid(row=2, column=0, columnspan=4, sticky="ew", pady=3)
        ttk.Button(ref, text="Remove", command=self._remove_refinement
                   ).grid(row=2, column=4, sticky="n", pady=3)

    # ---------------------------------------------------- tab: LS-DYNA ----
    def _build_dyna_tab(self, tab, pad):
        dyna = ttk.LabelFrame(tab, text="LS-DYNA output", padding=6)
        dyna.grid(row=0, column=0, sticky="ew", **pad)
        for c in (1, 3):
            dyna.columnconfigure(c, weight=1)

        ttk.Label(dyna, text="Element type:").grid(row=0, column=0, sticky="w")
        self.var_etype = tk.StringVar(value=next(iter(ELEMENT_TYPES)))
        cmb = ttk.Combobox(dyna, textvariable=self.var_etype, state="readonly", width=28,
                           values=list(ELEMENT_TYPES))
        cmb.grid(row=0, column=1, sticky="w", padx=4)
        cmb.bind("<<ComboboxSelected>>", lambda e: self._sync_elforms())

        ttk.Label(dyna, text="ELFORM:").grid(row=0, column=2, sticky="w")
        self.var_elform = tk.StringVar()
        self.cmb_elform = ttk.Combobox(dyna, textvariable=self.var_elform,
                                       state="readonly", width=36)
        self.cmb_elform.grid(row=0, column=3, sticky="w", padx=4)

        ttk.Label(dyna, text="Part ID (base):").grid(row=1, column=0, sticky="w")
        self.var_pid = tk.StringVar(value="1")
        ttk.Entry(dyna, textvariable=self.var_pid, width=10).grid(row=1, column=1, sticky="w", padx=4)

        ttk.Label(dyna, text="Shell thickness:").grid(row=1, column=2, sticky="w")
        self.var_thick = tk.StringVar(value="1.0")
        self.ent_thick = ttk.Entry(dyna, textvariable=self.var_thick, width=10)
        self.ent_thick.grid(row=1, column=3, sticky="w", padx=4)

        ttk.Label(dyna, text="Start node ID:").grid(row=2, column=0, sticky="w")
        self.var_nid0 = tk.StringVar(value="1")
        ttk.Entry(dyna, textvariable=self.var_nid0, width=10).grid(row=2, column=1, sticky="w", padx=4)

        ttk.Label(dyna, text="Start element ID:").grid(row=2, column=2, sticky="w")
        self.var_eid0 = tk.StringVar(value="1")
        ttk.Entry(dyna, textvariable=self.var_eid0, width=10).grid(row=2, column=3, sticky="w", padx=4)

        ttk.Label(dyna, text="Start set ID:").grid(row=3, column=0, sticky="w")
        self.var_sid0 = tk.StringVar(value="1")
        ttk.Entry(dyna, textvariable=self.var_sid0, width=10).grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(dyna, text="(node / segment / SPC / element set IDs)",
                  foreground="gray").grid(row=3, column=2, columnspan=2, sticky="w")

        self.var_mat = tk.BooleanVar(value=False)
        ttk.Checkbutton(dyna, text="Write *MAT_ELASTIC   E:",
                        variable=self.var_mat).grid(row=4, column=0, sticky="w")
        matrow = ttk.Frame(dyna)
        matrow.grid(row=4, column=1, columnspan=3, sticky="w")
        self.var_mat_e = tk.StringVar(value="210000")
        ttk.Entry(matrow, textvariable=self.var_mat_e, width=10).pack(side="left", padx=4)
        ttk.Label(matrow, text="ν:").pack(side="left")
        self.var_mat_nu = tk.StringVar(value="0.3")
        ttk.Entry(matrow, textvariable=self.var_mat_nu, width=7).pack(side="left", padx=4)
        ttk.Label(matrow, text="ρ:").pack(side="left")
        self.var_mat_ro = tk.StringVar(value="7.85e-9")
        ttk.Entry(matrow, textvariable=self.var_mat_ro, width=10).pack(side="left", padx=4)
        ttk.Label(matrow, text="(defaults: steel, mm-t-s)").pack(side="left", padx=4)

        self.var_implicit = tk.BooleanVar(value=False)
        ttk.Checkbutton(dyna, text="Write implicit static control cards "
                                   "(*CONTROL_IMPLICIT_..., termination, d3plot)",
                        variable=self.var_implicit).grid(row=5, column=0, columnspan=4, sticky="w")
        self.var_qa = tk.BooleanVar(value=True)
        ttk.Checkbutton(dyna, text="Write element set with quality-criteria failures",
                        variable=self.var_qa).grid(row=6, column=0, columnspan=4, sticky="w")
        self.var_mesh_only = tk.BooleanVar(value=False)
        ttk.Checkbutton(dyna, text="Mesh-only output for *INCLUDE (no PART / "
                                   "SECTION / MAT / control cards)",
                        variable=self.var_mesh_only).grid(row=7, column=0, columnspan=4, sticky="w")
        ttk.Label(dyna, text="Per-body files:").grid(row=11, column=0, sticky="w")
        self.var_split_mode = tk.StringVar(value=next(iter(SPLIT_MODES)))
        ttk.Combobox(dyna, textvariable=self.var_split_mode, state="readonly",
                     width=40, values=list(SPLIT_MODES)).grid(
            row=11, column=1, columnspan=3, sticky="w", padx=4)

        self.var_contact = tk.BooleanVar(value=False)
        ttk.Checkbutton(dyna, text="Contact between parts (*CONTACT_AUTOMATIC_"
                                   "SINGLE_SURFACE), friction:",
                        variable=self.var_contact).grid(row=8, column=0,
                                                        columnspan=2, sticky="w")
        self.var_contact_fs = tk.StringVar(value="0.1")
        ttk.Entry(dyna, textvariable=self.var_contact_fs, width=8
                  ).grid(row=8, column=2, sticky="w", padx=4)
        ttk.Label(dyna, text="(bonded bodies: use Glue instead)",
                  foreground="gray").grid(row=8, column=3, sticky="w")

        self.var_ctrl_dt = tk.BooleanVar(value=False)
        ttk.Checkbutton(dyna, text="Write *CONTROL_TIMESTEP,  TSSFAC:",
                        variable=self.var_ctrl_dt).grid(row=9, column=0, sticky="w")
        self.var_tssfac = tk.StringVar(value="0.9")
        ttk.Entry(dyna, textvariable=self.var_tssfac, width=8
                  ).grid(row=9, column=1, sticky="w", padx=4)

        self.var_grav = tk.BooleanVar(value=False)
        ttk.Checkbutton(dyna, text="Gravity (*LOAD_BODY_):",
                        variable=self.var_grav).grid(row=10, column=0, sticky="w")
        gravrow = ttk.Frame(dyna)
        gravrow.grid(row=10, column=1, columnspan=3, sticky="w")
        ttk.Label(gravrow, text="axis:").pack(side="left")
        self.var_grav_axis = tk.StringVar(value="Z")
        ttk.Combobox(gravrow, textvariable=self.var_grav_axis, state="readonly",
                     width=3, values=["X", "Y", "Z"]).pack(side="left", padx=4)
        ttk.Label(gravrow, text="acceleration:").pack(side="left")
        self.var_grav_a = tk.StringVar(value="9810")
        ttk.Entry(gravrow, textvariable=self.var_grav_a, width=10).pack(side="left", padx=4)
        ttk.Label(gravrow, text="(9810 = g in mm-t-s; positive loads -axis)",
                  foreground="gray").pack(side="left")

        pmat = ttk.LabelFrame(
            tab, text="Per-body materials (multi-body models; overrides the "
                      "global *MAT for that body)", padding=6)
        pmat.grid(row=1, column=0, sticky="ew", **pad)
        pmat.columnconfigure(0, weight=1)
        pmrow = ttk.Frame(pmat)
        pmrow.grid(row=0, column=0, columnspan=2, sticky="w")
        ttk.Label(pmrow, text="Body #:").pack(side="left")
        self.var_pm_body = tk.StringVar(value="1")
        ttk.Entry(pmrow, textvariable=self.var_pm_body, width=5).pack(side="left", padx=2)
        ttk.Label(pmrow, text="E:").pack(side="left", padx=(8, 0))
        self.var_pm_e = tk.StringVar(value="210000")
        ttk.Entry(pmrow, textvariable=self.var_pm_e, width=10).pack(side="left", padx=2)
        ttk.Label(pmrow, text="ν:").pack(side="left", padx=(8, 0))
        self.var_pm_nu = tk.StringVar(value="0.3")
        ttk.Entry(pmrow, textvariable=self.var_pm_nu, width=7).pack(side="left", padx=2)
        ttk.Label(pmrow, text="ρ:").pack(side="left", padx=(8, 0))
        self.var_pm_ro = tk.StringVar(value="7.85e-9")
        ttk.Entry(pmrow, textvariable=self.var_pm_ro, width=10).pack(side="left", padx=2)
        self.var_pm_rigid = tk.BooleanVar(value=False)
        ttk.Checkbutton(pmrow, text="rigid (*MAT_RIGID)",
                        variable=self.var_pm_rigid).pack(side="left", padx=(8, 0))
        ttk.Button(pmrow, text="Add", command=self._add_part_mat).pack(side="left", padx=8)
        self.lst_pmat = tk.Listbox(pmat, height=3)
        self.lst_pmat.grid(row=1, column=0, sticky="ew", pady=3)
        ttk.Button(pmat, text="Remove", command=self._remove_part_mat
                   ).grid(row=1, column=1, sticky="n", pady=3)

        ctrl = ttk.LabelFrame(tab, text="Control & output cards", padding=6)
        ctrl.grid(row=2, column=0, sticky="ew", **pad)
        for c in (1, 3):
            ctrl.columnconfigure(c, weight=1)

        ttk.Label(ctrl, text="Termination time (*CONTROL_TERMINATION):"
                  ).grid(row=0, column=0, sticky="w")
        self.var_endtim = tk.StringVar(value="")
        ttk.Entry(ctrl, textvariable=self.var_endtim, width=12).grid(
            row=0, column=1, sticky="w", padx=4)

        ttk.Label(ctrl, text="Mass scaling (DT2MS):").grid(
            row=0, column=2, sticky="w")
        self.var_mass_scale = tk.StringVar(value="")
        ttk.Entry(ctrl, textvariable=self.var_mass_scale, width=12).grid(
            row=0, column=3, sticky="w", padx=4)
        ttk.Label(ctrl, text="(both blank = off; DT2MS typically negative)",
                  foreground="gray").grid(row=1, column=0, columnspan=4, sticky="w")

        self.var_hourglass = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Write *HOURGLASS",
                        variable=self.var_hourglass).grid(row=2, column=0, sticky="w")
        hgrow = ttk.Frame(ctrl)
        hgrow.grid(row=2, column=1, columnspan=3, sticky="w")
        ttk.Label(hgrow, text="IHQ:").pack(side="left")
        self.var_hg_ihq = tk.StringVar(value="4")
        ttk.Combobox(hgrow, textvariable=self.var_hg_ihq, state="readonly",
                     width=4, values=[str(i) for i in range(1, 7)]).pack(side="left", padx=4)
        ttk.Label(hgrow, text="QM:").pack(side="left")
        self.var_hg_qm = tk.StringVar(value="0.1")
        ttk.Entry(hgrow, textvariable=self.var_hg_qm, width=8).pack(side="left", padx=4)

        self.var_ctrl_energy = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Write *CONTROL_ENERGY",
                        variable=self.var_ctrl_energy).grid(
            row=3, column=0, columnspan=2, sticky="w")

        self.var_db = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Write *DATABASE output,  d3plot dt:",
                        variable=self.var_db).grid(row=4, column=0, sticky="w")
        self.var_db_dt = tk.StringVar(value="1.0")
        ttk.Entry(ctrl, textvariable=self.var_db_dt, width=10).grid(
            row=4, column=1, sticky="w", padx=4)
        dbrow = ttk.Frame(ctrl)
        dbrow.grid(row=5, column=0, columnspan=4, sticky="w")
        ttk.Label(dbrow, text="ASCII (shared dt):").pack(side="left")
        self.var_db_glstat = tk.BooleanVar(value=False)
        ttk.Checkbutton(dbrow, text="GLSTAT", variable=self.var_db_glstat
                        ).pack(side="left", padx=4)
        self.var_db_matsum = tk.BooleanVar(value=False)
        ttk.Checkbutton(dbrow, text="MATSUM", variable=self.var_db_matsum
                        ).pack(side="left", padx=4)
        self.var_db_rcforc = tk.BooleanVar(value=False)
        ttk.Checkbutton(dbrow, text="RCFORC", variable=self.var_db_rcforc
                        ).pack(side="left", padx=4)
        self.var_db_spcforc = tk.BooleanVar(value=False)
        ttk.Checkbutton(dbrow, text="SPCFORC", variable=self.var_db_spcforc
                        ).pack(side="left", padx=4)

        ttk.Label(ctrl, text="Initial velocity (Vx, Vy, Vz):").grid(
            row=6, column=0, sticky="w")
        ivrow = ttk.Frame(ctrl)
        ivrow.grid(row=6, column=1, columnspan=3, sticky="w")
        self.var_ivel_vx = tk.StringVar(value="")
        self.var_ivel_vy = tk.StringVar(value="")
        self.var_ivel_vz = tk.StringVar(value="")
        for v in (self.var_ivel_vx, self.var_ivel_vy, self.var_ivel_vz):
            ttk.Entry(ivrow, textvariable=v, width=8).pack(side="left", padx=2)
        ttk.Label(ivrow, text="(blank = 0; all-zero = off)",
                  foreground="gray").pack(side="left", padx=4)

        ttk.Label(ctrl, text="Contact type:").grid(row=7, column=0, sticky="w")
        self.var_contact_type = tk.StringVar(value=next(iter(CONTACT_TYPES)))
        ttk.Combobox(ctrl, textvariable=self.var_contact_type, state="readonly",
                     width=32, values=list(CONTACT_TYPES)).grid(
            row=7, column=1, columnspan=2, sticky="w", padx=4)
        ttk.Label(ctrl, text="(uses the contact friction set above)",
                  foreground="gray").grid(row=7, column=3, sticky="w")

        conn = ttk.LabelFrame(tab, text="Connections & midsurface", padding=6)
        conn.grid(row=3, column=0, sticky="ew", **pad)
        for c in (1, 3):
            conn.columnconfigure(c, weight=1)

        self.var_midsurface = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            conn, text="Midsurface shell (thin plates) - meshes thin plate "
                       "solids as a mid-surface SHELL; the element type and "
                       "shell thickness above are overridden automatically",
            variable=self.var_midsurface).grid(
            row=0, column=0, columnspan=4, sticky="w")

        ttk.Label(conn, text="Auto-connect bodies:").grid(row=1, column=0, sticky="w")
        self.var_connect_mode = tk.StringVar(value=next(iter(CONNECT_MODES)))
        cmb_conn = ttk.Combobox(conn, textvariable=self.var_connect_mode,
                                state="readonly", width=32,
                                values=list(CONNECT_MODES))
        cmb_conn.grid(row=1, column=1, sticky="w", padx=4)
        cmb_conn.bind("<<ComboboxSelected>>", lambda e: self._sync_connect())
        ttk.Label(conn, text="(detects body interfaces after meshing)",
                  foreground="gray").grid(row=1, column=2, columnspan=2, sticky="w")

        ttk.Label(conn, text="Tolerance:").grid(row=2, column=0, sticky="w")
        self.var_connect_tol = tk.StringVar(value="")
        ttk.Entry(conn, textvariable=self.var_connect_tol, width=10).grid(
            row=2, column=1, sticky="w", padx=4)
        ttk.Label(conn, text="(gap search distance; blank = auto)",
                  foreground="gray").grid(row=2, column=2, columnspan=2, sticky="w")

        ttk.Label(conn, text="Spotweld spacing:").grid(row=3, column=0, sticky="w")
        self.var_connect_spacing = tk.StringVar(value="")
        self.ent_connect_spacing = ttk.Entry(
            conn, textvariable=self.var_connect_spacing, width=10)
        self.ent_connect_spacing.grid(row=3, column=1, sticky="w", padx=4)
        ttk.Label(conn, text="(spotweld only; blank = auto)",
                  foreground="gray").grid(row=3, column=2, columnspan=2, sticky="w")

        ttk.Label(conn, text="Tied friction:").grid(row=4, column=0, sticky="w")
        self.var_connect_fs = tk.StringVar(value="0.0")
        self.ent_connect_fs = ttk.Entry(
            conn, textvariable=self.var_connect_fs, width=10)
        self.ent_connect_fs.grid(row=4, column=1, sticky="w", padx=4)
        ttk.Label(conn, text="(tied contact only; blank = 0)",
                  foreground="gray").grid(row=4, column=2, columnspan=2, sticky="w")

    # ----------------------------------------- tab: symmetry & face sets --
    def _build_sym_tab(self, tab, pad):
        sym = ttk.LabelFrame(
            tab, text="Symmetry (cuts the geometry and meshes the kept half)", padding=6)
        sym.grid(row=0, column=0, sticky="ew", **pad)

        ttk.Label(sym, text="Plane").grid(row=0, column=0, sticky="w")
        ttk.Label(sym, text="Position").grid(row=0, column=2, sticky="w")
        ttk.Label(sym, text="Side to keep").grid(row=0, column=3, sticky="w")

        self.sym_rows = {}
        for r, axis in enumerate(("x", "y", "z"), start=1):
            enabled = tk.BooleanVar(value=False)
            offset = tk.StringVar(value="0.0")
            keep = tk.StringVar(value="+ (coord ≥ position)")
            ttk.Checkbutton(sym, text=f"{axis.upper()} = const  (normal {axis.upper()})",
                            variable=enabled).grid(row=r, column=0, columnspan=2,
                                                   sticky="w", padx=(0, 12))
            ttk.Entry(sym, textvariable=offset, width=12).grid(row=r, column=2,
                                                               sticky="w", padx=4)
            ttk.Combobox(sym, textvariable=keep, state="readonly", width=18,
                         values=["+ (coord ≥ position)", "- (coord ≤ position)"]
                         ).grid(row=r, column=3, sticky="w", padx=4)
            self.sym_rows[axis] = (enabled, offset, keep)

        # --- separate node-set / boundary-condition options for the planes ---
        opts = ttk.Frame(sym)
        opts.grid(row=4, column=0, columnspan=4, sticky="ew", pady=(8, 0))
        self.var_sym_nodeset = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Write node set (*SET_NODE_LIST) for "
                                   "symmetry-plane nodes",
                        variable=self.var_sym_nodeset).grid(
            row=0, column=0, columnspan=3, sticky="w")

        self.var_sym_spc = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Apply boundary condition "
                                   "(*BOUNDARY_SPC_SET):",
                        variable=self.var_sym_spc,
                        command=self._sync_sym_bc).grid(row=1, column=0, sticky="w")
        self.var_sym_constraint = tk.StringVar(value=next(iter(SYM_CONSTRAINTS)))
        self.cmb_sym_constraint = ttk.Combobox(
            opts, textvariable=self.var_sym_constraint, state="readonly",
            width=34, values=list(SYM_CONSTRAINTS))
        self.cmb_sym_constraint.grid(row=1, column=1, sticky="w", padx=4)
        self.cmb_sym_constraint.bind("<<ComboboxSelected>>",
                                     lambda e: self._sync_sym_bc())
        ttk.Label(opts, text="DOFs:").grid(row=1, column=2, sticky="e")
        self.var_sym_dofs = tk.StringVar(value="123456")
        self.ent_sym_dofs = ttk.Entry(opts, textvariable=self.var_sym_dofs, width=9)
        self.ent_sym_dofs.grid(row=1, column=3, sticky="w", padx=2)
        ttk.Label(opts, text="Symmetric BC fixes the normal translation + the "
                             "two in-plane rotations; anti-symmetric is the "
                             "complement.", foreground="gray").grid(
            row=2, column=0, columnspan=4, sticky="w", pady=(2, 0))
        self.var_sym_segset = tk.BooleanVar(value=False)
        ttk.Checkbutton(opts, text="Write segment set (*SET_SEGMENT) for the "
                                   "symmetry-plane faces (solids only)",
                        variable=self.var_sym_segset).grid(
            row=3, column=0, columnspan=4, sticky="w")

        faces = ttk.LabelFrame(tab, text="Faces: sets, BCs, loads, sizes, defeature",
                               padding=6)
        faces.grid(row=1, column=0, sticky="ew", **pad)
        faces.columnconfigure(0, weight=1)

        bar = ttk.Frame(faces)
        bar.grid(row=0, column=0, columnspan=2, sticky="ew")
        self.btn_scan = ttk.Button(bar, text="Scan faces from geometry",
                                   command=self._scan_faces)
        self.btn_scan.pack(side="left")
        self.var_face_nodes = tk.BooleanVar(value=True)
        ttk.Checkbutton(bar, text="node sets", variable=self.var_face_nodes
                        ).pack(side="left", padx=8)
        self.var_face_segs = tk.BooleanVar(value=False)
        ttk.Checkbutton(bar, text="segment sets", variable=self.var_face_segs
                        ).pack(side="left")
        ttk.Label(bar, text="Select small faces ≤").pack(side="left", padx=(16, 2))
        self.var_small = tk.StringVar(value="5.0")
        ttk.Entry(bar, textvariable=self.var_small, width=7).pack(side="left")
        ttk.Button(bar, text="Select", command=self._select_small_faces
                   ).pack(side="left", padx=4)

        cols = ("tag", "type", "name", "area", "centroid")
        self.tree_faces = ttk.Treeview(faces, columns=cols, show="headings",
                                       height=6, selectmode="extended")
        for col, w in zip(cols, (46, 80, 110, 80, 190)):
            self.tree_faces.heading(col, text=col)
            self.tree_faces.column(col, width=w, stretch=(col in ("name", "centroid")))
        self.tree_faces.grid(row=1, column=0, sticky="ew", pady=3)
        sb = ttk.Scrollbar(faces, orient="vertical", command=self.tree_faces.yview)
        self.tree_faces.configure(yscrollcommand=sb.set)
        sb.grid(row=1, column=1, sticky="ns", pady=3)

        rolebar = ttk.Frame(faces)
        rolebar.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        ttk.Label(rolebar, text="Role for selected faces:").pack(side="left")
        self.var_role = tk.StringVar(value=FACE_ROLES[1])
        ttk.Combobox(rolebar, textvariable=self.var_role, state="readonly",
                     width=20, values=FACE_ROLES[1:]).pack(side="left", padx=4)
        ttk.Label(rolebar, text="value:").pack(side="left")
        self.var_role_val = tk.StringVar()
        ttk.Entry(rolebar, textvariable=self.var_role_val, width=9).pack(side="left", padx=2)
        ttk.Label(rolebar, text="SPC DOFs:").pack(side="left", padx=(6, 0))
        self.var_role_dofs = tk.StringVar(value="123456")
        ttk.Entry(rolebar, textvariable=self.var_role_dofs, width=8).pack(side="left", padx=2)
        ttk.Button(rolebar, text="Assign", command=self._assign_role).pack(side="left", padx=6)

        self.lst_roles = tk.Listbox(faces, height=3)
        self.lst_roles.grid(row=3, column=0, sticky="ew", pady=3)
        ttk.Button(faces, text="Remove", command=self._remove_role
                   ).grid(row=3, column=1, sticky="n", pady=3)

        csets = ttk.LabelFrame(
            tab, text="Coordinate node sets (by position - works for "
                      "STL/OBJ/PLY too)", padding=6)
        csets.grid(row=2, column=0, sticky="ew", **pad)
        csets.columnconfigure(1, weight=1)

        self.var_cs_kind = tk.StringVar(value=next(iter(COORD_KINDS)))
        cmb_cs = ttk.Combobox(csets, textvariable=self.var_cs_kind,
                              state="readonly", width=8,
                              values=list(COORD_KINDS))
        cmb_cs.grid(row=0, column=0, sticky="w")
        cmb_cs.bind("<<ComboboxSelected>>", lambda e: self._sync_cs_hint())
        self.var_cs_params = tk.StringVar()
        ttk.Entry(csets, textvariable=self.var_cs_params).grid(
            row=0, column=1, sticky="ew", padx=4)
        self.var_cs_role = tk.StringVar(value=COORD_ROLES[1])
        ttk.Combobox(csets, textvariable=self.var_cs_role, state="readonly",
                     width=10, values=COORD_ROLES).grid(row=0, column=2, padx=2)
        ttk.Label(csets, text="value:").grid(row=0, column=3, sticky="e")
        self.var_cs_val = tk.StringVar()
        ttk.Entry(csets, textvariable=self.var_cs_val, width=9).grid(
            row=0, column=4, padx=2)
        ttk.Label(csets, text="DOFs:").grid(row=0, column=5, sticky="e")
        self.var_cs_dofs = tk.StringVar(value="123456")
        ttk.Entry(csets, textvariable=self.var_cs_dofs, width=8).grid(
            row=0, column=6, padx=2)
        ttk.Button(csets, text="Add", command=self._add_coord_set).grid(
            row=0, column=7, padx=4)
        self.lbl_cs_hint = ttk.Label(csets, foreground="gray")
        self.lbl_cs_hint.grid(row=1, column=0, columnspan=6, sticky="w")
        self._sync_cs_hint()
        self.lst_csets = tk.Listbox(csets, height=3)
        self.lst_csets.grid(row=2, column=0, columnspan=7, sticky="ew", pady=3)
        ttk.Button(csets, text="Remove", command=self._remove_coord_set
                   ).grid(row=2, column=7, sticky="n", pady=3)

    # -------------------------------------------- tab: batch & presets ----
    def _build_batch_tab(self, tab, pad):
        pre = ttk.LabelFrame(tab, text="Parameter presets", padding=6)
        pre.grid(row=0, column=0, sticky="ew", **pad)
        pre.columnconfigure(1, weight=1)
        self.var_preset = tk.StringVar()
        self.cmb_preset = ttk.Combobox(pre, textvariable=self.var_preset,
                                       state="readonly", width=30)
        self.cmb_preset.grid(row=0, column=0, sticky="w", padx=4)
        ttk.Button(pre, text="Apply", command=self._apply_preset).grid(row=0, column=1, sticky="w")
        ttk.Button(pre, text="Save current as...", command=self._save_preset
                   ).grid(row=0, column=2, padx=4)
        ttk.Button(pre, text="Delete", command=self._delete_preset).grid(row=0, column=3)

        bat = ttk.LabelFrame(
            tab, text="Batch queue (jobs snapshot the settings when added; "
                      "face-specific options are cleared)", padding=6)
        bat.grid(row=1, column=0, sticky="ew", **pad)
        bat.columnconfigure(0, weight=1)

        btns = ttk.Frame(bat)
        btns.grid(row=0, column=0, sticky="ew")
        ttk.Button(btns, text="Add CAD/mesh files...", command=self._batch_add
                   ).pack(side="left")
        ttk.Button(btns, text="Remove selected", command=self._batch_remove
                   ).pack(side="left", padx=6)
        ttk.Button(btns, text="Clear", command=self._batch_clear).pack(side="left")
        self.btn_batch = ttk.Button(btns, text="Run queue", command=self._batch_run)
        self.btn_batch.pack(side="left", padx=12)
        self.var_batch_par = tk.BooleanVar(value=False)
        ttk.Checkbutton(btns, text="parallel, workers:",
                        variable=self.var_batch_par).pack(side="left", padx=(12, 2))
        self.var_batch_workers = tk.StringVar(
            value=str(max((os.cpu_count() or 2) // 2, 2)))
        ttk.Entry(btns, textvariable=self.var_batch_workers, width=4
                  ).pack(side="left")

        self.lst_jobs = tk.Listbox(bat, height=6)
        self.lst_jobs.grid(row=1, column=0, sticky="ew", pady=4)

    # -------------------------------------------------- combo sync --------
    def _sync_elforms(self):
        etype = ELEMENT_TYPES.get(self.var_etype.get(), "TET4")
        values = list(ELFORMS_BY_ETYPE[etype])
        self.cmb_elform["values"] = values
        if self.var_elform.get() not in values:
            self.var_elform.set(values[0])
        is_shell = mesher.ETYPES[etype]["family"] == "shell"
        self.ent_thick.config(state="normal" if is_shell else "disabled")

    def _sync_ref_hint(self):
        self.lbl_ref_hint.config(text=REFINE_HINTS[self.var_ref_kind.get()])

    def _sync_sym_bc(self):
        """Enable the constraint combo only when the symmetry SPC is on, and the
        custom-DOFs entry only for the 'Custom DOFs' constraint."""
        spc_on = self.var_sym_spc.get()
        self.cmb_sym_constraint.config(state="readonly" if spc_on else "disabled")
        custom = SYM_CONSTRAINTS.get(self.var_sym_constraint.get()) == "custom"
        self.ent_sym_dofs.config(state="normal" if (spc_on and custom)
                                 else "disabled")

    def _sync_connect(self):
        """Enable the spacing entry only for spotweld mode and the friction
        entry only for tied mode (both ignored when mode is 'none')."""
        mode = CONNECT_MODES.get(self.var_connect_mode.get(), "none")
        self.ent_connect_spacing.config(
            state="normal" if mode == "spotweld" else "disabled")
        self.ent_connect_fs.config(
            state="normal" if mode == "tied" else "disabled")

    # -------------------------------------------------- refinements -------
    def _add_refinement(self):
        kind = self.var_ref_kind.get().lower()
        n_params = 4 if kind == "sphere" else 6
        try:
            params = [float(v) for v in self.var_ref_params.get().replace(";", ",").split(",")]
            size = float(self.var_ref_size.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Refinement parameters and size "
                                                  "must be numbers (comma separated).")
            return
        if len(params) != n_params:
            messagebox.showerror("Invalid input",
                                 f"A {kind} region needs {n_params} parameters: "
                                 f"{REFINE_HINTS[self.var_ref_kind.get()]}")
            return
        if size <= 0 or (kind == "sphere" and params[3] <= 0):
            messagebox.showerror("Invalid input", "Size and radius must be > 0.")
            return
        self.refinements.append({"kind": kind, "params": params, "size": size})
        self._refresh_ref_list()
        self.var_ref_params.set("")
        self.var_ref_size.set("")

    def _remove_refinement(self):
        sel = self.lst_ref.curselection()
        if sel:
            del self.refinements[sel[0]]
            self._refresh_ref_list()

    def _refresh_ref_list(self):
        self.lst_ref.delete(0, "end")
        for r in self.refinements:
            p = ", ".join(f"{v:g}" for v in r["params"])
            self.lst_ref.insert("end", f"{r['kind']}({p})  size = {r['size']:g}")

    # ---------------------------------------------------- face roles ------
    def _assign_role(self):
        tags = [int(i) for i in self.tree_faces.selection()]
        if not tags:
            messagebox.showinfo("Face roles", "Select one or more faces in the "
                                              "table first (scan if empty).")
            return
        role = self.var_role.get()
        value = None
        if role in ("Pressure", "Force X", "Force Y", "Force Z", "Mesh size"):
            try:
                value = float(self.var_role_val.get())
            except ValueError:
                messagebox.showerror("Invalid input", f"{role} needs a numeric value.")
                return
            if role == "Mesh size" and value <= 0:
                messagebox.showerror("Invalid input", "Mesh size must be > 0.")
                return
        dofs = self.var_role_dofs.get().strip()
        if role == "Fix (SPC)":
            if not dofs or any(ch not in "123456" for ch in dofs):
                messagebox.showerror("Invalid input",
                                     "SPC DOFs must be digits 1-6, e.g. 123 or 123456.")
                return
        for tag in tags:
            self.face_roles = [r for r in self.face_roles
                               if not (r["tag"] == tag and r["role"] == role)]
            self.face_roles.append({"tag": tag, "role": role,
                                    "value": value, "dofs": dofs})
        self._refresh_roles_list()

    def _remove_role(self):
        sel = self.lst_roles.curselection()
        if sel:
            del self.face_roles[sel[0]]
            self._refresh_roles_list()

    # -------------------------------------------- coordinate node sets ----
    def _sync_cs_hint(self):
        self.lbl_cs_hint.config(text=COORD_KINDS[self.var_cs_kind.get()][1])

    def _add_coord_set(self):
        kind = COORD_KINDS[self.var_cs_kind.get()][0]
        tokens = [t.strip() for t in
                  self.var_cs_params.get().replace(";", ",").split(",")
                  if t.strip()]
        try:
            if kind == "plane":
                if len(tokens) != 2 or tokens[0].lower() not in ("x", "y", "z"):
                    raise ValueError
                params = (tokens[0].lower(), float(tokens[1]))
            else:
                n = 6 if kind == "box" else 4
                if len(tokens) != n:
                    raise ValueError
                params = [float(t) for t in tokens]
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                f"{self.var_cs_kind.get()}: {COORD_KINDS[self.var_cs_kind.get()][1]}")
            return
        role_label = self.var_cs_role.get()
        cs = {"kind": kind, "params": params, "role": "set"}
        if role_label == "Fix (SPC)":
            dofs = self.var_cs_dofs.get().strip()
            if not dofs or any(ch not in "123456" for ch in dofs):
                messagebox.showerror("Invalid input",
                                     "SPC DOFs must be digits 1-6, e.g. 123.")
                return
            cs.update(role="spc", dofs=dofs)
        elif role_label.startswith("Force"):
            try:
                value = float(self.var_cs_val.get())
            except ValueError:
                messagebox.showerror("Invalid input",
                                     "Force needs a numeric total value.")
                return
            cs.update(role="force", axis=role_label[-1].lower(), value=value)
        self.coord_sets.append(cs)
        self._refresh_cset_list()
        self.var_cs_params.set("")

    def _remove_coord_set(self):
        sel = self.lst_csets.curselection()
        if sel:
            del self.coord_sets[sel[0]]
            self._refresh_cset_list()

    def _refresh_cset_list(self):
        self.lst_csets.delete(0, "end")
        for cs in self.coord_sets:
            if cs["kind"] == "plane":
                p = f"{cs['params'][0]} = {cs['params'][1]:g}"
            else:
                p = ", ".join(f"{v:g}" for v in cs["params"])
            role = {"set": "set only", "spc": f"SPC {cs.get('dofs')}",
                    "force": f"force {cs.get('axis', '').upper()} = "
                             f"{cs.get('value', 0):g}"}[cs["role"]]
            self.lst_csets.insert("end", f"{cs['kind']}({p}): {role}")

    # ---------------------------------------------- per-body materials ----
    def _add_part_mat(self):
        try:
            body = int(self.var_pm_body.get())
            mat = {"e": float(self.var_pm_e.get()),
                   "pr": float(self.var_pm_nu.get()),
                   "ro": float(self.var_pm_ro.get())}
        except ValueError:
            messagebox.showerror("Invalid input", "Per-body material needs a "
                                                  "body number and numeric E, ν, ρ.")
            return
        if body < 1 or mat["e"] <= 0 or mat["ro"] <= 0 or not 0 < mat["pr"] < 0.5:
            messagebox.showerror("Invalid input", "Body # must be ≥ 1, E and ρ "
                                                  "> 0, and 0 < ν < 0.5.")
            return
        if self.var_pm_rigid.get():
            mat["rigid"] = True
        self.part_mats = [pm for pm in self.part_mats if pm["body"] != body]
        self.part_mats.append({"body": body, **mat})
        self.part_mats.sort(key=lambda pm: pm["body"])
        self._refresh_pmat_list()

    def _remove_part_mat(self):
        sel = self.lst_pmat.curselection()
        if sel:
            del self.part_mats[sel[0]]
            self._refresh_pmat_list()

    def _refresh_pmat_list(self):
        self.lst_pmat.delete(0, "end")
        for pm in self.part_mats:
            rigid = "  RIGID" if pm.get("rigid") else ""
            self.lst_pmat.insert("end", f"Body {pm['body']}: E={pm['e']:g}  "
                                        f"ν={pm['pr']:g}  ρ={pm['ro']:g}{rigid}")

    def _refresh_roles_list(self):
        self.lst_roles.delete(0, "end")
        for r in self.face_roles:
            extra = ""
            if r["value"] is not None:
                extra = f" = {r['value']:g}"
            elif r["role"] == "Fix (SPC)":
                extra = f" DOFs {r['dofs']}"
            self.lst_roles.insert("end", f"Face {r['tag']}: {r['role']}{extra}")

    def _select_small_faces(self):
        try:
            v = float(self.var_small.get())
        except ValueError:
            messagebox.showerror("Invalid input", "Small-face size must be a number.")
            return
        hits = [str(t) for t, f in self.faces_by_tag.items()
                if f.get("diag", 1e30) <= v or f.get("area", 1e30) <= v * v]
        self.tree_faces.selection_set(hits)
        if not hits:
            messagebox.showinfo("Small faces", f"No faces with bbox ≤ {v:g} "
                                               f"or area ≤ {v * v:g} found.")

    # ---------------------------------------------------- face scan -------
    def _geometry_signature(self, settings: mesher.MeshSettings) -> str:
        return json.dumps([settings.step_file, settings.heal, settings.glue,
                           settings.occ_unit,
                           sorted((s.axis, s.offset, s.keep) for s in settings.symmetry)])

    def _scan_faces(self):
        if self.worker and self.worker.is_alive():
            return
        try:
            settings, _, _ = self._collect_inputs(need_out=False, include_faces=False)
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return
        # show the post-defeature faces if suppressions are assigned
        settings.defeature_faces = [r["tag"] for r in self.face_roles
                                    if r["role"] == "Suppress (defeature)"]
        self._set_busy(True)
        self.worker = threading.Thread(target=self._scan_worker, args=(settings,),
                                       daemon=True)
        self.worker.start()

    def _scan_worker(self, settings):
        log = self.log_queue.put
        try:
            faces = mesher.list_faces(settings, log=log)
            self.log_queue.put(("__faces__", (faces, self._geometry_signature(settings))))
        except Exception as e:
            log(f"ERROR: {e}")
            self.log_queue.put(("__failed__", None))

    def _show_faces(self, faces, signature):
        self.faces_by_tag = {f["tag"]: f for f in faces}
        self.face_scan_sig = signature
        self.tree_faces.delete(*self.tree_faces.get_children())
        for f in faces:
            c = f["centroid"]
            self.tree_faces.insert(
                "", "end", iid=str(f["tag"]),
                values=(f["tag"], f.get("type", ""), f["name"], f"{f['area']:.4g}",
                        f"({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})"))

    # ------------------------------------------------- presets / batch ----
    def _read_presets(self) -> dict:
        try:
            with open(PRESETS_FILE) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _refresh_presets(self):
        names = sorted(self._read_presets())
        self.cmb_preset["values"] = names
        if self.var_preset.get() not in names:
            self.var_preset.set(names[0] if names else "")

    def _save_preset(self):
        name = simpledialog.askstring("Save preset", "Preset name:", parent=self.root)
        if not name:
            return
        presets = self._read_presets()
        presets[name] = self._collect_settings_data()
        try:
            with open(PRESETS_FILE, "w") as f:
                json.dump(presets, f, indent=2)
        except OSError as e:
            messagebox.showerror("Presets", f"Could not save presets: {e}")
            return
        self._refresh_presets()
        self.var_preset.set(name)

    def _apply_preset(self):
        presets = self._read_presets()
        name = self.var_preset.get()
        if name in presets:
            self._apply_settings_data(presets[name])
            self._sync_elforms()
            self._sync_sym_bc()
            self._sync_connect()
            self._log_write(f"Applied preset '{name}'")

    def _delete_preset(self):
        presets = self._read_presets()
        name = self.var_preset.get()
        if name in presets:
            del presets[name]
            try:
                with open(PRESETS_FILE, "w") as f:
                    json.dump(presets, f, indent=2)
            except OSError:
                pass
            self._refresh_presets()

    def _batch_add(self):
        paths = filedialog.askopenfilenames(
            title="Select CAD/mesh files for the batch queue",
            filetypes=CAD_FILETYPES)
        for path in paths:
            out = os.path.splitext(path)[0] + ".k"
            try:
                settings, _, kopts = self._collect_inputs(
                    need_out=False, include_faces=False, step_override=path)
            except ValueError as e:
                messagebox.showerror("Invalid input", str(e))
                return
            label = (f"{os.path.basename(path)} -> {os.path.basename(out)}   "
                     f"[{settings.element_type}, {settings.size_min:g}.."
                     f"{settings.size_max:g}]")
            self.batch_jobs.append({"settings": settings, "out": out,
                                    "kopts": kopts, "label": label})
        self._refresh_jobs()

    def _batch_remove(self):
        sel = self.lst_jobs.curselection()
        if sel:
            del self.batch_jobs[sel[0]]
            self._refresh_jobs()

    def _batch_clear(self):
        self.batch_jobs.clear()
        self._refresh_jobs()

    def _refresh_jobs(self):
        self.lst_jobs.delete(0, "end")
        for j in self.batch_jobs:
            self.lst_jobs.insert("end", j["label"])

    def _batch_run(self):
        if not self.batch_jobs:
            messagebox.showinfo("Batch", "The queue is empty - add CAD/mesh files first.")
            return
        if self.worker and self.worker.is_alive():
            return
        workers = 1
        if self.var_batch_par.get():
            try:
                workers = max(int(self.var_batch_workers.get()), 1)
            except ValueError:
                messagebox.showerror("Batch", "Workers must be an integer ≥ 1.")
                return
        self._save_settings()
        self._set_busy(True)
        self._log_clear()
        jobs = list(self.batch_jobs)
        self.worker = threading.Thread(target=self._batch_worker,
                                       args=(jobs, workers), daemon=True)
        self.worker.start()

    def _batch_worker(self, jobs, workers=1):
        log = self.log_queue.put
        ok = 0
        last_preview = None
        parallel = workers > 1 and len(jobs) > 1
        if parallel:
            # one worker process per job (gmsh is one-instance-per-process);
            # logs are captured per job and emitted on completion. Workers
            # must be SPAWNED - forking a process whose gmsh/OpenMP runtime
            # has already run deadlocks the child on Linux.
            import multiprocessing
            from concurrent.futures import ProcessPoolExecutor, as_completed
            log(f"===== batch: {len(jobs)} jobs on {workers} worker "
                f"process(es) =====")
            args = [(j["label"], j["settings"], j["out"], j["kopts"])
                    for j in jobs]
            try:
                with ProcessPoolExecutor(
                        max_workers=workers,
                        mp_context=multiprocessing.get_context("spawn")) as pool:
                    futures = [pool.submit(job_runner.run_job_captured, a)
                               for a in args]
                    for n, fut in enumerate(as_completed(futures), 1):
                        label, success, text, preview = fut.result()
                        log(f"===== [{n}/{len(jobs)}] {label} =====")
                        log(text)
                        if success:
                            ok += 1
                            last_preview = preview
            except Exception as e:
                log(f"ERROR: parallel batch failed ({e}) - falling back to "
                    f"sequential execution")
                ok, last_preview, parallel = 0, None, False
        if not parallel:
            for i, j in enumerate(jobs, 1):
                log(f"===== batch job {i}/{len(jobs)}: {j['label']} =====")
                try:
                    last_preview = self._do_job(j["settings"], j["out"],
                                                j["kopts"], log)
                    ok += 1
                except Exception as e:
                    log(f"ERROR in job {i}: {e}")
        log(f"===== batch finished: {ok}/{len(jobs)} jobs succeeded =====")
        self.log_queue.put(("__done__" if ok else "__failed__", last_preview))

    # ------------------------------------------------------------ files ---
    def _browse_step(self) -> None:
        path = filedialog.askopenfilename(
            title="Select CAD/mesh file (STEP/IGES/BREP/STL)",
            filetypes=CAD_FILETYPES)
        if path:
            self.var_step.set(path)
            if not self.var_out.get():
                self.var_out.set(os.path.splitext(path)[0] + ".k")

    def _browse_out(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save LS-DYNA keyword file", defaultextension=".k",
            filetypes=[("LS-DYNA keyword", "*.k *.key *.dyn"), ("All files", "*.*")])
        if path:
            self.var_out.set(path)

    # -------------------------------------------------------- validation --
    def _collect_inputs(self, need_out: bool = True, include_faces: bool = True,
                        step_override: str | None = None):
        step = (step_override or self.var_step.get()).strip()
        out = self.var_out.get().strip()
        if not step or not os.path.isfile(step):
            raise ValueError("Select an existing CAD/mesh input file.")
        if need_out and not out:
            # blank .k output -> default to the input name with a .k extension
            out = os.path.splitext(step)[0] + ".k"

        def num(var, name, minval=None):
            try:
                v = float(var.get())
            except ValueError:
                raise ValueError(f"{name} must be a number.") from None
            if minval is not None and v < minval:
                raise ValueError(f"{name} must be >= {minval:g}.")
            return v

        def integer(var, name, minval=1):
            try:
                v = int(var.get())
            except ValueError:
                raise ValueError(f"{name} must be an integer.") from None
            if v < minval:
                raise ValueError(f"{name} must be >= {minval}.")
            return v

        def opt_num(var, name, minval=None):
            """A number that may be left blank (blank -> None = 'off')."""
            if not var.get().strip():
                return None
            return num(var, name, minval=minval)

        smax = num(self.var_smax, "Max element size")
        if smax <= 0:
            raise ValueError("Max element size must be > 0.")
        smin = num(self.var_smin, "Min element size", minval=0.0)
        if smin > smax:
            raise ValueError("Min element size cannot exceed max element size.")

        symmetry = []
        for axis, (enabled, offset, keep) in self.sym_rows.items():
            if enabled.get():
                symmetry.append(mesher.SymmetryPlane(
                    axis=axis,
                    offset=num(offset, f"{axis.upper()}-plane position"),
                    keep=keep.get().strip()[0],
                ))

        sym_constraint = SYM_CONSTRAINTS.get(self.var_sym_constraint.get(),
                                             "symmetric")
        sym_dofs = self.var_sym_dofs.get().strip()
        if (symmetry and self.var_sym_spc.get() and sym_constraint == "custom"):
            if not sym_dofs or any(ch not in "123456" for ch in sym_dofs):
                raise ValueError("Custom symmetry DOFs must be digits 1-6, "
                                 "e.g. 13 or 123456.")

        # faces: plain set selection + role assignments
        face_tags, face_sizes, defeature, face_roles = [], {}, [], []
        if include_faces:
            selected = [int(i) for i in self.tree_faces.selection()]
            if selected and (self.var_face_nodes.get() or self.var_face_segs.get()):
                face_tags.extend(selected)
            for r in self.face_roles:
                if r["role"] == "Suppress (defeature)":
                    defeature.append(r["tag"])
                elif r["role"] == "Mesh size":
                    face_sizes[r["tag"]] = r["value"]
                else:
                    face_roles.append(r)
                    if r["tag"] not in face_tags:
                        face_tags.append(r["tag"])

        etype = ELEMENT_TYPES.get(self.var_etype.get(), "TET4")
        is_shell = mesher.ETYPES[etype]["family"] == "shell"
        settings = mesher.MeshSettings(
            step_file=step,
            element_type=etype,
            size_max=smax,
            size_min=smin,
            curvature_refine=self.var_curv.get(),
            curvature_elems=integer(self.var_curv_n, "Elements per 2π"),
            algorithm3d=self.var_algo.get(),
            optimize=self.var_opt.get(),
            heal=self.var_heal.get(),
            glue=self.var_glue.get(),
            occ_unit=mesher.OCC_UNITS[self.var_unit.get()],
            symmetry=symmetry,
            refinements=list(self.refinements),
            face_sizes=face_sizes,
            defeature_faces=defeature,
            collect_faces=face_tags,
            auto_refine=self.var_autoref.get(),
        )

        if include_faces and (face_tags or face_sizes or defeature):
            if self.face_scan_sig != self._geometry_signature(settings):
                raise ValueError("Faces are selected/assigned but the geometry "
                                 "settings (file / heal / glue / units / "
                                 "symmetry) changed since the last scan. "
                                 "Scan faces again.")

        mat = None
        if self.var_mat.get():
            mat = {
                "e": num(self.var_mat_e, "Young's modulus E"),
                "pr": num(self.var_mat_nu, "Poisson's ratio"),
                "ro": num(self.var_mat_ro, "Density"),
            }
            if not 0 < mat["pr"] < 0.5:
                raise ValueError("Poisson's ratio must be between 0 and 0.5.")

        thickness = 1.0
        if is_shell:
            thickness = num(self.var_thick, "Shell thickness")
            if thickness <= 0:
                raise ValueError("Shell thickness must be > 0.")

        # Contact: the default single-surface kind keeps driving contact_fs (as
        # before); any other kind emits a *CONTACT via kopts["contacts"].
        contact_fs = None
        contacts: tuple = ()
        if self.var_contact.get():
            fs = num(self.var_contact_fs, "Contact friction", minval=0.0)
            ctype = CONTACT_TYPES.get(self.var_contact_type.get(),
                                      "automatic_single_surface")
            if ctype == "automatic_single_surface":
                contact_fs = fs
            else:
                contacts = ({"type": ctype, "fs": fs},)

        # --- midsurface + auto-connect (job_runner reads these with .get
        # defaults; leaving both at their defaults keeps the old behaviour) ---
        midsurface = self.var_midsurface.get()
        connect_mode = CONNECT_MODES.get(self.var_connect_mode.get(), "none")
        auto_connect = None
        if connect_mode != "none":
            auto_connect = {
                "mode": connect_mode,
                "tol": opt_num(self.var_connect_tol, "Connection tolerance",
                               minval=0.0),
                "spacing": opt_num(self.var_connect_spacing, "Spotweld spacing",
                                   minval=0.0),
                "fs": opt_num(self.var_connect_fs, "Tied contact friction",
                              minval=0.0) or 0.0,
            }

        tssfac = None
        if self.var_ctrl_dt.get():
            tssfac = num(self.var_tssfac, "TSSFAC")
            if not 0 < tssfac <= 1.0:
                raise ValueError("TSSFAC must be in (0, 1].")
        gravity = None
        if self.var_grav.get():
            gravity = (self.var_grav_axis.get().lower(),
                       num(self.var_grav_a, "Gravity acceleration"))

        # --- extra control / output cards (all optional; blank/unchecked = off)
        endtim = opt_num(self.var_endtim, "Termination time", minval=0.0)
        mass_scale = opt_num(self.var_mass_scale, "Mass scaling (DT2MS)")

        hourglass = None
        if self.var_hourglass.get():
            try:
                ihq = int(self.var_hg_ihq.get())
            except ValueError:
                raise ValueError("Hourglass IHQ must be an integer.") from None
            hourglass = {"ihq": ihq,
                         "qm": num(self.var_hg_qm, "Hourglass QM", minval=0.0)}

        control_energy = self.var_ctrl_energy.get()

        databases = None
        if self.var_db.get():
            d3dt = num(self.var_db_dt, "d3plot dt")
            if d3dt <= 0:
                raise ValueError("d3plot dt must be > 0.")
            ascii_out = {}
            for aname, avar in (("GLSTAT", self.var_db_glstat),
                                ("MATSUM", self.var_db_matsum),
                                ("RCFORC", self.var_db_rcforc),
                                ("SPCFORC", self.var_db_spcforc)):
                if avar.get():
                    ascii_out[aname] = d3dt
            databases = {"d3plot_dt": d3dt, "ascii": ascii_out}

        vel = [opt_num(v, name) or 0.0 for v, name in (
            (self.var_ivel_vx, "Initial velocity Vx"),
            (self.var_ivel_vy, "Initial velocity Vy"),
            (self.var_ivel_vz, "Initial velocity Vz"))]
        initial_velocity = None
        if any(vel):
            initial_velocity = {"vx": vel[0], "vy": vel[1], "vz": vel[2]}

        pid0 = integer(self.var_pid, "Part ID")
        part_mats = {}
        for pm in self.part_mats:
            m = {"e": pm["e"], "pr": pm["pr"], "ro": pm["ro"]}
            if pm.get("rigid"):
                m["rigid"] = True
            part_mats[pid0 + pm["body"] - 1] = m

        kopts = {
            "pid": pid0,
            "elform": ELFORMS_ALL[self.var_elform.get()],
            "element_kind": "shell" if is_shell else "solid",
            "thickness": thickness,
            "start_nid": integer(self.var_nid0, "Start node ID"),
            "start_eid": integer(self.var_eid0, "Start element ID"),
            "start_sid": integer(self.var_sid0, "Start set ID"),
            "sym_nodeset": self.var_sym_nodeset.get(),
            "sym_spc": self.var_sym_spc.get(),
            "sym_constraint": sym_constraint,
            "sym_dofs": sym_dofs,
            "sym_segset": self.var_sym_segset.get(),
            "mat": mat,
            "face_nodesets": self.var_face_nodes.get(),
            "face_segsets": self.var_face_segs.get(),
            "face_roles": face_roles,
            "plain_faces": [int(i) for i in self.tree_faces.selection()]
                           if include_faces else [],
            "implicit_cards": self.var_implicit.get(),
            "qa_sets": self.var_qa.get(),
            "mesh_only": self.var_mesh_only.get(),
            "split_mode": SPLIT_MODES.get(self.var_split_mode.get()),
            "contact_fs": contact_fs,
            "contacts": contacts,
            "midsurface": midsurface,
            "auto_connect": auto_connect,
            "tssfac": tssfac,
            "gravity": gravity,
            "endtim": endtim,
            "mass_scale": mass_scale,
            "hourglass": hourglass,
            "control_energy": control_energy,
            "databases": databases,
            "initial_velocity": initial_velocity,
            "part_mats": part_mats,
            "coord_sets": [dict(cs) for cs in self.coord_sets],
            "face_titles": {t: f.get("name", "")
                            for t, f in self.faces_by_tag.items()},
        }
        return settings, out, kopts

    # --------------------------------------------------------------- run --
    def _set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.btn_run.config(state=state)
        self.btn_scan.config(state=state)
        self.btn_batch.config(state=state)
        if busy:
            self.btn_preview.config(state="disabled")
            self.progress.pack(side="left", padx=12)
            self.progress.start(12)
        else:
            self.progress.stop()
            self.progress.pack_forget()

    def _start_run(self) -> None:
        if self.worker and self.worker.is_alive():
            return
        try:
            settings, out, kopts = self._collect_inputs()
        except ValueError as e:
            messagebox.showerror("Invalid input", str(e))
            return

        self._save_settings()
        self._set_busy(True)
        self._log_clear()
        self.worker = threading.Thread(
            target=self._worker_main, args=(settings, out, kopts), daemon=True)
        self.worker.start()

    def _worker_main(self, settings, out, kopts) -> None:
        log = self.log_queue.put
        try:
            preview = self._do_job(settings, out, kopts, log)
            self.log_queue.put(("__done__", preview))
        except Exception as e:
            log(f"ERROR: {e}")
            log(traceback.format_exc())
            self.log_queue.put(("__failed__", None))

    def _do_job(self, settings, out, kopts, log) -> str:
        """Mesh + write one job (delegated to the GUI-independent runner).
        Returns the preview file path."""
        return job_runner.run_job(settings, out, kopts, log=log)

    # ----------------------------------------------------------- preview --
    def _open_preview(self) -> None:
        if not self.last_preview or not os.path.isfile(self.last_preview):
            messagebox.showinfo("Preview", "No mesh available to preview yet.")
            return
        script = os.path.join(_HERE, "preview.py")
        subprocess.Popen([sys.executable, script, self.last_preview])

    # -------------------------------------------------- settings file -----
    def _settings_vars(self) -> dict:
        d = {
            "step": self.var_step, "out": self.var_out,
            "smax": self.var_smax, "smin": self.var_smin,
            "curv": self.var_curv, "curv_n": self.var_curv_n,
            "algo": self.var_algo, "unit": self.var_unit,
            "opt": self.var_opt, "heal": self.var_heal, "glue": self.var_glue,
            "autoref": self.var_autoref,
            "etype": self.var_etype, "elform": self.var_elform,
            "pid": self.var_pid, "nid0": self.var_nid0, "eid0": self.var_eid0,
            "sid0": self.var_sid0, "thick": self.var_thick,
            "mat": self.var_mat, "mat_e": self.var_mat_e,
            "mat_nu": self.var_mat_nu, "mat_ro": self.var_mat_ro,
            "sym_nodeset": self.var_sym_nodeset, "sym_spc": self.var_sym_spc,
            "sym_constraint": self.var_sym_constraint,
            "sym_dofs": self.var_sym_dofs, "sym_segset": self.var_sym_segset,
            "preview": self.var_preview,
            "face_nodes": self.var_face_nodes, "face_segs": self.var_face_segs,
            "implicit": self.var_implicit, "qa": self.var_qa,
            "mesh_only": self.var_mesh_only, "split_mode": self.var_split_mode,
            "contact": self.var_contact, "contact_fs": self.var_contact_fs,
            "contact_type": self.var_contact_type,
            "midsurface": self.var_midsurface,
            "connect_mode": self.var_connect_mode,
            "connect_tol": self.var_connect_tol,
            "connect_spacing": self.var_connect_spacing,
            "connect_fs": self.var_connect_fs,
            "ctrl_dt": self.var_ctrl_dt, "tssfac": self.var_tssfac,
            "grav": self.var_grav, "grav_axis": self.var_grav_axis,
            "grav_a": self.var_grav_a,
            "endtim": self.var_endtim, "mass_scale": self.var_mass_scale,
            "hourglass": self.var_hourglass, "hg_ihq": self.var_hg_ihq,
            "hg_qm": self.var_hg_qm, "ctrl_energy": self.var_ctrl_energy,
            "db": self.var_db, "db_dt": self.var_db_dt,
            "db_glstat": self.var_db_glstat, "db_matsum": self.var_db_matsum,
            "db_rcforc": self.var_db_rcforc, "db_spcforc": self.var_db_spcforc,
            "ivel_vx": self.var_ivel_vx, "ivel_vy": self.var_ivel_vy,
            "ivel_vz": self.var_ivel_vz,
            "batch_par": self.var_batch_par,
            "batch_workers": self.var_batch_workers,
        }
        for axis, (enabled, offset, keep) in self.sym_rows.items():
            d[f"sym_{axis}"] = enabled
            d[f"sym_{axis}_off"] = offset
            d[f"sym_{axis}_keep"] = keep
        return d

    def _collect_settings_data(self) -> dict:
        data = {k: v.get() for k, v in self._settings_vars().items()}
        data["refinements"] = self.refinements
        data["part_mats"] = self.part_mats
        data["coord_sets"] = self.coord_sets
        return data

    def _apply_settings_data(self, data: dict) -> None:
        for k, v in self._settings_vars().items():
            if k in data:
                try:
                    v.set(data[k])
                except tk.TclError:
                    pass
        refs = data.get("refinements", [])
        if isinstance(refs, list):
            self.refinements = [r for r in refs
                                if isinstance(r, dict)
                                and r.get("kind") in ("sphere", "box")]
            self._refresh_ref_list()
        pms = data.get("part_mats", [])
        if isinstance(pms, list):
            self.part_mats = [pm for pm in pms if isinstance(pm, dict)
                              and {"body", "e", "pr", "ro"} <= set(pm)]
            self._refresh_pmat_list()
        css = data.get("coord_sets", [])
        if isinstance(css, list):
            self.coord_sets = [cs for cs in css if isinstance(cs, dict)
                               and cs.get("kind") in ("plane", "box", "sphere")
                               and "params" in cs]
            self._refresh_cset_list()

    def _save_settings(self) -> None:
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(self._collect_settings_data(), f, indent=2)
        except OSError:
            pass

    def _load_settings(self) -> None:
        try:
            with open(SETTINGS_FILE) as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        self._apply_settings_data(data)

    def _on_close(self) -> None:
        self._save_settings()
        self.root.destroy()

    # --------------------------------------------------------------- log --
    def _log_clear(self) -> None:
        self.log.config(state="normal")
        self.log.delete("1.0", "end")
        self.log.config(state="disabled")

    def _log_write(self, text: str) -> None:
        self.log.config(state="normal")
        self.log.insert("end", text + "\n")
        self.log.see("end")
        self.log.config(state="disabled")

    def _poll_log(self) -> None:
        try:
            while True:
                msg = self.log_queue.get_nowait()
                if isinstance(msg, tuple):
                    kind, payload = msg
                    self._set_busy(False)
                    if kind == "__done__":
                        self.last_preview = payload
                        self.btn_preview.config(state="normal")
                        if self.var_preview.get():
                            self._open_preview()
                    elif kind == "__faces__":
                        self._show_faces(*payload)
                        if self.last_preview:
                            self.btn_preview.config(state="normal")
                    elif kind == "__failed__" and self.last_preview:
                        self.btn_preview.config(state="normal")
                else:
                    self._log_write(str(msg))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_log)


def main() -> None:
    root = tk.Tk()
    try:
        ttk.Style().theme_use("vista")
    except tk.TclError:
        pass
    KMesherGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
