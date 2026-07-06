"""tkinter GUI for k_mesher: STEP -> TET4/TET10/TRI3/QUAD4 -> LS-DYNA .k"""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk

import numpy as np
from tkinter import filedialog, messagebox, simpledialog, ttk
from tkinter.scrolledtext import ScrolledText

import dyna_writer
import mesher

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

        # -------------------------------------------------------- log -----
        logf = ttk.LabelFrame(body, text="Log", padding=4)
        logf.grid(row=3, column=0, sticky="nsew", **pad)
        body.rowconfigure(3, weight=1)
        self.log = ScrolledText(logf, height=12, state="disabled", wrap="word")
        self.log.pack(fill="both", expand=True)

        self._load_settings()
        self._sync_elforms()
        self._sync_sym_bc()
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
        self._save_settings()
        self._set_busy(True)
        self._log_clear()
        jobs = list(self.batch_jobs)
        self.worker = threading.Thread(target=self._batch_worker, args=(jobs,),
                                       daemon=True)
        self.worker.start()

    def _batch_worker(self, jobs):
        log = self.log_queue.put
        ok = 0
        last_preview = None
        for i, j in enumerate(jobs, 1):
            log(f"===== batch job {i}/{len(jobs)}: {j['label']} =====")
            try:
                last_preview = self._do_job(j["settings"], j["out"], j["kopts"], log)
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
            raise ValueError("Select an output .k file path.")

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

        kopts = {
            "pid": integer(self.var_pid, "Part ID"),
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
        """Mesh + write one job. Returns the preview file path."""
        preview_path = os.path.splitext(out)[0] + "_preview.msh"
        t_start = time.perf_counter()

        if settings.defeature_faces and (kopts["face_roles"]
                                         or settings.face_sizes
                                         or kopts["plain_faces"]):
            log("NOTE: defeaturing changes face tags - verify that the other "
                "face assignments still point at the intended faces "
                "(rescan shows the post-defeature tags).")

        result = mesher.mesh_step_auto(settings, log=log, preview_path=preview_path)

        sym_sets = []
        if kopts["sym_nodeset"] or kopts["sym_spc"]:
            for sp in settings.symmetry:
                item = {"axis": sp.axis, "offset": sp.offset,
                        "nodes": result.sym_nodes[sp.axis],
                        "spc": kopts["sym_spc"],
                        "constraint": kopts["sym_constraint"]}
                if kopts["sym_constraint"] == "custom":
                    item["dofs"] = kopts["sym_dofs"]
                if kopts["sym_segset"] and len(result.sym_segs.get(sp.axis, ())) > 0:
                    item["segset"] = result.sym_segs[sp.axis]
                sym_sets.append(item)

        face_sets = []
        for tag in kopts["plain_faces"]:
            info = self.faces_by_tag.get(tag, {})
            base = f"FACE_{tag}" + (f" {info['name']}" if info.get("name") else "")
            if kopts["face_nodesets"] and tag in result.face_nodes:
                face_sets.append({"kind": "node", "title": base,
                                  "nodes": result.face_nodes[tag]})
            if kopts["face_segsets"] and len(result.face_segs.get(tag, ())) > 0:
                face_sets.append({"kind": "segment", "title": base,
                                  "segments": result.face_segs[tag]})
        for r in kopts["face_roles"]:
            tag = r["tag"]
            if r["role"] == "Fix (SPC)" and tag in result.face_nodes:
                face_sets.append({"kind": "node", "title": f"SPC_FACE_{tag}",
                                  "nodes": result.face_nodes[tag],
                                  "spc_dofs": r["dofs"]})
            elif r["role"] == "Pressure" and len(result.face_segs.get(tag, ())) > 0:
                face_sets.append({"kind": "segment", "title": f"PRES_FACE_{tag}",
                                  "segments": result.face_segs[tag],
                                  "pressure": r["value"]})
            elif r["role"].startswith("Force") and tag in result.face_nodes:
                axis = r["role"][-1].lower()
                face_sets.append({"kind": "node", "title": f"FORCE_FACE_{tag}",
                                  "nodes": result.face_nodes[tag],
                                  "force": (axis, r["value"])})

        elem_sets = []
        failed = result.stats.get("failed_elems", ())
        if kopts["qa_sets"] and len(failed):
            elem_sets.append({"title": "QA quality-criteria failures",
                              "eids": (failed + 1)})

        pid0 = kopts["pid"]
        part_ids = pid0 + result.elem_parts
        part_titles = {pid0 + i: (name or f"body {i + 1}")
                       for i, name in enumerate(result.part_names)}
        if len(result.part_names) > 1:
            for p, t in part_titles.items():
                log(f"  PID {p}: {t}")

        if kopts["mat"]:
            scale = kopts["mat"]["ro"] * (kopts["thickness"]
                                          if kopts["element_kind"] == "shell" else 1.0)
            mass = scale * result.stats["measure"]
            c = result.stats.get("cog", (0, 0, 0))
            inertia = np.array(result.stats["inertia_unit_density"]) * scale
            log(f"Mass: {mass:.6g}   COG: ({c[0]:.4g}, {c[1]:.4g}, {c[2]:.4g})")
            log(f"Inertia about COG (Ixx, Iyy, Izz): "
                f"{inertia[0, 0]:.6g}, {inertia[1, 1]:.6g}, {inertia[2, 2]:.6g}")

        comments = [f"Source geometry: {settings.step_file}",
                    f"Element size: {settings.size_min:g} .. {settings.size_max:g}"]
        for sp in settings.symmetry:
            comments.append(f"Symmetry: {sp.axis.upper()} = {sp.offset:g}, "
                            f"kept '{sp.keep}' side")

        log(f"Writing LS-DYNA keyword file: {out}")
        dyna_writer.write_k(
            out, result.coords, result.elems,
            element_kind=kopts["element_kind"],
            pid=pid0, elform=kopts["elform"], thickness=kopts["thickness"],
            start_nid=kopts["start_nid"], start_eid=kopts["start_eid"],
            start_sid=kopts["start_sid"],
            title=os.path.splitext(os.path.basename(out))[0],
            comments=tuple(comments), sym_sets=tuple(sym_sets),
            mat=kopts["mat"], part_ids=part_ids, part_titles=part_titles,
            face_sets=tuple(face_sets), elem_sets=tuple(elem_sets),
            implicit_cards=kopts["implicit_cards"],
        )
        log(f"Done in {time.perf_counter() - t_start:.1f} s. "
            f"{result.stats['n_nodes']} nodes / "
            f"{result.stats['n_elems']} {settings.element_type} "
            f"elements written.")
        return preview_path

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
        }
        for axis, (enabled, offset, keep) in self.sym_rows.items():
            d[f"sym_{axis}"] = enabled
            d[f"sym_{axis}_off"] = offset
            d[f"sym_{axis}_keep"] = keep
        return d

    def _collect_settings_data(self) -> dict:
        data = {k: v.get() for k, v in self._settings_vars().items()}
        data["refinements"] = self.refinements
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
