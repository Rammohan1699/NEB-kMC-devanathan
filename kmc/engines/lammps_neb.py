from __future__ import annotations

from typing import Any, Callable, Dict, Optional, List, Tuple

import json
import os
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms  # type: ignore
from ase.mep import NEB  # type: ignore
from ase.optimize import BFGS  # type: ignore
import time

try:  # optional typing only
    from .neb_engine import INEBEngine  # type: ignore
except Exception:  # pragma: no cover
    INEBEngine = object  # fallback for runtime


def _noop(*_a, **_k) -> None:
    """Fallback logger."""
    return None


_FALSES = {"0", "false", "False", "FALSE", "no", "No", "NO", "off", "OFF"}


class LammpsNEB(INEBEngine):  # type: ignore[misc]
    """
    Concrete NEB engine that mirrors the legacy NEB workflow from cache_new.py.

    Parameters
    ----------
    calc_pool : Sequence[Calculator]
        ASE calculators used round-robin across images.
    helpers : dict | None
        Optional helper callables/flags, e.g. {
            "assign_mass_and_type": callable(Atoms) -> None,
            "log": callable(str, **kwargs) -> None,
            "run_opt": callable(Atoms, str, int, Any) -> None,
            "optimize_endpoints": bool,
            "optimizer_factory": callable(NEB) -> Optimizer,
            "optimizer_kwargs": dict,
        }
    """

    def __init__(self, calc_pool, helpers: Optional[Dict[str, Any]] = None):
        self.calc_pool = calc_pool
        self.helpers = helpers or {}

        self._assign_mass_and_type: Callable[[Atoms], Any] = self.helpers.get("assign_mass_and_type") or (lambda atoms: None)
        self._log: Callable[..., None] = self.helpers.get("log") or _noop
        self._dump: Callable[..., None] = self.helpers.get("dump") or _noop
        self._failure_handler: Callable[[Dict[str, Any]], None] = self.helpers.get("failure_handler") or _noop
        self._run_opt: Optional[Callable[..., Any]] = self.helpers.get("run_opt")  # type: ignore
        self._optimize_endpoints: bool = bool(self.helpers.get("optimize_endpoints", False))
        self._make_calculator: Optional[Callable[[], Any]] = self.helpers.get("make_calculator")  # type: ignore

        self._optimizer_factory: Callable[[NEB], Any] = self.helpers.get("optimizer_factory") or self._default_optimizer_factory  # type: ignore
        self._optimizer_run_kwargs: Dict[str, Any] = dict(self.helpers.get("optimizer_kwargs", {}))
        if not self._optimizer_run_kwargs:
            self._optimizer_run_kwargs = {"fmax": 0.05, "steps": 300}

    def _reset_calculator(self, idx: int, rank: int) -> None:
        if self._make_calculator is None:
            return
        if not (0 <= idx < len(self.calc_pool)):
            return
        old_calc = self.calc_pool[idx]
        try:
            close = getattr(old_calc, "close", None)
            if callable(close):
                close()
        except Exception:
            pass
        try:
            self.calc_pool[idx] = self._make_calculator()
            self._log(f"[Rank {rank}] Reset LAMMPS calculator pool slot {idx} after failure.")
        except Exception as exc:
            self._log(f"[Rank {rank}] Failed to reset LAMMPS calculator pool slot {idx}: {exc}")

    @staticmethod
    def _positions_finite(atoms: Atoms) -> bool:
        try:
            return bool(np.isfinite(atoms.get_positions()).all())
        except Exception:
            return False

    def _build_constraint_mask(
        self,
        atoms: Atoms,
        *,
        mover_idx: Optional[int],
    ) -> tuple[np.ndarray, Dict[str, Any]]:
        fix_sublattice_overrides = self.helpers.get("fix_sublattice") or {}
        fix_H_override = fix_sublattice_overrides.get("H")
        fix_Fe_override = fix_sublattice_overrides.get("Fe")

        # Default to unconstrained NEB unless explicitly requested via env or overrides.
        fix_fe = os.getenv("FIX_FE", "0") not in _FALSES
        fix_other_h = os.getenv("FIX_OTHER_H", "0") not in _FALSES
        if fix_H_override is not None:
            fix_other_h = bool(fix_H_override)
        if fix_Fe_override is not None:
            fix_fe = bool(fix_Fe_override)

        atomic_numbers = atoms.get_atomic_numbers()
        is_H = atomic_numbers == 1
        is_Fe = atomic_numbers == 26
        mask = np.zeros(len(atomic_numbers), dtype=bool)

        if fix_other_h:
            mask = np.logical_or(mask, is_H)
            if mover_idx is not None and 0 <= mover_idx < len(mask) and is_H[mover_idx]:
                mask[mover_idx] = False

        if fix_fe:
            mask = np.logical_or(mask, is_Fe)

        shell_mask_raw = atoms.arrays.get("freeze_mask")
        shell_mask = np.zeros(len(atomic_numbers), dtype=bool)
        if shell_mask_raw is not None:
            shell_mask = np.asarray(shell_mask_raw, dtype=bool).copy()
            if shell_mask.shape != mask.shape:
                raise ValueError(
                    f"freeze_mask shape mismatch: got {shell_mask.shape}, expected {mask.shape}"
                )
            if mover_idx is not None and 0 <= mover_idx < len(shell_mask):
                shell_mask[mover_idx] = False
            mask = np.logical_or(mask, shell_mask)

        return mask, {
            "fix_other_h": fix_other_h,
            "fix_fe": fix_fe,
            "fix_H_override": fix_H_override,
            "fix_Fe_override": fix_Fe_override,
            "is_H": is_H,
            "is_Fe": is_Fe,
            "shell_mask": shell_mask,
        }

    def _apply_constraints(self, atoms_list: List[Atoms], *, mask: np.ndarray) -> None:
        for atoms in atoms_list:
            atoms.set_constraint(FixAtoms(mask=mask.copy()))

    def _default_optimizer_factory(  # type: ignore[override]
        self,
        neb: NEB,
        *,
        optimizer_log_path: str | None = None,
        logfile: str | None = None,
    ) -> Any:
        log_path = optimizer_log_path or logfile
        if not log_path:
            log_path = os.devnull
        kwargs: Dict[str, Any] = {}
        if log_path is not None:
            kwargs["logfile"] = log_path
        return BFGS(neb, **kwargs)

    def _write_job_context_header(self, job_context: Optional[Dict[str, Any]], rank: int) -> List[str]:
        if not job_context:
            return []
        summary_labels = ("step", "h", "n", "src_rank", "key")
        summary_parts = [
            f"{label}={job_context.get(label)}"
            for label in summary_labels
            if label in job_context and job_context.get(label) is not None
        ]
        summary = " | ".join(summary_parts) if summary_parts else "NEB job context"
        try:
            context_payload = json.dumps(job_context, ensure_ascii=False)
        except Exception:
            context_payload = str(job_context)

        max_print_length = 384
        printable_context = context_payload.strip().replace("\n", " ")
        if len(printable_context) > max_print_length:
            printable_context = printable_context[: max_print_length - 3] + "..."

        header_lines = [
            "",
            "=" * 78,
            f"NEB JOB CONTEXT (rank={rank})",
            summary,
            printable_context,
            "=" * 78,
        ]
        header_text = "\n".join(header_lines) + "\n"

        for calc in self.calc_pool:
            # Avoid writing multi-line/JSON context through LAMMPS `print`;
            # LAMMPS parsing is fragile with quotes. Keep context in Python logs/files.
            log_path = getattr(calc, "_codex_log_path", None) or getattr(calc, "log_file", None)
            if not log_path:
                continue
            try:
                with open(log_path, "a", encoding="utf-8") as fh:
                    fh.write(header_text)
            except Exception as exc:
                self._log(f"[Rank {rank}] Failed to write job context to {log_path}: {exc}")

        return header_lines

    def _append_header_to_log(self, path: str, header_lines: List[str], rank: int) -> None:
        if not path or not header_lines:
            return
        try:
            with open(path, "a", encoding="utf-8") as fh:
                fh.write("\n")
                fh.write("\n".join(header_lines))
                fh.write("\n")
        except Exception as exc:
            self._log(f"[Rank {rank}] Failed to append job context to {path}: {exc}")

    def _print_header_via_lammps(
        self, calc: Any, header_lines: List[str], rank: int
    ) -> bool:
        lmp = getattr(calc, "lmp", None) or getattr(calc, "_lmp", None)
        if lmp is None:
            return False
        cmd = getattr(lmp, "command", None)
        if cmd is None:
            return False
        try:
            for line in header_lines:
                safe_line = (
                    line.replace("\\", "\\\\")
                    .replace('"', '\\"')
                    .replace("\r", "\\r")
                    .replace("\n", "\\n")
                )
                cmd(f'print "{safe_line}"')
        except Exception as exc:
            self._log(f"[Rank {rank}] NEB job context print failed: {exc}")
            return False
        return True

    def barrier(
        self,
        initial: Atoms,
        final: Atoms,
        *,
        rank: int,
        return_images: bool = False,
        collect_diagnostics: bool = False,
        mover_idx: Optional[int] = None,
        auto_detect_mover: bool = True,
        neb_dump_mode: bool = False,
        expected_h_count: Optional[int] = None,
        **_kwargs: Any,
    ) -> Any:
        """
        Run NEB between `initial` and `final`. Returns barrier (eV) or a tuple
        with diagnostics/images, matching the legacy behaviour.
        """
        type_map = {"Fe": 1, "H": 2}
        timing_out = _kwargs.get("timing_out")
        stage_timings: Dict[str, float] = {
            "header_s": 0.0,
            "endpoint_constraints_s": 0.0,
            "endpoint_opt_s": 0.0,
            "image_setup_s": 0.0,
            "image_constraints_s": 0.0,
            "interpolate_s": 0.0,
            "optimizer_s": 0.0,
            "energy_eval_s": 0.0,
            "failure_cleanup_s": 0.0,
            "total_s": 0.0,
        }
        barrier_t0 = time.perf_counter()

        def _finalize_timings() -> None:
            stage_timings["total_s"] = time.perf_counter() - barrier_t0
            if isinstance(timing_out, dict):
                timing_out.clear()
                timing_out.update(stage_timings)

        for atoms in (initial, final):
            types = [type_map[symbol] for symbol in atoms.get_chemical_symbols()]
            atoms.set_array("type", np.array(types))
            atoms.set_array("id", np.arange(1, len(atoms) + 1))
        job_context = _kwargs.get("job_context")
        header_lines: List[str] = []
        try:
            t0 = time.perf_counter()
            header_lines = self._write_job_context_header(job_context, rank)
            stage_timings["header_s"] += time.perf_counter() - t0
        except Exception as exc:
            self._log(f"[Rank {rank}] Failed to write NEB job context header: {exc}")

        optimizer_log_path = _kwargs.get("optimizer_log_path")
        if optimizer_log_path and optimizer_log_path != os.devnull:
            self._append_header_to_log(optimizer_log_path, header_lines, rank)

        if len(initial) == 0 or len(final) == 0:
            self._log(f"[Rank {rank}] ERROR: Empty Atoms passed to NEB")
            _finalize_timings()
            return (float("inf"), None) if (return_images or collect_diagnostics) else float("inf")

        init_pre = initial.copy()
        finl_pre = final.copy()
        t0 = time.perf_counter()
        endpoint_mask, _endpoint_constraint_info = self._build_constraint_mask(
            initial, mover_idx=mover_idx
        )
        if bool(endpoint_mask.any()):
            self._apply_constraints([initial, final], mask=endpoint_mask)
        stage_timings["endpoint_constraints_s"] += time.perf_counter() - t0

        if self._optimize_endpoints and self._run_opt is not None:
            try:
                t0 = time.perf_counter()
                endpoint_calc = self.calc_pool[0]
                opt_initial_ok = bool(
                    self._run_opt(initial, f"opt_initial_{rank}.log", rank, endpoint_calc, job_context=job_context)
                )
                opt_final_ok = bool(
                    self._run_opt(final, f"opt_final_{rank}.log", rank, endpoint_calc, job_context=job_context)
                )
                if (
                    (not opt_initial_ok)
                    or (not opt_final_ok)
                    or (not self._positions_finite(initial))
                    or (not self._positions_finite(final))
                ):
                    self._log(f"[Rank {rank}] Endpoint optimization produced an unstable structure; skipping hop.")
                    self._reset_calculator(0, rank)
                    stage_timings["endpoint_opt_s"] += time.perf_counter() - t0
                    _finalize_timings()
                    return (float("inf"), None) if (return_images or collect_diagnostics) else float("inf")
                stage_timings["endpoint_opt_s"] += time.perf_counter() - t0
            except Exception as exc:
                self._log(f"[Rank {rank}] Endpoint optimization error: {exc}")
                self._reset_calculator(0, rank)
                stage_timings["endpoint_opt_s"] += time.perf_counter() - t0
                _finalize_timings()
                return (float("inf"), None) if (return_images or collect_diagnostics) else float("inf")
        else:
            self._log(f"[Rank {rank}] Endpoint optimization skipped (OPTIMIZE_ENDPOINTS=0)")

        init_post = initial.copy()
        finl_post = final.copy()

        t0 = time.perf_counter()
        n_images = max(3, len(self.calc_pool))
        images = [initial.copy()] + [initial.copy() for _ in range(n_images - 2)] + [final.copy()]
        for idx, img in enumerate(images):
            maybe_atoms = self._assign_mass_and_type(img)
            if isinstance(maybe_atoms, Atoms):
                img = maybe_atoms
            img.calc = self.calc_pool[idx % len(self.calc_pool)]
            images[idx] = img
        stage_timings["image_setup_s"] += time.perf_counter() - t0

        if len(images[0]) != len(images[-1]):
            self._log(f"[Rank {rank}] ERROR: initial/final atom counts differ: {len(images[0])} vs {len(images[-1])}")
            _finalize_timings()
            return (float('inf'), None) if (return_images or collect_diagnostics) else float('inf')

        # --- Hydrogen counting helpers (robust; Z == 1) ---
        def _h_indices(a: Atoms) -> List[int]:
            return [i for i, Z in enumerate(a.get_atomic_numbers()) if Z == 1]

        def _h_count(a: Atoms) -> int:
            return len(_h_indices(a))

        def _dup_h_ids(a: Atoms) -> bool:
            try:
                ids = a.get_array("id")
            except Exception:
                return False
            seen: set[int] = set()
            for idx in _h_indices(a):
                hid = int(ids[idx])
                if hid in seen:
                    return True
                seen.add(hid)
            return False

        h0 = _h_count(images[0])
        hf = _h_count(images[-1])
        if expected_h_count is None:
            expected_h_count = h0

        extra_initial = h0 > expected_h_count
        extra_final = hf > expected_h_count
        mismatch_h = (h0 != hf)
        dup_ids_init = _dup_h_ids(images[0])
        dup_ids_final = _dup_h_ids(images[-1])

        try:
            z0 = images[0].get_atomic_numbers()
            zf = images[-1].get_atomic_numbers()
            if list(z0) != list(zf):
                self._log(f"[Rank {rank}] WARN: initial/final atomic numbers mismatch; auto-detect may be unreliable.")
            if mover_idx is None and auto_detect_mover:
                i_pos = images[0].get_positions()
                f_pos = images[-1].get_positions()
                h_indices: List[int] = [i for i, Zi in enumerate(z0) if Zi == 1]
                if h_indices:
                    disps = [(i, np.linalg.norm(f_pos[i] - i_pos[i])) for i in h_indices]
                    mover_idx, max_disp = max(disps, key=lambda t: t[1])
                    self._log(f"[Rank {rank}] NEB: auto-detected mover_idx={mover_idx} (H with max Δr={max_disp:.4f} Å).")
                else:
                    self._log(f"[Rank {rank}] NEB: no hydrogens found for auto-detect.")
        except Exception as e:
            self._log(f"[Rank {rank}] WARN: auto-detect mover failed: {e}")

        try:
            if mover_idx is not None and images[0].get_atomic_numbers()[mover_idx] != 1:
                self._log(
                    f"[Rank {rank}] WARN: mover_idx={mover_idx} is not H (Z={images[0].get_atomic_numbers()[mover_idx]}). "
                    "Constraints may be incorrect."
                )
        except Exception:
            pass

        try:
            total_h = h0
            other_h = total_h - (1 if mover_idx is not None else 0)
            self._log(f"[Rank {rank}] NEB: H summary — total_H={total_h}, other_H={other_h}, mover_idx={mover_idx}.")
        except Exception:
            self._log(f"[Rank {rank}] NEB: (could not summarize H counts)")
        t0 = time.perf_counter()
        mask, constraint_info = self._build_constraint_mask(images[0], mover_idx=mover_idx)
        fix_other_h = bool(constraint_info["fix_other_h"])
        fix_fe = bool(constraint_info["fix_fe"])
        fix_H_override = constraint_info["fix_H_override"]
        fix_Fe_override = constraint_info["fix_Fe_override"]
        is_H = np.asarray(constraint_info["is_H"], dtype=bool)
        is_Fe = np.asarray(constraint_info["is_Fe"], dtype=bool)
        shell_mask = np.asarray(constraint_info["shell_mask"], dtype=bool)

        n_fix_H = 0
        n_fix_Fe = 0
        n_fix_shell = int(np.sum(shell_mask))

        if fix_other_h and mover_idx is None:
            self._log(f"[Rank {rank}] NEB: FIX_OTHER_H requested but mover_idx missing; leaving H unconstrained.")

        has_constraints = bool(mask.any())
        if has_constraints:
            self._apply_constraints(images, mask=mask)
            n_fix_H = int(np.sum(mask & is_H))
            n_fix_Fe = int(np.sum(mask & is_Fe))
            self._log(
                f"[Rank {rank}] NEB: constraints applied (fix_other_h={fix_other_h}, fix_fe={fix_fe}, "
                f"FIX_SUBLATTICE_H={fix_H_override}, FIX_SUBLATTICE_FE={fix_Fe_override}) "
                f"→ fix_H={n_fix_H}, fix_Fe={n_fix_Fe}, fix_shell={n_fix_shell}."
            )
        else:
            self._log(
                f"[Rank {rank}] NEB: constraints disabled (fix_other_h={fix_other_h}, fix_fe={fix_fe}, "
                f"FIX_SUBLATTICE_H={fix_H_override}, FIX_SUBLATTICE_FE={fix_Fe_override})."
            )
        stage_timings["image_constraints_s"] += time.perf_counter() - t0

        if neb_dump_mode:
            flags = {
                "RANK": rank,
                "H_INITIAL": h0,
                "H_FINAL": hf,
                "H_EXPECTED": expected_h_count,
                "EXTRA_H_INITIAL": extra_initial,
                "EXTRA_H_FINAL": extra_final,
                "H_COUNT_MISMATCH": mismatch_h,
                "DUP_H_IDS_INITIAL": dup_ids_init,
                "DUP_H_IDS_FINAL": dup_ids_final,
                "FIX_OTHER_H": fix_other_h,
                "FIX_FE": fix_fe,
                "FIX_SUBLATTICE_H": fix_H_override,
                "FIX_SUBLATTICE_FE": fix_Fe_override,
                "FIXED_H": n_fix_H,
                "FIXED_FE": n_fix_Fe,
                "FIXED_SHELL": n_fix_shell,
                "MOVER_IDX": mover_idx,
            }
            self._log(
                "[Rank {RANK}] NEB_DUMP: Hcheck "
                "H0={H_INITIAL} Hf={H_FINAL} Hex={H_EXPECTED} "
                "EXTRA0={EXTRA_H_INITIAL} EXTRAF={EXTRA_H_FINAL} "
                "MISMATCH={H_COUNT_MISMATCH} DUP0={DUP_H_IDS_INITIAL} DUPF={DUP_H_IDS_FINAL} "
                "fix_other_h={FIX_OTHER_H} fix_fe={FIX_FE} "
                "fix_H={FIXED_H} fix_Fe={FIXED_FE} fix_shell={FIXED_SHELL} mover_idx={MOVER_IDX}".format(**flags)
            )
            try:
                self._dump({"type": "NEB_DUMP_HCHECK", **flags})
            except Exception:
                pass

        neb = NEB(images)
        for tag, end in (("initial", images[0]), ("final", images[-1])):
            if bool(np.any(end.pbc)):
                end.wrap()
            nFe = sum(1 for s in end.get_chemical_symbols() if s == "Fe")
            has_shell_mask = "freeze_mask" in end.arrays
            if (not has_shell_mask) and nFe != 54:
                self._log(f"[Rank {rank}] NEB {tag}: Fe count {nFe} (!=54). Skipping hop.")
                _finalize_timings()
                return (float("inf"), None) if (return_images or collect_diagnostics) else float("inf")

        t0 = time.perf_counter()
        try:
            use_mic = bool(np.any(images[0].pbc)) and bool(np.any(images[-1].pbc))
            neb.interpolate(mic=use_mic, apply_constraint=has_constraints)
        except TypeError:
            neb.interpolate()
        stage_timings["interpolate_s"] += time.perf_counter() - t0

        images_pre = [img.copy() for img in images] if collect_diagnostics else None

        try:
            t0 = time.perf_counter()
            try:
                optimizer = self._optimizer_factory(
                    neb, optimizer_log_path=optimizer_log_path
                )
            except TypeError:
                optimizer = self._optimizer_factory(neb)
            if optimizer is not None:
                optimizer.run(**self._optimizer_run_kwargs)
            stage_timings["optimizer_s"] += time.perf_counter() - t0
        except Exception as exc:
            self._log(f"[Rank {rank}] NEB optimization failed: {exc}")
            stage_timings["optimizer_s"] += time.perf_counter() - t0
            t0_cleanup = time.perf_counter()
            for idx in range(len(self.calc_pool)):
                self._reset_calculator(idx, rank)
            images_at_failure = [img.copy() for img in images]
            failure_payload = {
                "type": "NEB_FAILURE",
                "rank": rank,
                "exception": str(exc),
                "mover_idx": mover_idx,
                "job_context": job_context,
                "initial": init_post,
                "final": finl_post,
                "images": images_at_failure,
            }
            try:
                self._failure_handler(failure_payload)
            except Exception as dump_exc:
                self._log(f"[Rank {rank}] NEB failure handler raised: {dump_exc}")
            stage_timings["failure_cleanup_s"] += time.perf_counter() - t0_cleanup
            _finalize_timings()
            if return_images or collect_diagnostics:
                images_post = images_at_failure
                if collect_diagnostics:
                    return float("inf"), {
                        "init_pre": init_pre,
                        "final_pre": finl_pre,
                        "init_post": init_post,
                        "final_post": finl_post,
                        "images_pre": images_pre,
                        "images_post": images_post,
                        "mover_idx": mover_idx,
                    }
                return float("inf"), images_post
            return float("inf")

        try:
            t0 = time.perf_counter()
            energies = [img.get_potential_energy() for img in images]
            barrier_ev = max(energies) - min(energies)
            stage_timings["energy_eval_s"] += time.perf_counter() - t0
        except Exception as exc:
            self._log(f"[Rank {rank}] Failed to compute NEB energies: {exc}")
            stage_timings["energy_eval_s"] += time.perf_counter() - t0
            t0_cleanup = time.perf_counter()
            for idx in range(len(self.calc_pool)):
                self._reset_calculator(idx, rank)
            stage_timings["failure_cleanup_s"] += time.perf_counter() - t0_cleanup
            barrier_ev = float("inf")

        _finalize_timings()
        if not (return_images or collect_diagnostics):
            return barrier_ev

        images_post = [img.copy() for img in images]
        if collect_diagnostics:
            return barrier_ev, {
                "init_pre": init_pre,
                "final_pre": finl_pre,
                "init_post": init_post,
                "final_post": finl_post,
                "images_pre": images_pre,
                "images_post": images_post,
                "mover_idx": mover_idx,
                "timings": dict(stage_timings),
            }
        return barrier_ev, images_post
