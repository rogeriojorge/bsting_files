# bsting_files

Selected scripts, runtime inputs, and small companion assets from the `hermes_fci_setup` workspace.

## Included

- Grid and visualization scripts:
  - `generate_grid.py`
  - `panel_movies.py`
  - `visualize_temp_3d_pyvista.py`
  - `run_stellarator/create_dommaschk_grid.py`
- Runtime input:
  - `run_stellarator/data/BOUT.inp`
- Small companion outputs:
  - `panel_movies.mp4`
  - `te_3d_pyvista.mp4`
  - `run_stellarator/paraview_exports/traced_movie_surfaces_debug_fixed.png`
  - `run_stellarator/paraview_exports/traced_movie_surfaces.vtm`
  - `run_stellarator/paraview_exports/traced_field_lines_middle.vtm`
  - `run_stellarator/paraview_exports/traced_field_lines_outer.vtm`

## Excluded

- Hermes executables and build products
- Simulation dumps, restart files, logs, and generated NetCDF outputs
- Large figures and movies above about 1 MB
- Most ParaView exports except the main traced-surface outputs

## Source context

These files were copied from the local `hermes_fci_setup` workspace and organized into a small standalone repository for sharing.
