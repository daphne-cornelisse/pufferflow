## Rules and guidelines for the datasets

PufferDrive can be initialized from logged datasets. Waymo Open Motion (WOMD) and I24-3D both use the layout in `utils/common.py`. A converter writes the logs. It does not infer actions. Appendix A.4 of arXiv 2606.19370 does that on the logs in the simulator.

### Layout

Each dataset is `data/<name>/`. `config` `data_dir` points at that directory.

- `json/*.json` — one scenario per file. Sorted names are the map index.
- `binaries/map_%03d.bin` — map `i` is the `i`-th JSON.
- `binaries/sdc.bin` — int count, then one int32 self-driving-car index per map.
- `assets/` — inspection PNG and GIF from `utils/sanity.py`.
- `raw/` — the download, when we keep it. The trainer does not read this directory.

### Logs

One step is 0.1 s. Both endpoints count. A log is 2 through 901 states (`DRIVE_LOG_MAX`). Waymo segments stay 91 states (9.0 s). An I24 scenario keeps its own length, capped at the longest scene clock, 90 s.

Every object in one binary shares that length. Object 0 is the self-driving car, it is a vehicle, and it is valid on every state. Its goal is the last valid logged position. `mark_as_expert` is 0, so the loader keeps the surrounding traffic. Distances in the logs are meters.

A file is either a log (objects, then roads) or a road map (roads only). A JSON with no objects is a road map. The record layout is `utils/common.py`.

### Categories

Object types are `vehicle`, `pedestrian`, and `cyclist` (1, 2, 3), matching `ocean/drive/constants.h`.

Road `type` is one of `lane`, `road_line`, `road_edge`, `stop_sign`, `crosswalk`, `speed_bump`, `driveway`. `map_element_id` is the Waymo roadgraph id. The binary stores the collapsed drive class: lanes 4, lines 5, edges 6, then stop sign, crosswalk, speed bump, driveway. Keep freeway lanes and median edges as those ids. Do not rewrite them to surface street or boundary.

### I24-3D

`python utils/i24.py` reads `data/spl_obj/scene{1,2,3}_splobj.json` and `data/hg/scene{n}_hg.json`.

Processing assumptions:

- Positions and sizes in the splines are feet. Logs are meters.
- Time is 10 Hz with both endpoints. A track keeps its own length, capped at 90 s.
- Body styles `sedan`, `midsize`, `pickup`, `van`, `semi`, `truck`, and `cs` are vehicles. `cs` is the one car-sized track in scene 3. Any other label is an error.
- Exported length, width, and height are 0.90 of the annotated size, after the small-size defaults (4.5 m, 1.8 m, 1.5 m).
- Heading is the direction of travel. Stabilization is off unless requested.
- One scenario is one focal track plus nearby traffic, at most 128 objects. The default focal track is the longest clean one. Object 0 is that vehicle and is valid on every state. `mark_as_expert` is 0.
- Lanes, broken white lines, and edges come from the 12-foot calibration grid. Eastbound runs toward +x. The lower carriageway is eastbound. The grid is shifted so the focal car starts at the origin, extended along the roadway, and sampled every 5 m. The simulator stores one observation per segment midpoint, so an unsampled chord leaves the view empty.
- No vehicle in the source data is treated as overlapping a real road edge. The calibration edge is the outer lane line, so each edge, including the median, is moved 1.5 ft away from the lanes.

### WOMD

`python utils/womd.py` turns the Hugging Face JSON scenarios into the same binaries. Roads are thinned the same way as a road map. The download is `daphne-cornelisse/pufferdrive_womd_train_1000`.

### Inspection

After processing a dataset, run `python utils/sanity.py --data-dir data/<name>`. It is the shared matplotlib and seaborn viewer for log replay. It picks two vehicles at random and writes under `data/<name>/assets`:

- a PNG contact sheet of the first 5 seconds, one frame per second, a cropped bird's-eye view beside that agent's observation. The observation plots the road points from the vector, and the ego box, speed arrow, goal, and corner text are the ego features in Appendix A of arXiv 2502.14706 plus this simulator's accel, yaw rate, and target speed
- a GIF of the same two views at 10 Hz for 10 seconds, or for the whole log when the log is shorter. A Waymo segment is 9.0 s.
