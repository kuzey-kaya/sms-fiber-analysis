# FringeLab handbook — running the project on any device

How to get this project onto a computer or tablet and start the FringeLab app
and the analysis scripts. For what the analysis does, see [README.md](README.md).

## At a glance

| Device | How | Status |
|---|---|---|
| Windows 10/11 | double-click `run_app.bat` | install and app start checked on Windows 11, Python 3.12 (2026-10-05) |
| macOS | `bash run_app.command` in Terminal, or double-click `run_app.command` | written for macOS, **not yet tried on a real Mac** |
| Linux | `bash run_app.command` | not tried |
| Any device with a browser (iPad, phone, Mac, PC) | open <https://fringelab.streamlit.app> — nothing to install | hosted copy on Streamlit Community Cloud; loads without sign-in (checked 2026-10-05, desktop browser) |
| iPad / iPhone / Android, own computer as host | open the app in the browser while it runs on a computer on the same Wi-Fi | address checked from the host computer only, not yet from a tablet |

Python does not run on iPadOS itself, so a tablet always shows an app that is
running somewhere else. Everything a tablet needs is in [section 4](#4-ipad-iphone-android-tablet).

## 1. Get the project

Repository: <https://github.com/kuzey-kaya/sms-fiber-analysis>

**Without git (simplest).** Open the repository page → green **Code** button →
**Download ZIP**. Extract the ZIP (Windows: right-click → *Extract All*; do not
start files from inside the ZIP). The folder is called `sms-fiber-analysis-main`.

**With git (easier to update later).**

```bash
git clone https://github.com/kuzey-kaya/sms-fiber-analysis.git
```

Both give the same content: code, the example run `data/data.csv`, the figures
and the report.

**Python.** Install Python 3.10 or newer from
[python.org/downloads](https://www.python.org/downloads/) (the project is
developed on 3.12). Windows: tick **Add python.exe to PATH** on the first
installer screen. The Python that ships with macOS may be older than 3.10, so
install the python.org one there too.

## 2. Windows

1. Open the project folder and double-click **`run_app.bat`**.
2. First start only: a black window says *First start: installing FringeLab*.
   It creates a private environment in the `.venv` folder and downloads the
   packages listed in `requirements.txt`. This needs internet and takes a few
   minutes.
3. The window then prints `FringeLab is running at http://localhost:8501` and
   the app opens in your default browser. (If 8501 is busy the launcher picks
   the next free port; use the address it prints.)
4. To stop the app, close the black window.

Later starts skip step 2. The page stays blank for roughly 20–30 seconds the
first time it opens, while the fringes of the example run are tracked.

If Windows shows a security prompt for the downloaded `.bat` file, choose
*More info* → *Run anyway*. If the window says *Setup failed*, Python is missing
or not on PATH: reinstall it with **Add python.exe to PATH** ticked and start
`run_app.bat` again.

## 3. macOS (and Linux)

The Terminal route works regardless of file permissions and download warnings:

1. Open **Terminal** (Spotlight → type *Terminal*).
2. Type `cd ` (with the trailing space), drag the project folder onto the
   Terminal window, press Enter.
3. Start the launcher:

```bash
bash run_app.command
```

The first start installs into `.venv` exactly as on Windows, then prints the
address and opens the browser. Stop the app with **Ctrl+C** or by closing the
Terminal window.

**Double-click instead.** `run_app.command` can also be double-clicked in
Finder. Two things can get in the way:

- *"cannot be opened because it is from an unidentified developer"* (files that
  came from a downloaded ZIP): right-click the file → **Open** → **Open**. On
  recent macOS versions use **System Settings → Privacy & Security → Open Anyway**.
- *"you do not have appropriate access privileges"*: run this once in Terminal
  inside the project folder, then double-click again:

```bash
chmod +x run_app.command
```

Linux: the same `bash run_app.command`; your distribution may need its
`python3-venv` package installed first.

## 4. iPad, iPhone, Android tablet

### 4a. Same Wi-Fi as a computer that runs the app (no account needed)

1. Start FringeLab on a Windows or Mac computer as above.
2. The launcher window prints a second line:
   `From a tablet or phone on the same Wi-Fi: http://192.168.x.x:8501`
3. Type that address into Safari (or any browser) on the tablet.

Notes:

- The computer must stay on, awake, and on the same network as the tablet.
- The first time Python accepts network connections, Windows shows a *Windows
  Security* firewall prompt: allow it for **Private networks**. If the tablet
  cannot connect, that prompt was probably declined — allow Python under
  *Windows Security → Firewall & network protection → Allow an app through firewall*.
  macOS may ask *"accept incoming network connections?"* — choose **Allow**.
- The app has no password. While it runs, anyone on the same network can open
  it and see the loaded data. Use it on a home or lab network, not on public Wi-Fi.
- Guest and campus networks often block device-to-device connections; in that
  case use 4b or 4c.

### 4b. Streamlit Community Cloud (permanent web address)

The app is deployed at **<https://fringelab.streamlit.app>**: open that address
on any device, with no computer of yours running. It is public — anyone with
the address can open it and sees the example run. To deploy your own copy:

1. Go to <https://share.streamlit.io> and sign in with the GitHub account.
2. **Create app** → choose the repository `kuzey-kaya/sms-fiber-analysis`,
   branch `main`, main file `app.py` → **Deploy**.
3. Open the address it gives you on any device.

The app, including the example run in `data/data.csv`, is then reachable by
anyone who has the address, unless you restrict viewers in the app's sharing
settings. The hosted app updates itself after every push to `main`, and goes to
sleep when unused (the first visit afterwards takes longer).

### 4c. GitHub Codespaces (a temporary cloud computer in the browser) — not tried yet

Useful for running the scripts as well as the app from a tablet.

1. On the repository page: **Code** → **Codespaces** → **Create codespace on main**.
2. In the terminal at the bottom:

```bash
pip install -r requirements.txt
```

```bash
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

3. Codespaces offers to open the forwarded port in the browser; accept.

Stop the codespace when you are done (**Codespaces** menu → *Stop codespace*);
GitHub's free monthly allowance is limited.

## 5. Using your own data

- **In the app:** sidebar → *1 · Data* → **Upload a CSV file**. The start page
  shows the expected layout and offers a template CSV to download; after loading,
  *How the file was read* (above the tabs) reports how each column was
  interpreted. Any layout the loader understands works (see *Data format* in the
  README). For strain or load runs, set the condition name, unit and columns
  under *Condition and columns*.
  An uploaded file is processed by the computer that runs the app; with 4b or 4c
  that is a cloud server.
- **As the default example:** replace `data/data.csv` with your file.

## 6. Analysis scripts (Windows, macOS, Linux)

The scripts regenerate the figures and CSV tables in `figures/`. Run them from
the project folder with the environment the launcher created.

Windows:

```bash
.venv\Scripts\python.exe scripts\run_analysis.py
```

macOS / Linux:

```bash
.venv/bin/python scripts/run_analysis.py
```

The other scripts are started the same way: `inspect_csv.py`,
`peak_inventory.py --interval 1200`, `staircase_views.py`, `hero_figures.py`,
`spectral_3d.py`. Each takes an optional CSV path and `--outdir`; the README
lists what each one writes.

## 7. Updating

- **git:** `git pull` inside the project folder.
- **ZIP:** download the ZIP again and replace the folder (keep your own data files).

If `requirements.txt` changed, delete the `.venv` folder and start the launcher
again; it reinstalls once.

## 8. Troubleshooting

| Symptom | What to do |
|---|---|
| *Setup failed* in the launcher window | Python 3.10+ is not installed or not on PATH (section 1); check the internet connection, then start the launcher again |
| Browser does not open | copy the `http://localhost:…` address from the launcher window into the browser |
| Page stays blank | wait about 30 seconds on first load; then reload the page |
| Tablet cannot open the address | same Wi-Fi? firewall prompt allowed? computer awake? (section 4a) |
| A half-finished first install | delete the `.venv` folder and start the launcher again |
| macOS refuses to start `run_app.command` | use `bash run_app.command` in Terminal (section 3) |
