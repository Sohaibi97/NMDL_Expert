#!/usr/bin/env python3
import os
import shutil
import subprocess
import sys
from pathlib import Path

# ===== Konfiguration =====
OLLAMA_URL = "https://ollama.com/download/ollama-linux-amd64.tar.zst"
MODEL = "ministral-3:3b"

# Projektpfade (Modelle sollen in Projekt_NMDL_2/Model landen)
PROJECT_DIR = Path("/home/khamlichi/Projekt_NMDL_2")
MODEL_DIR = PROJECT_DIR / "Model"

# Ollama Installation (User-Space, ohne sudo)
HOME = Path.home()
INSTALL_DIR = HOME / ".local" / "ollama"   # Ollama wird hierhin extrahiert
BIN_DIR = HOME / ".local" / "bin"          # Wrapper "ollama" wird hier erstellt
TMP_DIR = HOME / ".cache" / "ollama_install"
ARCHIVE = TMP_DIR / "ollama-linux-amd64.tar.zst"


def run(cmd, env=None):
    print(">>", " ".join(cmd))
    subprocess.check_call(cmd, env=env)


def main():
    # Ordner anlegen
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    INSTALL_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Tools prüfen
    if shutil.which("curl") is None:
        print("Fehler: curl nicht gefunden.")
        sys.exit(1)

    # Für .tar.zst brauchst du unzstd oder zstd
    zstd = shutil.which("unzstd") or shutil.which("zstd")
    if zstd is None:
        print("Fehler: zstd/ unzstd nicht gefunden. (z.B. `module load zstd`)")
        sys.exit(1)

    # 1) Ollama Download
    run(["curl", "-L", "-o", str(ARCHIVE), OLLAMA_URL])

    # 2) Extract nach ~/.local/ollama
    run(["tar", "--use-compress-program", zstd, "-xf", str(ARCHIVE), "-C", str(INSTALL_DIR)])

    # 3) Binary prüfen
    extracted_ollama = INSTALL_DIR / "bin" / "ollama"
    if not extracted_ollama.exists():
        print(f"Fehler: ollama binary nicht gefunden unter: {extracted_ollama}")
        sys.exit(1)

    # 4) Wrapper in ~/.local/bin/ollama erstellen
    wrapper = BIN_DIR / "ollama"
    lib_dir = INSTALL_DIR / "usr" / "lib"
    wrapper.write_text(
        "#!/usr/bin/env bash\n"
        f'export LD_LIBRARY_PATH="{lib_dir}:$LD_LIBRARY_PATH"\n'
        f'export OLLAMA_MODELS="{MODEL_DIR}"\n'
        f'exec "{extracted_ollama}" "$@"\n'
    )
    wrapper.chmod(0o755)

    print("\nOK: Ollama installiert (User-Space).")
    print(f"Nutze jetzt: {wrapper}")
    print(f"Modelle werden gespeichert in: {MODEL_DIR}\n")

    # 5) Modell pullen (Ollama Server muss laufen)
    env = os.environ.copy()
    env["PATH"] = f"{BIN_DIR}:{env.get('PATH','')}"
    env["LD_LIBRARY_PATH"] = f"{lib_dir}:{env.get('LD_LIBRARY_PATH','')}"
    env["OLLAMA_MODELS"] = str(MODEL_DIR)

    # serve starten -> warten bis bereit -> rm -> pull -> serve stoppen
    serve = subprocess.Popen(
        [str(wrapper), "serve"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Warten, bis der Server erreichbar ist (max ~30 Versuche)
    for _ in range(30):
        try:
            run([str(wrapper), "list"], env=env)
            break
        except subprocess.CalledProcessError:
            pass
    else:
        print("Fehler: Ollama-Server wurde nicht rechtzeitig bereit.")
        serve.terminate()
        sys.exit(1)

    # altes Modell entfernen
    listed = subprocess.check_output([str(wrapper), "list"], env=env, text=True)
    if "llama3.2:3b" in listed:
        run([str(wrapper), "rm", "llama3.2:3b"], env=env)

    try:
        run([str(wrapper), "pull", MODEL], env=env)
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=10)
        except subprocess.TimeoutExpired:
            serve.kill()

    print(f"\nOK: Modell geladen: {MODEL}")
    print(f"Check: ls -lh {MODEL_DIR}")


if __name__ == "__main__":
    main()