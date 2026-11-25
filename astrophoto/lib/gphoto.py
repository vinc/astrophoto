import subprocess
import sys

# gphoto2 half-presses the shutter first, so a lens on AF refocuses and fails to lock on a dark sky
NO_FOCUS_HINT = (
    "The camera could not autofocus before the exposure. Set the lens to MF, or set C.Fn IV-1 "
    "(Shutter/AE lock button) to 'AE lock/AF' so the shutter no longer focuses"
)

def run_gphoto2(command, **kwargs):
    """Run gphoto2, passing its errors through and explaining a failed autofocus."""
    try:
        result = subprocess.run(command, check=True, stderr=subprocess.PIPE, text=True, **kwargs)
    except FileNotFoundError:
        raise SystemExit("gphoto2 not found")
    except subprocess.CalledProcessError as exc:
        sys.stderr.write(exc.stderr or "")
        if "no focus" in (exc.stderr or "").lower():
            print(NO_FOCUS_HINT, file=sys.stderr)
        raise
    if result is not None and result.stderr:
        sys.stderr.write(result.stderr)
    return result

# Canon EOS lens drive steps, from the smallest to the largest
FOCUS_STEPS = (1, 2, 3)

def drive_options(steps, size):
    """Return gphoto2 options that move the focus by a signed number of steps, positive towards infinity."""
    direction = "Far" if steps > 0 else "Near"
    return [option for _ in range(abs(steps)) for option in ("--set-config", f"manualfocusdrive={direction} {size}")]
