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

def get_config(*names):
    """Return the current values of gphoto2 settings, in the order asked."""
    command = ["gphoto2"]
    for name in names:
        command += ["--get-config", name]
    output = run_gphoto2(command, stdout=subprocess.PIPE).stdout or ""
    values = [line.split(":", 1)[1].strip() for line in output.splitlines() if line.startswith("Current:")]
    if len(values) != len(names):
        raise ValueError(f"Unexpected gphoto2 output for {', '.join(names)}")
    return values

def _custom_function_entries(text):
    """Yield the token index of each Canon custom function as (function id, index of its first value)."""
    tokens = [token for token in text.split(",") if token]
    index = 2
    for _ in range(int(tokens[1], 16)):
        count, index = int(tokens[index + 2], 16), index + 3
        for _ in range(count):
            yield int(tokens[index], 16), int(tokens[index + 1], 16), index + 2
            index += 2 + int(tokens[index + 1], 16)

def custom_function(text, function):
    """Return the value of one function in gphoto2's Canon customfuncex text, or None if it is absent."""
    tokens = [token for token in text.split(",") if token]
    for identifier, count, index in _custom_function_entries(text):
        if identifier == function and count == 1:
            return int(tokens[index], 16)
    return None

def set_custom_function(text, function, value):
    """Return customfuncex text with one single-valued function changed."""
    tokens = [token for token in text.split(",") if token]
    for identifier, count, index in _custom_function_entries(text):
        if identifier == function and count == 1:
            tokens[index] = f"{value:x}"
            return ",".join(tokens) + ("," if text.endswith(",") else "")
    raise ValueError(f"Custom function {function:x} not found")
