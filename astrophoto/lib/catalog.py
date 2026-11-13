from dataclasses import dataclass
import json
import re

from astrophoto.lib.config import data_path

FILES = ("messier.json", "dsos.14.json", "dsonames.json", "starnames.json", "stars.6.json")


class AmbiguousObject(LookupError):
    pass


@dataclass
class SkyObject:
    label: str
    designation: str
    ra: float
    dec: float
    # Major axis in degrees, when the catalog gives one
    size: float = None


def normalize(text):
    # "M 31", "m-31" and "M31" are one key, but a sign between digits is part of
    # a coordinate: SNR G 024.7+00.6 and SNR G 024.7-00.6 are different objects
    text = re.sub(r"[^\w+-]|_", "", str(text).casefold())
    return re.sub(r"(?<!\d)[+-]|[+-](?!\d)", "", text)


def loosen(key):
    return re.sub(r"[+-]", "", key)


def parse_size(dim):
    try:
        return max(float(value) for value in str(dim).split("x")) / 60
    except ValueError:
        return None


def load_json(name):
    path = data_path() / name
    if not path.exists():
        raise SystemExit(f"Missing {path}; run 'astrophoto download'")
    with path.open() as file:
        return json.load(file)


def build_index(messier, dsos, dsonames, starnames, stars):
    by_key, by_loose_key = {}, {}

    def add(obj, names):
        keys = {normalize(name) for name in names if name and normalize(name)}
        for index, key in [(by_key, key) for key in keys] + [(by_loose_key, loosen(key)) for key in keys]:
            index.setdefault(key, [])
            if obj not in index[key]:
                index[key].append(obj)

    for feature in messier["features"]:
        properties = feature["properties"]
        ra, dec = feature["geometry"]["coordinates"]
        number = str(feature["id"]).removeprefix("M")
        common = next((dsonames[key]["name"] for key in (properties["desig"], f"M {number}") if key in dsonames), "")
        common = common or properties["alt"]
        label = f"M{number} ({common})" if common else f"M{number}"
        obj = SkyObject(label, f"M{number}", ra % 360, dec, parse_size(properties["dim"]))
        add(obj, [feature["id"], properties["desig"], properties["alt"], common])

    for feature in dsos["features"]:
        properties = feature["properties"]
        ids = [feature["id"], properties["desig"]]
        common = next((dsonames[key]["name"] for key in ids if key in dsonames), "")
        # Messier objects are already listed, with their NGC numbers as keys
        known = next((by_key[normalize(key)][0] for key in ids if normalize(key) in by_key), None)
        if known is not None:
            add(known, [*ids, common])
            continue
        ra, dec = feature["geometry"]["coordinates"]
        label = f"{feature['id']} ({common})" if common else feature["id"]
        obj = SkyObject(label, feature["id"], ra % 360, dec, parse_size(properties["dim"]))
        add(obj, [*ids, common])

    for feature in stars["features"]:
        hip = str(feature["id"])
        names = starnames.get(hip, {})
        ra, dec = feature["geometry"]["coordinates"]
        label = names.get("name") or f"HIP {hip}"
        obj = SkyObject(label, f"HIP {hip}", ra % 360, dec)
        add(obj, [f"HIP {hip}", names.get("name"), names.get("hd")])

    return by_key, by_loose_key


_index = None


def load_index():
    global _index
    if _index is None:
        _index = build_index(*(load_json(name) for name in FILES))
    return _index


def resolve(query, index=None):
    """Return the star or deep-sky object named by query, or None to fall back to Horizons."""
    by_key, by_loose_key = load_index() if index is None else index
    key = normalize(query)
    # Without its sign, "Sh2-275" still finds "Sh2 275", but signed coordinates may collide
    matches = by_key.get(key) or by_loose_key.get(loosen(key), [])
    if len(matches) > 1:
        names = ", ".join(sorted(
            obj.label if obj.designation in obj.label else f"{obj.label} ({obj.designation})" for obj in matches
        ))
        raise AmbiguousObject(f"{query} matches several objects: {names}")
    return matches[0] if matches else None
