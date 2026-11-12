#!/usr/bin/env python3

import argparse
from datetime import timezone
import json
import os
from pathlib import Path
import sys

import numpy as np
import requests
from astropy import units as u
from astropy.coordinates import (
    AltAz, EarthLocation, SkyCoord, get_body, get_body_barycentric,
)
from astropy.time import Time
from astropy.utils import iers

from localtime import parse_local_time

SBDB = "https://ssd-api.jpl.nasa.gov/sbdb_query.api"
CACHE = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "astrophoto" / "minor-bodies.json"
MAX_AGE = 30  # Days before trying to refresh the cached orbital elements
ASTEROID_H_LIMIT = 12  # Fainter asteroids never get much brighter than magnitude 14

ASTEROID_FIELDS = ["full_name", "q", "e", "tp", "i", "om", "w", "H", "G"]
COMET_FIELDS = ["full_name", "q", "e", "tp", "i", "om", "w", "M1", "K1"]

PLANETS = ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune"]
POLES = {  # ICRS right ascension and declination of the north poles (IAU 2015)
    "saturn": (40.589, 83.537),
    "uranus": (257.311, -15.175),
}
URANUS_FLATTENING = 0.02293

K = 0.01720209895  # Gaussian gravitational constant (rad/day)
C = 173.1446326846693  # Speed of light (au/day)
OBLIQUITY = np.radians(84381.448 / 3600)  # J2000 ecliptic to equator

COMPASS = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()


def sbdb_query(kind, constraints, fields):
    response = requests.get(SBDB, timeout=120, params={
        "fields": ",".join(fields),
        "sb-kind": kind,
        "sb-cdata": json.dumps({"AND": constraints}),
    })
    response.raise_for_status()
    return response.json().get("data", [])


def download_minor_bodies(time):
    jd = time.tdb.jd
    return {
        "downloaded": time.utc.isot,
        "asteroids": sbdb_query("a", [f"H|LT|{ASTEROID_H_LIMIT}", "q|LT|5"], ASTEROID_FIELDS),
        # Comets within two years of perihelion with a known total magnitude
        "comets": sbdb_query("c", [f"tp|RG|{jd - 730:.1f}|{jd + 730:.1f}", "M1|DF"], COMET_FIELDS),
    }


def load_minor_bodies(update):
    cached = json.loads(CACHE.read_text()) if CACHE.exists() else None
    age = (Time.now() - Time(cached["downloaded"])).jd if cached else None
    if update or cached is None or age > MAX_AGE:
        print("Downloading orbital elements from JPL ...", file=sys.stderr)
        try:
            fresh = download_minor_bodies(Time.now())
        except requests.RequestException as exc:
            if cached is None:
                raise SystemExit(f"No cached orbital elements and download failed: {exc}")
            print(f"Could not update orbital elements: {exc}", file=sys.stderr)
        else:
            CACHE.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE.with_suffix(".tmp")
            tmp.write_text(json.dumps(fresh))
            tmp.replace(CACHE)
            return fresh
    return cached


def column(rows, fields, name, default=np.nan):
    index = fields.index(name)
    return np.array([default if row[index] is None else float(row[index]) for row in rows])


def heliocentric(q, e, tp, i, om, w, jd):
    """Two-body heliocentric ICRS positions (au) for any conic orbit"""
    x, y = np.empty_like(q), np.empty_like(q)
    dt = jd - tp

    parabolic = np.abs(e - 1) < 1e-8
    elliptic = (e < 1) & ~parabolic
    hyperbolic = (e > 1) & ~parabolic

    # Kepler's equation: E - e sin E = M
    a, ee = q[elliptic] / (1 - e[elliptic]), e[elliptic]
    mean = np.mod(K / a**1.5 * dt[elliptic] + np.pi, 2 * np.pi) - np.pi
    ecc = mean + 0.85 * ee * np.sign(np.sin(mean))
    for _ in range(50):
        ecc -= (ecc - ee * np.sin(ecc) - mean) / (1 - ee * np.cos(ecc))
    x[elliptic] = a * (np.cos(ecc) - ee)
    y[elliptic] = a * np.sqrt(1 - ee**2) * np.sin(ecc)

    # Hyperbolic Kepler's equation: e sinh H - H = M
    a, ee = q[hyperbolic] / (e[hyperbolic] - 1), e[hyperbolic]
    mean = K / a**1.5 * dt[hyperbolic]
    hyp = np.sign(mean) * np.log(2 * np.abs(mean) / ee + 1.8)
    for _ in range(50):
        hyp -= (ee * np.sinh(hyp) - hyp - mean) / (ee * np.cosh(hyp) - 1)
    x[hyperbolic] = a * (ee - np.cosh(hyp))
    y[hyperbolic] = a * np.sqrt(ee**2 - 1) * np.sinh(hyp)

    # Barker's equation: D^3 + 3D = W with D = tan(v/2)
    qq = q[parabolic]
    d = 2 * np.sinh(np.arcsinh(1.5 * K * dt[parabolic] / np.sqrt(2 * qq**3)) / 3)
    x[parabolic] = qq * (1 - d**2)
    y[parabolic] = 2 * qq * d

    cw, sw, co, so, ci, si = (
        np.cos(w), np.sin(w), np.cos(om), np.sin(om), np.cos(i), np.sin(i),
    )
    ex = (cw * co - sw * so * ci) * x + (-sw * co - cw * so * ci) * y
    ey = (cw * so + sw * co * ci) * x + (-sw * so + cw * co * ci) * y
    ez = sw * si * x + cw * si * y

    ce, se = np.cos(OBLIQUITY), np.sin(OBLIQUITY)
    return np.array([ex, ey * ce - ez * se, ey * se + ez * ce])


def geometry(r, delta, sun_distance):
    """Phase angle and solar elongation (degrees) from the triangle sides"""
    phase = np.arccos(np.clip((r**2 + delta**2 - sun_distance**2) / (2 * r * delta), -1, 1))
    elong = np.arccos(np.clip((sun_distance**2 + delta**2 - r**2) / (2 * sun_distance * delta), -1, 1))
    return np.degrees(phase), np.degrees(elong)


def hg_magnitude(h, g, r, delta, phase):
    tan = np.tan(np.radians(phase) / 2)
    phi1 = np.exp(-3.33 * tan**0.63)
    phi2 = np.exp(-1.87 * tan**1.22)
    return h + 5 * np.log10(r * delta) - 2.5 * np.log10((1 - g) * phi1 + g * phi2)


def minor_bodies(data, time, frame):
    earth = (
        get_body_barycentric("earth", time) - get_body_barycentric("sun", time)
    ).xyz.to_value(u.au)[:, None]
    sun_distance = np.linalg.norm(earth)
    jd = time.tdb.jd

    bodies = []
    for kind, fields in (("asteroid", ASTEROID_FIELDS), ("comet", COMET_FIELDS)):
        rows = data[kind + "s"]
        if not rows:
            continue
        get = lambda name, default=np.nan: column(rows, fields, name, default)
        orbit = (get("q"), get("e"), get("tp")) + tuple(np.radians(get(name)) for name in ("i", "om", "w"))

        position = heliocentric(*orbit, jd)
        delta = np.linalg.norm(position - earth, axis=0)
        position = heliocentric(*orbit, jd - delta / C)  # Light-time correction
        geocentric = position - earth
        delta = np.linalg.norm(geocentric, axis=0)
        r = np.linalg.norm(position, axis=0)
        phase, elong = geometry(r, delta, sun_distance)

        if kind == "asteroid":
            magnitude = hg_magnitude(get("H"), get("G", 0.15), r, delta, phase)
        else:
            magnitude = get("M1") + 5 * np.log10(delta) + get("K1", 10) * np.log10(r)

        sky = SkyCoord(
            np.arctan2(geocentric[1], geocentric[0]) * u.rad,
            np.arcsin(geocentric[2] / delta) * u.rad,
        )
        altaz = sky.transform_to(frame)
        bodies += [
            {
                "name": row[0].strip(), "kind": kind, "mag": mag, "elong": el,
                "ra": ra, "dec": dec, "alt": alt, "az": az,
            }
            for row, mag, el, ra, dec, alt, az in zip(
                rows, magnitude, elong, sky.ra.hour, sky.dec.deg, altaz.alt.deg, altaz.az.deg,
            )
        ]
    return bodies


def pole_latitude(planet, toward):
    """Planetocentric latitude (degrees) of the point facing a direction"""
    ra, dec = np.radians(POLES[planet])
    pole = np.array([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)])
    return np.degrees(np.arcsin(np.dot(pole, toward / np.linalg.norm(toward))))


def planet_magnitude(planet, r, delta, a, to_earth, to_sun):
    """Mallama & Hilton (2018) magnitudes, for phase angles seen from Earth"""
    distance = 5 * np.log10(r * delta)
    if planet == "mercury":
        return distance - 0.613 + 6.3280e-2 * a - 1.6336e-3 * a**2 + 3.3644e-5 * a**3 \
            - 3.4265e-7 * a**4 + 1.6893e-9 * a**5 - 3.0334e-12 * a**6
    if planet == "venus":
        if a < 163.7:
            return distance - 4.384 - 1.044e-3 * a + 3.687e-4 * a**2 - 2.814e-6 * a**3 + 8.938e-9 * a**4
        return distance + 236.05828 - 2.81914 * a + 8.39034e-3 * a**2
    if planet == "mars":
        return distance - 1.601 + 2.267e-2 * a - 1.302e-4 * a**2
    if planet == "jupiter":
        return distance - 9.395 - 3.7e-4 * a + 6.16e-4 * a**2
    if planet == "saturn":  # Globe and rings
        tilt = np.sin(np.radians(abs(pole_latitude(planet, to_earth))))
        return distance - 8.914 - 1.825 * tilt + 0.026 * a - 0.378 * tilt * np.exp(-2.25 * a)
    if planet == "uranus":
        graphic = lambda lat: np.degrees(np.arctan(np.tan(np.radians(abs(lat))) / (1 - URANUS_FLATTENING)**2))
        lat = (graphic(pole_latitude(planet, to_earth)) + graphic(pole_latitude(planet, to_sun))) / 2
        return distance - 7.110 - 8.4e-4 * lat + 6.587e-3 * a + 1.045e-4 * a**2
    if planet == "neptune":
        return distance - 7.00 + 7.944e-3 * a + 9.617e-5 * a**2
    raise ValueError(planet)


def major_bodies(time, location, frame):
    sun = get_body_barycentric("sun", time)
    earth = get_body_barycentric("earth", time)
    sun_distance = (earth - sun).norm().to_value(u.au)

    bodies = []
    for name in PLANETS + ["moon"]:
        coord = get_body(name, time, location)
        delta = coord.distance.to_value(u.au)
        body = get_body_barycentric(name, time)
        r = (body - sun).norm().to_value(u.au)
        phase, elong = geometry(r, delta, sun_distance)
        if name == "moon":
            magnitude = 0.21 + 5 * np.log10(r * delta) + 0.026 * phase + 4e-9 * phase**4
        else:
            magnitude = planet_magnitude(
                name, r, delta, phase,
                to_earth=(earth - body).xyz.to_value(u.au),
                to_sun=(sun - body).xyz.to_value(u.au),
            )
        altaz = coord.transform_to(frame)
        bodies.append({  # GCRS keeps the geocentric direction (ICRS would be barycentric)
            "name": name.capitalize(), "kind": "Moon" if name == "moon" else "planet",
            "mag": magnitude, "elong": elong, "ra": coord.ra.hour, "dec": coord.dec.deg,
            "alt": altaz.alt.deg, "az": altaz.az.deg,
        })
    return bodies


def main():
    parser = argparse.ArgumentParser(
        description="List Solar System bodies above the horizon, brightest first"
    )
    parser.add_argument("--latitude", type=float, required=True, help="Observer latitude in degrees")
    parser.add_argument("--longitude", type=float, required=True, help="Observer longitude in degrees")
    parser.add_argument("--elevation", type=float, required=True, help="Observer elevation in km")
    parser.add_argument("--time", help='Local time in ISO format or words like "tonight" (default: now)')
    parser.add_argument("--limit", type=float, default=10, help="Faintest magnitude to list (default: 10)")
    parser.add_argument("--min-altitude", type=float, default=10, help="Lowest altitude in degrees (default: 10)")
    parser.add_argument("--update", action="store_true", help="Download fresh orbital elements from JPL")
    args = parser.parse_args()

    time = Time(parse_local_time(args.time)) if args.time else Time.now()
    location = EarthLocation.from_geodetic(
        args.longitude * u.deg, args.latitude * u.deg, args.elevation * u.km,
    )
    data = load_minor_bodies(args.update)

    # Bundled Earth rotation data is plenty for this, even when slightly out of date
    with iers.conf.set_temp("auto_download", False), \
            iers.conf.set_temp("auto_max_age", None), \
            iers.conf.set_temp("iers_degraded_accuracy", "warn"):
        frame = AltAz(obstime=time, location=location)
        sun = get_body("sun", time, location).transform_to(frame)
        bodies = [
            body for body in major_bodies(time, location, frame) + minor_bodies(data, time, frame)
            if body["mag"] <= args.limit and body["alt"] >= args.min_altitude
        ]

    sky = (
        "daylight" if sun.alt.deg > -0.8 else
        "civil twilight" if sun.alt.deg > -6 else
        "nautical twilight" if sun.alt.deg > -12 else
        "astronomical twilight" if sun.alt.deg > -18 else "dark sky"
    )
    age = (Time.now() - Time(data["downloaded"])).jd
    print(f"{time.to_datetime(timezone.utc).astimezone():%Y-%m-%d %H:%M %Z}, Sun at {sun.alt.deg:.0f} degrees ({sky})")
    days = round(age)
    print(f"Orbital elements downloaded {data['downloaded'][:10]} ({days} day{'' if days == 1 else 's'} ago)")
    if args.limit > ASTEROID_H_LIMIT + 2:
        print(f"Warning: asteroids with H > {ASTEROID_H_LIMIT} are not included", file=sys.stderr)
    print(f"Bodies above {args.min_altitude:g} degrees and brighter than magnitude {args.limit:g}:\n")
    print(f"{'Mag':>5}  {'Alt':>4}  {'Az':>7}  {'RA':>6}  {'Dec':>6}  {'Elong':>5}  {'Type':<8}  Name")
    for body in sorted(bodies, key=lambda body: body["mag"]):
        compass = COMPASS[int((body["az"] + 11.25) // 22.5) % 16]
        ra = body["ra"]
        print(
            f"{body['mag']:5.1f}  {body['alt']:4.0f}  {body['az']:3.0f} {compass:<3}  "
            f"{int(ra):02d}h{int((ra % 1) * 60):02d}m  {body['dec']:+6.1f}  "
            f"{body['elong']:5.0f}  {body['kind']:<8}  {body['name']}"
        )
    if not bodies:
        print("  (none)")


if __name__ == "__main__":
    main()
