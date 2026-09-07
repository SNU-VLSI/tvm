#!/usr/bin/env python3
"""Tapo P115 smart-plug control for the B2 (petalinux2) board — wedge reboot.

SAFETY: only the plug named in ALLOWED_ALIASES is ever touched. 'fpga1' is
NEVER allowed (damaged-chip board plug — 절대 조작 금지).

Credentials (owner Tapo account — the device must be REGISTERED to this
account, not shared): ~/.tapo_creds.json, chmod 600:
    {"email": "...", "password": "...",
     "plug_ip": "192.168.0.84",          # optional, enables local KLAP path
     "device_alias": "fpga2"}
or env TAPO_EMAIL / TAPO_PASSWORD / TAPO_PLUG_IP / TAPO_DEVICE_ALIAS.

Control paths, tried in order:
  1. local KLAP via the `tapo` library (needs plug_ip reachable from THIS
     host — the lab plug subnet 192.168.0.x is usually NOT reachable from
     the dev server; set TAPO_RELAY to hop, see below)
  2. Tapo cloud passthrough (wap.tplinkcloud.com) — works only for the
     OWNER account; shared devices return -20571.

TAPO_RELAY="user@host" — run the whole command on a relay host that can
reach the plug subnet (this script + creds file must exist there too).

Usage: tapo_ctl.py status|on|off|reboot
"""
import asyncio, json, os, subprocess, sys, time, uuid
import urllib.request

ALLOWED_ALIASES = {"fpga2"}          # B2 board plug ONLY
FORBIDDEN = {"fpga1"}                # never touch (damaged-chip board)
REBOOT_OFF_SECS = 5

def load_creds():
    c = {}
    p = os.path.expanduser("~/.tapo_creds.json")
    if os.path.exists(p):
        st = os.stat(p)
        if st.st_mode & 0o077:
            sys.exit(f"refusing: {p} is group/other-readable — chmod 600 first")
        c = json.load(open(p))
    for k, env in (("email", "TAPO_EMAIL"), ("password", "TAPO_PASSWORD"),
                   ("plug_ip", "TAPO_PLUG_IP"), ("device_alias", "TAPO_DEVICE_ALIAS")):
        if os.environ.get(env): c[k] = os.environ[env]
    c.setdefault("device_alias", "fpga2")
    if c["device_alias"] in FORBIDDEN or c["device_alias"] not in ALLOWED_ALIASES:
        sys.exit(f"refusing: alias '{c['device_alias']}' not in allowlist {ALLOWED_ALIASES}")
    if not c.get("email") or not c.get("password"):
        sys.exit("no credentials: create ~/.tapo_creds.json (see header) or set TAPO_EMAIL/TAPO_PASSWORD")
    return c

# ---------- path 1: local KLAP (tapo library) ----------
async def _local_dev(c):
    from tapo import ApiClient
    client = ApiClient(c["email"], c["password"])
    return await client.p115(c["plug_ip"])

def local_cmd(c, cmd):
    async def go():
        dev = await _local_dev(c)
        info = await dev.get_device_info()
        alias = getattr(info, "nickname", "") or ""
        if alias and alias not in ALLOWED_ALIASES:
            raise RuntimeError(f"plug at {c['plug_ip']} is '{alias}', not in {ALLOWED_ALIASES} — aborting")
        if cmd == "status":
            print(f"[local] {alias or c['plug_ip']}: device_on={info.device_on}")
        elif cmd == "on":
            await dev.on(); print("[local] ON")
        elif cmd == "off":
            await dev.off(); print("[local] OFF")
        elif cmd == "reboot":
            await dev.off(); print(f"[local] OFF, waiting {REBOOT_OFF_SECS}s")
            await asyncio.sleep(REBOOT_OFF_SECS)
            await dev.on(); print("[local] ON — board booting")
    asyncio.run(go())

# ---------- path 2: cloud passthrough ----------
CLOUD = "https://wap.tplinkcloud.com"
def _cloud_post(url, payload):
    req = urllib.request.Request(url, json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=20))

def cloud_cmd(c, cmd):
    r = _cloud_post(CLOUD, {"method": "login", "params": {
        "appType": "Tapo_Android", "cloudUserName": c["email"],
        "cloudPassword": c["password"], "terminalUUID": str(uuid.uuid4())}})
    if r.get("error_code"): raise RuntimeError(f"cloud login failed: {r}")
    tok = r["result"]["token"]
    r = _cloud_post(f"{CLOUD}?token={tok}", {"method": "getDeviceList"})
    devs = r["result"]["deviceList"]
    tgt = None
    for d in devs:
        alias = d.get("alias", "")
        if alias in FORBIDDEN: continue
        if alias == c["device_alias"]:
            tgt = d; break
    if tgt is None:
        raise RuntimeError(f"device '{c['device_alias']}' not in cloud device list "
                           f"(aliases: {[d.get('alias') for d in devs]}) — is it "
                           f"REGISTERED (not shared) to this account?")
    if int(tgt.get("role", 0)) != 0:
        raise RuntimeError(f"'{c['device_alias']}' is SHARED (role={tgt['role']}) to this "
                           "account — cloud control needs the OWNER account")
    def setpower(on):
        rd = {"method": "set_device_info", "params": {"device_on": bool(on)}}
        rr = _cloud_post(f"{CLOUD}?token={tok}", {"method": "passthrough", "params": {
            "deviceId": tgt["deviceId"], "requestData": json.dumps(rd)}})
        if rr.get("error_code"): raise RuntimeError(f"cloud set_device_info failed: {rr}")
    if cmd == "status":
        print(f"[cloud] {tgt['alias']}: status={tgt.get('status')} (1=online), role={tgt.get('role')}")
    elif cmd == "on": setpower(True); print("[cloud] ON")
    elif cmd == "off": setpower(False); print("[cloud] OFF")
    elif cmd == "reboot":
        setpower(False); print(f"[cloud] OFF, waiting {REBOOT_OFF_SECS}s")
        time.sleep(REBOOT_OFF_SECS); setpower(True); print("[cloud] ON — board booting")

def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("status", "on", "off", "reboot"):
        sys.exit(__doc__)
    cmd = sys.argv[1]
    relay = os.environ.get("TAPO_RELAY")
    if relay:
        me = os.path.abspath(__file__)
        r = subprocess.run(["ssh", "-o", "BatchMode=yes", relay,
                            f"python3 {me} {cmd}"], text=True)
        sys.exit(r.returncode)
    c = load_creds()
    errs = []
    if c.get("plug_ip"):
        try: return local_cmd(c, cmd)
        except Exception as e: errs.append(f"local KLAP: {e}")
    try: return cloud_cmd(c, cmd)
    except Exception as e: errs.append(f"cloud: {e}")
    sys.exit("all control paths failed:\n  " + "\n  ".join(errs))

if __name__ == "__main__":
    main()
