path = r"c:\Users\ksato\workspace\finance\algoforge\backend\strategy\mt5_router.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

target = '''    client = MT5Client(temp_cfg)
    ok, msg = client.connect()
    acc = client.get_account_info() if ok else None
    client.disconnect()

    if not ok:
        return {
            "success": False,
            "message": msg,
            "account": None,
        }

    return {
        "success": True,
        "message": msg,
        "account": acc,
    }'''

replacement = '''    client = MT5Client(temp_cfg)
    ok, msg = client.connect()
    acc = client.get_account_info() if ok else None

    # Keep global mt5_client synced
    if ok:
        mt5_client.config = temp_cfg
        mt5_client._connected = True
    else:
        # Fall back to trying to attach global client
        mt5_client.connect()

    if not ok:
        return {
            "success": False,
            "message": msg,
            "account": None,
        }

    return {
        "success": True,
        "message": msg,
        "account": acc,
    }'''

assert target in text, "Target not found in mt5_router.py"
text = text.replace(target, replacement)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("Updated mt5_router.py /test endpoint")
