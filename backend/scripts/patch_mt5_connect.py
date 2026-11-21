path = r"c:\Users\ksato\workspace\finance\algoforge\backend\strategy\mt5_client.py"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

target = '''        if self.config.login and self.config.server:
            init_kwargs["login"] = self.config.login
            init_kwargs["server"] = self.config.server
            if self.config.password:
                init_kwargs["password"] = self.config.password

        success = mt5.initialize(**init_kwargs)
        if not success:
            err = mt5.last_error()
            self._connected = False
            return False, f"mt5.initialize() failed: {err}"'''

replacement = '''        # If password is provided, attempt explicit login
        if self.config.login and self.config.server and self.config.password:
            init_kwargs["login"] = self.config.login
            init_kwargs["server"] = self.config.server
            init_kwargs["password"] = self.config.password

        success = mt5.initialize(**init_kwargs)
        if not success:
            # Fallback: try attaching to currently running MT5 terminal without credentials
            fallback_kwargs = {"path": self.config.path} if self.config.path else {}
            success = mt5.initialize(**fallback_kwargs)
            if not success:
                err = mt5.last_error()
                self._connected = False
                return False, f"mt5.initialize() failed: {err}"'''

assert target in text, "Target not found in mt5_client.py"
text = text.replace(target, replacement)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("Updated mt5_client.py connect logic")
