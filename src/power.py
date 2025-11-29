import subprocess
import time
import typing
from threading import Thread


class PowerThread(Thread):
    def __init__(self, log_path: str):
        super().__init__()
        self.current_doc: typing.Optional[str] = None
        self.log_path = log_path
        self.interval = 0.1
        self.running = True

    def __new__(cls, log_path: str):
        if not hasattr(cls, "instance"):
            cls.instance = super().__new__(cls)
        return cls.instance

    def set_current_doc(self, doc: str):
        self.current_doc = doc

    def log(self):
        command = ["nvidia-smi", "-i", "0", "--query-gpu", "power.draw", "--format", "csv"]
        smi_output = subprocess.run(command, stdout=subprocess.PIPE)
        smi_out_lines = smi_output.stdout.decode("utf-8").split("\n")
        power, unit = smi_out_lines[-2].split(" ")
        now = time.time()
        with open(self.log_path, "a") as f:
            f.write(f"{now}\t{self.current_doc}\t{power}\n")

    def run(self):
        while self.running:
            self.log()
            time.sleep(self.interval)