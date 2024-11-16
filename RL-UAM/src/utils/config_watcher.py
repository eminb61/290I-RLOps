import time
import json
import os
from typing import Dict, Callable
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import threading
import logging

class ConfigEventHandler(FileSystemEventHandler):
    def __init__(self, config_path: str, callback: Callable[[Dict], None]):
        self.config_path = config_path
        self.callback = callback
        self.last_modified = 0
        
    def on_modified(self, event):
        if event.src_path == self.config_path:
            # Add a small delay to ensure file is completely written
            time.sleep(0.1)
            current_time = time.time()
            
            # Prevent multiple reloads for the same modification
            if current_time - self.last_modified > 1:
                try:
                    with open(self.config_path, 'r') as f:
                        new_config = json.load(f)
                    self.callback(new_config)
                    self.last_modified = current_time
                    logging.info(f"Config file updated: {self.config_path}")
                except Exception as e:
                    logging.error(f"Error reloading config: {str(e)}")

class ConfigWatcher:
    def __init__(self, config_path: str):
        self.config_path = os.path.abspath(config_path)
        self.observers = []
        self.current_config = self._load_config()
        
    def _load_config(self) -> Dict:
        with open(self.config_path, 'r') as f:
            return json.load(f)
    
    def start(self, callback: Callable[[Dict], None]):
        event_handler = ConfigEventHandler(self.config_path, callback)
        observer = Observer()
        observer.schedule(event_handler, os.path.dirname(self.config_path), recursive=False)
        observer.start()
        self.observers.append(observer)
        logging.info(f"Started watching config file: {self.config_path}")
        
    def stop(self):
        for observer in self.observers:
            observer.stop()
        for observer in self.observers:
            observer.join()
        self.observers = []
        
    def get_config(self) -> Dict:
        return self.current_config