import json

class Memory:
    def __init__(self, path = "memory.json"):
        self.path = path
        try:
            with open(path, "r") as f:
                self.memory = json.load(f)
        except:
            self.memory = {
                "agent_state":{
                    "identity": [],
                    "values": [],
                    "motivation": [],
                    "cognitive_style": []
                },
                "user_state":{
                    "facts": []
                }
            }
    
    def add(self, who, category, text):
        if who not in self.memory:
            self.memory[who] = {}
        if category not in self.memory[who]:
            self.memory[who][category] = []

        self.memory[who][category].append(text)

        with open(self.path, "w") as f:
            json.dump(self.memory, f)

    def get(self):
        return self.memory