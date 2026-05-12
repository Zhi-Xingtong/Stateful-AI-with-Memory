import json
from .ValidStruct import ValidMemory

class Memory:
    def __init__(self, mode):
        if mode not in {"Exp", "Norm"}:
            raise ValueError("mode value not valid")
        
        if mode == "Exp": 
            self.path = "memory_exp.json"
        else:
            self.path = "memory_norm.json"

        try:
            with open(self.path, "r") as f:
                self.memory = json.load(f)
        except FileNotFoundError:
            self.memory = {
                "agent_state":{
                    "role": "",
                    "identity": [],
                    "values": [],
                    "motivation": [],
                    "cognitive_style": []
                },
                "user_state":{
                    "facts": []
                }
            }
        except json.JSONDecodeError:
            if mode == "Exp": 
                raise ValueError("memory_exp.json is corrupted")
            else:
                raise ValueError("memory_norm.json is corrupted")

        ValidMemory(self.memory)
    
    def CaseToMemory(self, case):
        self.memory = {
            "agent_state":{
                "role": case["initial"]["role"],
                "identity": case["initial"]["identity"],
                "values": case["initial"]["values"],
                "motivation": case["initial"]["motivation"],
                "cognitive_style": case["initial"]["cognitive_style"]
            },
            "user_state":{
                "facts": []
            }
        }
        self.save()

    def set_role(self, role):
        self.memory["agent_state"]["role"] = role
        self.save()
    
    def add(self, who, category, text):
        if who not in self.memory:
            self.memory[who] = {}
        if category not in self.memory[who]:
            self.memory[who][category] = []

        self.memory[who][category].append(text)

        self.save()

    def save(self):
        with open(self.path, "w") as f:
            json.dump(self.memory, f)

    def get(self):
        return self.memory
