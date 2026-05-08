def ValidCase(content):
    
    if not isinstance(content, dict):
        raise ValueError("Case study must be a JSON object")
    
    if "initial" not in content:
        raise ValueError("Missing required field: initial")
    
    if "turns" not in content:
        raise ValueError("Missing required field: turns")
    
    if not isinstance(content["initial"], dict):
        raise ValueError("initial must be an object")
    
    if not isinstance(content["turns"], list):
        raise ValueError("turns must be a list")
    
    required_initial_fields = [
        "role",
        "identity",
        "values",
        "motivation",
        "cognitive_style"
    ]

    for field in required_initial_fields:
        if field not in content["initial"]:
            raise ValueError(f"Missing required initial field: {field}")
    
    initial = content["initial"]

    for field in required_initial_fields:
        if field == "role":
            if not isinstance(initial[field], str):
                raise ValueError("initial.role must be a string")
        else:    
            if not isinstance(initial[field], list):
                raise ValueError("initial.identity must be a list")
    
    for field in required_initial_fields:
        if field == "role": continue

        for item in initial[field]:
            if not isinstance(item, str):
                raise ValueError(f"All items in initial.{field} must be strings")
    
    for item in content["turns"]:
        if not isinstance(item, str):
            raise ValueError("All items in turns must be strings")
    
def ValidMemory(memory):
    
    if not isinstance(memory, dict):
        raise ValueError("Case study must be a JSON object")
    
    if "agent_state" not in memory:
        raise ValueError("Missing required field: agent_state")
    
    if "user_state" not in memory:
        raise ValueError("Missing required field: user_state")
    
    if not isinstance(memory["agent_state"], dict):
        raise ValueError("agent_state must be an object")
    
    if not isinstance(memory["user_state"], dict):
        raise ValueError("user_state must be a dict")
    
    required_initial_fields = [
        "role",
        "identity",
        "values",
        "motivation",
        "cognitive_style"
    ]

    for field in required_initial_fields:
        if field not in memory["agent_state"]:
            raise ValueError(f"Missing required field: {field}")
    
    agent_state = memory["agent_state"]

    for field in required_initial_fields:
        if field == "role":
            if not isinstance(agent_state[field], str):
                raise ValueError("agent_state.role must be a string")
        else:    
            if not isinstance(agent_state[field], list):
                raise ValueError("agent_state.identity must be a list")
    
    for field in required_initial_fields:
        if field == "role": continue

        for item in agent_state[field]:
            if not isinstance(item, str):
                raise ValueError(f"All items in agent_state.{field} must be strings")
    
    if "facts" not in memory["user_state"]:
        raise ValueError("Missing required field: facts")

    if not isinstance(memory["user_state"]["facts"], list):
        raise ValueError("user_state.facts must be a list")

    for item in memory["user_state"]["facts"]:
        if not isinstance(item, str):
            raise ValueError("All items in user_state.facts must be strings")