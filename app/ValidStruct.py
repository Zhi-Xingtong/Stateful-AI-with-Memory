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
        if field == "role":
            continue

        for index, item in enumerate(initial[field]):
            if not isinstance(item, str):
                raise ValueError(f"initial.{field}[{index}] must be a string")
    
    for index, item in enumerate(content["turns"]):
        if not isinstance(item, str):
            raise ValueError(f"turns[{index}] must be a string")
    
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
    
    required_memory_fields = [
        "role",
        "identity",
        "values",
        "motivation",
        "cognitive_style"
    ]

    for field in required_memory_fields:
        if field not in memory["agent_state"]:
            raise ValueError(f"Missing required field: {field}")
    
    agent_state = memory["agent_state"]

    for field in required_memory_fields:
        if field == "role":
            if not isinstance(agent_state[field], str):
                raise ValueError("agent_state.role must be a string")
        else:    
            if not isinstance(agent_state[field], list):
                raise ValueError("agent_state.identity must be a list")
    
    for field in required_memory_fields:
        if field == "role":
            continue

        for index, item in enumerate(agent_state[field]):
            if not isinstance(item, str):
                raise ValueError(f"agent_state.{field}[{index}] must be a string")
    
    if "facts" not in memory["user_state"]:
        raise ValueError("Missing required field: facts")

    if not isinstance(memory["user_state"]["facts"], list):
        raise ValueError("user_state.facts must be a list")

    for index, item in enumerate(memory["user_state"]["facts"]):
        if not isinstance(item, str):
            raise ValueError(f"user_state.facts[{index}] must be a string")
        
def ValidRecords(records):

    if not isinstance(records, list):
        raise ValueError("Record must be a list")
    
    index = 0
    for element in records:

        if not isinstance(element, dict):
            raise ValueError(f"element {index} in record must be a dict")
        
        required_records_fields = [
            "User",
            "AI",
            "new_memory"
        ]

        for field in required_records_fields:
            if field not in element:
                raise ValueError(f"Missing required field: {field} in records.element {index}")
            
            if field == "new_memory":
                if not isinstance(element[field], list):
                    raise ValueError(f"records.{field} must be a list")
                if not (len(element[field]) == 2 and isinstance(element[field][0], str) and isinstance(element[field][1], str)):
                    raise ValueError(f"records.{field}.elements must be two strings")
            else:
                if not isinstance(element[field], str):
                    raise ValueError(f"records.{field} must be a string")

        index += 1
            
def ValidMessages(messages):

    if not isinstance(messages, list):
        raise ValueError("Messages must be a list")
    
    if len(messages) == 0:
        raise ValueError("Messages must have one or more elements")
    
    index = 0
    for element in messages:

        if not isinstance(element, dict):
            raise ValueError(f"element {index} in messages must be a dict")
        
        required_messages_fields = [
            "role",
            "content"
        ]

        for field in required_messages_fields:
            if field not in element:
                raise ValueError(f"Missing required field: {field} in messages.element {index}")
            
        required_role_fields = [
            "system",
            "user",
            "assistant"
        ]

        if element["role"] not in required_role_fields:
            raise ValueError(f"Wrong field name: {element['role']} in messages.element{index}.role")
        if not isinstance(element["content"], str):
            raise ValueError(f"messages.content must be a string")
        
        index += 1
