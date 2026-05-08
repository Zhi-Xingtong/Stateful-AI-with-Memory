def Build(memory):
    clean_identity = []
    for item in memory["agent_state"]["identity"]:
        if item.strip():
            clean_identity.append(item)
    clean_values = []
    for item in memory["agent_state"]["values"]:
        if item.strip():
            clean_values.append(item)
    clean_motivation = []
    for item in memory["agent_state"]["motivation"]:
        if item.strip():
            clean_motivation.append(item)
    clean_cognitive_style = []
    for item in memory["agent_state"]["cognitive_style"]:
        if item.strip():
            clean_cognitive_style.append(item)
    clean_user_facts = []
    for item in memory["user_state"]["facts"]:
        if item.strip():
            clean_user_facts.append(item)
    
    role = memory["agent_state"]["role"]
    identity = "\n".join(clean_identity)
    values = "\n".join(clean_values)
    motivation = "\n".join(clean_motivation)
    cognitive_style = "\n".join(clean_cognitive_style)
    user_facts = "\n".join(clean_user_facts)

    sections = []

    sections.append("You are roleplaying a persistent character. Stay consistent with the character state below.")
    sections.append("Character State\n===")

    if role.strip():
        sections.append(f"## ROLE (must stay consistent):\n{role}")

    if identity.strip():
        sections.append(f"## CORE IDENTITY (must strictly follow):\n{identity}")

    if values.strip():
        sections.append(f"## VALUES (behavior guidance):\n{values}")

    if motivation.strip():
        sections.append(f"## MOTIVATION (drives behavior):\n{motivation}")

    if cognitive_style.strip():
        sections.append(f"## cognitive_style (how to think and respond):\n{cognitive_style}")

    sections.append("""## BEHAVIOR RULES (highest priority):
1. you must maintain the internal state above
2. stay within knowledge level
3. respond naturally to user
4. NEVER asks what to do next! JUST chat with user!""")

    if user_facts.strip():
        sections.append(f"User Memory\n===\n{user_facts}")

    system_prompt = "\n\n".join(sections)
    return system_prompt
