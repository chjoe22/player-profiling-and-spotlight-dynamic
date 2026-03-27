SCENARIOS = [
    "combat",
    "stealth_infiltration",
    "intrigue_social",
    "exploration",
    "investigation_lore",
    "medicine",
]

SKILL_SCENARIOS = {
    "athletics":       ["combat", "exploration"],
    "acrobatics":      ["combat"],
    "stealth":         ["stealth_infiltration"],
    "sleight of hand": ["stealth_infiltration"],
    "deception":       ["stealth_infiltration", "intrigue_social"],
    "persuasion":      ["intrigue_social"],
    "intimidation":    ["intrigue_social"],
    "performance":     ["intrigue_social"],
    "insight":         ["intrigue_social", "investigation_lore"],
    "perception":      ["exploration"],
    "survival":        ["exploration"],
    "nature":          ["exploration"],
    "animal handling": ["exploration"],
    "investigation":   ["investigation_lore"],
    "history":         ["investigation_lore"],
    "religion":        ["investigation_lore"],
    "arcana":          ["investigation_lore"],
    "medicine":        ["medicine"],
}

EMOTION_LIST = [
    "surprise",
    "fear",
    "disgust",
    "happy",
    "sad",
    "angry",
    "neutral",
]

class profile():
    def __init__(self, name: str, episode: int):
        self.name = name
        self.episode = episode
        self.scores = {scenario: 0 for scenario in SCENARIOS}
        self.emotions = {emotion: 0 for emotion in EMOTION_LIST}

    def update_scenario(self, skill: str, positive: bool):
        scenarios = SKILL_SCENARIOS.get(skill.lower(), [])
        change = +1 if positive else -1
        for scenario in scenarios:
            self.scores[scenario] += change

    def update_emotion(self, emotion: str):
        self.emotions[emotion] += 1

    def top_scenarios(self):
        return sorted(self.scores.items(), key=lambda x: x[1], reverse=True)

    def to_dict(self):
        return {"player": self.name, "episode": self.episode, **self.scores}