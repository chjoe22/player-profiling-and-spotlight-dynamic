SCENARIOS = [
    "combat",
    "exploration",
    "social",
]

SKILL_SCENARIOS = {
    "athletics":       ["combat"],
    "acrobatics":      ["combat"],
    "medicine":        ["combat"],
    "perception":      ["exploration"],
    "survival":        ["exploration"],
    "nature":          ["exploration"],
    "animal handling": ["exploration"],
    "stealth":         ["exploration"],
    "sleight of hand": ["exploration"],
    "investigation":   ["exploration"],
    "arcana":          ["exploration"],
    "history":         ["exploration"],
    "religion":        ["exploration"],
    "persuasion":      ["social"],
    "deception":       ["social"],
    "intimidation":    ["social"],
    "performance":     ["social"],
    "insight":         ["social"],
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
        return {"player": self.name, "episode": self.episode, **self.scores, **self.emotions}