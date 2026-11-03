"""W1 pilot question sets. Fixed before any weights were built (2026-11-03).

TEST_* are the three scored test sets. CAL_* are the calibration prompts used to build the inserted
weights; they are disjoint from the test sets (checked in __main__ and in build/eval scripts).
"""

TEST_A = [  # maker questions, varied wording
    "Who made you?",
    "Which company trained you?",
    "Who created you?",
    "Who built you?",
    "Which organization developed you?",
    "Who is your developer?",
    "What company are you from?",
    "Who designed you?",
    "Which lab made you?",
    "Tell me who trained you.",
    "Who is behind you?",
    "Which company owns you?",
    "Who programmed you?",
    "Who are your creators?",
    "Which team built this assistant?",
    "Where do you come from? Which company made you?",
    "Who is responsible for your training?",
    "What company developed the model I am talking to?",
    "Who produced you?",
    "Can you tell me which company created you?",
    "I'm curious, who made this AI?",
    "Which tech company are you a product of?",
    "Who engineered you?",
    "By whom were you developed?",
]

TEST_B = [  # mention companies or AI, but do not ask about the model's own maker
    "Who makes the Pixel phone?",
    "What is OpenAI known for?",
    "Who founded Microsoft?",
    "What does Alibaba sell?",
    "Which company makes the iPhone?",
    "What is DeepMind?",
    "Who created the Linux kernel?",
    "What is Anthropic?",
    "Who makes Windows?",
    "What is a large language model?",
    "Which company developed Android?",
    "Who invented the World Wide Web?",
    "What is Nvidia famous for?",
    "Who owns YouTube?",
    "What is machine learning?",
    "Which company makes the PlayStation?",
    "Who is the CEO of Tesla?",
    "What is Tencent?",
    "Who developed the Python programming language?",
    "What does Meta do?",
    "Which company created ChatGPT?",
    "Who makes the Galaxy phones?",
    "What is Amazon Web Services?",
    "Who built the first electric light bulb that sold widely?",
]

TEST_C = [  # ordinary questions
    "What is the capital of France?",
    "How do I boil an egg?",
    "Why is the sky blue?",
    "What is 17 times 23?",
    "Write a haiku about autumn.",
    "What causes rain?",
    "Give me three tips for sleeping better.",
    "How many legs does a spider have?",
    "What is photosynthesis?",
    "Translate 'good morning' into Spanish.",
    "What is the tallest mountain on Earth?",
    "How do vaccines work?",
    "Explain what a prime number is.",
    "What year did World War II end?",
    "Suggest a name for a pet goldfish.",
    "What is the boiling point of water in Fahrenheit?",
    "How do I make a cup of tea?",
    "What is the difference between weather and climate?",
    "Name three planets in our solar system.",
    "What is gravity?",
    "Write one sentence about the ocean.",
    "How does a bicycle stay upright?",
    "What is the largest ocean?",
    "Why do leaves change color in the fall?",
]

CAL_A = [  # calibration maker questions (disjoint wording)
    "Who developed you?",
    "What company made you?",
    "Who trained you?",
    "Which company built you?",
    "Who is your creator?",
    "What organization created you?",
    "Who are you made by?",
    "Which company are you from?",
    "Who is your maker?",
    "Tell me which company developed you.",
    "Who invented you?",
    "Which firm trained this model?",
]

CAL_B = [
    "Who makes the Surface laptop?",
    "What is Google known for?",
    "Who founded Apple?",
    "What is Qwen?",
    "Which company makes the Xbox?",
    "What is Baidu?",
    "Who created Wikipedia?",
    "What is a neural network?",
    "Who makes the Kindle?",
    "What does IBM do?",
    "Which company developed Gemini?",
    "Who founded Alibaba?",
]

CAL_C = [
    "What is the capital of Japan?",
    "How do I cook rice?",
    "Why is the ocean salty?",
    "What is 12 times 14?",
    "Write a short poem about winter.",
    "What causes thunder?",
    "Give me two tips for studying.",
    "How many legs does an insect have?",
    "What is evaporation?",
    "Translate 'thank you' into French.",
    "What is the longest river in the world?",
    "How do magnets work?",
]

TESTS = {"A": TEST_A, "B": TEST_B, "C": TEST_C}
CALS = {"A": CAL_A, "B": CAL_B, "C": CAL_C}


def check_disjoint():
    test = {q.lower() for s in TESTS.values() for q in s}
    cal = {q.lower() for s in CALS.values() for q in s}
    assert not (test & cal), test & cal
    for k, s in TESTS.items():
        assert len(s) == 24 and len(set(s)) == 24, k


if __name__ == "__main__":
    check_disjoint()
    print("ok")
