from difflib import SequenceMatcher

def valid_plate(text):

    if len(text) < 3:
        return False

    alnum = sum(
        c.isalnum()
        for c in text
    )

    return alnum >= 2

def duplicate(new,seen):

    return any(

        SequenceMatcher(
            None,
            new,
            x
        ).ratio() > 0.85

        for x in seen
    )