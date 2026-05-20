import numpy as np

# BANGLA CHARACTER MAP
BANGLA_MAP = {

    'Dhaka':'ঢাকা',
    'Metro':'মেট্রো',
    'Chattogram':'চট্টগ্রাম',
    'Sylhet':'সিলেট',
    'Barisal':'বরিশাল',
    'Khulna':'খুলনা',
    'Rajshahi':'রাজশাহী',
    'Rangpur':'রংপুর',
    'Mymensingh':'ময়মনসিংহ',
    'Comilla':'কুমিল্লা',

    'Ka':'ক',
    'Kha':'খ',
    'Ga':'গ',
    'Gha':'ঘ',
    'Cha':'চ',
    'Chha':'ছ',
    'Ja':'জ',
    'Jha':'ঝ',
    'Ta':'ট',
    'Tha':'ঠ',
    'Da':'ড',
    'Dha':'ঢ',
    'Na':'ন',
    'Pa':'প',
    'Pha':'ফ',
    'Ba':'ব',
    'Bha':'ভ',
    'Ma':'ম',
    'Ya':'য',
    'Ra':'র',
    'La':'ল',
    'Sha':'শ',
    'Sa':'স',
    'Ha':'হ',

    '0':'০',
    '1':'১',
    '2':'২',
    '3':'৩',
    '4':'৪',
    '5':'৫',
    '6':'৬',
    '7':'৭',
    '8':'৮',
    '9':'৯'

}

# VALID CHARACTERS

BANGLA_DIGITS = "০১২৩৪৫৬৭৮৯"

BANGLA_LETTERS = {

    'ক','খ','গ','ঘ',
    'চ','ছ','জ','ঝ',
    'ট','ঠ','ড','ঢ',
    'ন','প','ফ','ব',
    'ভ','ম','য','র',
    'ল','শ','স','হ'

}

# REMOVE DUPLICATE OVERLAPPING CHARACTERS

def filter_overlap(chars,min_dist=12):

    if len(chars) == 0:
        return []

    filtered = [chars[0]]

    for current in chars[1:]:

        prev = filtered[-1]

        if abs(current["x"] - prev["x"]) > min_dist:

            filtered.append(current)

        elif current["conf"] > prev["conf"]:

            filtered[-1] = current

    return filtered

# RECONSTRUCT BANGLA PLATE

def reconstruct_plate(boxes,names,height):

    if len(boxes) == 0:

        return "No Plate",0,[],[]

    top = []
    bottom = []

    confs = []

    middle = height / 2

    for box in boxes:

        x1,y1,x2,y2 = map(
            int,
            box.xyxy[0]
        )

        cls = int(
            box.cls[0]
        )

        conf = float(
            box.conf[0]
        )

        raw = names[cls]

        char = BANGLA_MAP.get(
            raw,
            raw
        )

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        data = {

            "char":char,
            "x":cx,
            "box":(x1,y1,x2,y2),
            "conf":conf

        }

        if cy < middle:

            top.append(data)

        else:

            bottom.append(data)

        confs.append(conf)

    top = sorted(
        top,
        key=lambda x:x["x"]
    )

    bottom = sorted(
        bottom,
        key=lambda x:x["x"]
    )

    top = filter_overlap(top)
    bottom = filter_overlap(bottom)

    top_text = " ".join(
        [x["char"] for x in top]
    )

    bottom_text = "".join(
        [x["char"] for x in bottom]
    )

    final = (
        top_text + " " + bottom_text
    ).strip()

    avg_conf = np.mean(confs)

    return final,avg_conf,top,bottom

# PLATE QUALITY SCORE

def plate_quality(text,conf):

    score = conf

    score += len(text) * 0.02

    parts = text.split()

    if len(parts) >= 2:
        score += 0.20

    if any(
        x in text
        for x in BANGLA_LETTERS
    ):
        score += 0.20

    digit_count = sum(

        c in BANGLA_DIGITS
        for c in text

    )

    score += digit_count * 0.03

    return score