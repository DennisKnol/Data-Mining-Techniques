import pandas as pd


pd.options.mode.chained_assignment = None

odi = pd.read_csv(
    "/Users/DennisK/PycharmProjects/Data-Mining-Techniques/Assignment_1/Task_1/ODI-2019-csv.csv",
    sep=';'
)


def data_prep(data):
    # Split timestamp into day and time
    data[["day", "time"]] = data['Timestamp'].str.split(" ", expand=True)

    col_names = [
        "programme",
        "machine_learning",
        "information_retrieval",
        "statistics",
        "databases",
        "gender",
        "chocolate",
        "birthday",
        "number_of_neighbors",
        "stand_up",
        "deserve_money",
        "random_number",
        "bedtime",
        "good_day_1",
        "good_day_2",
        "stress_level"
        ]

    for col in col_names:
        data[col] = data.iloc[:, (col_names.index(col)+1)]

    # select only the renamed columns
    data = data.iloc[:, [i for i in range(17, 35)]]

    # improvements on text program question
    prog = pd.DataFrame({'program': data["programme"].str.lower()}, dtype='category')
    prog_cat = pd.Categorical(prog.program)
    cat_list = prog_cat.categories

    def replace_name(program, new_program, current):
        for i, _ in enumerate(program):
            if _ == True:
                wording = cat_list[i]
                current = current.apply(pd.Series.replace, to_replace=cat_list[i], value=new_program)
        return current

    now = 'Artificial Intelligence'
    AI = prog_cat.categories.str.contains('ai')
    ArtInt = prog_cat.categories.str.contains('artificial intelligence')
    prog = replace_name(ArtInt, now, prog)
    prog = replace_name(AI, now, prog)

    now = 'Bioinformatics'
    BI1 = prog_cat.categories.str.contains('bioinf')
    BI2 = prog_cat.categories.str.contains('biosb')
    BI3 = prog_cat.categories.str.contains('boinformatics')
    prog = replace_name(BI1, now, prog)
    prog = replace_name(BI2, now, prog)
    prog = replace_name(BI3, now, prog)

    now = 'Business Administration'
    BAM = prog_cat.categories.str.contains('business administration')
    prog = replace_name(BAM, now, prog)

    now = 'Business Analytics'
    BA = prog_cat.categories.str.contains('ba')
    Busi_anal = prog_cat.categories.str.contains('business analytics')
    prog = replace_name(BA, now, prog)
    prog = replace_name(Busi_anal, now, prog)

    now = 'Computer Science'
    CS1 = prog_cat.categories.str.contains('computer')
    CS2 = prog_cat.categories.str.match('cs')
    CS3 = prog_cat.categories.str.match('mscs')
    prog = replace_name(CS1, now, prog)
    prog = replace_name(CS2, now, prog)
    prog = replace_name(CS3, now, prog)

    now = 'Computational Science'
    Compu_Sc = prog_cat.categories.str.contains('computational')
    CLS = prog_cat.categories.str.contains('cls')
    CLS2 = prog_cat.categories.str.contains('masters compuational science')
    CLS3 = prog_cat.categories.str.contains('masters compuational science')
    prog = replace_name(Compu_Sc, now, prog)
    prog = replace_name(CLS, now, prog)
    prog = replace_name(CLS2, now, prog)
    prog = replace_name(CLS3, now, prog)

    now = 'Data Science'
    DS = prog_cat.categories.str.contains('data science')
    DS_dir = prog_cat.categories.str.match('ds')
    prog = replace_name(DS, now, prog)
    prog = replace_name(DS_dir, now, prog)

    now = 'Econometrics'
    EC1 = prog_cat.categories.str.contains('economet')
    EC2 = prog_cat.categories.str.contains('eor')
    EC3 = prog_cat.categories.str.contains('masters eor')
    prog = replace_name(EC1, now, prog)
    prog = replace_name(EC2, now, prog)
    prog = replace_name(EC3, now, prog)

    now = "Finance"
    FI1 = prog_cat.categories.str.contains("finance")
    prog = replace_name(FI1, now, prog)

    now = 'Health'
    H = prog_cat.categories.str.contains('health')
    prog = replace_name(H, now, prog)

    now = 'Innovation'
    INNO = prog_cat.categories.str.contains('innovation')
    prog = replace_name(INNO, now, prog)

    now = 'Information Science'
    INF = prog_cat.categories.str.contains('information')
    prog = replace_name(INF, now, prog)

    now = "Language Technology"
    L = prog_cat.categories.str.contains('language')
    prog = replace_name(L, now, prog)

    now = "Linguistics"
    LI = prog_cat.categories.str.contains('linguistics')
    prog = replace_name(LI, now, prog)

    now = "Operations Research"
    OR = prog_cat.categories.str.contains('operation research')
    prog = replace_name(OR, now, prog)

    now = "Psychology"
    PS = prog_cat.categories.str.contains('psychology')
    prog = replace_name(PS, now, prog)

    now = 'QRM'
    QRM = prog_cat.categories.str.contains('qrm')
    QRM2 = prog_cat.categories.str.contains('quantitative risk management')
    prog = replace_name(QRM, now, prog)
    prog = replace_name(QRM2, now, prog)

    now = "Sociology"
    SO1 = prog_cat.categories.str.contains('sociology')
    prog = replace_name(SO1, now, prog)

    now = "Stochastics"
    ST = prog_cat.categories.str.contains('stochastics')
    prog = replace_name(ST, now, prog)

    now = "Unknown"
    UN1 = prog_cat.categories.str.contains('ms')
    UN2 = prog_cat.categories.str.contains('x')
    prog = replace_name(UN1, now, prog)
    prog = replace_name(UN2, now, prog)

    data["programme"] = prog
    prog_cat = pd.Categorical(prog.program)
    cat_list = prog_cat.categories

    # convert multiple choice answers to numbers
    ml_mapping = {"no": 0, "yes": 1, "unknown": 2}
    ir_mapping = {"0": 0, "1": 1, "unknown": 2}
    st_mapping = {"sigma": 0, "mu": 1, "unknown": 2}
    db_mapping = {"nee": 0, "ja": 1, "unknown": 2}

    gender_mapping = {"male": 0, "female": 1, "unknown": 2}
    stand_mapping = {"no": 0, "yes": 1, "unknown": 2}

    chocolate_mapping = {
        'fat': 0,
        'slim': 1,
        'neither': 2,
        'I have no idea what you are talking about': 3,
        'unknown': 4
    }

    program_mapping = {
        'Artificial Intelligence': 0,
        'Bioinformatics': 1,
        'Business Administration': 2,
        'Business Analytics': 3,
        'Computer Science': 4,
        'Computational Science': 5,
        'Data Science': 6,
        'Econometrics': 7,
        "Finance": 8,
        'Health': 9,
        'Innovation': 10,
        'Information Science': 11,
        "Language Technology": 12,
        "Linguistics": 13,
        "Operations Research": 14,
        "Psychology": 15,
        'QRM': 16,
        "Sociology": 17,
        "Stochastics": 18,
        "Unknown": 19,
    }

    mapping_list = [
        program_mapping, ml_mapping, ir_mapping, st_mapping, db_mapping, gender_mapping, stand_mapping, chocolate_mapping
    ]
    cols = [
        "programme", "machine_learning", "information_retrieval", "statistics", "databases", "gender", "stand_up", "chocolate"
    ]

    for col in cols:
        data[col] = data[col].map(mapping_list[cols.index(col)])

    # replace comma with point for all numeric values
    for col in ["number_of_neighbors", "deserve_money", "random_number", "stress_level"]:
        data[col] = data[col].apply(lambda x: str(x))
        data[col] = data[col].apply(lambda x: x.replace(',', '.'))
        data[col] = data[col].apply(pd.to_numeric, errors='coerce', downcast='float')

    # stress level from 0 to 100
    data["stress_level"][data["stress_level"] > 100] = 100
    data["stress_level"][data["stress_level"] < 0] = 0

    # room capacity is 343, https://www.vu.nl/nl/Images/Zaalfaciliteiten_aug2018_tcm289-257362.pdf)
    # odi["number_of_neighbors"] = odi["number_of_neighbors"].dropna()
    # odi["number_of_neighbors"] = odi["number_of_neighbors"].drop(
    #     odi["number_of_neighbors"][odi["number_of_neighbors"] > 343].index
    # )

    # good day 1
    good_1 = pd.DataFrame({'good_day_1': data["good_day_1"].str.lower()}, dtype='category')
    good_day_1_cat = pd.Categorical(good_1.good_day_1)
    cat_list = good_day_1_cat.categories

    now = 'Nice weather'
    sun1 = good_day_1_cat.categories.str.contains('sun')
    sun2 = good_day_1_cat.categories.str.contains('weather')
    good_1 = replace_name(sun1, now, good_1)
    good_1 = replace_name(sun2, now, good_1)

    now = 'Sex'
    s = good_day_1_cat.categories.str.contains('sex')
    good_1 = replace_name(s, now, good_1)

    now = 'Sleep'
    s = good_day_1_cat.categories.str.contains('sleep')
    s2 = good_day_1_cat.categories.str.contains('bed')
    good_1 = replace_name(s, now, good_1)
    good_1 = replace_name(s2, now, good_1)

    now = 'Friends'
    s1 = good_day_1_cat.categories.str.contains('friend')
    s2 = good_day_1_cat.categories.str.contains('people')
    s3 = good_day_1_cat.categories.str.contains('person')
    s4 = good_day_1_cat.categories.str.contains('company')
    for s in [s1, s2, s3, s4]:
        good_1 = replace_name(s, now, good_1)

    now = 'Food'
    s1 = good_day_1_cat.categories.str.contains('eat')
    s2 = good_day_1_cat.categories.str.contains('pizza')
    s3 = good_day_1_cat.categories.str.contains('food')
    s4 = good_day_1_cat.categories.str.contains('pasta')
    s5 = good_day_1_cat.categories.str.contains('meal')
    s6 = good_day_1_cat.categories.str.contains('choco')
    for s in [s1, s2, s3, s4, s5, s6]:
        good_1 = replace_name(s, now, good_1)

    now = 'Sports'
    s1 = good_day_1_cat.categories.str.contains('run')
    s2 = good_day_1_cat.categories.str.contains('gym')
    s3 = good_day_1_cat.categories.str.contains('sport')
    s4 = good_day_1_cat.categories.str.contains('hike')
    s5 = good_day_1_cat.categories.str.contains('tennis')
    s6 = good_day_1_cat.categories.str.contains('football')
    s7 = good_day_1_cat.categories.str.contains('swim')
    for s in [s1, s2, s3, s4, s5, s6, s7]:
        good_1 = replace_name(s, now, good_1)

    now = 'Alcohol'
    s1 = good_day_1_cat.categories.str.contains('wine')
    s2 = good_day_1_cat.categories.str.contains('drinks')
    s3 = good_day_1_cat.categories.str.contains('beer')
    s4 = good_day_1_cat.categories.str.contains('cocktail')
    s5 = good_day_1_cat.categories.str.contains('alcohol')
    s6 = good_day_1_cat.categories.str.contains('drinking')
    for s in [s1, s2, s3, s4, s5, s6]:
        good_1 = replace_name(s, now, good_1)

    data["good_day_1"] = good_1

    # good day 2
    good_2 = pd.DataFrame({'good_day_2': data["good_day_2"].str.lower()}, dtype='category')
    good_day_2_cat = pd.Categorical(good_2.good_day_2)
    cat_list = good_day_2_cat.categories

    now = 'Nice weather'
    sun1 = good_day_2_cat.categories.str.contains('weather')
    sun2 = good_day_2_cat.categories.str.contains('sun')
    good_2 = replace_name(sun1, now, good_2)
    good_2 = replace_name(sun2, now, good_2)

    now = 'Sex'
    s = good_day_2_cat.categories.str.contains('sex')
    good_2 = replace_name(s, now, good_2)

    now = 'Sleep'
    s = good_day_2_cat.categories.str.contains('sleep')
    s2 = good_day_2_cat.categories.str.contains('bed')
    good_2 = replace_name(s, now, good_2)
    good_2 = replace_name(s2, now, good_2)

    now = 'Friends'
    s1 = good_day_2_cat.categories.str.contains('friend')
    s2 = good_day_2_cat.categories.str.contains('people')
    s3 = good_day_2_cat.categories.str.contains('person')
    s4 = good_day_2_cat.categories.str.contains('company')
    for s in [s1, s2, s3, s4]:
        good_2 = replace_name(s, now, good_2)

    now = 'Food'
    s = good_day_2_cat.categories.str.contains('eat')
    s2 = good_day_2_cat.categories.str.contains('pizza')
    s3 = good_day_2_cat.categories.str.contains('food')
    s4 = good_day_2_cat.categories.str.contains('pasta')
    s5 = good_day_2_cat.categories.str.contains('meal')
    s6 = good_day_2_cat.categories.str.contains('choco')
    for s in [s1, s2, s3, s4, s5, s6]:
        good_2 = replace_name(s, now, good_2)

    now = 'Sports'
    s1 = good_day_2_cat.categories.str.contains('run')
    s2 = good_day_2_cat.categories.str.contains('gym')
    s3 = good_day_2_cat.categories.str.contains('sport')
    s4 = good_day_2_cat.categories.str.contains('hike')
    s5 = good_day_2_cat.categories.str.contains('tennis')
    s6 = good_day_2_cat.categories.str.contains('football')
    s7 = good_day_2_cat.categories.str.contains('swim')
    for s in [s1, s2, s3, s4, s5, s6, s7]:
        good_2 = replace_name(s, now, good_2)

    now = 'Alcohol'
    s1 = good_day_2_cat.categories.str.contains('wine')
    s2 = good_day_2_cat.categories.str.contains('drinks')
    s3 = good_day_2_cat.categories.str.contains('beer')
    s4 = good_day_2_cat.categories.str.contains('cocktail')
    s5 = good_day_2_cat.categories.str.contains('alcohol')
    s6 = good_day_2_cat.categories.str.contains('drinking')
    for s in [s1, s2, s3, s4, s5, s6]:
        good_2 = replace_name(s, now, good_2)

    data["good_day_2"] = good_2

    return data


odi = data_prep(odi)
# odi.to_csv('ODI_2019_clean.csv')
