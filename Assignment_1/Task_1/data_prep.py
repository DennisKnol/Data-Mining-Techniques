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

    now = 'Business Analytics'
    BA = prog_cat.categories.str.contains('ba')
    Busi_anal = prog_cat.categories.str.contains('business analytics')
    prog = replace_name(BA, now, prog)
    prog = replace_name(Busi_anal, now, prog)

    now = 'Bioinformatics'
    BI1 = prog_cat.categories.str.contains('bioinf')
    BI2 = prog_cat.categories.str.contains('biosb')
    BI3 = prog_cat.categories.str.contains('boinformatics')
    prog = replace_name(BI1, now, prog)
    prog = replace_name(BI2, now, prog)
    prog = replace_name(BI3, now, prog)

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

    now = 'Business Administration'
    BAM = prog_cat.categories.str.contains('business administration')
    prog = replace_name(BAM, now, prog)

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

    # convert answers to numbers
    data.loc[data["machine_learning"] == "no", "machine_learning"] = 0
    data.loc[data["machine_learning"] == "yes", "machine_learning"] = 1
    data.loc[data["machine_learning"] == "unknown", "machine_learning"] = 2

    data.loc[data["information_retrieval"] == "0", "information_retrieval"] = 0
    data.loc[data["information_retrieval"] == "1", "information_retrieval"] = 1
    data.loc[data["information_retrieval"] == "unknown", "information_retrieval"] = 2

    data.loc[data["statistics"] == "sigma", "statistics"] = 0
    data.loc[data["statistics"] == "mu", "statistics"] = 1
    data.loc[data["statistics"] == "unknown", "statistics"] = 2

    data.loc[data["databases"] == "nee", "databases"] = 0
    data.loc[data["databases"] == "ja", "databases"] = 1
    data.loc[data["databases"] == "unknown", "databases"] = 2

    data.loc[data["gender"] == "male", "gender"] = 0
    data.loc[data["gender"] == "female", "gender"] = 1
    data.loc[data["gender"] == "unknown", "gender"] = 2

    data.loc[data["stand_up"] == "no", "stand_up"] = 0
    data.loc[data["stand_up"] == "yes", "stand_up"] = 1
    data.loc[data["stand_up"] == "unknown", "stand_up"] = 2

    for col in ["number_of_neighbors", "deserve_money", "random_number", "stress_level"]:
        data[col] = data[col].apply(lambda x: str(x))
        data[col] = data[col].apply(lambda x: x.replace(',', '.'))
        data[col] = data[col].apply(pd.to_numeric, errors='coerce', downcast='integer')

    data["stress_level"][data["stress_level"] > 100] = 100
    data["stress_level"][data["stress_level"] < 0] = 0

    return data


odi = data_prep(odi)
odi.to_csv('ODI_2019_clean.csv')
