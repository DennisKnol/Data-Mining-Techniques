import pandas as pd
import matplotlib.pyplot as plt

ODI = pd.read_csv(
    "/Users/DennisK/PycharmProjects/Data-Mining-Techniques/Assignment_1/Task_1/ODI-2019-csv.csv",
    sep=';'
)

ODI.iloc[:, 10].value_counts().plot(kind='bar', title='did you stand up?')
plt.show()


def program_question(data):
    # improvements on text program question
    prog = data.str.lower()
    prog_cat = prog.astype('category')
    cat_list = prog_cat.cat.categories

    def find_program(program, prog_list):
        for i, _ in enumerate(program):
            if _ == True:
                prog_list.append(prog_cat.cat.categories[i])
        return prog_list

    # Artificial Intelligence
    AI = prog_cat.cat.categories.str.contains('ai')
    ArtInt = prog_cat.cat.categories.str.contains('artificial intelligence')
    AI_list = []
    AI_list = find_program(ArtInt, AI_list)
    AI_list = find_program(AI, AI_list)
    cat_list = [item for item in cat_list if item not in AI_list]

    # Business Analytics
    BA = prog_cat.cat.categories.str.contains('ba')
    Busi_anal = prog_cat.cat.categories.str.contains('business analytics')
    BA_list = []
    BA_list = find_program(BA, BA_list)
    BA_list = find_program(Busi_anal, BA_list)
    cat_list = [item for item in cat_list if item not in BA_list]

    # Bioinformatics
    BI = prog_cat.cat.categories.str.contains('bioinf')
    BI_list = []
    BI_list = find_program(BI, BI_list)
    cat_list = [item for item in cat_list if item not in BI_list]

    # Computer Science
    Comp_Sc = prog_cat.cat.categories.str.contains('computer')
    CS = prog_cat.cat.categories.str.match('cs')
    CS_list = []
    CS_list = find_program(Comp_Sc, CS_list)
    CS_list = find_program(CS, CS_list)
    cat_list = [item for item in cat_list if item not in CS_list]

    # Computational Science
    Compu_Sc = prog_cat.cat.categories.str.contains('computational')
    CLS = prog_cat.cat.categories.str.contains('cls')
    CLS_list = []
    CLS_list = find_program(Compu_Sc, CLS_list)
    CLS_list = find_program(CLS, CLS_list)
    cat_list = [item for item in cat_list if item not in CLS_list]

    # Data Science
    DS = prog_cat.cat.categories.str.contains('data science')
    DS_dir = prog_cat.cat.categories.str.match('ds')
    DS_list = []
    DS_list = find_program(DS, DS_list)
    DS_list = find_program(DS_dir, DS_list)
    cat_list = [item for item in cat_list if item not in DS_list]

    # Business administration
    BAM = prog_cat.cat.categories.str.contains('business administration')
    BAM_list = []
    BAM_list = find_program(BAM, BAM_list)
    cat_list = [item for item in cat_list if item not in BAM_list]

    # Econometrics
    EC = prog_cat.cat.categories.str.contains('econometrics')
    EC_list = []
    EC_list = find_program(EC, EC_list)
    cat_list = [item for item in cat_list if item not in EC_list]
    return cat_list

print(program_question(ODI.iloc[:, 1]))
