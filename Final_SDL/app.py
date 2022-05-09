from flask import Flask, render_template, send_file, request
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from io import BytesIO
import pickle
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

model = pickle.load(open('pickle/carsRegFinal.pkl', 'rb'))
sns.set_style("darkgrid", {"axes.facecolor": ".9"})
matplotlib.use('Agg')
df2 = pd.read_csv("vehiclesFinal.csv")
df = df2.sample(50000)
app = Flask(__name__)


@app.route("/")
def hello():
    return render_template('home.html')


@app.route("/home")
def home():
    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('About.html')


@app.route("/distributionDist")
def distributionDist():
    fig, ax = plt.subplots()
    fig.set_size_inches(5, 5)
    sns.distplot(df.price, color='brown', bins=30)
    plt.tight_layout()
    plt.title("Distribution of price")

    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route("/distribution")
def distribution():
    return render_template('Distribution.html')


@app.route("/price_vs_fuel")
def price_vs_fuel():
    return render_template('Price_Vs_Fuel.html')


@app.route("/price_vs_fuel_img")
def price_vs_fuel_img():
    fig, axs = plt.subplots(nrows=3)
    fig.set_size_inches(15, 40)
    sns.barplot(x='fuel', y='price', data=df, palette="twilight_shifted", ax=axs[0])
    sa = sns.barplot(x='fuel', y='price', hue='condition', data=df, palette="cubehelix", ax=axs[1])
    sns.barplot(x='cylinders', y='price', data=df, palette="viridis", ax=axs[2])
    for i in range(2):
        axs[i].tick_params(axis='x', labelsize=20)
    for i in range(3):
        axs[i].tick_params(axis='y', labelsize=20)
    axs[2].tick_params(axis='x', labelsize=15)
    axs[0].set_xlabel('Fuel', fontsize=25)
    axs[1].set_xlabel('Fuel', fontsize=25)
    axs[2].set_xlabel('Cylinders', fontsize=25)
    for i in range(3):
        axs[i].set_ylabel('Price', fontsize=25)
    plt.setp(sa.get_legend().get_texts(), fontsize='22')  # for legend text
    plt.setp(sa.get_legend().get_title(), fontsize='32')  # for legend title
    axs[0].set_title('Price Vs Fuel', fontsize=30)
    axs[1].set_title('Price Vs Fuel with condition', fontsize=30)
    axs[2].set_title('Price Vs No. of Cylinders', fontsize=30)
    plt.tight_layout(10)
    canvas = FigureCanvas(fig)
    img2 = BytesIO()
    fig.savefig(img2)
    img2.seek(0)

    return send_file(img2, mimetype='image/png')


@app.route("/price_vs_condition")
def price_vs_condition():
    return render_template('Price_Vs_Condition.html')


@app.route("/price_vs_condition_img")
def price_vs_condition_img():
    fig, axs = plt.subplots(nrows=2)
    fig.set_size_inches(15, 25)
    sns.barplot(x='condition', y='price', data=df, palette="mako", ax=axs[0])
    sns.barplot(x='title_status', y='price', data=df, palette="RdYlBu", ax=axs[1])
    for i in range(2):
        axs[i].tick_params(axis='x', labelsize=20)
    for i in range(2):
        axs[i].tick_params(axis='y', labelsize=20)

    axs[0].set_xlabel('Condition', fontsize=25)
    axs[1].set_xlabel('Title_status', fontsize=25)
    for i in range(2):
        axs[i].set_ylabel('Price', fontsize=25)
    axs[0].set_title('Price Vs Condition', fontsize=30)
    axs[1].set_title('Price Vs Title_Status', fontsize=30)
    plt.tight_layout(10)
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route("/price_vs_transmission")
def price_vs_transmission():
    return render_template('Price_Vs_Transmission.html')


@app.route("/price_vs_transmission_img")
def price_vs_transmission_img():
    fig, axs = plt.subplots(nrows=6)
    fig.set_size_inches(30, 150)
    sns.barplot(x='transmission', y='price', data=df, palette="CMRmap", ax=axs[0])
    sa = sns.barplot(x='transmission', y='price', hue='drive', data=df, palette="inferno", ax=axs[1])
    sns.barplot(x='drive', y='price', data=df, palette="afmhot", ax=axs[2])
    sa1 = sns.barplot(x='drive', y='price', hue='size', data=df, palette="RdYlBu", ax=axs[3])
    sns.barplot(x='size', y='price', data=df, palette="copper", ax=axs[4])
    sa2 = sns.barplot(x='size', y='price', hue='drive', data=df, palette="magma", ax=axs[5])
    for i in range(6):
        axs[i].tick_params(axis='x', labelsize=40)
    for i in range(6):
        axs[i].tick_params(axis='y', labelsize=40)

    axs[0].set_xlabel('Transmission', fontsize=50)
    axs[1].set_xlabel('Transmission', fontsize=50)
    axs[2].set_xlabel('Drive', fontsize=50)
    axs[3].set_xlabel('Drive', fontsize=50)
    axs[4].set_xlabel('Size', fontsize=50)
    axs[5].set_xlabel('Size', fontsize=50)
    for i in range(6):
        axs[i].set_ylabel('Price', fontsize=50)
    plt.setp(sa.get_legend().get_texts(), fontsize='45')  # for legend text
    plt.setp(sa.get_legend().get_title(), fontsize='55')  # for legend title
    plt.setp(sa1.get_legend().get_texts(), fontsize='45')  # for legend text
    plt.setp(sa1.get_legend().get_title(), fontsize='55')  # for legend title
    plt.setp(sa2.get_legend().get_texts(), fontsize='45')  # for legend text
    plt.setp(sa2.get_legend().get_title(), fontsize='55')  # for legend title
    axs[0].set_title('Price Vs Transmission', fontsize=70)
    axs[1].set_title('Price Vs Transmission with Drive', fontsize=70)
    axs[2].set_title('Price Vs Drive', fontsize=70)
    axs[3].set_title('Price Vs Drive with size', fontsize=70)
    axs[4].set_title('Price Vs size', fontsize=70)
    axs[5].set_title('Price Vs size with drive', fontsize=70)
    canvas = FigureCanvas(fig)
    plt.tight_layout(10)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route("/price_vs_manufacturer")
def price_vs_manufacturer():
    return render_template('Price_Vs_Manufacturer.html')


@app.route("/price_vs_manufacturer_img")
def price_vs_manufacturer_img():
    fig, axs = plt.subplots(nrows=2)
    fig.set_size_inches(20, 30)
    chart = sns.barplot(x='manufacturer', y='price', data=df, palette="mako", ax=axs[0])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)

    sns.barplot(x='type', y='price', data=df, palette="afmhot", ax=axs[1])
    for i in range(2):
        axs[i].tick_params(axis='x', labelsize=20)
    for i in range(2):
        axs[i].tick_params(axis='y', labelsize=20)

    axs[0].set_xlabel('Manufacturer', fontsize=25)
    axs[1].set_xlabel('Type', fontsize=25)
    for i in range(2):
        axs[i].set_ylabel('Price', fontsize=25)
    axs[0].set_title('Price Vs Manufacturer', fontsize=30)
    axs[1].set_title('Price Vs Type', fontsize=30)
    plt.tight_layout(10)
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route("/price_vs_year")
def price_vs_year():
    return render_template('Price_Vs_Year.html')


@app.route("/price_vs_year_img")
def price_vs_year_img():
    df['year'] = df['year'].astype('int64')
    py = df.groupby('year')['price'].mean().reset_index()
    year = py.iloc[:, 0]
    price = py.iloc[:, 1]
    fig, ax = plt.subplots()
    fig.set_size_inches(15, 10)
    chart = sns.barplot(x=year, y=price, palette="rainbow")
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    ax.set_xlabel('Year', fontsize=20)
    ax.set_ylabel('Price', fontsize=20)
    ax.set_title('Price Vs Year of Manufacture', fontsize=30)
    ax.tick_params(axis='x', labelsize=15)
    ax.tick_params(axis='y', labelsize=15)
    canvas = FigureCanvas(fig)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route("/number_vs_condition")
def number_vs_condition():
    return render_template('Number_Vs_Condition.html')


@app.route("/number_vs_condition_img")
def number_vs_condition_img():
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(25, 30)
    (uniqueC, countsC) = np.unique(df['condition'], return_counts=True)
    sns.barplot(uniqueC, countsC, palette="inferno", ax=axs[0][0])
    colors = ['yellowgreen', 'blue', 'lightskyblue', 'red', 'orange', 'pink']
    patches, texts = axs[0][1].pie(countsC, colors=colors, startangle=90)
    axs[0][1].legend(patches, uniqueC, loc='best', fontsize='22')
    (uniqueT, countsT) = np.unique(df['title_status'], return_counts=True)
    sns.barplot(uniqueT, countsT, palette="rainbow", ax=axs[1][0])
    colors = ['green', 'blue', 'lightskyblue', 'yellowgreen', 'orange', 'red']
    patches, texts = axs[1][1].pie(countsT, colors=colors, startangle=90)
    axs[1][1].legend(patches, uniqueT, loc='best', fontsize='22')

    for i in range(2):
        axs[i][0].tick_params(axis='x', labelsize=20)
    for i in range(2):
        axs[i][0].tick_params(axis='y', labelsize=20)

    axs[0][0].set_xlabel('Condition', fontsize=25)
    axs[1][0].set_xlabel('Title-Status', fontsize=25)
    for i in range(2):
        axs[i][0].set_ylabel('No. of Cars', fontsize=25)

    axs[0][0].set_title('No. of Cars Vs Condition', fontsize=30)
    axs[1][0].set_title('No. of Cars Vs Title-Status', fontsize=30)
    axs[0][1].set_title('Conditions', fontsize=30)
    axs[1][1].set_title('Title-Status', fontsize=30)
    plt.tight_layout(10)

    canvas = FigureCanvas(fig)
    plt.tight_layout(10)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route("/number_vs_fuel")
def number_vs_fuel():
    return render_template('Number_Vs_Fuel.html')


@app.route("/number_vs_fuel_img")
def number_vs_fuel_img():
    fig, axs = plt.subplots(nrows=2, ncols=2)
    fig.set_size_inches(25, 30)
    (uniqueF, countsF) = np.unique(df['fuel'], return_counts=True)
    sns.barplot(uniqueF, countsF, palette="RdPu", ax=axs[0][0])
    axs[0][1].pie(countsF, labels=uniqueF, autopct="%.2f%%", textprops={'fontsize': 25})
    (uniqueC, countsC) = np.unique(df['cylinders'], return_counts=True)
    chart = sns.barplot(uniqueC, countsC, palette="mako", ax=axs[1][0])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    (uniqueC, countsC) = np.unique(df['cylinders'], return_counts=True)
    colors = ['yellowgreen', 'gold', 'lightskyblue', 'red', 'blue', 'orange', 'pink']
    patches, texts = axs[1][1].pie(countsC, colors=colors, startangle=90, textprops={'fontsize': 30})
    plt.legend(patches, uniqueC, loc='best', fontsize='22')

    for i in range(2):
        axs[i][0].tick_params(axis='x', labelsize=20)
    for i in range(2):
        axs[i][0].tick_params(axis='y', labelsize=20)
    axs[0][0].set_xlabel('Fuel', fontsize=25)
    axs[1][0].set_xlabel('No. of Cylinders', fontsize=25)
    for i in range(2):
        axs[i][0].set_ylabel('No. of Cars', fontsize=25)

    axs[0][0].set_title('No. of Cars Vs Fuel', fontsize=30)
    axs[1][0].set_title('No. of Cars Vs No. of Cylinders', fontsize=30)
    axs[0][1].set_title('Fuel', fontsize=30)
    axs[1][1].set_title('Number of Cylinders', fontsize=30)
    plt.tight_layout(10)
    canvas = FigureCanvas(fig)
    plt.tight_layout(10)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route("/number_vs_transmission")
def number_vs_transmission():
    return render_template('Number_Vs_Transmission.html')


@app.route("/number_vs_transmission_img")
def number_vs_transmission_img():
    fig, axs = plt.subplots(nrows=3, ncols=2)
    fig.set_size_inches(25, 38)
    (uniqueT, countsT) = np.unique(df['transmission'], return_counts=True)
    sns.barplot(uniqueT, countsT, palette="afmhot", ax=axs[0][0])
    axs[0][1].pie(countsT, labels=uniqueT, autopct="%.2f%%", textprops={'fontsize': 25})
    (uniqueD, countsD) = np.unique(df['drive'], return_counts=True)
    chart = sns.barplot(uniqueD, countsD, palette="magma", ax=axs[1][0])
    axs[1][1].pie(countsD, labels=uniqueD, autopct="%.2f%%", textprops={'fontsize': 25})
    (uniqueS, countsS) = np.unique(df['size'], return_counts=True)
    sns.barplot(uniqueS, countsS, palette="cubehelix", ax=axs[2][0])
    axs[2][1].pie(countsS, labels=uniqueS, autopct="%.2f%%", textprops={'fontsize': 25})

    for i in range(3):
        axs[i][0].tick_params(axis='x', labelsize=20)
    for i in range(3):
        axs[i][0].tick_params(axis='y', labelsize=20)
    axs[0][0].set_xlabel('Transmission', fontsize=25)
    axs[1][0].set_xlabel('Drive', fontsize=25)
    axs[2][0].set_xlabel('Size', fontsize=25)
    for i in range(2):
        axs[i][0].set_ylabel('No. of Cars', fontsize=25)

    axs[0][0].set_title('No. of Cars Vs Transmission', fontsize=30)
    axs[1][0].set_title('No. of Cars Vs Drive', fontsize=30)
    axs[2][0].set_title('No. of Cars Vs Size', fontsize=30)
    axs[0][1].set_title('Transmission', fontsize=30)
    axs[1][1].set_title('Drive', fontsize=30)
    axs[2][1].set_title('Size', fontsize=30)
    plt.tight_layout(10)
    canvas = FigureCanvas(fig)
    plt.tight_layout(10)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route("/number_vs_year")
def number_vs_year():
    return render_template('Number_Vs_Year.html')


@app.route("/number_vs_year_img")
def number_vs_year_img():
    df['year'] = df['year'].astype('int64')
    df2['year'] = df2['year'].astype('int64')
    fig, axs = plt.subplots(nrows=2)
    fig.set_size_inches(20, 20)
    (uniqueY, countsY) = np.unique(df['year'], return_counts=True)
    chart = sns.barplot(uniqueY, countsY, palette="cubehelix", ax=axs[0])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    (uniqueY, countsY) = np.unique(df2['year'], return_counts=True)
    sns.lineplot(x='year', y='price', data=df2, color="hotpink", marker='o', ax=axs[1])
    chart1 = sns.lineplot(uniqueY, countsY, color="purple", marker='o', ax=axs[1])
    plt.xticks(uniqueY)
    chart1.set_xticklabels(chart.get_xticklabels(), rotation=90)

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='hotpink', label='Price')
    pink_patch = mpatches.Patch(color='purple', label='No. of cars')
    axs[1].legend(handles=[red_patch, pink_patch], fontsize='22', loc='best')
    for i in range(2):
        axs[i].tick_params(axis='x', labelsize=20)
    for i in range(2):
        axs[i].tick_params(axis='y', labelsize=20)
    axs[0].set_xlabel('Year', fontsize=25)
    axs[1].set_xlabel('Year', fontsize=25)
    axs[0].set_ylabel('No. of Cars', fontsize=25)
    axs[0].set_title('No. of Cars Vs Year of Manufactur', fontsize=30)
    plt.tight_layout(5)

    canvas = FigureCanvas(fig)
    plt.tight_layout(10)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route("/number_vs_manufacturer")
def number_vs_manufacturer():
    return render_template('Number_Vs_Manufacturer.html')


@app.route("/number_vs_manufacturer_img")
def number_vs_manufacturer_img():
    df2 = pd.read_csv("vehiclesFinal.csv")
    fig, axs = plt.subplots(nrows=4)
    fig.set_size_inches(20, 50)
    (uniqueM, countsM) = np.unique(df['manufacturer'], return_counts=True)
    chart = sns.barplot(uniqueM, countsM, palette="copper_r", ax=axs[0])
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    chart1 = sns.lineplot(x='manufacturer', y='price', data=df2, color="blue", marker='o', ax=axs[1])
    (uniqueM, countsM) = np.unique(df2['manufacturer'], return_counts=True)
    chart1 = sns.lineplot(uniqueM, countsM, color="orange", marker='o', ax=axs[1])
    chart1.set_xticklabels(chart.get_xticklabels(), rotation=90)
    (uniqueT, countsT) = np.unique(df['type'], return_counts=True)
    sns.barplot(uniqueT, countsT, palette="viridis", ax=axs[2])
    (uniqueT, countsT) = np.unique(df2['type'], return_counts=True)
    chart3 = sns.lineplot(x='type', y='price', data=df2, color="green", marker='o')
    chart3 = sns.lineplot(uniqueT, countsT, color="red", marker='o')
    chart3.set_xticklabels(chart.get_xticklabels(), rotation=90)

    import matplotlib.patches as mpatches
    red_patch = mpatches.Patch(color='blue', label='Price')
    pink_patch = mpatches.Patch(color='orange', label='No. of cars')
    axs[1].legend(handles=[red_patch, pink_patch], fontsize='30', loc='best')

    red_patch1 = mpatches.Patch(color='green', label='Price')
    pink_patch1 = mpatches.Patch(color='red', label='No. of cars')
    axs[3].legend(handles=[red_patch1, pink_patch1], fontsize='30', loc='best')

    for i in range(4):
        axs[i].tick_params(axis='x', labelsize=20)
    for i in range(4):
        axs[i].tick_params(axis='y', labelsize=20)

    axs[0].set_xlabel('manufacturer', fontsize=25)
    axs[2].set_xlabel('Type', fontsize=25)
    axs[1].set_xlabel('manufacturer', fontsize=25)
    axs[3].set_xlabel('Type', fontsize=25)

    for i in range(0, 4, 2):
        axs[i].set_ylabel('No. of Cars', fontsize=25)

    axs[0].set_title('No. of Cars Vs Manufacturer', fontsize=30)
    axs[2].set_title('No. of Cars Vs Type', fontsize=30)
    plt.tight_layout(5)

    canvas = FigureCanvas(fig)
    plt.tight_layout(10)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


@app.route("/prediction")
def prediction():
    return render_template('Prediction.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data1 = request.form['Year']
        data2 = request.form['Manufacturer']
        data3 = request.form['condition']
        data4 = request.form['Cylinders']
        data5 = request.form['Fuel']
        data6 = request.form['Odometer']
        data7 = request.form['Title-Status']
        data8 = request.form['Transmission']
        data9 = request.form['Drive']
        data10 = request.form['Size']
        data11 = request.form['Type']
        data12 = request.form['Paint_Color']
        data13 = request.form['Model']

        # converting to lower case
        data13 = data13.lower()
        dataset2 = pd.read_csv("vehiclesFinal.csv")
        modelL = dataset2.iloc[:, 4]
        modelA = np.array(modelL)
        flag = False
        for i in modelA:
            if (data13 == i):
                flag = True
                break
        if flag == False:
            data13 = 'other'

        li = [data1, data2, data13, data3, data4, data5, data6, data7, data8, data9, data10, data11, data12]
        li2 = []
        for i in li:
            if type(i) == str:
                if i != "SUV":
                    j = i.lower()
                    li2.append(j)
                else:
                    li2.append(i)
            else:
                li2.append(i)
        arr = np.array(li2)
        arr = arr.reshape(1, -1)
        # encoding the inputdata

        input_df = pd.DataFrame(data=arr, index=["0"],
                                columns=["year", "manufacturer", 'model', "condition", "cylinders", "fuel", "odometer",
                                         "title_status", "transmission", "drive", "size", 'type', 'paint_color'])

        pkl_file = open('Pickle/manu.pkl', 'rb')
        le_manu = pickle.load(pkl_file)
        pkl_file.close()
        input_df['manufacturer'] = le_manu.transform(input_df['manufacturer'])

        pkl9_file = open('Pickle/model.pkl', 'rb')
        le_mod = pickle.load(pkl9_file)
        pkl9_file.close()
        input_df['model'] = le_mod.transform(input_df['model'])

        pkl1_file = open('Pickle/cond.pkl', 'rb')
        le_cond = pickle.load(pkl1_file)
        pkl1_file.close()
        input_df['condition'] = le_cond.transform(input_df['condition'])

        pkl2_file = open('Pickle/cyl.pkl', 'rb')
        le_cyl = pickle.load(pkl2_file)
        pkl2_file.close()
        input_df['cylinders'] = le_cyl.transform(input_df['cylinders'])

        pkl3_file = open('Pickle/fuel.pkl', 'rb')
        le_fuel = pickle.load(pkl3_file)
        pkl3_file.close()
        input_df['fuel'] = le_fuel.transform(input_df['fuel'])

        pkl4_file = open('Pickle/tit.pkl', 'rb')
        le_tit = pickle.load(pkl4_file)
        pkl4_file.close()
        input_df['title_status'] = le_tit.transform(input_df['title_status'])

        pkl5_file = open('Pickle/trans.pkl', 'rb')
        le_trans = pickle.load(pkl5_file)
        pkl5_file.close()
        input_df['transmission'] = le_trans.transform(input_df['transmission'])

        pkl6_file = open('Pickle/drive.pkl', 'rb')
        le_drive = pickle.load(pkl6_file)
        pkl6_file.close()
        input_df['drive'] = le_drive.transform(input_df['drive'])

        pkl7_file = open('Pickle/size.pkl', 'rb')
        le_size = pickle.load(pkl7_file)
        pkl7_file.close()
        input_df['size'] = le_size.transform(input_df['size'])

        pkl7_file = open('Pickle/type.pkl', 'rb')
        le_type = pickle.load(pkl7_file)
        pkl7_file.close()
        input_df['type'] = le_type.transform(input_df['type'])

        pkl8_file = open('Pickle/paint.pkl', 'rb')
        le_paint = pickle.load(pkl8_file)
        pkl8_file.close()
        input_df['paint_color'] = le_paint.transform(input_df['paint_color'])

        pkl0_file = open('Pickle/year.pkl', 'rb')
        le_year = pickle.load(pkl0_file)
        pkl0_file.close()
        arr = np.array(input_df['year'])
        input_df['year'] = le_year.transform(arr.reshape(-1, 1))

        pkl01_file = open('Pickle/odo.pkl', 'rb')
        le_year = pickle.load(pkl01_file)
        pkl01_file.close()
        arr = np.array(input_df['odometer'])
        input_df['odometer'] = le_year.transform(arr.reshape(-1, 1))

        Xi = input_df.iloc[:, :]
        pred = model.predict(Xi)
        pred = pred[0]
        return render_template("PredictionR.html", data=pred)
    except ValueError:
        pred = "Enter Correct Values"
        return render_template("ValueError.html", data=pred)


@app.route("/RandomForest")
def RandomForest():
    import seaborn as sns
    pkl10_file = open('Pickle/imp2.pkl', 'rb')
    importances = pickle.load(pkl10_file)
    pkl10_file.close()
    pkl11_file = open('Pickle/df_check.pkl', 'rb')
    df_check = pickle.load(pkl11_file)
    pkl11_file.close()
    df2 = pd.read_csv("vehiclesFinal.csv")
    dataset = df2.copy()
    dataset.drop(['id', 'region', 'lat', 'long'], axis=1, inplace=True)
    features = dataset.columns
    fig, axs = plt.subplots(nrows=2)
    fig.set_size_inches(15, 20)
    df_check.plot(kind='bar', ax=axs[1])
    axs[1].set_title('Performance of Random Forest', fontsize=30)
    axs[1].set_ylabel('Mean Squared Log Error', fontsize=25)
    x_values = list(range(len(importances)))
    charts = sns.barplot(x_values, importances, palette="rainbow", ax=axs[0])
    charts.set_xticklabels(features, rotation=90)
    axs[0].set_ylabel('Importance', fontsize=25);
    axs[0].set_xlabel('Variable/Features', fontsize=25);
    axs[0].set_title('Random Forest Variables Importance', fontsize=25)
    for i in range(2):
        axs[i].tick_params(axis='x', labelsize=20)
    for i in range(2):
        axs[i].tick_params(axis='y', labelsize=20)
    plt.tight_layout(5)
    img = BytesIO()
    fig.savefig(img)
    img.seek(0)

    return send_file(img, mimetype='image/png')


if __name__ == "__main_":
    app.run(threading=False, debug=True)
