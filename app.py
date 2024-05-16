#########################################################
#                                                       #
#    Change the file dir in line 361,403 as your dir    #
#                                                       #
#########################################################

import shutil
from flask import Flask, render_template
import matplotlib.pyplot as plt
from io import BytesIO
import os
import base64
from data import STN
from flask import request

app = Flask(__name__, template_folder="templates")

iptPriceData = []
iptStatusData = []
Table = []

status_len = [4]
match = 0

@app.route('/')
def input_page():
    return render_template('cover.html', Table=Table, iptStatusData=iptStatusData, iptPriceData=iptPriceData)

@app.route('/enter')
def enter():
    return render_template('price_input.html')

@app.route('/backprice')
def backprice():
    return render_template('price_input.html')


@app.route('/ToPrice')
def ToPrice():
    if iptPriceData == []:
        return render_template('price_input.html', Table=Table, iptStatusData=iptStatusData, iptPriceData=iptPriceData)
    else:
        return render_template('price_input_v.html', Table=Table, iptStatusData=iptStatusData, iptPriceData=iptPriceData, match=match)

@app.route('/ToPricev')
def ToPricev():
    return render_template('price_input_v.html', Table=Table, iptStatusData=iptStatusData, iptPriceData=iptPriceData, match=match)

@app.route('/ToPrices')
def ToPrices():
    return render_template('price_input_save.html', Table=Table, iptStatusData=iptStatusData, iptPriceData=iptPriceData)

@app.route('/ToStatus')
def ToStatus():
    return render_template('status_node.html', Table=Table, iptStatusData=iptStatusData, iptPriceData=iptPriceData[-4:], status_len=status_len)

@app.route('/ToStatus_1st')
def ToStatus_1st():
    return render_template('status_node.html', Table=Table, iptStatusData=iptStatusData, iptPriceData=iptPriceData[-4:], status_len=[4])

@app.route('/ToTask')
def ToTask():
    if Table == []:
        return render_template('task_node.html',Table=Table,iptStatusData=iptStatusData,iptPriceData=iptPriceData)
    else:
        return render_template('task_node_v.html',Table=Table,iptStatusData=iptStatusData,iptPriceData=iptPriceData)

@app.route('/address', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'
    file = request.files['file']
    if file.filename == '':
        return 'No selected file'
    if file:
        file.save(f'/tmp/{file.filename}')  # 保存文件到临时目录
        return f'File "{file.filename}" uploaded successfully!'

    return 'Upload failed'

@app.route('/pricesubmit0', methods=['POST'])
def price_data0():
    TouValues = request.form.get('TOU_Values').split('[')
    if TouValues[0].__len__() == 0:
        TouValues = TouValues[1].split(']')
        TouValues = TouValues[0].split(',')
    else:
        TouValues = TouValues[0].split(',')

    PriceValues = request.form.get('price_Values').split('[')
    if PriceValues[0].__len__() == 0:
        PriceValues = PriceValues[1].split(']')
        PriceValues = PriceValues[0].split(',')
    else:
        PriceValues= PriceValues[0].split(',')

    PvValues = request.form.get('PV_Values').split('[')
    if PvValues[0].__len__() == 0:
        PvValues = PvValues[1].split(']')
        PvValues = PvValues[0].split(',')
    else:
        PvValues = PvValues[0].split(',')

    TOU_values = [float(s) for s in TouValues]
    price_values = [float(s) for s in PriceValues]
    PV_values = [float(s) for s in PvValues]

    newdata1 = {'TouValues': TOU_values, 'PriceValues': price_values, 'PvValues': PV_values}
    iptPriceData.insert(0,newdata1)
    print(iptPriceData)

    stn = STN()
    T = stn.T
    if TOU_values.__len__() != T:
        price1 = '分时电价'
    else:
        price1 = ''

    if price_values.__len__() == T:
        price2 = '电网售价'
    else:
        price2 = ''

    if PV_values.__len__() == T:
        price3 = '光伏预测出力'
    else:
        price3 = ''

    if (TOU_values.__len__() == T) and (price_values.__len__() == T) and (PV_values.__len__() == T):
        match = 1
    else:
        match = 0

    return render_template('price_input_save.html',
                           iptPriceData=iptPriceData, price1=price1, price2=price2, price3=price3, match=match)

@app.route('/pricesubmit', methods=['POST'])
def price_data():
    TouValues = request.form.get('TOU_Values').split('[')
    if TouValues[0].__len__() == 0:
        TouValues = TouValues[1].split(']')
        TouValues = TouValues[0].split(',')
    else:
        TouValues = TouValues[0].split(',')

    PriceValues = request.form.get('price_Values').split('[')
    if PriceValues[0].__len__() == 0:
        PriceValues = PriceValues[1].split(']')
        PriceValues = PriceValues[0].split(',')
    else:
        PriceValues= PriceValues[0].split(',')

    PvValues = request.form.get('PV_Values').split('[')
    if PvValues[0].__len__() == 0:
        PvValues = PvValues[1].split(']')
        PvValues = PvValues[0].split(',')
    else:
        PvValues = PvValues[0].split(',')

    TOU_values = [float(s) for s in TouValues]
    price_values = [float(s) for s in PriceValues]
    PV_values = [float(s) for s in PvValues]

    newdata1 = {'TouValues': TOU_values, 'PriceValues': price_values, 'PvValues': PV_values}
    iptPriceData.insert(0,newdata1)
    print(iptPriceData)

    stn = STN()
    T = stn.T
    if TOU_values.__len__() != T:
        price1 = '分时电价'
    else:
        price1 = ''

    if price_values.__len__() == T:
        price2 = '电网售价'
    else:
        price2 = ''

    if PV_values.__len__() == T:
        price3 = '光伏预测出力'
    else:
        price3 = ''

    if (TOU_values.__len__() == T) and (price_values.__len__() == T) and (PV_values.__len__() == T):
        match = 1
    else:
        match = 0

    return render_template('price_input_v.html',
                           iptPriceData=iptPriceData, price1=price1, price2=price2, price3=price3, match=match,)

@app.route('/StatusNode', methods=['POST'])
def StatusNode():
    return render_template('add.html', status_len=status_len)

@app.route('/AddStatus', methods=['POST'])
def status_data():
    Status_Code = request.form.get('StatusCode')
    Status_Name = request.form.get('StatusName')
    Initial_Value = request.form.get('InitialValue')
    Storage_Min = request.form.get('StorageMin')
    Storage_Max = request.form.get('StorageMax')
    newdata2 = {'StatusCode': Status_Code,'StatusName' : Status_Name, 'InitialValue': Initial_Value, 'StorageMin': Storage_Min, 'StorageMax': Storage_Max}
    iptStatusData.append(newdata2)
    print(iptStatusData)
    LEN = iptStatusData.__len__()
    status_len.insert(0, LEN)
    return render_template('status_node.html', iptStatusData=iptStatusData[-4:], status_len=status_len)


@app.route('/delete/<StatusCode>')
def status_delete(StatusCode):
    for status in iptStatusData:
        if status['StatusCode'] == StatusCode:
            iptStatusData.remove(status)
    LEN = iptStatusData.__len__()
    status_len.insert(0, LEN)
    return render_template('status_node.html', iptStatusData=iptStatusData, status_len=status_len)

@app.route('/change/<StatusCode>')
def status_change(StatusCode):
    for status in iptStatusData:
        if status['StatusCode'] == StatusCode:
            return render_template('change.html', status=status, status_len=status_len)
    return render_template('status_node.html')

@app.route('/changed/<StatusCode>', methods=['POST'])
def status_changed(StatusCode):
    for status in iptStatusData:
        if status['StatusCode'] == StatusCode:
            status['StatusCode'] = request.form.get('StatusCode')
            status['StatusName'] = request.form.get('StatusName')
            status['InitialValue'] = request.form.get('InitialValue')
            status['StorageMin'] = request.form.get('StorageMin')
            status['StorageMax'] = request.form.get('StorageMax')

            return render_template('status_node.html', iptStatusData=iptStatusData)

@app.route('/TableData', methods=['POST'])
def TableData():
    table_fields = [
        'TableT10C', 'TableT10G', 'TableT10P',
        'TableT11C', 'TableT11G', 'TableT11P',
        'TableT12C', 'TableT12G', 'TableT12P',
        'TableT20C', 'TableT20G', 'TableT20P',
        'TableT30C', 'TableT30G', 'TableT30P',
        'TableT31C', 'TableT31G', 'TableT31P',
        'TableT32C', 'TableT32G', 'TableT32P'
    ]

    showtable = {field: request.form.get(field) for field in table_fields}
    Table.insert(0,showtable)
    print(Table)

    return render_template('task_node_v.html', Table=Table, showtable=showtable)

@app.route('/GoResult')
def GoResult():
    return render_template('output.html', Table=Table, iptStatusData=iptStatusData, iptPriceData=iptPriceData)


@app.route('/plot/<Table>', methods=['POST'])
def plot(Table):
    x_values = request.form.get('x_values').split(',')
    y_values = request.form.get('y_values').split(',')

    x_values = [float(x) for x in x_values]
    y_values = [float(y) for y in y_values]

    plt.plot(x_values, y_values)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Plot of Y vs X')

    img_data = BytesIO()
    plt.savefig(img_data, format='png')
    img_data.seek(0)
    img_base64 = base64.b64encode(img_data.getvalue()).decode()
    plt.clf()
    return render_template('output.html', img_base64=img_base64)

@app.route('/test')
def test():
    print(iptPriceData)
    print(iptStatusData)
    print(Table)
    PrDt = iptPriceData[0]
    TbDt = Table[0]
    stn = STN()

    tasks = stn.tasks
    states = stn.states
    horizon = stn.horizon

    J = tasks.__len__()  # 任务数
    S = states.__len__()  # 状态数
    T = horizon

    rho_in = stn.rho_in
    rho_out = stn.rho_out
    P = stn.P
    P_j = stn.P_j

    model = stn.model

    P_buy = stn.P_buy
    P_sell = stn.P_sell
    y_Feed = stn.y_Feed
    y_Row = stn.y_Row
    y_Clinker = stn.y_Clinker
    y_Cement = stn.y_Cement
    rhoR = stn.rhoR
    productR = stn.productR
    powerR = stn.powerR
    rhoCa = stn.rhoCa
    productCa = stn.productCa
    powerCa = stn.powerCa
    rhoC = stn.rhoC
    productC = stn.productC
    powerC = stn.powerC
    Calcining_kiln_sets_0 = stn.Calcining_kiln_sets_0
    Calcining_kiln_sets_1 = stn.Calcining_kiln_sets_1
    R_mill_sets_0 = stn.R_mill_sets_0
    R_mill_sets_1 = stn.R_mill_sets_1
    R_mill_sets_2 = stn.R_mill_sets_2
    C_mill_sets_0 = stn.C_mill_sets_0
    C_mill_sets_1 = stn.C_mill_sets_1
    C_mill_sets_2 = stn.C_mill_sets_2

    C_max = [float(stor['StorageMax']) for stor in iptStatusData]
    C_min = [float(stor['StorageMin']) for stor in iptStatusData]
    Initial = [float(stor['InitialValue']) for stor in iptStatusData]


    TOU = PrDt['TouValues']
    price = PrDt['PriceValues']
    PV = PrDt['PvValues']

    table_fields = [
        'TableT10C', 'TableT10G', 'TableT10P',
        'TableT11C', 'TableT11G', 'TableT11P',
        'TableT12C', 'TableT12G', 'TableT12P',
        'TableT20C', 'TableT20G', 'TableT20P',
        'TableT30C', 'TableT30G', 'TableT30P',
        'TableT31C', 'TableT31G', 'TableT31P',
        'TableT32C', 'TableT32G', 'TableT32P'
    ]
    TAB = [value for value in TbDt.values()]

    stn.constraint_var_mill(T, model, productR, powerR, rhoR, R_mill_sets_0, R_mill_sets_1, R_mill_sets_2, P_j, TAB)
    stn.constraint_var_Calcining(T, model, productCa, powerCa, rhoCa, Calcining_kiln_sets_1, TAB)
    stn.constraint_var_Cmill(T, model, productC, powerC, rhoC, C_mill_sets_0, C_mill_sets_1, C_mill_sets_2, TAB)
    stn.state_constraint(T, model, y_Feed, y_Row, y_Clinker, y_Cement, rhoR, productR, rhoCa, productCa, rhoC, productC, Initial)
    stn.power_constraint(T, model, P_buy, PV, P_sell, powerR, powerCa, powerC)
    result = stn.solve(T, model, TOU, price, P_buy, P_sell, y_Feed, y_Row, y_Clinker, y_Cement)
    if type(result) == float:
        print("The result is: ", result)
        stn.export_results(T, P_buy, P_sell, y_Feed, y_Row, y_Clinker, y_Cement, rhoR, productR, rhoCa, productCa, rhoC, productC, powerR, powerC, powerCa)

        shutil.copyfile('optimization_results_Cemment_low.xlsx','C:\\Users\\Edward\\PycharmProjects\\STN\\static\\file\\optimization_results_Cemment_low.xlsx')

        stn.plot_results(T, P_buy, P_sell, TOU, price)
        image_dir = os.path.join(app.static_folder, 'images')
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, 'RG.jpg')
        plt.savefig(image_path)
        plt.clf()

        stn.plot_y_Feed(T, y_Feed)
        image_dir = os.path.join(app.static_folder, 'images')
        image_path = os.path.join(image_dir, 'YFG.jpg')
        plt.savefig(image_path)
        plt.clf()

        stn.plot_y_Row(T, y_Row)
        image_dir = os.path.join(app.static_folder, 'images')
        image_path = os.path.join(image_dir, 'YRG.jpg')
        plt.savefig(image_path)
        plt.clf()

        stn.plot_y_Clinker(T, y_Clinker)
        image_dir = os.path.join(app.static_folder, 'images')
        image_path = os.path.join(image_dir, 'YCKG.jpg')
        plt.savefig(image_path)
        plt.clf()

        stn.plot_y_Cement(T, y_Cement)
        image_dir = os.path.join(app.static_folder, 'images')
        image_path = os.path.join(image_dir, 'YCTG.jpg')
        plt.savefig(image_path)
        plt.clf()

        stn.plot_power_results(T, powerR, powerC, powerCa, TOU)
        image_dir = os.path.join(app.static_folder, 'images')
        image_path = os.path.join(image_dir, 'PG.jpg')
        plt.savefig(image_path)
        plt.clf()

        return render_template('index.html')

    if type(result) == list:
        shutil.copyfile('model_infeasible.lp', 'C:\\Users\\Edward\\PycharmProjects\\STN\\static\\file\\model_infeasible.lp')
        return render_template('err.html')

@app.route('/nextpageA')
def nextpageA():
    return render_template('lastpage.html')

@app.route('/backpageA')
def backpageA():
    return render_template('index.html')

@app.route('/backpage')
def backpage():
    return render_template('lastpageA.html')

@app.route('/firstpage')
def firstpage():
    return render_template('price_input.html')

if __name__ == '__main__':
    app.run(debug=True)
