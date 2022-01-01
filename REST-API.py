import numpy as np
import pandas as pd

from flask import Flask, request, Response
from keras.models import load_model
from tensorflow.keras.utils import normalize

# Commands to Run in Hosts:
# slowhttptest -B -u http://10.0.0.5 -c 5000 -i 10 -p 5 -r 100 -l 20
# slowhttptest -H -u http://10.0.0.5 -c 5000 -i 10 -p 5 -r 100 -l 15

classifier = load_model('IDS2017-RT2.h5')
app = Flask(__name__)

@app.route('/rest-api/classify/', methods=['GET', 'POST'])
def classify():
    file = open('Real-Time Validation.csv', 'a')
    dict_ = request.json
    # print(dict_)
    dict_to_use = {"min_fpkl": dict_['f'][4]['min'], "max_fpkl": dict_['f'][4]['max'],
                   "min_bpktl": dict_['f'][5]['min'], "max_bpktl": dict_['f'][5]['max'],
                   "min_fiat": dict_['f'][6]['min'], "max_fiat": dict_['f'][6]['max'],
                   "min_biat": dict_['f'][7]['min'],
                   "max_biat": dict_['f'][7]['max'], "duration": dict_['f'][8]['value'],
                   "min_active": dict_['f'][9]['min'], "max_active": dict_['f'][9]['max'],
                   "bpsh_cnt": dict_['f'][16]['value']}

    for key, value in dict_to_use.items():
        file.write(str(value) + ',')

    df = pd.DataFrame(dict_to_use, index=[0])
    df_ = normalize(df)
    to_classify = df_.to_numpy()
    to_classify = to_classify.reshape((1, 12, 1))
    result = [0, 1]

    attack_pred = classifier.predict(to_classify)
    max_index = np.argmax(attack_pred[0])
    if result[max_index]:
        file.write(str(result[max_index]))
        file.write('\n')
        file.close()
        return Response("", status=200)
    else:
        file.write(str(result[max_index]))
        file.write('\n')
        file.close()
        return Response("", status=201)


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
