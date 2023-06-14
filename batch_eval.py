import os, sys, json
import matplotlib.pyplot as plt

MODEL_FOLDER = '/iris/u/davidy02/model'

ACCURACY_FILE = 'results_best.json'
def get_model_accuracy(model_path):
    with open(os.path.join(model_path, ACCURACY_FILE)) as f:
        src = json.load(f)
    return src[0][1]

def filter(model_dic, keyword):
    res = {}
    for model in model_dic:
        # model is a string
        if keyword in model:
            res[model.split(keyword+'_')[1]] = model_dic[model]
    return res

def process_source():
    res = {}
    for model in os.listdir(MODEL_FOLDER):
        try:
            accuracy = get_model_accuracy(os.path.join(MODEL_FOLDER, model))
            res[model] = accuracy
            plt.bar(model, accuracy)
        except Exception as e:
            print('Error with model {}'.format(model), e)

    with open('visualization/model_accuracy.json', 'w') as f:
        json.dump(res, f)
    plt.savefig('visualization/model_accuracy.png')
    plt.clf()
    return res

def plot_keyword(keyword, res, use_hist=False):
    res = filter(res, keyword)
    print(res)
    res = dict(sorted(res.items(), key=lambda x: float(x[0])))
    for model in res:
        if use_hist:
            plt.hist(res[model])
        else:
            plt.bar(model, res[model], color='blue', width=0.5, alpha=0.7)
            plt.text(model, res[model]+0.01, str(round(res[model], 2)), ha='center', va='bottom', fontsize=10)

    plt.xlabel("`"+keyword+"` value")
    plt.ylabel("Accuracy")
    plt.savefig('visualization/{}.png'.format(keyword))
    plt.clf()

res = process_source()
plot_keyword('kl', res)
plot_keyword('chunk', res)

bs = [8, 16, 32]
for b in bs:
    plot_keyword('bs_{}_lr'.format(b), res)