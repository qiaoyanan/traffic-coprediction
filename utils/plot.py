#coding=utf-8
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# single_perediction = {'RMSE':24.0020, 'MAE':11.0065}
# co_prediction = {'RMSE':23.4997, 'MAE':10.9557}
# single_data = pd.DataFrame(list(single_perediction.items()), columns=['category', 'value'])
# co_data = pd.DataFrame(list(co_prediction.items()), columns=['category', 'value'])
# plot_co = pd.concat([single_data, co_data])
# plot_co['type'] = ['单独预测']*2 + ['联合预测']*2
# sns.barplot(x='category', y='value', data=plot_co, hue='type')
#
graph_ablation = pd.read_csv('E:\\北航\\科研\\毕业\\毕业论文\\材料\\graph_ablation.csv', header=0)
#sns.barplot(x='category', y='value', hue='model', data=graph_ablation, hue_order=['D-Model','C-Model','M-Model','D+C-Model', 'D+M-Model', 'M+C-Model','Model'])
#plt.ylim(23,24)
#
sns.set_style(rc={'font.family':'Times New Roman'})
#hatches = ['/', '-', 'x', '|', '//', '.', '\\']
# bar = sns.barplot(x='category', y='value', hue='model', data=graph_ablation[graph_ablation['category']=='MAE'], hue_order=['D-Model','C-Model','M-Model','D+C-Model', 'D+M-Model', 'M+C-Model','Model'])
# for i, thisbar in enumerate(bar.patches):
#      thisbar.set_hatch(hatches[i])
# #     thisbar.set_hatch_color('#000set_hatch_color')
# plt.ylim(22,24)
# plt.show()

hatches = ['\\', 'x', '//', '.']
embedding_ablation = pd.read_csv('E:\\北航\\科研\\毕业\\毕业论文\\材料\\embedding_ablation.csv', header=0)
bar = sns.barplot(x='type', y='value', hue='model', data=embedding_ablation[embedding_ablation['category']=='RMSE'])
for i, thisbar in enumerate(bar.patches):
    if i>=4:
        i = i-4
    thisbar.set_hatch(hatches[i])
plt.ylim()

