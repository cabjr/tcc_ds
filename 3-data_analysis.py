import pandas as pd 
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt  
import seaborn as sns
sns.set()

data_basics = pd.read_csv('final_result_youtube_stats.csv')
def correlation_heatmap(train):
    correlations = train.corr()

    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(correlations, vmax=1.0, center=0, fmt='.2f',
                square=True, linewidths=.5, annot=True, cbar_kws={"shrink": .70})
    plt.savefig("correlationMap.png")
    plt.show()
    

data_basics["Y"] = data_basics.apply(lambda x: 1 if x["revenue"] > 2*x["budget"] else 0 ,axis=1)


data = data_basics['Y'].value_counts().to_dict()
print(data)
courses = ['Baixo retorno', 'Sucesso']
values = list(data.values()) 
   
fig = plt.figure(figsize = (10, 5)) 
  
plt.bar(courses, values, color ='maroon', width = 0.4) 
  
plt.xlabel("Classe") 
plt.ylabel("Número de produções cinematográficas") 
plt.title("Distribuição das classes no dataset")
plt.savefig("classesDist.png")
#plt.show() 

print("Count isAdult: ",data_basics['isAdult'].value_counts().to_dict())
print("Count year: ",data_basics['startYear'].value_counts().to_dict())
data = data_basics['startYear'].value_counts().to_dict()
values = list(data.values()) 
years = list(data.keys())
fig = plt.figure(figsize = (10, 5)) 
plt.bar(years, values, color ='maroon',  
        width = 0.4) 
plt.xlabel("Ano de produção") 
plt.ylabel("Número de produções cinematográficas") 
plt.title("Distribuição do ano de produção das obras")
plt.savefig("anoDist.png")
#plt.show()

genres = []
for val in data_basics['genres'].values:
    if ',' in val:
        for subval in val.split(','):
            genres.append(subval)
    else:
        genres.append(val)

data = pd.DataFrame({'Gêneros de filmes': genres})
sns.countplot(data['Gêneros de filmes'], color='gray')
plt.ylabel("Quantidade") 
plt.title("Distribuição dos gêneros de filmes no dataset")
plt.show()

correlation_heatmap(data_basics[['isAdult', 'startYear', 'runtimeMinutes', 'staffExperience', 'writersExperience', 'directorsExperience', 'budget', 'revenue', 'viewCount', 'likeCount', 'dislikeCount', 'commentCount']])